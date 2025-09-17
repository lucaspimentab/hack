#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data import SeqConfig, build_pair_week_matrix, DemandSeqDataset
from .model import DemandTransformer
from .train import TrainConfig, train_model
from .utils_io import wmape


def autotune(args, base_weekly: pd.DataFrame, device: str) -> dict:
    import logging
    logging.info("🔎 AutoTune: iniciando busca rápida de hiperparâmetros.")
    rng = random.Random(123)

    treat_opts = ["zero", "remove", "feature"]
    neg_rates  = [0.05, 0.10, 0.20]
    cls_pos_w  = [3.0, 6.0]
    caps_post  = [1.0, 1.5, 2.0]
    context    = [8, 16]
    d_models   = [48]
    n_layers   = [2]

    combos = []
    for t in treat_opts:
        for nr in neg_rates:
            for cw in cls_pos_w:
                for cap in caps_post:
                    for L in context:
                        for dm in d_models:
                            for ly in n_layers:
                                combos.append((t, nr, cw, cap, L, dm, ly))
    rng.shuffle(combos)
    trials = min(int(args.tune_trials or 12), len(combos))
    combos = combos[:trials]
    logging.info(f"🔬 AutoTune: {trials} combinações serão testadas.")

    best = {"wmape": float("inf"), "cfg": None}

    for i, (treat, neg_r, cw, cap_post, L, dm, ly) in enumerate(combos, start=1):
        logging.info(f"▶️  Trial {i}/{trials}: treat={treat} neg_r={neg_r} cw={cw} cap_post={cap_post} L={L} d_model={dm} layers={ly}")
        weekly = base_weekly.copy()

        if args.only_jan_pairs:
            jan = weekly[weekly["semana_iso"].isin([1,2,3,4,5])]
            jan_pairs = jan.loc[jan["quantidade"]>0, ["pdv","produto"]].drop_duplicates()
            weekly = weekly.merge(jan_pairs.assign(_keep=1), on=["pdv","produto"], how="inner")
        if args.max_pairs and args.max_pairs > 0:
            tot = (weekly.groupby(["pdv","produto"], as_index=False)["quantidade"].sum()
                     .sort_values("quantidade", ascending=False))
            top_pairs = tot.head(int(args.max_pairs))[["pdv","produto"]]
            weekly = weekly.merge(top_pairs.assign(_keep=1), on=["pdv","produto"], how="inner")

        seq_cfg = SeqConfig(context_len=int(L), use_returns_feature=bool(args.use_returns_feature), neg_sample_rate=float(neg_r))
        pairs, M = build_pair_week_matrix(weekly, include_returns=bool(args.use_returns_feature))
        M["quantidade"] = np.maximum(np.nan_to_num(M["quantidade"].astype(float), nan=0.0), 0.0)
        if seq_cfg.use_returns_feature:
            if "devol_flag" not in M.columns: M["devol_flag"] = 0

        ds_train = DemandSeqDataset(M, seq_cfg, split="train")
        ds_val   = DemandSeqDataset(M, seq_cfg, split="val")

        in_dim = (1 + (1 if seq_cfg.use_returns_feature else 0)) + 2
        model = DemandTransformer(in_dim=in_dim, d_model=int(dm), nhead=4, num_layers=int(ly), dropout=0.1,
                                  num_pdv=len(ds_train.pdv2id) or 1, num_prod=len(ds_train.prod2id) or 1, emb_dim=32)

        # prior
        tmp_loader = DataLoader(ds_train, batch_size=4096, shuffle=False, num_workers=0)
        pos, tot = 0.0, 0.0
        with torch.no_grad():
            for b in tmp_loader:
                pos += float((b["y_active"] > 0.5).sum().item()); tot += float(len(b["y_active"]))
        prior = max(1e-6, min(1.0 - 1e-6, (pos / tot) if tot > 0 else 0.01))
        logit = math.log(prior / (1.0 - prior))
        with torch.no_grad():
            if hasattr(model.head_act[-2], 'bias') and model.head_act[-2].bias is not None:
                model.head_act[-2].bias.fill_(logit)

        tcfg = TrainConfig(
            batch_size=min(args.batch_size, 512),
            epochs=int(args.tune_epochs),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            cls_pos_weight=float(cw),
            max_train_batches=int(args.tune_max_train_batches),
            max_val_batches=0,
            num_workers=min(args.num_workers, 2),
            pin_memory=bool(args.pin_memory),
            thr_eval_mode=args.thr_eval_mode,
            early_stopping=bool(args.early_stopping),
            patience=int(args.patience),
        )
        best_trial = train_model(model, ds_train, ds_val, tcfg, device=device)

        val_loader = DataLoader(ds_val, batch_size=tcfg.batch_size, shuffle=False,
                                num_workers=tcfg.num_workers, pin_memory=tcfg.pin_memory)
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device, non_blocking=tcfg.pin_memory)
                pdv_id = batch["pdv_id"].to(device, non_blocking=tcfg.pin_memory)
                prod_id = batch["prod_id"].to(device, non_blocking=tcfg.pin_memory)
                p_active, yhat_log = model(x, pdv_id, prod_id)
                yhat = torch.relu(torch.expm1(yhat_log)).cpu().numpy()
                p = torch.clamp(p_active, 1e-6, 1-1e-6).cpu().numpy()
                y = batch["y"].cpu().numpy()
                ypred = np.where(p >= best_trial["best_thr"], yhat, 0.0)
                preds.append(ypred); trues.append(y)
        Yhat = np.concatenate(preds, axis=0) if preds else np.array([])
        Ytrue = np.concatenate(trues, axis=0) if trues else np.array([])
        score = wmape(Ytrue, Yhat) if len(Ytrue) else float("inf")
        logging.info(f"   ▶ WMAPE trial = {score:.3f}% @ thr={best_trial['best_thr']:.2f}")

        if score < best["wmape"]:
            best["wmape"] = float(score)
            best["cfg"] = {"treat_negatives": treat, "neg_sample_rate": float(neg_r),
                           "cls_pos_weight": float(cw), "cap_mult_pair_q90": float(cap_post),
                           "context_len": int(L), "d_model": int(dm), "num_layers": int(ly)}

    logging.info(f"🏆 AutoTune melhor configuração: {best['cfg']} | WMAPE={best['wmape']:.3f}%")
    return best["cfg"] or {}