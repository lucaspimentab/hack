#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import numpy as np
import torch
from dataclasses import dataclass
from typing import Dict, Optional
from torch import nn
from torch.utils.data import DataLoader

from .utils_io import timer, mem_report, search_threshold


@dataclass
class TrainConfig:
    batch_size: int = 1024
    epochs: int = 8
    lr: float = 2e-3
    weight_decay: float = 1e-4
    cls_pos_weight: float = 3.0
    max_train_batches: Optional[int] = None
    max_val_batches: Optional[int] = None
    num_workers: int = 0
    pin_memory: bool = False
    thr_eval_mode: str = "weighted"   # 'weighted' | 'pos_only' | 'all'
    early_stopping: bool = False
    patience: int = 1
        # === NOVOS CAMPOS ===
    resume: Optional[str] = None               # caminho do checkpoint para retomar
    save_ckpt_dir: Optional[str] = "checkpoints"  # diretório para salvar checkpoints

def train_model(model,
                ds_train,
                ds_val,
                cfg: TrainConfig,
                device: str) -> Dict:
    import os, torch, logging

    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    bce = nn.BCELoss(reduction="none")
    mse = nn.MSELoss(reduction="none")

    def make_loader(ds, shuffle):
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle,
                          num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, drop_last=False)

    train_loader = make_loader(ds_train, True)
    val_loader   = make_loader(ds_val,   False)

    best = {"val_wmape": float("inf"), "state_dict": None, "best_thr": 0.5}
    no_improve = 0

    logging.info(f"🔧 Treino: device={device} | epochs={cfg.epochs} | batch_size={cfg.batch_size} "
                 f"| workers={cfg.num_workers} | pin_memory={cfg.pin_memory}")
    logging.info(f"📊 Samples: train={len(ds_train)} | val={len(ds_val)}")

    # ======== RETOMADA DE CHECKPOINT ========
    start_epoch = 1
    if getattr(cfg, "resume", None):
        if os.path.exists(cfg.resume):
            ckpt = torch.load(cfg.resume, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            opt.load_state_dict(ckpt["optimizer_state"])
            start_epoch = ckpt.get("epoch", 0) + 1
            logging.info(f"▶️ Retomando treino do checkpoint {cfg.resume} a partir da época {start_epoch}")
        else:
            logging.warning(f"⚠️ Arquivo de checkpoint não encontrado: {cfg.resume}")

    for epoch in range(start_epoch, cfg.epochs + 1):
        # ========== Treino ==========
        with timer(f"Treino - época {epoch}"):
            model.train()
            tot_loss = 0.0
            n_batches = 0
            for bi, batch in enumerate(train_loader, start=1):
                x       = batch["x"].to(device, non_blocking=cfg.pin_memory)
                pdv_id  = batch["pdv_id"].to(device, non_blocking=cfg.pin_memory)
                prod_id = batch["prod_id"].to(device, non_blocking=cfg.pin_memory)
                y_active= batch["y_active"].to(device, non_blocking=cfg.pin_memory)
                y_log   = batch["y_log"].to(device, non_blocking=cfg.pin_memory)

                p_active, yhat_log = model(x, pdv_id, prod_id)
                p_active = torch.clamp(p_active, 1e-6, 1 - 1e-6)

                w = torch.where(y_active > 0.5,
                                torch.tensor(cfg.cls_pos_weight, device=device),
                                torch.tensor(1.0, device=device))
                loss_cls = (bce(p_active, y_active) * w).mean()
                mask = (y_active > 0.5).float()
                loss_reg = (mse(yhat_log, y_log) * mask).sum() / (mask.sum() + 1e-6)

                # >>> NOVO: loss para valor esperado E[y] = p * size <<<
                yhat_sz = torch.relu(torch.expm1(yhat_log))
                y_true_cont = batch["y"].to(device)
                y_exp = p_active * yhat_sz
                loss_ev = torch.nn.L1Loss()(y_exp, y_true_cont)

                # Combinação: classificação + regressão (positivos) + valor esperado
                loss = loss_cls + loss_reg + 0.1 * loss_ev

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

                tot_loss += float(loss.detach().cpu().item())
                n_batches += 1

                if bi % 50 == 0:
                    logging.info(f"  Época {epoch} | Lote {bi} | loss={loss.item():.4f}")
                    mem_report(f"epoch {epoch} batch {bi}")

                if cfg.max_train_batches and bi >= cfg.max_train_batches:
                    logging.warning(f"⏭️ Interrompendo treino cedo (max_train_batches={cfg.max_train_batches})")
                    break

        # ========== Validação + threshold ==========
        with timer(f"Validação - época {epoch} (threshold search)"):
            model.eval()
            all_probs, all_sizes, all_trues = [], [], []
            with torch.no_grad():
                for vi, batch in enumerate(val_loader, start=1):
                    x       = batch["x"].to(device, non_blocking=cfg.pin_memory)
                    pdv_id  = batch["pdv_id"].to(device, non_blocking=cfg.pin_memory)
                    prod_id = batch["prod_id"].to(device, non_blocking=cfg.pin_memory)
                    p_active, yhat_log = model(x, pdv_id, prod_id)
                    probs = torch.clamp(p_active, 1e-6, 1 - 1e-6).cpu().numpy()
                    sizes = torch.relu(torch.expm1(yhat_log)).cpu().numpy()
                    ytrue = batch["y"].cpu().numpy()
                    all_probs.append(probs); all_sizes.append(sizes); all_trues.append(ytrue)
                    if cfg.max_val_batches and vi >= cfg.max_val_batches:
                        logging.warning(f"⏭️ Interrompendo validação cedo (max_val_batches={cfg.max_val_batches})")
                        break

            import numpy as np
            P = np.concatenate(all_probs, axis=0) if all_probs else np.array([])
            S = np.concatenate(all_sizes, axis=0) if all_sizes else np.array([])
            Y = np.concatenate(all_trues, axis=0) if all_trues else np.array([])
            if len(Y) == 0:
                best_w, best_thr = float("inf"), 0.5
            else:
                best_thr, best_w = search_threshold(P, S, Y, mode=cfg.thr_eval_mode)

        # ========== Atualiza best + Early stopping ==========
        if best_w < best["val_wmape"] - 1e-12:
            best["val_wmape"] = float(best_w)
            best["state_dict"] = {k: v.cpu() for k, v in model.state_dict().items()}
            best["best_thr"] = float(best_thr)
            no_improve = 0
        else:
            no_improve += 1
            if cfg.early_stopping and no_improve >= cfg.patience:
                logging.info(f"⏹️ Early stopping: sem melhora por {cfg.patience} época(s).")
                break

        logging.info(
            f"[Epoch {epoch}/{cfg.epochs}] "
            f"train_loss={tot_loss / max(1, n_batches):.4f} | "
            f"val_WMAPE={best_w:.3f}% | best={best['val_wmape']:.3f}% @ thr={best['best_thr']:.2f} "
            f"| thr_mode={cfg.thr_eval_mode}"
        )
        mem_report(f"epoch {epoch} end")

        # ======== SALVAR CHECKPOINT ========
        if getattr(cfg, "save_ckpt_dir", None):
            os.makedirs(cfg.save_ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(cfg.save_ckpt_dir, f"epoch{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": opt.state_dict(),
            }, ckpt_path)
            logging.info(f"💾 Checkpoint salvo em {ckpt_path}")

    if best["state_dict"] is not None:
        model.load_state_dict(best["state_dict"])
    return best
