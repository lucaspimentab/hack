#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .utils_io import make_week_sincos, timer, log_df


def autoreg_predict_jan(model,
                        ds_pred,
                        seq_cfg,
                        best_thr: float,
                        device: str) -> pd.DataFrame:
    model.eval(); rows = []; L = seq_cfg.context_len
    with torch.no_grad():
        for idx, s in enumerate(ds_pred.samples, start=1):
            if idx % 100000 == 0:
                import logging
                logging.info(f"  Pred {idx}/{len(ds_pred.samples)} ...")
            ctx = np.array(s["ctx"], dtype=float)
            ctx = np.maximum(np.nan_to_num(ctx, nan=0.0), 0.0)
            if seq_cfg.use_returns_feature:
                ctx_ret = np.array(s.get("ctx_ret", np.zeros_like(ctx)), dtype=float)
                ctx_ret = np.clip(np.nan_to_num(ctx_ret, nan=0.0), 0.0, 1.0)
            else:
                ctx_ret = None
            pdv_id = torch.tensor([s["pdv_id"]], dtype=torch.long, device=device)
            prod_id = torch.tensor([s["prod_id"]], dtype=torch.long, device=device)
            last_week = int(s["t_week"])  # 52
            seq_q = ctx.copy()
            seq_r = ctx_ret.copy() if ctx_ret is not None else None

            for step, target_week in enumerate(seq_cfg.target_weeks, start=1):
                ctx_now = seq_q[-L:]
                weeks = np.arange(last_week - L + 1, last_week + 1, dtype=int)
                w_next = ((weeks + step - 1) % 52) + 1
                sincos = make_week_sincos(w_next)
                x_ch = [np.log1p(ctx_now)]
                if seq_cfg.use_returns_feature:
                    ctx_ret_now = (seq_r[-L:] if seq_r is not None and len(seq_r) >= L else np.zeros_like(ctx_now))
                    x_ch.append(ctx_ret_now)
                x_np = np.concatenate([np.stack(x_ch, axis=-1), sincos], axis=-1)[None, ...]
                x = torch.tensor(x_np, dtype=torch.float32, device=device)
                p_active, yhat_log = model(x, pdv_id, prod_id)
                yhat = float(torch.relu(torch.expm1(yhat_log)).cpu().numpy()[0])
                p = float(torch.clamp(p_active, 1e-6, 1-1e-6).cpu().numpy()[0])
                ypred = p * yhat
                rows.append({"semana": int(target_week), "pdv": s["pdv"], "produto": s["produto"],
                             "quantidade": int(round(max(0.0, ypred)))})
                seq_q = np.concatenate([seq_q, [max(0.0, ypred)]], axis=0)
                if seq_cfg.use_returns_feature:
                    seq_r = np.concatenate([seq_r, [0.0]], axis=0)
    out = pd.DataFrame(rows).sort_values(["semana","pdv","produto"]).reset_index(drop=True)
    return out
