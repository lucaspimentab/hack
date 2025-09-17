#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .utils_io import make_week_sincos


def build_pair_week_matrix(weekly: pd.DataFrame, include_returns: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pairs = weekly[["pdv","produto"]].drop_duplicates().reset_index(drop=True)
    all_weeks = pd.DataFrame({"semana_iso": np.arange(1, 53, dtype=int)})
    grid = (pairs.assign(key=1).merge(all_weeks.assign(key=1), on="key").drop("key", axis=1))
    cols = ["pdv","produto","semana_iso","quantidade"] + (["devol_flag"] if include_returns and ("devol_flag" in weekly.columns) else [])
    wk = weekly[[c for c in cols if c in weekly.columns]]
    M = (grid.merge(wk, on=["pdv","produto","semana_iso"], how="left")
              .fillna({"quantidade":0.0, **({"devol_flag":0} if include_returns else {})}))
    if include_returns and "devol_flag" not in M.columns:
        M["devol_flag"] = 0
    return pairs, M

@dataclass
class SeqConfig:
    context_len: int = 16
    target_weeks: List[int] = (1,2,3,4,5)
    use_returns_feature: bool = False
    neg_sample_rate: float = 0.10

class DemandSeqDataset(Dataset):
    def __init__(self, M: pd.DataFrame, seq_cfg: SeqConfig, split: str):
        self.split = split
        self.cfg = seq_cfg
        self.M = M.copy()
        self.pdv2id = {v:i for i,v in enumerate(self.M["pdv"].unique())}
        self.prod2id = {v:i for i,v in enumerate(self.M["produto"].unique())}
        self.M["pdv_id"] = self.M["pdv"].map(self.pdv2id).astype(int)
        self.M["prod_id"] = self.M["produto"].map(self.prod2id).astype(int)

        group = self.M.groupby(["pdv","produto","pdv_id","prod_id"], as_index=False)
        self.series = []
        for _, g in group:
            q = np.zeros(52, dtype=float)
            ret = np.zeros(52, dtype=float)
            weeks = g["semana_iso"].to_numpy()
            q[weeks - 1] = g["quantidade"].to_numpy(dtype=float)
            if self.cfg.use_returns_feature and ("devol_flag" in g.columns):
                ret[weeks - 1] = g["devol_flag"].to_numpy(dtype=float)
            self.series.append({
                "pdv": g["pdv"].iloc[0],
                "produto": g["produto"].iloc[0],
                "pdv_id": int(g["pdv_id"].iloc[0]),
                "prod_id": int(g["prod_id"].iloc[0]),
                "q": q,
                "ret": ret,
            })

        self.samples = []
        L = self.cfg.context_len
        if split in ["train", "val"]:
            val_weeks = set(self.cfg.target_weeks)  # {1..5}
            for s in self.series:
                q = s["q"]
                for t in range(1, 53):
                    is_val = t in val_weeks
                    if split == "train" and is_val: continue
                    if split == "val"   and not is_val: continue
                    if t <= L:
                        if split == "val":
                            prefix = q[:max(0, t-1)]
                            pad = np.zeros(L - len(prefix), dtype=float)
                            ctx = np.concatenate([pad, prefix], axis=0)
                        else:
                            continue
                    else:
                        ctx = q[(t-L):t]
                    y = q[t-1]
                    if self.split == "train" and y <= 0.0:
                        import random
                        if random.random() > max(0.0, min(1.0, self.cfg.neg_sample_rate)):
                            continue
                    item = {
                        "pdv_id": s["pdv_id"],
                        "prod_id": s["prod_id"],
                        "pdv": s["pdv"],
                        "produto": s["produto"],
                        "ctx": ctx.astype(float),
                        "t_week": t,
                        "y_active": 1.0 if y > 0 else 0.0,
                        "y_log": np.log1p(max(y, 0.0)),
                        "y": float(y),
                    }
                    if self.cfg.use_returns_feature:
                        if t <= L and split == "val":
                            prefix_ret = s["ret"][:max(0, t-1)]
                            pad_ret = np.zeros(L - len(prefix_ret), dtype=float)
                            item["ctx_ret"] = np.concatenate([pad_ret, prefix_ret], axis=0).astype(float)
                        else:
                            item["ctx_ret"] = s["ret"][(t-L):t].astype(float)
                    self.samples.append(item)
        elif split == "predict":
            for s in self.series:
                q = s["q"]
                ctx = q[-L:] if len(q) >= L else np.concatenate([np.zeros(L - len(q)), q])
                item = {
                    "pdv_id": s["pdv_id"],
                    "prod_id": s["prod_id"],
                    "pdv": s["pdv"],
                    "produto": s["produto"],
                    "ctx": ctx.astype(float),
                    "t_week": 52,
                }
                if self.cfg.use_returns_feature:
                    r = s["ret"]
                    item["ctx_ret"] = (r[-L:] if len(r) >= L else np.concatenate([np.zeros(L - len(r)), r])).astype(float)
                self.samples.append(item)
        else:
            raise ValueError("split must be train/val/predict")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        ctx = np.array(s["ctx"], dtype=float)
        ctx = np.maximum(np.nan_to_num(ctx, nan=0.0), 0.0)
        L = len(ctx)
        weeks = np.arange(s["t_week"] - L, s["t_week"], dtype=int)
        sincos = make_week_sincos(weeks)
        x_ch = [np.log1p(ctx)]
        if self.cfg.use_returns_feature:
            ctx_ret = np.array(s.get("ctx_ret", np.zeros_like(ctx)), dtype=float)
            ctx_ret = np.clip(np.nan_to_num(ctx_ret, nan=0.0), 0.0, 1.0)
            x_ch.append(ctx_ret)
        x = np.stack(x_ch, axis=-1)
        x = np.concatenate([x, sincos], axis=-1)
        item = {
            "x": torch.tensor(x, dtype=torch.float32),
            "pdv_id": torch.tensor(s["pdv_id"], dtype=torch.long),
            "prod_id": torch.tensor(s["prod_id"], dtype=torch.long),
            "t_week": torch.tensor(s["t_week"], dtype=torch.long),
        }
        if "y" in s:
            item["y_active"] = torch.tensor(s["y_active"], dtype=torch.float32)
            item["y_log"] = torch.tensor(s["y_log"], dtype=torch.float32)
            item["y"] = torch.tensor(s["y"], dtype=torch.float32)
        return item