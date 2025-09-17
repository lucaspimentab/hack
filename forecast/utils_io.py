#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import math
import time
import json
import logging
from contextlib import contextmanager
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import torch

# ============ Logging & util ============
def setup_logger(level: str = "INFO"):
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S"
    )
    logging.getLogger("numexpr").setLevel(logging.WARNING)
    logging.getLogger("fsspec").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

@contextmanager
def timer(msg: str):
    t0 = time.time()
    logging.info(f"⏳ {msg} ...")
    try:
        yield
    finally:
        dt = time.time() - t0
        logging.info(f"✅ {msg} concluído em {dt:.2f}s")

def mem_report(tag: str = ""):
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        logging.info(f"🧠 CUDA mem {tag}: allocated={alloc:.3f} GB | reserved={reserved:.3f} GB")

def log_df(df: pd.DataFrame, name: str, head: bool = False, n: int = 3):
    logging.info(f"📐 {name}: shape={df.shape} | cols={list(df.columns)}")
    if head:
        logging.info(f"{name} head:\n{df.head(n)}")

# ============ I/O & coerções ============
def read_any(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".parquet", ".pq"]:
        try:
            return pd.read_parquet(path)
        except Exception:
            return pd.read_parquet(path, engine="pyarrow")
    else:
        try:
            return pd.read_csv(path, sep=";", encoding="utf-8")
        except Exception:
            return pd.read_csv(path, sep=",", encoding="utf-8")

def coerce_transactions(
    df: pd.DataFrame,
    quantity_col: str | None = None,
    fallback_count_rows: bool = False
) -> pd.DataFrame:
    df = df.copy()
    orig = df.columns.tolist()
    df.columns = [c.strip().lower() for c in df.columns]
    logging.info(f"🔎 Colunas originais (normalizadas): {df.columns.tolist()} | (antes: {orig})")

    # IDs
    if "internal_store_id" in df.columns and "pdv" not in df.columns:
        df = df.rename(columns={"internal_store_id": "pdv"})
    if "internal_product_id" in df.columns and "produto" not in df.columns:
        df = df.rename(columns={"internal_product_id": "produto"})
    for cand in ["sku","product","codigo"]:
        if cand in df.columns and "produto" not in df.columns:
            df = df.rename(columns={cand:"produto"}); break

    # Datas
    date_candidates = []
    for cand in ["data", "transaction_date", "purchase_date", "reference_date"]:
        if cand in df.columns:
            date_candidates.append(pd.to_datetime(df[cand], errors="coerce"))
    if not date_candidates:
        raise ValueError("Não encontrei coluna de data entre: data, transaction_date, purchase_date, reference_date.")
    data_series = None
    for s in date_candidates:
        data_series = s if data_series is None else data_series.fillna(s)
    df["data"] = pd.to_datetime(data_series, errors="coerce")
    for cand in ["transaction_date", "purchase_date", "reference_date"]:
        if cand in df.columns:
            df = df.drop(columns=[cand])

    # Quantidade
    qty_col = None
    if quantity_col and quantity_col.lower() in df.columns:
        qty_col = quantity_col.lower()
    else:
        for cand in ["quantidade","quantity","qtd","qty","units","unit","sales_qty","volume","qte"]:
            if cand in df.columns:
                qty_col = cand; break
    if qty_col is None:
        if fallback_count_rows:
            df["quantidade"] = 1.0
        else:
            raise ValueError(
                "Transações precisam conter uma coluna de quantidade. "
                "Use --quantity_col NOME_DA_COLUNA, ou --fallback_count_rows para usar 1 unidade por linha.\n"
                f"Colunas disponíveis: {df.columns.tolist()}"
            )
    else:
        if qty_col != "quantidade":
            df = df.rename(columns={qty_col: "quantidade"})
        df["quantidade"] = pd.to_numeric(df["quantidade"], errors="coerce").fillna(0).astype(float)

    # Monetárias (se houver)
    ren = {}
    if "gross_value" in df.columns: ren["gross_value"] = "faturamento_bruto"
    if "net_value"   in df.columns: ren["net_value"]   = "faturamento_liquido"
    if "gross_profit" in df.columns: ren["gross_profit"] = "lucro_bruto"
    if ren: df = df.rename(columns=ren)

    # Checagem
    required = {"data","pdv","produto","quantidade"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Faltam colunas obrigatórias {required}. Ausentes: {missing} | cols={df.columns.tolist()}")

    logging.info(f"✅ Colunas finais de transações: {df.columns.tolist()}")
    return df

def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    iso = df["data"].dt.isocalendar()
    df["ano"] = iso["year"].astype(int)
    df["semana_iso"] = iso["week"].astype(int)
    df["ano_semana"] = df["ano"].astype(str) + "-" + df["semana_iso"].astype(str).str.zfill(2)
    return df

def weekly_agg(df: pd.DataFrame, returns_col: Optional[str] = None) -> pd.DataFrame:
    agg = (df.groupby(["pdv","produto","ano","semana_iso"], as_index=False)
             .agg(quantidade=("quantidade","sum")))
    if returns_col is not None and returns_col in df.columns:
        ret_week = (df.groupby(["pdv","produto","ano","semana_iso"], as_index=False)
                      .agg(devol_flag=(returns_col, lambda s: int((s.astype(bool)).any()))))
        agg = agg.merge(ret_week, on=["pdv","produto","ano","semana_iso"], how="left")
        agg["devol_flag"] = agg["devol_flag"].fillna(0).astype(int)
    return agg

# ============ Métricas & helpers ============
def wmape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true).sum()
    if denom <= 0:
        return np.nan
    return (np.abs(y_true - y_pred).sum() / denom) * 100.0

def clamp_nonneg_int(a: np.ndarray) -> np.ndarray:
    a = np.nan_to_num(a, nan=0.0)
    a = np.maximum(a, 0.0)
    return np.rint(a).astype(int)

def make_week_sincos(week_idx: np.ndarray) -> np.ndarray:
    w = week_idx.astype(float)
    ang = 2 * math.pi * (w - 1) / 52.0
    return np.stack([np.sin(ang), np.cos(ang)], axis=-1)

# ============ EDA numérica ============
def apply_weekly_eda_cleanup(
    weekly: pd.DataFrame,
    min_active_weeks: int,
    min_total_qty: float,
    max_zeros_share: float,
    train_cap_pair_quantile: float,
    train_cap_pair_mult: float,
    train_global_qclip: float,
    logger=logging,
) -> pd.DataFrame:
    w = weekly.copy()
    w["quantidade"] = np.maximum(np.nan_to_num(w["quantidade"].astype(float), nan=0.0), 0.0)

    stats = (
        w.groupby(["pdv","produto"], as_index=False)
        .agg(
            sum_q=("quantidade", "sum"),
            active_weeks=("quantidade", lambda s: int((np.asarray(s) > 0).sum())),
            zeros_share=("quantidade",  lambda s: float((np.asarray(s) == 0).mean())),
        )
    )
    before_pairs = stats.shape[0]
    mask = (
        (stats["active_weeks"] >= int(min_active_weeks)) &
        (stats["sum_q"]        >= float(min_total_qty)) &
        (stats["zeros_share"]  <= float(max_zeros_share))
    )
    kept_pairs = stats.loc[mask, ["pdv","produto"]]
    removed_pairs = before_pairs - kept_pairs.shape[0]
    logger.info(f"🧹 EDA: pares removidos por esparsidade/relevância = {removed_pairs} / {before_pairs}")
    w = w.merge(kept_pairs.assign(_keep=1), on=["pdv","produto"], how="inner")
    after_pairs = w[["pdv","produto"]].drop_duplicates().shape[0]
    logger.info(f"🧮 EDA: pares restantes = {after_pairs}")

    # CAP por par
    if 0.0 < float(train_cap_pair_quantile) < 1.0 and float(train_cap_pair_mult) > 0.0:
        q = (w.groupby(["pdv","produto"])["quantidade"]
               .quantile(float(train_cap_pair_quantile))
               .rename("q_cap").reset_index())
        q["cap"] = q["q_cap"] * float(train_cap_pair_mult)
        w = w.merge(q[["pdv","produto","cap"]], on=["pdv","produto"], how="left")
        before_clip = w["quantidade"].to_numpy(copy=True, dtype=float)
        cap_vals = w["cap"].replace([np.inf,-np.inf], np.inf).fillna(np.inf).to_numpy()
        w["quantidade"] = np.minimum(w["quantidade"].to_numpy(dtype=float), cap_vals)
        n_clipped = int((before_clip > w["quantidade"].to_numpy()).sum())
        logger.info(
            f"🪓 EDA: CAP por par → quantil={float(train_cap_pair_quantile):.2f} × mult={float(train_cap_pair_mult):.2f} "
            f"→ {n_clipped} valores truncados"
        )
        w = w.drop(columns=["cap"], errors="ignore")

    # Winsorize global
    if 0.0 < float(train_global_qclip) < 1.0:
        gcap = float(pd.Series(w["quantidade"], dtype=float).quantile(float(train_global_qclip)))
        before = w["quantidade"].to_numpy(copy=True, dtype=float)
        w["quantidade"] = np.minimum(w["quantidade"].to_numpy(dtype=float), gcap)
        n_clipg = int((before > w["quantidade"].to_numpy()).sum())
        logger.info(f"🪚 EDA: winsorize global → q={float(train_global_qclip):.2f} (cap={gcap:.3f}) → {n_clipg} valores truncados")

    w["quantidade"] = np.maximum(np.nan_to_num(w["quantidade"].astype(float), nan=0.0), 0.0)
    return w

# ============ Threshold search ============
def search_threshold(P: np.ndarray, S: np.ndarray, Y: np.ndarray, mode: str = "weighted") -> Tuple[float, float]:
    # Grid mais amplo para evitar colapso em p baixos
    thrs = np.linspace(0.01, 0.90, 90)

    def _wmape(y_true, y_pred):
        denom = np.abs(y_true).sum()
        if denom == 0:
            return np.nan
        return 100.0 * np.abs(y_true - y_pred).sum() / denom

    best_w, best_thr = float("inf"), 0.5
    for thr in thrs:
        yhat = np.where(P >= thr, S, 0.0)
        if mode == "pos_only":
            mask = (Y > 0)
            score = _wmape(Y[mask], yhat[mask]) if mask.any() else _wmape(Y, yhat)
        elif mode == "weighted":
            mask = (Y > 0)
            w_pos = _wmape(Y[mask], yhat[mask]) if mask.any() else np.nan
            w_all = _wmape(Y, yhat)
            score = w_all if np.isnan(w_pos) else 0.7 * w_pos + 0.3 * w_all
        else:
            score = _wmape(Y, yhat)
        if score < best_w:
            best_w, best_thr = float(score), float(thr)
    return best_thr, best_w


# ============ Baseline sazonal & blend ============
def seasonal_baseline_for_pairs(weekly_df: pd.DataFrame,
                                pairs_df: pd.DataFrame,
                                weeks_jan=(1,2,3,4,5)) -> pd.DataFrame:
    lvl_pair = (weekly_df.groupby(['pdv','produto','semana_iso'])['quantidade']
                .mean().rename('m_pair')).reset_index()
    lvl_prod = (weekly_df.groupby(['produto','semana_iso'])['quantidade']
                .mean().rename('m_prod')).reset_index()
    lvl_glob = (weekly_df.groupby(['semana_iso'])['quantidade']
                .mean().rename('m_glob')).reset_index()

    weeks = pd.DataFrame({'semana_iso': list(weeks_jan)})
    grid = (pairs_df[['pdv','produto']].drop_duplicates()
            .assign(_k=1).merge(weeks.assign(_k=1), on='_k').drop(columns=['_k']))
    out = (grid.merge(lvl_pair, on=['pdv','produto','semana_iso'], how='left')
                .merge(lvl_prod, on=['produto','semana_iso'], how='left')
                .merge(lvl_glob, on=['semana_iso'], how='left'))
    out['baseline_qty'] = out['m_pair'].fillna(out['m_prod']).fillna(out['m_glob']).fillna(0.0)
    out = out[['semana_iso','pdv','produto','baseline_qty']].rename(columns={'semana_iso':'semana'})
    out['baseline_qty'] = np.clip(np.round(out['baseline_qty']).astype(int), 0, None)
    return out

def apply_blend_on_villains(sub_df: pd.DataFrame,
                            villains_csv: str,
                            weekly_df: pd.DataFrame,
                            when_zero: bool = True) -> pd.DataFrame:
    try:
        v = pd.read_csv(villains_csv)
    except Exception:
        v = pd.read_csv(villains_csv, sep=';')
    if 'pdv' not in v.columns or 'produto' not in v.columns:
        raise ValueError("villains_csv precisa conter colunas 'pdv' e 'produto'.")
    pairs = v[['pdv','produto']].drop_duplicates()
    base = seasonal_baseline_for_pairs(weekly_df, pairs)
    key_cols = ['semana','pdv','produto']
    merged = sub_df.merge(base, on=key_cols, how='left')
    merged['baseline_qty'] = merged['baseline_qty'].fillna(0).astype(int)
    if when_zero:
        use_bl = (merged['quantidade'].astype(int) == 0) & (merged['baseline_qty'] > 0)
    else:
        use_bl = merged['baseline_qty'] > 0
    merged.loc[use_bl, 'quantidade'] = merged.loc[use_bl, 'baseline_qty']
    merged['quantidade'] = np.clip(merged['quantidade'].astype(int), 0, None)
    return merged[key_cols + ['quantidade']]

# ======= métricas extras para diagnóstico =======
def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.abs(y_true - y_pred).mean()

def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
