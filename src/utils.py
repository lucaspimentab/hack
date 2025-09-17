from __future__ import annotations
import pandas as pd
import numpy as np

WEEKS_JAN_2023 = [1,2,3,4,5]

def ensure_int(df: pd.DataFrame, cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(int)
    return df

def clamp_nonneg_int(series: pd.Series) -> pd.Series:
    arr = np.nan_to_num(series.values, nan=0.0)
    arr = np.maximum(arr, 0.0)
    return pd.Series(np.rint(arr).astype(int), index=series.index)

def coalesce(*series_list):
    out = None
    for s in series_list:
        if out is None:
            out = s.copy()
        else:
            mask = out.isna()
            out[mask] = s[mask]
    return out