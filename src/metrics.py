from __future__ import annotations
import pandas as pd
import numpy as np

def wmape(y_true, y_pred):
    y_true = pd.Series(y_true).astype(float)
    y_pred = pd.Series(y_pred).astype(float)
    denom = np.abs(y_true).sum()
    if denom == 0:
        return np.nan
    return (np.abs(y_true - y_pred).sum() / denom) * 100.0