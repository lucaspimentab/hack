from __future__ import annotations
import pandas as pd
import numpy as np

def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['data'] = pd.to_datetime(df['data'])
    iso = df['data'].dt.isocalendar()  # retorna colunas: year, week, day
    df['ano'] = iso['year'].astype(int)
    df['semana_iso'] = iso['week'].astype(int)
    # aqui estava o erro: precisa usar .str.zfill(2)
    df['ano_semana'] = df['ano'].astype(str) + '-' + df['semana_iso'].astype(str).str.zfill(2)
    return df


def weekly_agg(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby(['pdv','produto','ano','semana_iso'], as_index=False)
              .agg(quantidade=('quantidade','sum')))

def join_catalogs(trans_weekly: pd.DataFrame, products: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    out = trans_weekly.merge(products, on='produto', how='left', suffixes=('','_prod'))
    out = out.merge(stores, on='pdv', how='left', suffixes=('','_pdv'))
    return out