from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List

# Utilitários simples
def clamp_nonneg_int(series: pd.Series) -> pd.Series:
    arr = np.nan_to_num(series.values, nan=0.0)
    arr = np.maximum(arr, 0.0)
    return pd.Series(np.rint(arr).astype(int), index=series.index)

def _coalesce(a: pd.Series, b: pd.Series | None = None,
              c: pd.Series | None = None, d: pd.Series | None = None) -> pd.Series:
    out = a.copy()
    if b is not None:
        m = out.isna()
        out[m] = b[m]
    if c is not None:
        m = out.isna()
        out[m] = c[m]
    if d is not None:
        m = out.isna()
        out[m] = d[m]
    return out

def build_seasonal_tables(df_weekly: pd.DataFrame) -> dict:
    """
    df_weekly: colunas ['pdv','produto','ano','semana_iso','quantidade'] (agregado semanal de 2022).
    Retorna tabelas sazonais por níveis hierárquicos.
    """
    lvl_pair_week = df_weekly.groupby(['pdv','produto','semana_iso'])['quantidade'].mean().rename('pair_week_mean')
    lvl_pair_year = df_weekly.groupby(['pdv','produto'])['quantidade'].mean().rename('pair_year_mean')
    lvl_prod_week = df_weekly.groupby(['produto','semana_iso'])['quantidade'].mean().rename('prod_week_mean')
    lvl_prod_year = df_weekly.groupby(['produto'])['quantidade'].mean().rename('prod_year_mean')
    lvl_glob_week = df_weekly.groupby(['semana_iso'])['quantidade'].mean().rename('glob_week_mean')
    lvl_glob_year = pd.Series({'global_year_mean': df_weekly['quantidade'].mean()})
    return dict(
        pair_week=lvl_pair_week,
        pair_year=lvl_pair_year,
        prod_week=lvl_prod_week,
        prod_year=lvl_prod_year,
        glob_week=lvl_glob_week,
        glob_year=lvl_glob_year
    )

def blended_forecast_grid_w(
    pairs_df: pd.DataFrame,
    seasonal_tables: dict,
    target_weeks: List[int],
    w_pair_week: float = 0.7,
    w_pair_year: float = 0.2,
    w_prod_week: float = 0.1,
    # parâmetros anti-sparsidade / anti-outlier:
    min_pair_year_mean: float = 1.0,   # se média anual do par < limiar, derruba prob
    min_pos_weeks: int = 2,            # se par teve <2 semanas com venda no ano, derruba prob
    support_scale: float = 1.0,        # escala da prob baseada em suporte (0..1)
    jan_support_scale: float = 1.0,    # escala da prob baseada em atividade em jan (0..1)
    cap_mult_pair_q90: float = 1.05,   # limite = 1.05 × p90 semanal do par
    eps: float = 1e-6
) -> pd.DataFrame:
    pw  = seasonal_tables['pair_week']   # Series index=(pdv,produto,semana_iso) -> quantidade média
    py  = seasonal_tables['pair_year']   # Series index=(pdv,produto) -> média anual
    prw = seasonal_tables['prod_week']   # Series index=(produto,semana_iso)
    pry = seasonal_tables['prod_year']   # Series index=(produto)
    gw  = seasonal_tables['glob_week']   # Series index=(semana_iso)
    gy  = seasonal_tables['glob_year']['global_year_mean'] if 'global_year_mean' in seasonal_tables['glob_year'] else np.nan

    # ------ Tabelas auxiliares p/ travas ------
    # pair_week em DataFrame padronizado
    pw_df = pw.reset_index(name='quantidade')  # cols: pdv, produto, semana_iso, quantidade

    # suporte do par: semanas com venda > 0, total de semanas vistas
    pair_weeks_pos   = (pw_df.assign(pos=lambda d: (d['quantidade'] > 0).astype(int))
                             .groupby(['pdv','produto'])['pos'].sum())
    pair_weeks_total = pw_df.groupby(['pdv','produto'])['semana_iso'].nunique()

    # estatística de tamanho: p90 semanal por par (CAP)
    pair_q90 = pw_df.groupby(['pdv','produto'])['quantidade'].quantile(0.90)

    # atividade em janeiro (semanas 1..5) por par
    jan_set = [1,2,3,4,5]
    pw_jan = pw_df[pw_df['semana_iso'].isin(jan_set)]
    pair_jan_pos  = (pw_jan.assign(pos=lambda d: (d['quantidade'] > 0).astype(int))
                           .groupby(['pdv','produto'])['pos'].sum())
    pair_jan_mean = pw_jan.groupby(['pdv','produto'])['quantidade'].mean()

    # nível produto: fator de janeiro do produto (jan_mean / year_mean) p/ suavizar sazonalidade
    prw_df = prw.reset_index(name='quantidade')  # cols: produto, semana_iso, quantidade
    prod_jan_mean = prw_df[prw_df['semana_iso'].isin(jan_set)].groupby('produto')['quantidade'].mean()
    prod_year_mean = pry  # média anual do produto

    # ------ Grade alvo ------
    grid = pairs_df.assign(key=1).merge(
        pd.DataFrame({'semana_iso': target_weeks, 'key':[1]*len(target_weeks)}),
        on='key'
    ).drop('key', axis=1)
    idx = grid.index

    # Lookups nível por nível
    s_pw  = pd.Series(grid.set_index(['pdv','produto','semana_iso']).index.map(pw),  index=idx, dtype='float')
    s_py  = pd.Series(grid.set_index(['pdv','produto']).index.map(py),               index=idx, dtype='float')
    s_prw = pd.Series(grid.set_index(['produto','semana_iso']).index.map(prw),       index=idx, dtype='float')
    s_pry = pd.Series(grid.set_index(['produto']).index.map(pry),                    index=idx, dtype='float')
    s_gw  = pd.Series(grid.set_index(['semana_iso']).index.map(gw),                  index=idx, dtype='float')
    s_gy  = pd.Series([gy]*len(grid),                                                index=idx, dtype='float')

    def _coalesce(a, b=None, c=None, d=None):
        out = a.copy()
        if b is not None:
            m = out.isna(); out[m] = b[m]
        if c is not None:
            m = out.isna(); out[m] = c[m]
        if d is not None:
            m = out.isna(); out[m] = d[m]
        return out

    seasonal_anchor = _coalesce(s_pw,  s_prw, s_gw, s_gy)
    pair_anchor     = _coalesce(s_py,  s_pry, s_gy)
    prod_anchor     = _coalesce(s_prw, s_pry, s_gw, s_gy)

    pred = (w_pair_week * seasonal_anchor
          + w_pair_year * pair_anchor
          + w_prod_week * prod_anchor).fillna(0.0)

    # ------ GATE 1: probabilidade pelo suporte do par ------
    s_weeks_pos   = pd.Series(grid.set_index(['pdv','produto']).index.map(pair_weeks_pos),   index=idx, dtype='float')
    s_weeks_total = pd.Series(grid.set_index(['pdv','produto']).index.map(pair_weeks_total), index=idx, dtype='float')
    base_prob = (s_weeks_pos / (s_weeks_total + eps)).clip(0, 1).fillna(0)
    base_prob = (base_prob * support_scale).clip(0, 1)

    # penaliza pares muito raros no ano
    s_pair_year_mean = pd.Series(grid.set_index(['pdv','produto']).index.map(py), index=idx, dtype='float').fillna(0)
    rare_pair = (s_pair_year_mean < min_pair_year_mean) | (s_weeks_pos.fillna(0) < min_pos_weeks)
    base_prob = base_prob * np.where(rare_pair, 0.25, 1.0)   # derruba com força pares raros

    # ------ GATE 2: atividade específica em janeiro ------
    s_pair_jan_pos  = pd.Series(grid.set_index(['pdv','produto']).index.map(pair_jan_pos),  index=idx, dtype='float')
    jan_prob = (s_pair_jan_pos / 5.0).clip(0, 1).fillna(0)
    jan_prob = (jan_prob * jan_support_scale).clip(0, 1)

    # fator produto de janeiro (jan/ano) para suavizar
    s_prod_jan_mean  = pd.Series(grid.set_index(['produto']).index.map(prod_jan_mean),  index=idx, dtype='float')
    s_prod_year_mean = pd.Series(grid.set_index(['produto']).index.map(prod_year_mean), index=idx, dtype='float')
    prod_jan_factor = (s_prod_jan_mean / (s_prod_year_mean + eps)).fillna(0).clip(0, 1)

    # aplica gates
    pred = pred * base_prob * np.maximum(prod_jan_factor, 0.2) * np.maximum(jan_prob, 0.1)

    # ------ CAP: limita à 1.05 × p90 semanal histórico do par ------
    s_pair_q90 = pd.Series(grid.set_index(['pdv','produto']).index.map(pair_q90), index=idx, dtype='float')
    cap = (s_pair_q90 * cap_mult_pair_q90).replace([np.inf, -np.inf], np.inf).fillna(np.inf)
    pred = np.minimum(pred, cap)

    grid = grid.copy()
    grid['quantidade'] = clamp_nonneg_int(pd.Series(pred, index=idx))
    grid = grid.rename(columns={'semana_iso':'semana'})
    return grid[['semana','pdv','produto','quantidade']].sort_values(['semana','pdv','produto']).reset_index(drop=True)
