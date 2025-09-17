#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EDA focada em drivers de WMAPE com LOGS detalhados.
- Lê diretamente 3 arquivos Parquet (sem precisar de argumentos de terminal).
- Gera CSVs, PNGs e um console_report.txt com achados.
"""

import os
import time
import logging
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIGURAÇÃO (edite aqui)
# =========================
TRANSACTIONS_PATH = "data/part-00000-tid-5196563791502273604-c90d3a24-52f2-4955-b4ec-fb143aae74d8-4-1-c000.snappy.parquet"
PRODUCTS_PATH     = "data/part-00000-tid-7173294866425216458-eae53fbf-d19e-4130-ba74-78f96b9675f1-4-1-c000.snappy.parquet"
STORES_PATH       = "data/part-00000-tid-2779033056155408584-f6316110-4c9a-4061-ae48-69b77c7c8c36-4-1-c000.snappy.parquet"
OUTDIR            = "eda_out"

# =========================
# LOGGING
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("EDA_WMAPE")

def tic():
    return time.time()

def toc(t0, label=""):
    dt = time.time() - t0
    logger.info(f"✅ {label} concluído em {dt:.2f}s")
    return dt

# =========================
# FUNÇÕES
# =========================
def _read_any(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".parquet", ".pq"):
        try:
            return pd.read_parquet(path)
        except Exception:
            return pd.read_parquet(path, engine="pyarrow")
    elif ext in (".csv", ".txt"):
        # tenta ; depois autodetect
        try:
            return pd.read_csv(path, sep=';', encoding='utf-8')
        except Exception:
            return pd.read_csv(path, sep=None, engine="python")
    else:
        raise ValueError(f"Formato não suportado: {ext}")

def ensure_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors='coerce')

def pquantile(a, q):
    a = np.asarray(a)
    if a.size == 0:
        return 0.0
    return float(np.quantile(a, q))

def load_products(path: str) -> pd.DataFrame:
    t0 = tic()
    logger.info(f"⏳ Lendo produtos: {path}")
    df = _read_any(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if 'produto' not in df.columns:
        for cand in ['sku','id_produto','product','codigo','internal_product_id']:
            if cand in df.columns:
                df = df.rename(columns={cand:'produto'})
                break
    if 'produto' not in df.columns:
        logger.warning(f"Cadastro de produtos sem coluna 'produto'. Colunas: {list(df.columns)}")
    toc(t0, "Leitura de produtos")
    logger.info(f"📐 produtos: shape={df.shape} | cols={list(df.columns)[:10]}...")
    return df

def load_stores(path: str) -> pd.DataFrame:
    t0 = tic()
    logger.info(f"⏳ Lendo PDVs: {path}")
    df = _read_any(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if 'pdv' not in df.columns:
        for cand in ['loja','id_pdv','store','internal_store_id']:
            if cand in df.columns:
                df = df.rename(columns={cand:'pdv'})
                break
    if 'pdv' not in df.columns:
        logger.warning(f"Cadastro de PDVs sem coluna 'pdv'. Colunas: {list(df.columns)}")
    toc(t0, "Leitura de PDVs")
    logger.info(f"📐 pdvs: shape={df.shape} | cols={list(df.columns)[:10]}...")
    return df

def load_transactions(path: str) -> pd.DataFrame:
    t0 = tic()
    logger.info(f"⏳ Lendo transações: {path}")
    df = _read_any(path)
    df.columns = [c.strip().lower() for c in df.columns]

    # IDs
    if 'internal_store_id' in df.columns: df = df.rename(columns={'internal_store_id':'pdv'})
    if 'internal_product_id' in df.columns: df = df.rename(columns={'internal_product_id':'produto'})
    if 'sku' in df.columns and 'produto' not in df.columns: df = df.rename(columns={'sku':'produto'})
    if 'product' in df.columns and 'produto' not in df.columns: df = df.rename(columns={'product':'produto'})
    if 'codigo' in df.columns and 'produto' not in df.columns: df = df.rename(columns={'codigo':'produto'})

    # data
    date_col = None
    for cand in ['transaction_date','purchase_date','reference_date','data']:
        if cand in df.columns:
            date_col = cand
            break
    if date_col is None:
        raise ValueError(f"Transações precisam de coluna de data. cols={list(df.columns)}")

    if date_col != 'data':
        df = df.rename(columns={date_col:'data'})
    for other in ['transaction_date','purchase_date','reference_date']:
        if other in df.columns and other != 'data':
            df = df.drop(columns=[other])

    # quantias/valores
    if 'quantity' in df.columns: df = df.rename(columns={'quantity':'quantidade'})
    if 'qtd' in df.columns: df = df.rename(columns={'qtd':'quantidade'})
    if 'gross_value' in df.columns: df = df.rename(columns={'gross_value':'faturamento_bruto'})
    if 'net_value' in df.columns: df = df.rename(columns={'net_value':'faturamento_liquido'})
    if 'gross_profit' in df.columns: df = df.rename(columns={'gross_profit':'lucro_bruto'})

    required = ['data','pdv','produto','quantidade']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Transações precisam conter {required}. Faltando: {missing} | cols={list(df.columns)}")

    df['data'] = ensure_datetime(df['data'])
    toc(t0, "Leitura de transações")
    logger.info(f"📐 transações: shape={df.shape} | cols={list(df.columns)[:12]}...")
    logger.info(f"🔎 head:\n{df.head(3)}")
    return df

def compute_weekly(trans: pd.DataFrame):
    t0 = tic()
    logger.info("⏳ Construindo calendário ISO e agregando semanalmente...")
    iso = trans['data'].dt.isocalendar()
    trans = trans.assign(ano=iso['year'].astype(int),
                         semana_iso=iso['week'].astype(int),
                         is_return=(trans['quantidade'] < 0).astype(int))
    weekly = (trans.groupby(['pdv','produto','semana_iso'], as_index=False)
                    .agg(quantidade=('quantidade','sum'),
                         returns=('is_return','sum')))
    toc(t0, "Agregação semanal")
    logger.info(f"📐 weekly: shape={weekly.shape} | cols={list(weekly.columns)}")
    return weekly, trans

def build_pair_grid(weekly: pd.DataFrame) -> pd.DataFrame:
    t0 = tic()
    logger.info("⏳ Construindo grade 1..52 por par (pdv×produto)...")
    pairs = weekly[['pdv','produto']].drop_duplicates()
    weeks = pd.DataFrame({'semana_iso': np.arange(1,53,dtype=int)})
    grid = pairs.assign(_k=1).merge(weeks.assign(_k=1), on='_k').drop(columns=['_k'])
    wf = grid.merge(weekly, on=['pdv','produto','semana_iso'], how='left')
    wf['quantidade'] = wf['quantidade'].fillna(0.0)
    wf['returns'] = wf['returns'].fillna(0.0)
    toc(t0, "Grade por par/semana")
    logger.info(f"📐 weekly_full: shape={wf.shape}")
    return wf

def pair_level_metrics(weekly_full: pd.DataFrame) -> pd.DataFrame:
    t0 = tic()
    logger.info("⏳ Calculando métricas por par (drivers de WMAPE)...")
    g = weekly_full.groupby(['pdv','produto'])
    df = g.agg(
        total_qty=('quantidade','sum'),
        active_weeks=('quantidade', lambda x: int((x>0).sum())),
        zero_weeks=('quantidade', lambda x: int((x<=0).sum())),
        p90_week=('quantidade', lambda x: pquantile(x, 0.90)),
        p99_week=('quantidade', lambda x: pquantile(x, 0.99)),
        max_week=('quantidade','max'),
        mean_week=('quantidade','mean'),
        returns_count=('returns','sum')
    ).reset_index()
    df['zeros_share'] = df['zero_weeks'] / 52.0

    # janeiro 1..5
    jan = weekly_full[weekly_full['semana_iso'].isin([1,2,3,4,5])]
    jan_g = jan.groupby(['pdv','produto']).agg(
        jan_pos_weeks=('quantidade', lambda x: int((x>0).sum())),
        jan_sum=('quantidade','sum')
    ).reset_index()
    df = df.merge(jan_g, on=['pdv','produto'], how='left').fillna({'jan_pos_weeks':0,'jan_sum':0.0})
    toc(t0, "Métricas por par")
    logger.info(f"📐 pair_metrics: shape={df.shape}")
    return df

def enrich_with_catalogs(pair_metrics: pd.DataFrame, products: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    t0 = tic()
    logger.info("⏳ Enriquecendo com cadastros (produtos/PDVs)...")
    out = pair_metrics.merge(products, on='produto', how='left', suffixes=('','_prod'))
    out = out.merge(stores, on='pdv', how='left', suffixes=('','_pdv'))
    toc(t0, "Enriquecimento com cadastros")
    logger.info(f"📐 pair_metrics+cadastros: shape={out.shape}")
    return out

def category_summary(pair_metrics_cat: pd.DataFrame) -> pd.DataFrame:
    if 'categoria' not in pair_metrics_cat.columns:
        logger.info("ℹ️ Cadastro de produtos sem coluna 'categoria' — pulando resumo por categoria.")
        return pd.DataFrame()
    t0 = tic()
    logger.info("⏳ Resumindo por categoria de produto...")
    cs = (pair_metrics_cat.groupby('categoria', dropna=False)
          .agg(
              pairs=('produto','count'),
              mean_total_qty=('total_qty','mean'),
              median_total_qty=('total_qty','median'),
              mean_active_weeks=('active_weeks','mean'),
              frac_sparse=('active_weeks', lambda x: float((x<=2).mean())),
              mean_p90=('p90_week','mean'),
              mean_zeros_share=('zeros_share','mean'),
              mean_jan_pos=('jan_pos_weeks','mean')
          ).sort_values('mean_total_qty', ascending=False).reset_index())
    toc(t0, "Resumo por categoria")
    logger.info(f"📐 category_summary: shape={cs.shape}")
    return cs

def identify_villains(pm: pd.DataFrame) -> pd.DataFrame:
    t0 = tic()
    logger.info("⏳ Identificando pares 'vilões' (alto impacto no WMAPE)...")
    q_qty = pm['total_qty'].quantile(0.99) if len(pm)>0 else 0.0
    q_max = pm['max_week'].quantile(0.99) if len(pm)>0 else 0.0
    villains = pm[
        (pm['total_qty'] >= q_qty) |
        (pm['max_week'] >= q_max) |
        ((pm['zeros_share'] >= 0.90) & (pm['jan_pos_weeks'] == 0))
    ].copy()
    toc(t0, "Seleção de vilões")
    logger.info(f"📐 villains: shape={villains.shape} | q_qty@99={q_qty:.2f} | q_max@99={q_max:.2f}")
    return villains

def make_plots(outdir: str, trans: pd.DataFrame, weekly: pd.DataFrame, pair_metrics: pd.DataFrame):
    t0 = tic()
    logger.info("⏳ Gerando gráficos (PNG)...")
    os.makedirs(outdir, exist_ok=True)

    # 1) hist quantidades por transação (clamp para visual)
    plt.figure()
    trans['quantidade'].clip(lower=-50, upper=2000).hist(bins=60)
    plt.title("Distribuição de quantidades por transação (clamp -50..2000)")
    plt.xlabel("quantidade"); plt.ylabel("freq")
    plt.tight_layout(); plt.savefig(os.path.join(outdir,"hist_quantidades_transacao.png")); plt.close()

    # 2) hist log1p de quantidades semanais
    plt.figure()
    logw = np.log1p(weekly['quantidade'].clip(lower=0))
    logw = logw[np.isfinite(logw)]
    plt.hist(logw, bins=60)
    plt.title("Distribuição log1p das quantidades semanais (par-semana)")
    plt.xlabel("log1p(quantidade semanal)"); plt.ylabel("freq")
    plt.tight_layout(); plt.savefig(os.path.join(outdir,"hist_log1p_semanais.png")); plt.close()

    # 3) hist semanas ativas por par
    plt.figure()
    pair_metrics['active_weeks'].hist(bins=52)
    plt.title("Semanas ativas por par no ano")
    plt.xlabel("semanas com venda (>0)"); plt.ylabel("nº de pares")
    plt.tight_layout(); plt.savefig(os.path.join(outdir,"hist_weeks_ativas.png")); plt.close()

    # 4) hist zeros_share por par
    plt.figure()
    pair_metrics['zeros_share'].hist(bins=50)
    plt.title("Proporção de semanas com zero por par")
    plt.xlabel("share de zeros (0..1)"); plt.ylabel("nº de pares")
    plt.tight_layout(); plt.savefig(os.path.join(outdir,"hist_zeros_share.png")); plt.close()

    # 5) hist atividade em janeiro
    plt.figure()
    pair_metrics['jan_pos_weeks'].hist(bins=6, range=(0,6))
    plt.title("Atividade em janeiro (semanas 1..5) por par")
    plt.xlabel("semanas com venda em jan"); plt.ylabel("nº de pares")
    plt.tight_layout(); plt.savefig(os.path.join(outdir,"hist_jan_atividade.png")); plt.close()

    # 6) top-15 produtos
    top_prods = (pair_metrics.groupby('produto',as_index=False)['total_qty']
                 .sum().sort_values('total_qty', ascending=False).head(15))
    plt.figure()
    plt.bar(top_prods['produto'].astype(str), top_prods['total_qty'])
    plt.title("Top 15 produtos por quantidade total (ano)"); plt.xticks(rotation=90)
    plt.tight_layout(); plt.savefig(os.path.join(outdir,"top15_produtos.png")); plt.close()

    # 7) top-15 PDVs
    top_stores = (pair_metrics.groupby('pdv',as_index=False)['total_qty']
                 .sum().sort_values('total_qty', ascending=False).head(15))
    plt.figure()
    plt.bar(top_stores['pdv'].astype(str), top_stores['total_qty'])
    plt.title("Top 15 PDVs por quantidade total (ano)"); plt.xticks(rotation=90)
    plt.tight_layout(); plt.savefig(os.path.join(outdir,"top15_pdvs.png")); plt.close()

    toc(t0, "Geração de gráficos")

def write_console_report(path: str, trans: pd.DataFrame, pair_metrics: pd.DataFrame, villains: pd.DataFrame):
    lines = []
    lines.append("== Console Report: EDA focada em WMAPE ==\n")
    lines.append(f"Transações: rows={len(trans)}, cols={len(trans.columns)}")
    lines.append(f"Período: {trans['data'].min()} .. {trans['data'].max()}")
    negs = int((trans['quantidade']<0).sum())
    zeros = int((trans['quantidade']==0).sum())
    lines.append(f"Negativos (devoluções): {negs}")
    lines.append(f"Zeros (linhas): {zeros} ({(zeros/len(trans))*100:.2f}%)\n")

    lines.append(f"Pares distintos: {pair_metrics[['pdv','produto']].drop_duplicates().shape[0]}")
    lines.append(f"Semanas ativas (mediana): {int(pair_metrics['active_weeks'].median())}")
    lines.append(f"Share de zeros (mediana): {pair_metrics['zeros_share'].median():.3f}")
    lines.append(f"Pares com <=2 semanas ativas: {(pair_metrics['active_weeks']<=2).mean()*100:.2f}%")
    lines.append(f"Pares sem venda em jan (1..5): {(pair_metrics['jan_pos_weeks']==0).mean()*100:.2f}%\n")

    lines.append("Heurísticas que justificam ajustes no pipeline:")
    lines.append(" - Long-tail forte: poucos pares dominam o denominador do WMAPE.")
    lines.append(" - Sparsidade elevada: muitos pares com poucas semanas ativas → threshold tende a zero.")
    lines.append(" - Picos semanais (max/p99) pedem CAP (p90×mult) no pós-processamento.")
    lines.append(" - Devoluções precisam de tratamento (zero/remove/feature).")
    lines.append(f"\nPares 'vilões' listados: {len(villains)} (top 1% por total_qty/max_week ou zeros_share>=0.9 & jan=0)")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def main():
    os.makedirs(OUTDIR, exist_ok=True)

    # 1) Ler dados
    trans = load_transactions(TRANSACTIONS_PATH)
    prods = load_products(PRODUCTS_PATH)
    stores = load_stores(STORES_PATH)

    # 2) EDA
    weekly, trans = compute_weekly(trans)
    weekly_full = build_pair_grid(weekly)
    pm = pair_level_metrics(weekly_full)
    pm_cat = enrich_with_catalogs(pm, prods, stores)
    cs = category_summary(pm_cat)
    villains = identify_villains(pm_cat)

    # 3) Salvar artefatos
    t0 = tic()
    logger.info("⏳ Salvando artefatos (CSVs, PNGs e relatório)...")
    pm_cat.to_csv(os.path.join(OUTDIR, "pair_metrics.csv"), index=False)
    villains.to_csv(os.path.join(OUTDIR, "villains.csv"), index=False)
    if not cs.empty:
        cs.to_csv(os.path.join(OUTDIR, "category_summary.csv"), index=False)
    make_plots(OUTDIR, trans, weekly, pm_cat)
    write_console_report(os.path.join(OUTDIR, "console_report.txt"), trans, pm_cat, villains)
    toc(t0, "Salvamento de artefatos")

    # 4) Mensagem final no console
    print(textwrap.dedent(f"""
    ✅ EDA concluída.
    Saídas em: {OUTDIR}/
      - pair_metrics.csv        # diagnóstico por par (use para cap/blend/thresholds)
      - villains.csv            # pares de maior impacto no WMAPE
      - category_summary.csv    # (se houver 'categoria' no cadastro)
      - *.png                   # gráficos: distribuições e top-15
      - console_report.txt      # resumo textual com achados-chave

    Dicas de uso no pipeline:
      - CAP obrigatório nos pares do 'villains.csv' (p.ex. p90×1.5..2.0).
      - Busca de threshold ponderando Y>0 (evitar threshold~0.95).
      - Aumentar neg_sample_rate e reduzir cls_pos_weight para calibrar ativação.
      - Filtrar pares com active_weeks<=2, se necessário.
    """).strip())

if __name__ == "__main__":
    main()
