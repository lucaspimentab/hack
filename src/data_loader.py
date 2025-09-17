from __future__ import annotations
import pandas as pd
import os

def _read_any(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".parquet", ".pq"]:
        try:
            return pd.read_parquet(path)
        except Exception:
            return pd.read_parquet(path, engine="pyarrow")
    elif ext in [".csv", ".txt"]:
        # tenta ; depois ,
        try:
            return pd.read_csv(path, sep=';', encoding='utf-8')
        except Exception:
            return pd.read_csv(path, sep=',', encoding='utf-8')
    else:
        raise ValueError(f"Formato não suportado: {ext}")

def load_transactions(path: str) -> pd.DataFrame:
    df = _read_any(path)
    df.columns = [c.strip().lower() for c in df.columns]

    # pdv/produto
    if 'internal_store_id' in df.columns: df = df.rename(columns={'internal_store_id':'pdv'})
    if 'internal_product_id' in df.columns: df = df.rename(columns={'internal_product_id':'produto'})
    if 'sku' in df.columns and 'produto' not in df.columns: df = df.rename(columns={'sku':'produto'})
    if 'product' in df.columns and 'produto' not in df.columns: df = df.rename(columns={'product':'produto'})
    if 'codigo' in df.columns and 'produto' not in df.columns: df = df.rename(columns={'codigo':'produto'})

    # resolver coluna de data (prioridade para transaction_date)
    date_col = None
    if 'transaction_date' in df.columns:
        date_col = 'transaction_date'
    elif 'purchase_date' in df.columns:
        date_col = 'purchase_date'
    elif 'reference_date' in df.columns:
        date_col = 'reference_date'

    if date_col is not None:
        df = df.rename(columns={date_col: 'data'})
    # remover outras colunas de data para evitar duplicação
    for other in ['transaction_date','purchase_date','reference_date']:
        if other in df.columns and other != 'data':
            df = df.drop(columns=[other])

    # quantidade e faturamento
    if 'quantity' in df.columns: df = df.rename(columns={'quantity':'quantidade'})
    if 'qtd' in df.columns: df = df.rename(columns={'qtd':'quantidade'})
    if 'gross_value' in df.columns: df = df.rename(columns={'gross_value':'faturamento'})
    elif 'valor' in df.columns: df = df.rename(columns={'valor':'faturamento'})
    elif 'net_value' in df.columns: df = df.rename(columns={'net_value':'faturamento'})

    required = ['data','pdv','produto','quantidade']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Transações precisam conter {required}. Faltando: {missing} | cols={list(df.columns)}")

    # se por algum motivo houver duas colunas chamadas 'data', mantenha a primeira
    if isinstance(df['data'], pd.DataFrame):
        df['data'] = pd.to_datetime(df['data'].iloc[:,0], errors='coerce')
    else:
        df['data'] = pd.to_datetime(df['data'], errors='coerce')
    return df

def load_products(path: str) -> pd.DataFrame:
    df = _read_any(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if 'produto' not in df.columns:
        for cand in ['sku','id_produto','product','codigo','internal_product_id']:
            if cand in df.columns:
                df = df.rename(columns={cand:'produto'})
                break
    if 'produto' not in df.columns:
        raise ValueError(f"Cadastro de produtos deve conter 'produto'. cols={list(df.columns)}")
    return df

def load_stores(path: str) -> pd.DataFrame:
    df = _read_any(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if 'pdv' not in df.columns:
        for cand in ['loja','id_pdv','store','internal_store_id']:
            if cand in df.columns:
                df = df.rename(columns={cand:'pdv'})
                break
    if 'pdv' not in df.columns:
        raise ValueError(f"Cadastro de PDVs deve conter 'pdv'. cols={list(df.columns)}")
    return df
