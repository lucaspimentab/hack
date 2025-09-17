from __future__ import annotations
import os, math, json, argparse, numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader

from .utils_io import (
    setup_logger, timer, mem_report, log_df,
    read_any, coerce_transactions, add_calendar, weekly_agg,
    apply_weekly_eda_cleanup, wmape, search_threshold,
    apply_blend_on_villains, mae, rmse
)
from .data import SeqConfig, build_pair_week_matrix, DemandSeqDataset
from .model import DemandTransformer
from .train import TrainConfig, train_model
from .predict import autoreg_predict_jan
from .tune import autotune
import numpy as np

def main():
    import argparse  # garantir import local se não estiver no topo
    # crie o parser ANTES de usar ap.add_argument
    ap = argparse.ArgumentParser()

    ap.add_argument("--resume", type=str, default=None,
                help="Caminho para checkpoint .pt (retoma treino a partir dele)")
    ap.add_argument("--save_ckpt_dir", type=str, default="checkpoints",
                    help="Diretório onde checkpoints serão salvos")


    ap.add_argument("--val_report_csv", type=str, default=None, help="Se informado, salva um CSV detalhado da validação (pares, semanas, erros, contribuições WMAPE)")
    ap.add_argument("--transactions", required=True, help="CSV/Parquet de transações (2022)")
    ap.add_argument("--products", required=False, default=None)
    ap.add_argument("--stores", required=False, default=None)
    ap.add_argument("--out", required=True, help="Arquivo de saída (submission .csv)")

    # Quantidade / devoluções
    ap.add_argument("--quantity_col", type=str, default=None)
    ap.add_argument("--fallback_count_rows", action="store_true")
    ap.add_argument("--treat_negatives", type=str, default="zero", choices=["zero","remove","feature"])
    ap.add_argument("--use_returns_feature", action="store_true")

    # EDA / limpeza (treino/val)
    ap.add_argument("--min_active_weeks", type=int, default=2)
    ap.add_argument("--min_total_qty", type=float, default=2.0)
    ap.add_argument("--max_zeros_share", type=float, default=0.995)
    ap.add_argument("--train_cap_pair_quantile", type=float, default=0.90)
    ap.add_argument("--train_cap_pair_mult", type=float, default=1.5)
    ap.add_argument("--train_global_qclip", type=float, default=0.0)

    # seq / model / train
    ap.add_argument("--context_len", type=int, default=16)
    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--num_layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--emb_dim", type=int, default=32)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--cls_pos_weight", type=float, default=3.0)

    # logging / debug
    ap.add_argument("--log_level", type=str, default="INFO")
    ap.add_argument("--max_train_batches", type=int, default=None)
    ap.add_argument("--max_val_batches", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--pin_memory", action="store_true")

    # filtros/granularidade
    ap.add_argument("--only_jan_pairs", action="store_true")
    ap.add_argument("--max_pairs", type=int, default=None)
    ap.add_argument("--neg_sample_rate", type=float, default=0.10)

    # pós-processamento
    ap.add_argument("--cap_mult_pair_q90", type=float, default=2.0)

    # AutoTune
    ap.add_argument("--auto_tune", action="store_true")
    ap.add_argument("--tune_trials", type=int, default=12)
    ap.add_argument("--tune_epochs", type=int, default=2)
    ap.add_argument("--tune_max_train_batches", type=int, default=1500)

    # Extras: early stopping / threshold / blend vilões
    ap.add_argument("--early_stopping", action="store_true")
    ap.add_argument("--patience", type=int, default=1)
    ap.add_argument("--thr_eval_mode", type=str, default="weighted", choices=["weighted","pos_only","all"])
    ap.add_argument("--blend_villains_csv", type=str, default=None)
    ap.add_argument("--blend_when_zero", action="store_true")

    args = ap.parse_args()
    setup_logger(args.log_level)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    import logging
    logging.info(f"🚀 Iniciando no device={device}")
    mem_report("start")

    # 1) Carrega transações
    with timer("Leitura de transações"):
        trans_raw = read_any(args.transactions); log_df(trans_raw, "trans_raw", head=True)

    with timer("Coerção de colunas de transações"):
        trans = coerce_transactions(trans_raw, quantity_col=args.quantity_col, fallback_count_rows=bool(args.fallback_count_rows))
        log_df(trans, "trans coerced", head=True)

    # 2) Tratamento de negativos / devoluções
    with timer("Tratamento de negativos / devoluções"):
        trans["is_return"] = (trans["quantidade"] < 0).astype(int)
        if args.treat_negatives == "remove":
            before = len(trans); trans = trans[trans["quantidade"] >= 0].copy()
            logging.info(f"Removidas {before - len(trans)} linhas com quantidade negativa.")
        elif args.treat_negatives == "zero":
            nneg = int((trans["quantidade"] < 0).sum())
            trans.loc[trans["quantidade"] < 0, "quantidade"] = 0.0
            logging.info(f"Clampeadas {nneg} quantidades negativas para zero.")
        elif args.treat_negatives == "feature":
            logging.info("Mantendo valores negativos; usando flag de devolução como feature.")

    # 3) Filtro 2022 + calendário
    with timer("Filtrar 2022 + adicionar calendário"):
        trans = trans[(trans["data"] >= "2022-01-01") & (trans["data"] < "2023-01-01")].copy()
        if len(trans) == 0:
            logging.error("Sem dados em 2022 após filtro de datas. Verifique o dataset.")
        trans = add_calendar(trans); log_df(trans, "trans 2022 + cal", head=True)

    # 4) Agrega semanalmente
    with timer("Agregação semanal"):
        returns_col = "is_return" if args.use_returns_feature else None
        weekly = weekly_agg(trans, returns_col=returns_col); log_df(weekly, "weekly", head=True)

    # 4.1) EDA numérica
    with timer("Limpeza EDA (pares/quantidades para treino/val)"):
        weekly = apply_weekly_eda_cleanup(
            weekly=weekly,
            min_active_weeks=int(args.min_active_weeks),
            min_total_qty=float(args.min_total_qty),
            max_zeros_share=float(args.max_zeros_share),
            train_cap_pair_quantile=float(args.train_cap_pair_quantile),
            train_cap_pair_mult=float(args.train_cap_pair_mult),
            train_global_qclip=float(args.train_global_qclip),
            logger=logging,
        )
        log_df(weekly, "weekly (após EDA)", head=True)

    # 5) Filtros de pares
    if args.only_jan_pairs:
        jan = weekly[weekly["semana_iso"].isin([1,2,3,4,5])]
        jan_pairs = jan.loc[jan["quantidade"]>0, ["pdv","produto"]].drop_duplicates()
        before = len(weekly[["pdv","produto"]].drop_duplicates())
        weekly = weekly.merge(jan_pairs.assign(_keep=1), on=["pdv","produto"], how="inner")
        logging.info(f"✂️  only_jan_pairs=True → pares: {before:,} → {len(jan_pairs):,}")
    if args.max_pairs is not None and args.max_pairs > 0:
        tot = (weekly.groupby(["pdv","produto"], as_index=False)["quantidade"].sum()
                 .sort_values("quantidade", ascending=False))
        top_pairs = tot.head(int(args.max_pairs))[["pdv","produto"]]
        before = len(weekly[["pdv","produto"]].drop_duplicates())
        weekly = weekly.merge(top_pairs.assign(_keep=1), on=["pdv","produto"], how="inner")
        logging.info(f"🎯 max_pairs={args.max_pairs} → pares: {before:,} → {len(top_pairs):,}")

    # 5.5) AutoTune (opcional)
    if args.auto_tune:
        best_cfg = autotune(args, weekly.copy(), device=device)
        for k, v in (best_cfg or {}).items():
            setattr(args, k, v)

    # 6) Grade 1..52
    seq_cfg = SeqConfig(context_len=int(args.context_len),
                        use_returns_feature=bool(args.use_returns_feature),
                        neg_sample_rate=float(args.neg_sample_rate))
    with timer("Construir grade por par/semana 1..52"):
        pairs, M = build_pair_week_matrix(weekly, include_returns=bool(args.use_returns_feature))
        M["quantidade"] = np.maximum(np.nan_to_num(M["quantidade"].astype(float), nan=0.0), 0.0)
        if seq_cfg.use_returns_feature:
            M["devol_flag"] = np.clip(np.nan_to_num(M.get("devol_flag", 0)).astype(float), 0.0, 1.0)
        log_df(pairs, "pairs"); log_df(M, "M (pair-week grid)", head=True)

    # 7) Datasets
    with timer("Criar datasets (train/val/predict)"):
        ds_train = DemandSeqDataset(M, seq_cfg, split="train")
        ds_val   = DemandSeqDataset(M, seq_cfg, split="val")
        ds_pred  = DemandSeqDataset(M, seq_cfg, split="predict")
        logging.info(f"📦 Dataset sizes -> train={len(ds_train)} | val={len(ds_val)} | predict={len(ds_pred)}")

    # 8) Modelo
    with timer("Construir modelo"):
        num_pdv  = len(ds_train.pdv2id) or 1
        num_prod = len(ds_train.prod2id) or 1
        in_dim = (1 + (1 if seq_cfg.use_returns_feature else 0)) + 2
        model = DemandTransformer(in_dim=in_dim, d_model=int(args.d_model), nhead=int(args.nhead),
                                  num_layers=int(args.num_layers), dropout=float(args.dropout),
                                  num_pdv=num_pdv, num_prod=num_prod, emb_dim=int(args.emb_dim))
        logging.info(f"🧱 Model: in_dim={in_dim} d_model={args.d_model} nhead={args.nhead} "
                     f"layers={args.num_layers} emb_dim={args.emb_dim} "
                     f"| num_pdv={num_pdv} num_prod={num_prod}")
        mem_report("after model build")

    # 9) Prior na cabeça de ativação
    with timer("Inicialização do viés (prior) da cabeça de ativação"):
        tmp_loader = DataLoader(ds_train, batch_size=4096, shuffle=False, num_workers=0)
        pos, tot = 0.0, 0.0
        with torch.no_grad():
            for b in tmp_loader:
                pos += float((b["y_active"] > 0.5).sum().item()); tot += float(len(b["y_active"]))
        prior = max(1e-6, min(1.0 - 1e-6, (pos / tot) if tot > 0 else 0.01))
        logit = math.log(prior / (1.0 - prior))
        with torch.no_grad():
            if hasattr(model.head_act[-2], 'bias') and model.head_act[-2].bias is not None:
                model.head_act[-2].bias.fill_(logit)
        logging.info(f"🔥 prior={prior:.4f} (logit={logit:.3f}) | train_samples={int(tot)} | val_samples={len(ds_val)}")

    # 10) Treino
    tcfg = TrainConfig(
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        cls_pos_weight=float(args.cls_pos_weight),
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory),
        thr_eval_mode=args.thr_eval_mode,
        early_stopping=bool(args.early_stopping),
        patience=int(args.patience),
        resume=args.resume,
        save_ckpt_dir=args.save_ckpt_dir,
    )
    with timer("Treinamento total"):
        best = train_model(model, ds_train, ds_val, tcfg, device=device)

    # 11) Validação final (Jan/2022)
    with timer("Validação final Jan/2022 (valor esperado) + relatório"):
        val_loader = DataLoader(
            ds_val,
            batch_size=tcfg.batch_size,
            shuffle=False,
            num_workers=tcfg.num_workers,
            pin_memory=tcfg.pin_memory
        )

        model.eval()
        preds, trues = [], []
        all_p, all_size = [], []
        all_week, all_pdv_id, all_prod_id = [], [], []

        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device, non_blocking=tcfg.pin_memory)
                pdv_id = batch["pdv_id"].to(device, non_blocking=tcfg.pin_memory)
                prod_id = batch["prod_id"].to(device, non_blocking=tcfg.pin_memory)

                p_active, yhat_log = model(x, pdv_id, prod_id)
                size = torch.relu(torch.expm1(yhat_log)).cpu().numpy()
                p = torch.clamp(p_active, 1e-6, 1-1e-6).cpu().numpy()
                y = batch["y"].cpu().numpy()

                preds.append(p * size)  # valor esperado
                trues.append(y)

                all_p.append(p); all_size.append(size)
                all_week.append(batch["t_week"].cpu().numpy())
                all_pdv_id.append(batch["pdv_id"].cpu().numpy())
                all_prod_id.append(batch["prod_id"].cpu().numpy())

        # Arrays consolidados
        Yhat  = np.concatenate(preds, axis=0) if preds else np.array([])
        Ytrue = np.concatenate(trues, axis=0) if trues else np.array([])
        P     = np.concatenate(all_p, axis=0) if all_p else np.array([])
        SZ    = np.concatenate(all_size, axis=0) if all_size else np.array([])
        W     = np.concatenate(all_week, axis=0) if all_week else np.array([])
        PDV   = np.concatenate(all_pdv_id, axis=0) if all_pdv_id else np.array([])
        PROD  = np.concatenate(all_prod_id, axis=0) if all_prod_id else np.array([])

        # ====== Busca do threshold ótimo (gated) ANTES de logar/CSV ======
        def _wmape_gate(p, sz, y, thr):
            ypred = np.where(p >= thr, sz, 0.0)
            denom = np.abs(y).sum()
            return 100.0 * np.abs(y - ypred).sum() / (denom + 1e-9)

        if Ytrue.size == 0:
            best_thr = 0.0
            Yhat_thr = np.zeros_like(Yhat)
            val_wmape_exp = float("nan")
            val_wmape_gat = float("nan")
            val_mae = float("nan")
            val_rmse = float("nan")
        else:
            _thr_grid = np.unique(np.concatenate([
                np.array([0.00, 0.01, 0.02, 0.05]),
                np.linspace(0.10, 0.95, 18)
            ]))
            best_thr = float(min(_thr_grid, key=lambda t: _wmape_gate(P, SZ, Ytrue, t)))
            Yhat_thr = np.where(P >= best_thr, SZ, 0.0)

            val_wmape_exp = wmape(Ytrue, Yhat)      # p*size
            val_wmape_gat = wmape(Ytrue, Yhat_thr)  # gated
            val_mae       = mae(Ytrue, Yhat)
            val_rmse      = rmse(Ytrue, Yhat)

        logging.info(f"🔎 [VAL] Threshold ótimo (global) = {best_thr:.3f}")
        logging.info(f"🏁 [VAL] WMAPE (valor esperado p*size) = {val_wmape_exp:.3f}%")
        logging.info(f"🏁 [VAL] WMAPE (gated thr={best_thr:.3f}) = {val_wmape_gat:.3f}%")
        logging.info(f"📏 [VAL] MAE={val_mae:.4f} | RMSE={val_rmse:.4f}")
        mem_report("after final val")

        # ====== Relatório detalhado (CSV) ======
        inv_pdv  = {v:k for k,v in ds_val.pdv2id.items()}
        inv_prod = {v:k for k,v in ds_val.prod2id.items()}
        pdv_vals  = [inv_pdv.get(int(i), int(i)) for i in PDV]
        prod_vals = [inv_prod.get(int(i), int(i)) for i in PROD]

        df_val = pd.DataFrame({
            "pdv": pdv_vals,
            "produto": prod_vals,
            "semana": W.astype(int),
            "y_true": Ytrue.astype(float),
            "p_active": P.astype(float),
            "y_size_hat": SZ.astype(float),
            "y_pred": Yhat.astype(float),           # p*size
            "y_pred_gated": Yhat_thr.astype(float)  # gated (threshold ótimo)
        })
        df_val["abs_err"] = (df_val["y_true"] - df_val["y_pred"]).abs()
        denom = df_val["y_true"].abs().sum()
        df_val["wmape_contrib_%"] = 100.0 * df_val["abs_err"] / (denom if denom > 0 else 1.0)

        out_csv = args.val_report_csv or "eda_out/val_report.csv"
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        df_val.to_csv(out_csv, sep=";", index=False, encoding="utf-8")
        logging.info(f"🧾 Relatório de validação salvo em: {out_csv} (linhas={len(df_val):,})")

        # Resumos úteis
        top_pairs = (df_val.groupby(["pdv","produto"], as_index=False)["wmape_contrib_%"].sum()
                        .sort_values("wmape_contrib_%", ascending=False).head(20))
        logging.info("🔝 Top-20 pares por contribuição no WMAPE:")
        logging.info("\n" + top_pairs.to_string(index=False))

        df_cal = df_val.copy()
        df_cal["has_demand"] = (df_cal["y_true"] > 0).astype(int)
        df_cal["p_bin"] = pd.qcut(df_cal["p_active"], q=10, duplicates="drop")
        calib = (df_cal.groupby("p_bin", observed=False)
                .agg(p_mean=("p_active","mean"),
                    freq_pos=("has_demand","mean"),
                    n=("has_demand","size"))
                .reset_index()
                .sort_values("p_mean"))
        logging.info("📈 Calibração (p médio vs. frequência real de positivos) por decil:")
        logging.info("\n" + calib.to_string(index=False))

        def _wmape_group(g):
            return 100.0 * g["abs_err"].sum() / (g["y_true"].abs().sum() + 1e-9)

        by_week = (
            df_val.groupby("semana", as_index=False)
                .apply(lambda g: pd.Series({"wmape_sem": _wmape_group(g), "n": len(g)}),
                        include_groups=False)
                .reset_index(drop=True)
        )
        logging.info("🗓️  WMAPE por semana (val):")
        logging.info("\n" + by_week.to_string(index=False))





    # 12) Predição autoregressiva (1..5)
    with timer("Predição autoregressiva (semanas 1..5)"):
        sub = autoreg_predict_jan(model, ds_pred, seq_cfg, best_thr=float(best_thr), device=device)
        log_df(sub, "submission (pré-CAP)", head=True)

    # 13) CAP por p90 do par nas previsões
    if args.cap_mult_pair_q90 is not None and float(args.cap_mult_pair_q90) > 0:
        with timer("Aplicar CAP por p90 do par (pós-processamento)"):
            p90 = (M.groupby(["pdv","produto"])["quantidade"].quantile(0.90)
                     .rename("q90").reset_index())
            sub = sub.merge(p90, on=["pdv","produto"], how="left")
            sub["cap"] = (sub["q90"] * float(args.cap_mult_pair_q90)).replace([np.inf, -np.inf], np.inf).fillna(np.inf)
            sub["quantidade"] = np.minimum(sub["quantidade"].astype(float), sub["cap"].astype(float)).astype(int)
            sub = sub.drop(columns=["q90","cap"])
            log_df(sub, "submission (pós-CAP)", head=True)

    # 13.5) Blend com baseline nos vilões (opcional)
    if args.blend_villains_csv:
        with timer("Blend baseline nos vilões (pós-CAP)"):
            weekly_hist = M[['pdv','produto','semana_iso','quantidade']].copy()
            sub_before = sub.copy()
            sub = apply_blend_on_villains(
                sub_df=sub,
                villains_csv=args.blend_villains_csv,
                weekly_df=weekly_hist,
                when_zero=args.blend_when_zero
            )
            changed = int((sub_before['quantidade'].values != sub['quantidade'].values).sum())
            logging.info(f"🔁 Blend com baseline aplicado nos vilões; {changed} linhas alteradas")

    # 14) Salvar submissão
    with timer("Salvar submissão"):
        out_path = args.out
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        sub.to_csv(out_path, sep=";", index=False, encoding="utf-8")
        logging.info(f"📤 Submissão salva em: {out_path} com {len(sub):,} linhas.")

    # 15) Salvar metadados
    with timer("Salvar metadados (.meta.json)"):
        meta = {
            "best_val_wmape_jan_2022": float(val_wmape_gat),
            "best_thr": float(best_thr),
            "hparams": {
                "context_len": int(args.context_len),
                "d_model": int(args.d_model), "nhead": int(args.nhead), "num_layers": int(args.num_layers),
                "dropout": float(args.dropout), "emb_dim": int(args.emb_dim),
                "batch_size": int(args.batch_size), "epochs": int(args.epochs),
                "lr": float(args.lr), "weight_decay": float(args.weight_decay),
                "cls_pos_weight": float(args.cls_pos_weight),
                "cap_mult_pair_q90": float(args.cap_mult_pair_q90) if args.cap_mult_pair_q90 is not None else None,
                "quantity_col": args.quantity_col,
                "fallback_count_rows": bool(args.fallback_count_rows),
                "treat_negatives": args.treat_negatives,
                "use_returns_feature": bool(args.use_returns_feature),
                "min_active_weeks": int(args.min_active_weeks),
                "min_total_qty": float(args.min_total_qty),
                "max_zeros_share": float(args.max_zeros_share),
                "train_cap_pair_quantile": float(args.train_cap_pair_quantile),
                "train_cap_pair_mult": float(args.train_cap_pair_mult),
                "train_global_qclip": float(args.train_global_qclip),
                "only_jan_pairs": bool(args.only_jan_pairs),
                "max_pairs": args.max_pairs,
                "neg_sample_rate": float(args.neg_sample_rate),
                "auto_tune": bool(args.auto_tune),
                "tune_trials": int(args.tune_trials),
                "tune_epochs": int(args.tune_epochs),
                "tune_max_train_batches": int(args.tune_max_train_batches),
                "log_level": args.log_level,
                "max_train_batches": args.max_train_batches,
                "max_val_batches": args.max_val_batches,
                "num_workers": args.num_workers,
                "pin_memory": bool(args.pin_memory),
                "thr_eval_mode": args.thr_eval_mode,
                "early_stopping": bool(args.early_stopping),
                "patience": int(args.patience),
            }
        }
        meta_path = os.path.splitext(out_path)[0] + ".meta.json"
        with open(meta_path,"w",encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        logging.info(f"🧾 Meta salva: {meta_path}")

    logging.info("🎉 Pipeline finalizado com sucesso.")
    mem_report("end")

if __name__ == "__main__":
    main()