import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

def slice_by_dates(df, start, end):
    return df.loc[(df.index >= pd.to_datetime(start)) & (df.index < pd.to_datetime(end))]

def year_add(d, years):
    d = pd.to_datetime(d)
    return (d + pd.DateOffset(years=years)).strftime("%Y-%m-%d")

def make_folds(dates, start, end, train_y=5, val_y=1, test_y=1):
    win = []
    s = pd.to_datetime(start)
    e = pd.to_datetime(end)
    while True:
        tr_s = s
        tr_e = pd.to_datetime(year_add(tr_s, train_y))
        va_e = pd.to_datetime(year_add(tr_e, val_y))
        te_e = pd.to_datetime(year_add(va_e, test_y))
        if te_e > e: break
        win.append((tr_s, tr_e, va_e, te_e))
        s = pd.to_datetime(year_add(s, test_y))  # 슬라이딩
    return win

def train_q(X_tr, yq_tr, X_va, yq_va, X_te, alpha, seed):
    q = LGBMRegressor(
        objective="quantile", alpha=alpha,
        n_estimators=2000, learning_rate=0.03,
        num_leaves=63, min_data_in_leaf=1,
        subsample=0.9, colsample_bytree=0.9, random_state=seed
    )
    q.fit(X_tr, yq_tr, eval_set=[(X_va, yq_va)], eval_metric="l1",
          callbacks=[early_stopping(200), log_evaluation(200)])
    return q, q.predict(X_va), q.predict(X_te)

def train_er(X_tr, yR_tr, X_va, yR_va, X_te, seed):
    r = LGBMRegressor(
        objective="l2",
        n_estimators=1500, learning_rate=0.03,
        num_leaves=63, min_data_in_leaf=1,
        subsample=0.9, colsample_bytree=0.9, random_state=seed
    )
    r.fit(X_tr, yR_tr, eval_set=[(X_va, yR_va)], eval_metric="l2",
          callbacks=[early_stopping(200), log_evaluation(200)])
    return r, r.predict(X_va), r.predict(X_te)

def quantile_offset(y_true, q_pred, p):
    """Split-conformal style constant offset: delta = quantile_p(y - qhat)."""
    resid = (np.asarray(y_true) - np.asarray(q_pred))
    resid = resid[np.isfinite(resid)]
    if len(resid) == 0: return 0.0
    return float(np.quantile(resid, p))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/meta_train.parquet")
    ap.add_argument("--start", default="2008-01-01")
    ap.add_argument("--end",   default="2025-09-30")
    ap.add_argument("--train-years", type=int, default=5)
    ap.add_argument("--val-years",   type=int, default=1)
    ap.add_argument("--test-years",  type=int, default=1)
    ap.add_argument("--alpha", dest="alphas", action="append", type=float,
                    help="repeatable: --alpha 0.85 --alpha 0.90 --alpha 0.95")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="reports/wf_mae")
    ap.add_argument("--conformal", action="store_true",
                    help="compute delta on val: delta = quantile_p(y - qhat)")
    ap.add_argument("--cf-p", type=float, default=None,
                    help="quantile level p for delta (default: alpha)")
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    alphas = args.alphas if args.alphas else [0.90]

    df = pd.read_parquet(args.data).sort_index()
    feats = [c for c in df.columns if c not in ("symbol","y","y_mae","y_R","y_dur")]
    Xall = df[feats]; y_mae_all = df["y_mae"]; yR_all = df["y_R"]

    folds = make_folds(df.index, args.start, args.end,
                       train_y=args.train_years, val_y=args.val_years, test_y=args.test_years)

    meta = []
    for k,(tr_s,tr_e,va_e,te_e) in enumerate(folds,1):
        trX = slice_by_dates(Xall, tr_s, tr_e);     tr_yq = slice_by_dates(y_mae_all, tr_s, tr_e); tr_yR = slice_by_dates(yR_all, tr_s, tr_e)
        vaX = slice_by_dates(Xall, tr_e, va_e);     va_yq = slice_by_dates(y_mae_all, tr_e, va_e); va_yR = slice_by_dates(yR_all, tr_e, va_e)
        teX = slice_by_dates(Xall, va_e, te_e);     te_yq = slice_by_dates(y_mae_all, va_e, te_e); te_yR = slice_by_dates(yR_all, va_e, te_e)
        if len(teX)==0 or len(vaX)==0 or len(trX)==0: 
            continue

        # E[R]는 fold당 한 번 학습
        r, r_va, r_te = train_er(trX, tr_yR, vaX, va_yR, teX, args.seed)
        pd.DataFrame({"date": vaX.index, "E_R": r_va}).set_index("date").to_parquet(Path(args.outdir)/f"fold{k:02d}_val_E_R.parquet")
        pd.DataFrame({"date": teX.index, "E_R": r_te}).set_index("date").to_parquet(Path(args.outdir)/f"fold{k:02d}_test_E_R.parquet")

        # 각 α에 대해 q_mae 학습/저장 + (옵션) delta 저장
        cover = {}
        delta_dict = {}
        for a in alphas:
            q, q_va, q_te = train_q(trX, tr_yq, vaX, va_yq, teX, a, args.seed)
            tag = f"a{int(round(a*100)):02d}"
            # preds
            pd.DataFrame({"date": vaX.index, "q_mae": q_va}).set_index("date").to_parquet(Path(args.outdir)/f"fold{k:02d}_val_q_{tag}.parquet")
            pd.DataFrame({"date": teX.index, "q_mae": q_te}).set_index("date").to_parquet(Path(args.outdir)/f"fold{k:02d}_test_q_{tag}.parquet")
            # coverage
            cov_va = float((va_yq.values <= q_va).mean())
            cov_te = float((te_yq.values <= q_te).mean())
            cover[tag] = {"coverage_val":cov_va, "coverage_test":cov_te}

            # delta (Block-Conformal Offset)
            if args.conformal:
                p = a if (args.cf_p is None) else args.cf_p
                delta = quantile_offset(va_yq.values, q_va, p=p)
                delta_dict[tag] = delta
                # store a small json per-fold per-alpha (optional but handy)
                Path(args.outdir).mkdir(parents=True, exist_ok=True)
                (Path(args.outdir)/f"fold{k:02d}_val_q_{tag}_delta.json").write_text(json.dumps({"delta": delta}, indent=2), encoding="utf-8")

        meta.append({
            "fold": k,
            "train": [str(tr_s),str(tr_e)],
            "val":   [str(tr_e),str(va_e)],
            "test":  [str(va_e),str(te_e)],
            "n_tr": len(trX), "n_va": len(vaX), "n_te": len(teX),
            "alphas": alphas,
            "coverage": cover,
            "delta": delta_dict
        })

    with open(Path(args.outdir)/"wf_meta.json","w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[saved] wf_meta -> {args.outdir}/wf_meta.json")
    print(f"[saved] preds per fold per alpha -> {args.outdir}/fold##_val_q_aXX.parquet & fold##_test_q_aXX.parquet + E_R")
    if args.conformal:
        print("[note] conformal delta saved alongside (fold##_val_q_{tag}_delta.json)")

if __name__ == "__main__":
    main()
