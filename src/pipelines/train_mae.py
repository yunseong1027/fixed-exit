import argparse, json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

def time_split_index(n: int, test_ratio=0.2, val_ratio=0.1):
    i_test = int(n * (1 - test_ratio))
    i_val = int(i_test * (1 - val_ratio))
    return slice(0, i_val), slice(i_val, i_test), slice(i_test, n)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/processed/meta_train.parquet")
    ap.add_argument("--alpha", type=float, default=0.9)      # Qα(MAE)
    ap.add_argument("--test-ratio", type=float, default=0.2)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, default="reports/mae_q")
    args = ap.parse_args()

    rng_seed = args.seed
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.data).sort_index()
    X = df.drop(columns=["symbol","y","y_mae","y_R","y_dur"])
    y_mae = df["y_mae"].values
    y_R   = df["y_R"].values

    tr, va, te = time_split_index(len(df), test_ratio=args.test_ratio, val_ratio=args.val_ratio)
    Xtr,Xv,Xte = X.iloc[tr], X.iloc[va], X.iloc[te]
    yq_tr,yq_v,yq_te = y_mae[tr], y_mae[va], y_mae[te]
    yr_tr,yr_v,yr_te = y_R[tr],   y_R[va],   y_R[te]

    # 1) MAE 상위 분위수 회귀
    q = LGBMRegressor(
        objective="quantile", alpha=args.alpha,
        n_estimators=2000, learning_rate=0.03,
        num_leaves=63, min_data_in_leaf=1,
        subsample=0.9, colsample_bytree=0.9, random_state=rng_seed
    )
    q.fit(Xtr, yq_tr, eval_set=[(Xv, yq_v)], eval_metric="l1",
          callbacks=[early_stopping(200), log_evaluation(200)])
    q_pred_va = q.predict(Xv)
    q_pred_te = q.predict(Xte)

    # 2) 기대수익 회귀(선택)
    r = LGBMRegressor(
        objective="l2",
        n_estimators=1500, learning_rate=0.03,
        num_leaves=63, min_data_in_leaf=1,
        subsample=0.9, colsample_bytree=0.9, random_state=rng_seed
    )
    r.fit(Xtr, yr_tr, eval_set=[(Xv, yr_v)], eval_metric="l2",
          callbacks=[early_stopping(200), log_evaluation(200)])
    r_pred_va = r.predict(Xv)
    r_pred_te = r.predict(Xte)

    # diagnostics
    med_abs_err = float(np.median(np.abs(yq_te - q_pred_te)))
    coverage_va = float((yq_v  <= q_pred_va).mean())
    coverage_te = float((yq_te <= q_pred_te).mean())

    # save preds
    pd.DataFrame({"date": Xv.index, "q_mae": q_pred_va, "E_R": r_pred_va}) \
      .set_index("date").to_parquet(Path(args.outdir) / "val_preds.parquet")
    pd.DataFrame({"date": Xte.index, "q_mae": q_pred_te, "E_R": r_pred_te}) \
      .set_index("date").to_parquet(Path(args.outdir) / "test_preds.parquet")

    # save metrics
    with open(Path(args.outdir) / "metrics.json", "w") as f:
        json.dump({
            "alpha": args.alpha,
            "q_mae_median_abs_err": med_abs_err,
            "coverage_val": coverage_va,
            "coverage_test": coverage_te
        }, f, indent=2)

    joblib.dump(q, "models/q_mae.pkl"); joblib.dump(r, "models/r_exp.pkl")
    print(f"[saved] preds -> {args.outdir}/val_preds.parquet, test_preds.parquet")
    print("[saved] models -> models/q_mae.pkl, models/r_exp.pkl")
    print(f"[VAL]  coverage≈{coverage_va:.3f}  (target {args.alpha:.2f})")
    print(f"[TEST] coverage≈{coverage_te:.3f}  (target {args.alpha:.2f})")
    print(f"[TEST] Q{args.alpha:.2f}(MAE) median abs error = {med_abs_err:.6f}")

if __name__ == "__main__":
    main()
