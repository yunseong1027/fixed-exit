import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


# ---------------------------
# Utils
# ---------------------------
def seed_everything(seed: int = 42):
    np.random.seed(seed)


def encode_multiclass(y: pd.Series):
    """{-1,0,1} -> {0,1,2} 매핑"""
    label2id = {-1: 0, 0: 1, 1: 2}
    id2label = {v: k for k, v in label2id.items()}
    y_id = y.map(label2id).astype(int).values
    return y_id, id2label, label2id


def time_split_index(n: int, test_ratio=0.2, val_ratio=0.1):
    """시간순 분할: [0:train) [train:val) [val:test) (val은 train의 꼬리)"""
    i_test = int(n * (1 - test_ratio))
    i_val = int(i_test * (1 - val_ratio))
    return slice(0, i_val), slice(i_val, i_test), slice(i_test, n)


def expected_calibration_error_binary(p_hat: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    """positive-class 확률 기준 ECE"""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(p_hat, bins) - 1
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        m = idx == b
        if not np.any(m):
            continue
        conf = p_hat[m].mean()
        acc = y_true[m].mean()
        ece += float(abs(conf - acc)) * (m.sum() / n)
    return float(ece)


def brier_multiclass(y_true_id: np.ndarray, proba: np.ndarray, n_classes: int) -> float:
    """멀티클래스 Brier (mean(sum_k (p_k - y_k)^2))"""
    Y = label_binarize(y_true_id, classes=list(range(n_classes)))
    return float(np.mean(np.sum((proba - Y) ** 2, axis=1)))


def parse_pos_weight(option: str, y_train: np.ndarray, task: str) -> float | None:
    """
    --pos-weight: 'auto' | 'none' | float 문자열
    binary에서만 사용. multiclass면 None.
    """
    if task != "binary":
        return None
    opt = option.strip().lower()
    if opt == "none":
        return None
    if opt == "auto":
        pos_rate = float(y_train.mean())
        if pos_rate <= 0:
            return None
        return (1.0 - pos_rate) / max(pos_rate, 1e-6)
    try:
        return float(option)
    except Exception:
        return None


# ---------------------------
# Training / Calibration
# ---------------------------
def fit_lgbm(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    task: str,
    seed: int,
    n_estimators: int,
    learning_rate: float,
    num_leaves: int,
    min_data_in_leaf: int,
    subsample: float,
    colsample_bytree: float,
    pos_weight: float | None,
    es_rounds: int,
    log_every: int,
):
    params = dict(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_data_in_leaf=min_data_in_leaf,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        feature_pre_filter=False,   # 분할 억제 완화
        random_state=seed,
        n_jobs=-1,
    )
    if task == "binary":
        clf = LGBMClassifier(objective="binary", **params)
        if pos_weight is not None:
            clf.set_params(scale_pos_weight=pos_weight)
        clf.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="binary_logloss",
            callbacks=[early_stopping(es_rounds), log_evaluation(log_every)],
        )
        return clf
    else:
        n_classes = len(np.unique(y_train))
        clf = LGBMClassifier(objective="multiclass", num_class=n_classes, **params)
        clf.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="multi_logloss",
            callbacks=[early_stopping(es_rounds), log_evaluation(log_every)],
        )
        return clf


def calibrate(clf, X_val: pd.DataFrame, y_val: np.ndarray, method: str):
    """CalibratedClassifierCV 로 교정. method: 'none'|'sigmoid'|'isotonic'
       단, 검증셋이 단일 클래스면 교정 스킵"""
    if method == "none":
        return clf, None
    if len(np.unique(y_val)) < 2:
        print("[calib] skipped: validation has a single class.")
        return clf, None
    cal = CalibratedClassifierCV(clf, method=("sigmoid" if method == "sigmoid" else "isotonic"), cv="prefit")
    cal.fit(X_val, y_val)
    return cal, cal


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/processed/train.parquet")
    ap.add_argument("--task", type=str, choices=["binary", "multiclass"], default="binary")
    ap.add_argument("--target", type=str, choices=["up", "down", "notup"], default="up",
                    help="binary에서 양성 정의: up(+1) / down(-1) / notup(!=+1)")
    ap.add_argument("--calib", type=str, choices=["none", "sigmoid", "isotonic"], default="none")
    ap.add_argument("--test-ratio", type=float, default=0.2)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, default="reports/run1")
    # 모델 파라미터(소표본 친화 기본값)
    ap.add_argument("--n-estimators", type=int, default=800)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--num-leaves", type=int, default=63)
    ap.add_argument("--min-leaf", type=int, default=1)
    ap.add_argument("--subsample", type=float, default=0.9)
    ap.add_argument("--colsample", type=float, default=0.9)
    ap.add_argument("--pos-weight", type=str, default="none", help="'auto'|'none'|float (binary만)")
    ap.add_argument("--early-stopping", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=200)
    args = ap.parse_args()

    seed_everything(args.seed)
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    # ----- 데이터 -----
    df = pd.read_parquet(args.data).sort_index()
    if df.empty:
        raise RuntimeError(f"empty dataset: {args.data}")
    X = df.drop(columns=["symbol", "y"])
    y_raw = df["y"].astype(int)

    # 항상 up 라벨(평가용)은 별도로 만들어 둔다
    y_up_all = (y_raw == 1).astype(int).values

    # 학습 타깃 생성
    if args.task == "binary":
        if args.target == "up":
            y_all = (y_raw == 1).astype(int).values
        elif args.target == "down":
            y_all = (y_raw == -1).astype(int).values
        else:  # notup
            y_all = (y_raw != 1).astype(int).values
    else:
        y_all, id2label, label2id = encode_multiclass(y_raw)
        plus_one_id = label2id[1]  # +1 클래스 id

    # 시간순 분할
    tr_idx, val_idx, te_idx = time_split_index(len(df), test_ratio=args.test_ratio, val_ratio=args.val_ratio)
    X_train, X_val, X_test = X.iloc[tr_idx], X.iloc[val_idx], X.iloc[te_idx]
    y_train, y_val, y_test = y_all[tr_idx], y_all[val_idx], y_all[te_idx]
    y_up_train, y_up_val, y_up_te = y_up_all[tr_idx], y_up_all[val_idx], y_up_all[te_idx]
    dates_test = X_test.index
    symbols_test = df.iloc[te_idx]["symbol"].values

    print(f"[split] train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
    print(f"[pos-rate up] train={y_up_train.mean():.3f}  val={y_up_val.mean():.3f}  test={y_up_te.mean():.3f}")
    if args.task == "binary":
        print(f"[target] {args.target}")

    # pos weight 파싱 (binary만)
    pos_w = parse_pos_weight(args.pos_weight, y_train if args.task == "binary" else np.array([]), args.task)
    if args.task == "binary":
        print(f"[pos-weight] {args.pos_weight} -> {pos_w}")

    # ----- 학습 -----
    clf = fit_lgbm(
        X_train, y_train, X_val, y_val, args.task, args.seed,
        n_estimators=args.n_estimators, learning_rate=args.lr,
        num_leaves=args.num_leaves, min_data_in_leaf=args.min_leaf,
        subsample=args.subsample, colsample_bytree=args.colsample,
        pos_weight=pos_w, es_rounds=args.early_stopping, log_every=args.log_every
    )

    # (선택) 교정
    predictor, calibrator = calibrate(clf, X_val, y_val, args.calib)

    # 예측
    proba_val = predictor.predict_proba(X_val)
    proba_te  = predictor.predict_proba(X_test)

    # 확률 분포 요약(수축/상수예측 감지)
    def summarize_binary(p):
        pp = p[:, 1]
        if args.task == "binary" and args.target != "up":
            # down/notup으로 학습했다면 보수로 p_up을 만든다
            pp = 1.0 - pp
        return float(pp.mean()), float(pp.std())

    if args.task == "binary":
        m_val, s_val = summarize_binary(proba_val)
        m_te,  s_te  = summarize_binary(proba_te)
        print(f"[proba stats] p_up  val: mean={m_val:.4f}, std={s_val:.4f} | test: mean={m_te:.4f}, std={s_te:.4f}")
    else:
        m_val = float(proba_val[:, plus_one_id].mean()); s_val = float(proba_val[:, plus_one_id].std())
        m_te  = float(proba_te[:,  plus_one_id].mean()); s_te  = float(proba_te[:,  plus_one_id].std())
        print(f"[proba stats] p_up  val: mean={m_val:.4f}, std={s_val:.4f} | test: mean={m_te:.4f}, std={s_te:.4f}")

    # ----- 평가 (항상 p_up 기준) -----
    metrics = {}

    if args.task == "binary":
        # p_up 복원
        p_up_val = proba_val[:, 1]
        p_up_te  = proba_te[:, 1]
        if args.target != "up":
            p_up_val = 1.0 - p_up_val
            p_up_te  = 1.0 - p_up_te

        # Val 지표 (단일 클래스/상수확률에 견고)
        try:
            auc_val = float(roc_auc_score(y_up_val, p_up_val))
        except Exception:
            auc_val = float("nan")
        ll_val = float(log_loss(y_up_val, np.c_[1 - p_up_val, p_up_val], labels=[0, 1]))
        print(f"[VAL]  logloss={ll_val:.4f}  auc={auc_val:.4f}")

        # Test 지표 (단일 클래스에도 안전)
        metrics["logloss"]  = float(log_loss(y_up_te,  np.c_[1 - p_up_te, p_up_te], labels=[0, 1]))
        try:
            metrics["auc"]         = float(roc_auc_score(y_up_te, p_up_te))
            metrics["auc_flipped"] = float(roc_auc_score(y_up_te, 1.0 - p_up_te))
        except Exception:
            metrics["auc"] = metrics["auc_flipped"] = float("nan")
        metrics["brier"]     = float(brier_score_loss(y_up_te, p_up_te))
        metrics["ece_p_up"]  = expected_calibration_error_binary(p_up_te, y_up_te.astype(int))

        print(f"\n[TEST] logloss={metrics['logloss']:.4f}  auc={metrics['auc']:.4f}  "
              f"auc_flipped={metrics['auc_flipped']:.4f}  "
              f"brier={metrics['brier']:.4f}  ece={metrics['ece_p_up']:.4f}")

        out_pred = pd.DataFrame(
            {"date": dates_test, "symbol": symbols_test, "y_true_up": y_up_te.astype(int), "p_up": p_up_te}
        ).set_index("date")

    else:
        # 멀티클래스 기본 지표
        n_classes = proba_te.shape[1]
        y_pred = np.argmax(proba_te, axis=1)
        metrics["multi_logloss"] = float(log_loss(y_test, proba_te, labels=list(range(n_classes))))
        metrics["balanced_acc"]  = float(balanced_accuracy_score(y_test, y_pred))
        metrics["brier_multi"]   = brier_multiclass(y_test, proba_te, n_classes)

        print(f"\n[TEST] multi_logloss={metrics['multi_logloss']:.4f}  "
              f"balanced_acc={metrics['balanced_acc']:.4f}  brier_multi={metrics['brier_multi']:.4f}")
        print("\nclassification_report:")
        print(classification_report(y_test, y_pred, digits=4))
        print("confusion_matrix:\n", confusion_matrix(y_test, y_pred))

        # p_up(= +1 클래스 확률)로도 별도 저장/평가 가능
        p_up_te = proba_te[:, plus_one_id]
        out_pred = pd.DataFrame(
            {"date": dates_test, "symbol": symbols_test, "y_true_up": y_up_te.astype(int), "p_up": p_up_te}
        ).set_index("date")

    # ----- 중요도 저장 -----
    try:
        booster = clf.booster_
        fi = pd.DataFrame({
            "feature": X.columns,
            "gain": booster.feature_importance(importance_type="gain"),
            "split": booster.feature_importance(importance_type="split"),
        }).sort_values("gain", ascending=False)
        fi.to_csv(Path(args.outdir) / "feature_importance.csv", index=False)
        print(f"[saved] feature_importance -> {args.outdir}/feature_importance.csv")
    except Exception:
        pass

    # ----- 저장 -----
    joblib.dump(clf, Path("models") / "lgbm_model.pkl")
    if calibrator is not None:
        joblib.dump(calibrator, Path("models") / "calibrator.pkl")
    with open(Path(args.outdir) / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    out_pred.to_parquet(Path(args.outdir) / "test_preds.parquet")
    print(f"[saved] metrics -> {args.outdir}/metrics.json, preds -> {args.outdir}/test_preds.parquet")
    print("[saved] model -> models/lgbm_model.pkl" + (" + models/calibrator.pkl" if calibrator is not None else ""))


if __name__ == "__main__":
    main()
