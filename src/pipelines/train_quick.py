import argparse
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import log_loss, balanced_accuracy_score, classification_report, confusion_matrix

def encode_labels(y: pd.Series):
    # LightGBM은 0..K-1 정수 라벨을 권장 → {-1,0,1} → {0,1,2}
    label2id = {-1:0, 0:1, 1:2}
    id2label = {v:k for k,v in label2id.items()}
    y_id = y.map(label2id).astype(int).values
    return y_id, id2label

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/processed/train_aapl_qqq.parquet")
    ap.add_argument("--test-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_parquet(args.data).sort_index()
    X = df.drop(columns=["symbol","y"])
    y = df["y"].astype(int)

    y_id, id2label = encode_labels(y)

    # 시간순 홀드아웃: 마지막 20%를 테스트로
    n = len(df)
    split = int(n * (1 - args.test_ratio))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y_id[:split], y_id[split:]

    clf = LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=2000,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        max_depth=-1,
        random_state=args.seed,
        n_jobs=-1,
    )

    clf.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],                 # 스모크 테스트: 테스트셋을 검증셋으로 재사용(임시)
        eval_metric="multi_logloss",
        callbacks=[early_stopping(100), log_evaluation(100)]
    )

    proba = clf.predict_proba(X_test)
    y_pred = np.argmax(proba, axis=1)

    mlog = log_loss(y_test, proba, labels=[0, 1, 2])
    bal  = balanced_accuracy_score(y_test, y_pred)

    print("\n=== Metrics (holdout) ===")
    print(f"multi_logloss : {mlog:.4f}")
    print(f"balanced_acc  : {bal:.4f}")

    # 라벨 복원해 리포트/혼동행렬 출력
    y_test_lbl = np.vectorize(id2label.get)(y_test)
    y_pred_lbl = np.vectorize(id2label.get)(y_pred)

    from sklearn.metrics import classification_report, confusion_matrix
    print("\nclassification_report:")
    print(classification_report(y_test_lbl, y_pred_lbl, digits=4))
    print("confusion_matrix:\n", confusion_matrix(y_test_lbl, y_pred_lbl, labels=[-1,0,1]))

if __name__ == "__main__":
    main()

# python -m src.pipelines.train_quick --data data/processed/train_aapl_qqq.parquet