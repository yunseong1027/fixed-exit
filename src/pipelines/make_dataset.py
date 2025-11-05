import argparse, pathlib
import numpy as np
import pandas as pd
from src.features.make import make_features

def make_market_features(df_labeled_qqq: pd.DataFrame) -> pd.DataFrame:
    """
    QQQ 라벨 파일에서 시장(레짐) 피처 생성 (모두 shift(1)로 누수 방지)
    """
    close = df_labeled_qqq["close"]
    logret_1 = np.log(close / close.shift(1))
    sma20 = close.rolling(20, min_periods=20).mean()

    mkt = pd.DataFrame({
        "mkt_ret_5":  close.pct_change(5),
        "mkt_ret_20": close.pct_change(20),
        "mkt_rv_20":  logret_1.rolling(20, min_periods=20).std() * (252 ** 0.5),
        "mkt_dist_sma20": close / sma20 - 1.0,
    }, index=df_labeled_qqq.index)

    return mkt.shift(1)  # t 시점 예측에 t-1까지 정보만 쓰도록

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["AAPL","QQQ"])
    ap.add_argument("--in-dir",  type=str, default="data/interim")
    ap.add_argument("--out",     type=str, default="data/processed/train.parquet")
    ap.add_argument("--add-mkt", action="store_true", help="QQQ 시장 피처를 조인")
    args = ap.parse_args()

    # (선택) 시장 피처 준비
    mkt_feat = None
    if args.add_mkt:
        qqq_path = pathlib.Path(args.in_dir) / "QQQ_labeled.parquet"
        qqq_df = pd.read_parquet(qqq_path)
        mkt_feat = make_market_features(qqq_df)

    Xy_list = []
    for s in args.symbols:
        path = pathlib.Path(args.in_dir) / f"{s}_labeled.parquet"
        df = pd.read_parquet(path)

        X, y = make_features(df)           # 내부에서 shift(1) 처리됨
        if mkt_feat is not None:
            X = X.join(mkt_feat, how="left")

        X = X.dropna()                     # 조인 후 결측 제거
        y = y.reindex(X.index)

        X["symbol"] = s
        X["y"] = y
        Xy_list.append(X)

    all_df = pd.concat(Xy_list).dropna()
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    all_df.to_parquet(args.out)
    print(f"[ok] saved -> {args.out}, shape={all_df.shape}, cols={list(all_df.columns)}")

if __name__ == "__main__":
    main()
