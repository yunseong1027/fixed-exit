import argparse, pathlib
import numpy as np
import pandas as pd
from src.features.meta import make_meta_features

def make_market_features(df_labeled_qqq: pd.DataFrame) -> pd.DataFrame:
    close = df_labeled_qqq["close"]
    logret = np.log(close / close.shift(1))
    sma20 = close.rolling(20, min_periods=20).mean()
    mkt = pd.DataFrame({
        "mkt_ret_5":  close.pct_change(5),
        "mkt_ret_20": close.pct_change(20),
        "mkt_rv_20":  logret.rolling(20, min_periods=20).std() * (252 ** 0.5),
        "mkt_dist_sma20": close / sma20 - 1.0,
    }, index=df_labeled_qqq.index).shift(1)  # 누수 방지
    return mkt

def build_one(symbol, in_dir, add_mkt, mkt_feat):
    path = pathlib.Path(in_dir) / f"{symbol}_labeled_meta.parquet"
    df = pd.read_parquet(path).sort_index()

    # 후보=1 & ATR 준비된 구간만 '최종적으로' 사용할 계획
    mask = (df["candidate"] == 1) & (df["atr_14"].notna())

    # 메타-피처는 전체에서 계산 → 후보로 슬라이스
    X_full = make_meta_features(df)   # 내부에서 shift(1)
    if add_mkt and mkt_feat is not None:
        X_full = X_full.join(mkt_feat, how="left")

    X = X_full.loc[mask].copy()
    # 후보 타입 one-hot 추가
    cand_feats = df.loc[mask, ["cand_breakout","cand_rsi_bounce","cand_pullback"]].astype("Int8")
    X = X.join(cand_feats)

    # 라벨들: 성공/실패, MAE, R, 보유기간
    y_up  = (df.loc[mask, "label"] == 1).astype(int).rename("y")           # 성공(익절)
    y_mae = df.loc[mask, "mae"].astype(float).rename("y_mae")
    y_R   = df.loc[mask, "R"].astype(float).rename("y_R")
    y_dur = df.loc[mask, "dur"].astype(float).rename("y_dur")

    # 정렬/결측 제거
    valid = X.notna().all(axis=1) & y_mae.notna() & y_R.notna()
    X = X.loc[valid]
    y_up  = y_up.loc[valid]
    y_mae = y_mae.loc[valid]
    y_R   = y_R.loc[valid]
    y_dur = y_dur.loc[valid]

    X["symbol"] = symbol
    X["y"]      = y_up
    X["y_mae"]  = y_mae
    X["y_R"]    = y_R
    X["y_dur"]  = y_dur
    return X

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["AAPL","QQQ"])
    ap.add_argument("--in-dir", type=str, default="data/interim")
    ap.add_argument("--out", type=str, default="data/processed/meta_train.parquet")
    ap.add_argument("--add-mkt", action="store_true")
    args = ap.parse_args()

    # 시장 피처 준비(선택)
    mkt_feat = None
    if args.add_mkt:
        qqq = pd.read_parquet(pathlib.Path(args.in_dir)/"QQQ_labeled_meta.parquet")
        mkt_feat = make_market_features(qqq)

    frames = []
    for s in args.symbols:
        frames.append(build_one(s, args.in_dir, args.add_mkt, mkt_feat))

    out = pd.concat(frames).sort_index()
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out)
    print(f"[ok] saved -> {args.out}, shape={out.shape}, cols={list(out.columns)}")

if __name__ == "__main__":
    main()
