import pandas as pd
import numpy as np
from .tech import zscore, rsi, realized_vol

def make_features(df_labeled: pd.DataFrame):
    """
    입력: OHLCV + atr_14 + (tp, sl, label, event_end)
    출력: (X, y) — 정보누수 방지 위해 모두 shift(1)
    """
    df = df_labeled.copy()

    # 기본 파생
    close = df["close"]
    logret_1 = (close / close.shift(1)).apply(lambda x: 0.0 if pd.isna(x) else np.log(x))
    ret_1 = close.pct_change()
    ret_5 = close.pct_change(5)
    ret_20 = close.pct_change(20)

    sma_5  = close.rolling(5,  min_periods=5).mean()
    sma_20 = close.rolling(20, min_periods=20).mean()
    sma_60 = close.rolling(60, min_periods=60).mean()

    feat = pd.DataFrame({
        "ret_1": ret_1,
        "ret_5": ret_5,
        "ret_20": ret_20,
        "logret_1": logret_1,
        "dist_sma20": close / sma_20 - 1.0,
        "sma5_over_20": sma_5 / sma_20 - 1.0,
        "rsi_14": rsi(close, 14),
        "rv_20": realized_vol(logret_1, 20, annualize=True),
        "atr_14": df["atr_14"],
        "vol_z_60": zscore(df["volume"].astype(float), 60),
        "vol_ratio_20": df["volume"] / df["volume"].rolling(20, min_periods=20).mean() - 1.0,
    }, index=df.index)

    # 정보누수 방지: 예측 시점 t에서 사용 가능한 값만 쓰도록 모두 한 칸 미뤄서 라벨(t)과 align
    X = feat.shift(1)

    # 타깃
    y = df["label"].astype("Int8")

    # 유효 구간 필터: 피처/라벨 결측 제거 + atr 준비된 구간
    valid = X.notna().all(axis=1) & df["atr_14"].notna() & y.isin([-1,0,1])
    X = X.loc[valid]
    y = y.loc[valid]

    return X, y
