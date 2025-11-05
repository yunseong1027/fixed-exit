import numpy as np
import pandas as pd

def breakout_20(df: pd.DataFrame, eps: float = -0.002, use_close: bool = True) -> pd.Series:
    """
    20일 돌파 후보 (정보누수 방지: t에서 t-1 정보로 판정).
    eps<0면 '근접 돌파' 허용(완화), 예: eps=-0.002 → 0.2% 미만 부족도 통과.
    """
    if use_close:
        roll_max = df["close"].rolling(20, min_periods=20).max()
        ref = df["close"]
    else:
        roll_max = df["high"].rolling(20, min_periods=20).max()
        ref = df["high"]
    thresh = roll_max.shift(1) * (1.0 + eps)
    return (ref.shift(1) >= thresh).astype("Int8").rename("cand_breakout")

def rsi_oversold_bounce(df: pd.DataFrame, n=14, th=35) -> pd.Series:
    """RSI 과매도(<=th) 다음날 양봉을 '반등 후보'로."""
    delta = df["close"].diff()
    up = delta.clip(lower=0).rolling(n, min_periods=n).mean()
    dn = (-delta.clip(upper=0)).rolling(n, min_periods=n).mean()
    rs = up / dn.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    sig = ((rsi.shift(1) <= th) & (df["close"] > df["open"])).astype("Int8")
    return sig.rename("cand_rsi_bounce")

def pullback_after_breakout(df: pd.DataFrame, look=20, pb=0.003) -> pd.Series:
    """전일 종가가 20일 고점 근접(±pb)이고 당일 양봉인 '풀백 재돌파 후보'."""
    rh = df["close"].rolling(look, min_periods=look).max()
    near = (np.abs(df["close"].shift(1) - rh.shift(1)) / df["close"].shift(1) <= pb)
    sig = (near & (df["close"] > df["open"])).astype("Int8")
    return sig.rename("cand_pullback")
