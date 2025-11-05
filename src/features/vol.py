import numpy as np
import pandas as pd

def _true_range(high, low, prev_close):
    return np.maximum(
        high - low,
        np.maximum(np.abs(high - prev_close), np.abs(low - prev_close))
    )

def atr(df: pd.DataFrame, n: int = 14, high="high", low="low", close="close") -> pd.Series:
    """
    Wilder ATR(단순 이동평균 버전). n개 이상 쌓인 뒤부터 값이 나온다.
    반환 컬럼명은 'atr_{n}'.
    """
    h = df[high].to_numpy(dtype=float).reshape(-1)
    l = df[low].to_numpy(dtype=float).reshape(-1)
    c = df[close].to_numpy(dtype=float).reshape(-1)

    # 이전 종가 (첫 값은 NaN)
    prev_c = np.empty_like(c)
    prev_c[0] = np.nan
    prev_c[1:] = c[:-1]

    tr = _true_range(h, l, prev_c)
    s = pd.Series(tr, index=df.index).rolling(n, min_periods=n).mean()
    return s.rename(f"atr_{n}")
