import numpy as np
import pandas as pd

# ----- 범위 기반 변동성 -----
def parkinson_vol(df: pd.DataFrame, n=20) -> pd.Series:
    x = np.log(df["high"] / df["low"]) ** 2
    pv = (1.0 / (4.0 * np.log(2.0))) * x.rolling(n, min_periods=n).mean()
    return np.sqrt(pv).rename(f"pv_{n}")

def garman_klass_vol(df: pd.DataFrame, n=20) -> pd.Series:
    log_co = np.log(df["close"] / df["open"])
    log_ho = np.log(df["high"] / df["open"])
    log_lo = np.log(df["low"]  / df["open"])
    var_inst = 0.5 * (log_ho - log_lo) ** 2 - (2 * np.log(2) - 1) * (log_co ** 2)
    var_mean = var_inst.rolling(n, min_periods=n).mean()
    return np.sqrt(np.clip(var_mean, 0, None)).rename(f"gk_{n}")

def rogers_satchell_vol(df: pd.DataFrame, n=20) -> pd.Series:
    log_hc = np.log(df["high"] / df["close"])
    log_ho = np.log(df["high"] / df["open"])
    log_lc = np.log(df["low"]  / df["close"])
    log_lo = np.log(df["low"]  / df["open"])
    rs = (log_hc * log_ho + log_lc * log_lo).rolling(n, min_periods=n).mean()
    return np.sqrt(rs.clip(lower=0)).rename(f"rs_{n}")

# ----- 선형회귀 slope/R² -----
def _linreg_window(y: np.ndarray):
    n = len(y)
    x = np.arange(n)
    sx, sy = x.sum(), y.sum()
    sxx = (x * x).sum(); sxy = (x * y).sum()
    denom = n * sxx - sx * sx
    if denom == 0: return np.nan, np.nan
    slope = (n * sxy - sx * sy) / denom
    yhat = slope * (x - sx / n) + sy / n
    ss_tot = ((y - y.mean()) ** 2).sum()
    ss_res = ((y - yhat) ** 2).sum()
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    return slope, r2

def rolling_slope_r2(s: pd.Series, n: int, name_prefix: str):
    def f_slope(w): return _linreg_window(pd.Series(w).values)[0]
    def f_r2(w):    return _linreg_window(pd.Series(w).values)[1]
    slope = s.rolling(n, min_periods=n).apply(f_slope, raw=False).rename(f"{name_prefix}_slope_{n}")
    r2    = s.rolling(n, min_periods=n).apply(f_r2,    raw=False).rename(f"{name_prefix}_r2_{n}")
    return slope, r2

# ----- 메타-피처 묶음 -----
def make_meta_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]; high = df["high"]; low = df["low"]; vol = df["volume"].astype(float)
    atr = df["atr_14"]

    # 강도/거리
    rh20 = close.rolling(20, min_periods=20).max()
    dist_rh20 = (close - rh20) / atr
    sma20 = close.rolling(20, min_periods=20).mean()
    sma5  = close.rolling(5,  min_periods=5).mean()
    dist_sma20    = close / sma20 - 1.0
    sma5_over_20  = sma5  / sma20 - 1.0

    # 범위 기반 변동성
    pv20 = parkinson_vol(df, 20)
    gk20 = garman_klass_vol(df, 20)
    rs20 = rogers_satchell_vol(df, 20)

    # realized vol & 변화량
    rv20  = np.log(close / close.shift(1)).rolling(20, min_periods=20).std() * (252 ** 0.5)
    drv20 = rv20 - rv20.shift(5)

    # 추세 모양
    slope5,  r2_5  = rolling_slope_r2(close, 5,  "px")
    slope10, r2_10 = rolling_slope_r2(close, 10, "px")

    # 볼린저/스퀴즈
    bb_mid = sma20
    bb_std = close.rolling(20, min_periods=20).std()
    bb_width = (4 * bb_std) / bb_mid.replace(0, np.nan)  # 상하 2σ 대략화
    keltner = (atr.rolling(20, min_periods=20).mean()) / bb_mid.replace(0, np.nan)
    squeeze = bb_width / keltner.replace(0, np.nan)

    # 캔들 구조 & 갭
    body = (close - df["open"]).abs() / atr
    upper = (high - np.maximum(close, df["open"])) / atr
    lower = (np.minimum(close, df["open"]) - low) / atr
    clv   = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)

    overnight = (df["open"] / close.shift(1) - 1.0)
    intraday  = (close / df["open"] - 1.0)
    gap_norm  = (df["open"] - close.shift(1)) / atr
    gap_fill  = (np.maximum(close, df["open"]) - np.minimum(close, df["open"])) / atr

    feat = pd.DataFrame({
        "dist_rh20": dist_rh20,
        "dist_sma20": dist_sma20, "sma5_over_20": sma5_over_20,
        "pv20": pv20, "gk20": gk20, "rs20": rs20,
        "rv20": rv20, "drv20": drv20,
        "px_slope5": slope5, "px_r2_5": r2_5,
        "px_slope10": slope10, "px_r2_10": r2_10,
        "bb_width": bb_width, "squeeze": squeeze,
        "atr_14": atr, "vol_z60": (vol - vol.rolling(60, min_periods=60).mean()) /
                                  vol.rolling(60, min_periods=60).std(),
        "vol_ratio20": vol / vol.rolling(20, min_periods=20).mean() - 1.0,
        "overnight_ret": overnight, "intraday_ret": intraday,
        "gap_norm": gap_norm, "gap_fill": gap_fill,
        "body_atr": body, "upper_wick_atr": upper, "lower_wick_atr": lower, "clv": clv,
    }, index=df.index)

    # 정보누수 방지
    return feat.shift(1)
