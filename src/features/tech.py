import numpy as np
import pandas as pd

def zscore(s: pd.Series, n=60):
    m = s.rolling(n, min_periods=n).mean()
    v = s.rolling(n, min_periods=n).std()
    return (s - m) / v

def rsi(close: pd.Series, n=14):
    delta = close.diff()
    up = (delta.clip(lower=0)).rolling(n, min_periods=n).mean()
    dn = (-delta.clip(upper=0)).rolling(n, min_periods=n).mean()
    rs = up / dn
    return 100 - (100 / (1 + rs))

def realized_vol(logret: pd.Series, n=20, annualize=True):
    rv = logret.rolling(n, min_periods=n).std()
    return rv * (252**0.5) if annualize else rv
