import re
import pandas as pd

def standardize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    MultiIndex/접두사/대소문자 뒤섞인 컬럼을
    open, high, low, close, volume 로 표준화한다.
    """
    # 1) 컬럼 평탄화 + 문자열화
    if isinstance(df.columns, pd.MultiIndex):
        cols = ['_'.join(map(str, tup)).strip('_') for tup in df.columns.to_flat_index()]
        df = df.copy()
        df.columns = cols
    else:
        df = df.copy()
        df.columns = [str(c) for c in df.columns]

    low = [c.lower().strip() for c in df.columns]

    def pick(*aliases):
        # '^|_' + (alias1|alias2) + '(_|$)' 패턴으로 앞/중간/끝 어디에 있어도 매칭
        pat = re.compile(r'(^|_)(' + '|'.join(a.replace(' ', '_') for a in aliases) + r')($|_)')
        for i, c in enumerate(low):
            if pat.search(c):
                return df.columns[i]
        return None

    col_open  = pick("open")
    col_high  = pick("high")
    col_low   = pick("low")
    # adj close가 있으면 그걸 우선 사용
    col_close = pick("adj_close", "adj close", "close")
    col_vol   = pick("volume", "vol")

    missing = [name for name, col in
               [("open", col_open), ("high", col_high), ("low", col_low),
                ("close", col_close), ("volume", col_vol)] if col is None]
    if missing:
        raise KeyError(f"Cannot find columns {missing}. Available: {list(df.columns)[:10]}...")

    out = df[[col_open, col_high, col_low, col_close, col_vol]].copy()
    out.columns = ["open", "high", "low", "close", "volume"]
    return out
