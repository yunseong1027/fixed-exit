import yfinance as yf, pandas as pd, pathlib

def fetch(symbol, start="2008-01-01"):
    df = yf.download(symbol, start=start, auto_adjust=True, progress=False)
    df = df.rename(columns=str.lower)
    df.index.name = "date"
    return df[["open","high","low","close","volume"]]

if __name__ == "__main__":
    out = pathlib.Path("data/raw"); out.mkdir(parents=True, exist_ok=True)
    for s in ["QQQ","AAPL"]:
        fetch(s).to_parquet(out/f"{s}.parquet")
