import argparse, pathlib
import pandas as pd, yfinance as yf

def fetch(sym, start="2005-01-01"):
    df = yf.download(sym, start=start, auto_adjust=True, progress=False)
    df = df.rename(columns=str.lower)
    df.index.name = "date"
    return df[["open","high","low","close","volume"]]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", required=True,
                    help="예: AAPL MSFT AMZN GOOGL META NVDA TSLA SPY QQQ TLT GLD XLE XLF XLK IWM")
    ap.add_argument("--start", default="2005-01-01")
    ap.add_argument("--out", default="data/raw")
    args = ap.parse_args()

    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for s in args.symbols:
        try:
            df = fetch(s, start=args.start)
            df.to_parquet(out/f"{s}.parquet")
            print(f"[ok] {s}: {df.shape} -> {out}/{s}.parquet")
        except Exception as e:
            print(f"[warn] {s}: {e}")

if __name__ == "__main__":
    main()
