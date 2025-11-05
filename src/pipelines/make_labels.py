# src/pipelines/make_labels.py
import argparse, pathlib
import pandas as pd
from src.features.vol import atr
from src.utils.ohlcv import standardize_ohlcv
from src.labeling.triple_barrier import triple_barrier

def label_one(symbol: str, raw_dir="data/raw", out_dir="data/interim",
              atr_n=14, up_mult=1.5, dn_mult=1.0, max_holding=20):
    raw_path = pathlib.Path(raw_dir) / f"{symbol}.parquet"
    df = pd.read_parquet(raw_path).sort_index()
    df = standardize_ohlcv(df)

    atr_col = f"atr_{atr_n}"
    df[atr_col] = atr(df, n=atr_n)

    lab = triple_barrier(df, atr_col=atr_col, up_mult=up_mult,
                         dn_mult=dn_mult, max_holding=max_holding)
    out = pd.concat([df, lab], axis=1)

    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = pathlib.Path(out_dir) / f"{symbol}_labeled.parquet"
    out.to_parquet(out_path)
    return out_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["AAPL","QQQ"])
    ap.add_argument("--atr-n", type=int, default=14)
    ap.add_argument("--up-mult", type=float, default=1.5)
    ap.add_argument("--dn-mult", type=float, default=1.0)
    ap.add_argument("--hold", type=int, default=20)
    args = ap.parse_args()

    for s in args.symbols:
        path = label_one(s, atr_n=args.atr_n, up_mult=args.up_mult,
                         dn_mult=args.dn_mult, max_holding=args.hold)
        print(f"[ok] {s}: saved -> {path}")

if __name__ == "__main__":
    main()
