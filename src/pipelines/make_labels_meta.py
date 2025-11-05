import argparse, pathlib
import pandas as pd
from src.utils.ohlcv import standardize_ohlcv
from src.features.vol import atr
from src.signals.candidates import breakout_20, rsi_oversold_bounce, pullback_after_breakout
from src.labeling.triple_barrier import triple_barrier

def _trade_metrics_row(df: pd.DataFrame, i_entry: int, event_end_ts, label, tp_price, sl_price):
    """엔트리~종료창에서 MAE/MFE/실현수익/보유기간 계산."""
    # entry = 다음 날 시가
    if i_entry >= len(df): return None
    entry = float(df["open"].iloc[i_entry])

    # 종료 인덱스: event_end가 주어지면 그 위치, 없으면 창 끝(보유한도)
    if pd.isna(event_end_ts):
        i_exit = len(df) - 1
    else:
        try:
            i_exit = df.index.get_loc(event_end_ts)
        except KeyError:
            i_exit = min(i_entry + 20, len(df) - 1)  # 안전 가드

    # 창 최저/최고
    low_win  = df["low"].iloc[i_entry:i_exit+1].min()
    high_win = df["high"].iloc[i_entry:i_exit+1].max()
    mae = max((entry - low_win)  / entry, 0.0)    # 최대 불리 변동(+)
    mfe = max((high_win - entry) / entry, 0.0)    # 최대 유리 변동(+)

    # 실현 수익률 R (고정 출구 기준)
    if label == 1 and pd.notna(tp_price):
        R = float(tp_price) / entry - 1.0
    elif label == -1 and pd.notna(sl_price):
        R = float(sl_price) / entry - 1.0
    else:
        close_end = float(df["close"].iloc[i_exit])
        R = close_end / entry - 1.0

    dur = i_exit - i_entry + 1
    return mae, mfe, R, dur

def label_one(symbol: str, raw_dir="data/raw", out_dir="data/interim",
              atr_n=14, up_mult=2.0, dn_mult=0.8, max_holding=15,
              eps=-0.002, use_close=True):
    raw_path = pathlib.Path(raw_dir) / f"{symbol}.parquet"
    df = pd.read_parquet(raw_path).sort_index()
    df = standardize_ohlcv(df)

    atr_col = f"atr_{atr_n}"
    df[atr_col] = atr(df, n=atr_n)

    # 후보 3종 + 통합 플래그
    df["cand_breakout"]  = breakout_20(df, eps=eps, use_close=use_close)
    df["cand_rsi_bounce"]= rsi_oversold_bounce(df)
    df["cand_pullback"]  = pullback_after_breakout(df)
    df["candidate"] = ((df["cand_breakout"]==1) | (df["cand_rsi_bounce"]==1) | (df["cand_pullback"]==1)).astype("Int8")

    # 후보=1인 시점만 라벨링
    mask = df["candidate"] == 1
    lab = triple_barrier(
        df.loc[mask],
        atr_col=atr_col,
        up_mult=up_mult, dn_mult=dn_mult, max_holding=max_holding
    )

    # 결과 컬럼 초기화
    for col, dtype in [("label","Int8"),("t_hit","int32"),("tp","float64"),("sl","float64"),("event_end","datetime64[ns]"),
                       ("mae","float64"),("mfe","float64"),("R","float64"),("dur","float64")]:
        df[col] = pd.Series(index=df.index, dtype=dtype)

    # 값 반영(+ MAE/MFE/R/dur)
    df.loc[mask, ["label","t_hit","tp","sl","event_end"]] = lab[["label","t_hit","tp","sl","event_end"]]

    # 각 후보행에 대해 메트릭 계산
    idx = df.index.tolist()
    for i_ts in df.index[mask]:
        i = idx.index(i_ts)
        i_entry = i + 1
        event_end_ts = df.at[i_ts, "event_end"]
        label = df.at[i_ts, "label"]
        tp_price = df.at[i_ts, "tp"]
        sl_price = df.at[i_ts, "sl"]
        out = _trade_metrics_row(df, i_entry, event_end_ts, label, tp_price, sl_price)
        if out is not None:
            df.at[i_ts, "mae"] = out[0]
            df.at[i_ts, "mfe"] = out[1]
            df.at[i_ts, "R"]   = out[2]
            df.at[i_ts, "dur"] = out[3]

    # 저장
    out_dir = pathlib.Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{symbol}_labeled_meta.parquet"
    df.to_parquet(out_path)
    return out_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["AAPL","QQQ"])
    ap.add_argument("--atr-n", type=int, default=14)
    ap.add_argument("--up-mult", type=float, default=2.0)
    ap.add_argument("--dn-mult", type=float, default=0.8)
    ap.add_argument("--hold", type=int, default=15)
    ap.add_argument("--raw-dir", type=str, default="data/raw")
    ap.add_argument("--out-dir", type=str, default="data/interim")
    ap.add_argument("--eps", type=float, default=-0.002)
    ap.add_argument("--use-close", action="store_true")
    args = ap.parse_args()

    for s in args.symbols:
        p = label_one(s, raw_dir=args.raw_dir, out_dir=args.out_dir,
                      atr_n=args.atr_n, up_mult=args.up_mult,
                      dn_mult=args.dn_mult, max_holding=args.hold,
                      eps=args.eps, use_close=args.use_close)
        print(f"[ok] {s}: saved -> {p}")

if __name__ == "__main__":
    main()
