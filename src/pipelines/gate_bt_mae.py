import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

# -------------------- split --------------------
def time_split_index(n: int, test_ratio=0.2, val_ratio=0.1):
    i_test = int(n * (1 - test_ratio))
    i_val = int(i_test * (1 - val_ratio))
    return slice(0, i_val), slice(i_val, i_test), slice(i_test, n)

# -------------------- load --------------------
def load_data(data_path, preds_path, test_ratio=0.2, val_ratio=0.1, rv_cap_pct=None):
    df = pd.read_parquet(data_path).sort_index()
    preds = pd.read_parquet(preds_path).sort_index()  # ['q_mae','E_R'] by date

    # split
    n = len(df)
    _, _, te = time_split_index(n, test_ratio=test_ratio, val_ratio=val_ratio)
    df_te = df.iloc[te].copy()

    # regime filter (optional): exclude high-vol tails
    if (rv_cap_pct is not None) and ("mkt_rv_20" in df_te.columns):
        cap = np.nanpercentile(df_te["mkt_rv_20"].values, rv_cap_pct)
        df_te = df_te[df_te["mkt_rv_20"] <= cap]

    # join predictions (preds is test-only; inner join for safety)
    te_join = df_te.join(preds, how="inner")

    # required cols
    cols_needed = ["y_R", "y_mae", "y_dur", "q_mae", "E_R", "symbol"]
    te_join = te_join[[c for c in cols_needed if c in te_join.columns]].dropna()

    # ensure integer duration >=1
    te_join["y_dur"] = te_join["y_dur"].round().clip(lower=1).astype(int)

    return te_join

# -------------------- equity helpers --------------------
def mdd_from_equity(eq):
    peak = np.maximum.accumulate(eq)
    dd = 1.0 - eq/peak
    return float(np.max(dd))

def cagr_from_equity(eq, total_days):
    """연 252영업일 가정, 전체 타임라인 일수로 연율화."""
    if len(eq) < 2 or total_days <= 0:
        return 0.0
    years = total_days / 252.0
    total = eq[-1] / eq[0]
    return float(total**(1/years) - 1.0)

# -------------------- schedulers --------------------
def simulate_entry_only(df_te, pick_mask, cost):
    """간단 비교용: 선택된 '후보일'에만 전체 R을 일괄 반영(보유기간 무시)."""
    T = len(df_te)
    rets = np.zeros(T, dtype=float)
    rets[pick_mask] = df_te.loc[pick_mask, "y_R"].values

    eq = np.ones(T + 1, dtype=float)
    for i, r in enumerate(rets):
        eq[i+1] = eq[i] * (1.0 + r - (cost if pick_mask[i] else 0.0))

    total_days = T                        # ★ 전체 타임라인 기준
    mdd  = mdd_from_equity(eq)
    cagr = cagr_from_equity(eq, total_days)
    calmar = cagr/mdd if mdd>0 else np.inf
    return cagr, mdd, calmar

def simulate_exclusive_duration(df_te, pick_mask, cost):
    """
    현실감 ↑: 포지션 1개 독점. '캘린더 전체 영업일' 기준으로 누적.
    - 테스트 구간의 시작~끝을 영업일 캘린더로 생성
    - 후보일의 진입 날짜를 해당 영업일 인덱스로 매핑
    - 진입 시 y_dur 동안 일일 수익 (1+R)^(1/dur)-1 를 누적
    - per-trade cost는 '진입일 1회'만 차감
    - 보유 중 겹치는 후보는 스킵(동시보유=1)
    """
    # 1) 전체 캘린더 구성 (영업일)
    cal_start = pd.to_datetime(df_te.index.min()).normalize()
    cal_end   = pd.to_datetime(df_te.index.max()).normalize()
    cal = pd.date_range(cal_start, cal_end, freq="B")   # 영업일 가정

    # 2) 후보일 → 캘린더 위치 매핑
    idx_map = {pd.to_datetime(d).normalize(): i for i, d in enumerate(cal)}
    # 후보 진입일(= df_te.index의 해당 날짜)을 캘린더 인덱스로 변환
    entry_idx = []
    for d, use in zip(df_te.index, pick_mask):
        if not use:
            entry_idx.append(None)
            continue
        k = idx_map.get(pd.to_datetime(d).normalize(), None)
        entry_idx.append(k)

    # 3) 시뮬레이션
    T = len(cal)
    eq = np.ones(T + 1, dtype=float)
    occupancy = np.zeros(T, dtype=bool)  # 보유 중인 날 표시
    accepted = skipped = 0

    R = df_te["y_R"].values
    D = df_te["y_dur"].round().clip(lower=1).astype(int).values

    for k, epos in enumerate(entry_idx):
        if epos is None:            # 비선택 후보
            continue
        d = int(D[k]); r_total = float(R[k])
        # 보유 기간이 캘린더를 넘으면 잘라냄
        endpos = min(epos + d, T)

        # 겹치는지 확인
        if epos >= T or occupancy[epos:endpos].any():
            skipped += 1
            continue

        # 일일 수익률(균등 분할)
        daily_ret = (1.0 + r_total)**(1.0 / d) - 1.0

        # entry day ~ endpos-1까지 누적 (entry day에 cost 1회 차감)
        # 먼저 entry 이전 구간은 eq를 그대로 carry
        # (루프 밖에서 eq[i+1] = eq[i] 형태로 항상 carry 하도록 작성하면 더 깔끔)
        # 여기선 간단히 entry~endpos-1 구간만 처리하고 그 외는 0수익으로 carry된다고 가정
        eq[epos+1] = eq[epos] * (1.0 + daily_ret - cost)  # entry day
        occupancy[epos:endpos] = True
        for t in range(epos+1, endpos):
            eq[t+1] = eq[t] * (1.0 + daily_ret)
        accepted += 1

    # 4) 빈 구간 carry: 위에서 수익이 반영된 날만 eq가 채워졌으므로 나머지 비어있는 구간은 직전 값 carry
    for i in range(1, T+1):
        if eq[i] == 1.0 and eq[i-1] != 1.0:  # 초깃값만 남아있을 때를 구분
            eq[i] = eq[i-1]

    # 5) 지표
    total_days = T
    mdd  = mdd_from_equity(eq)
    cagr = cagr_from_equity(eq, total_days)
    calmar = cagr/mdd if mdd>0 else np.inf
    return cagr, mdd, calmar, accepted, skipped, total_days

def eval_from_mask(df_te, pick_mask, cost, scheduler="exclusive_duration"):
    if scheduler == "entry_only":
        cagr, mdd, calmar = simulate_entry_only(df_te, pick_mask, cost)
        return {"CAGR": cagr, "MDD": mdd, "Calmar": calmar,
                "accepted": int(pick_mask.sum()), "skipped": 0, "days": len(df_te)}
    else:
        cagr, mdd, calmar, acc, skp, days = simulate_exclusive_duration(df_te, pick_mask, cost)
        return {"CAGR": cagr, "MDD": mdd, "Calmar": calmar,
                "accepted": acc, "skipped": skp, "days": days}

# -------------------- evaluations --------------------
def evaluate_gate(df_te, tau, theta, cost, scheduler):
    pick = ((df_te["q_mae"] <= tau) & (df_te["E_R"] >= theta)).values
    acc_rate = float(pick.mean())
    if acc_rate == 0.0:
        return {"acc_rate":0.0,"n_trades":0,"CAGR":0.0,"MDD":1.0,"Calmar":0.0,"accepted":0,"skipped":0,"days":len(df_te)}
    res = eval_from_mask(df_te, pick, cost, scheduler=scheduler)
    return {"acc_rate": acc_rate, "n_trades": int(pick.sum()), **res}

def evaluate_random(df_te, target_acc_rate, cost, scheduler, seed=42):
    rng = np.random.default_rng(seed)
    pick = rng.random(len(df_te)) < target_acc_rate
    return eval_from_mask(df_te, pick, cost, scheduler=scheduler)

def evaluate_all(df_te, cost, scheduler):
    pick = np.ones(len(df_te), dtype=bool)  # accept all candidates
    return eval_from_mask(df_te, pick, cost, scheduler=scheduler)

# -------------------- bootstrap CI --------------------
def block_bootstrap_metrics(df_te, pick_mask, cost, scheduler, block=20, B=200, seed=42):
    rng = np.random.default_rng(seed)
    T = len(df_te)
    stats = []
    for _ in range(B):
        idx = []
        while len(idx) < T:
            start = rng.integers(0, max(T-block,1))
            idx.extend(range(start, min(start+block, T)))
        idx = np.array(idx[:T])
        pm = pick_mask[idx]
        dfb = df_te.iloc[idx]
        res = eval_from_mask(dfb, pm, cost, scheduler=scheduler)
        stats.append([res["CAGR"], res["MDD"], res["Calmar"]])
    arr = np.array(stats)
    def ci(col): return (float(np.nanpercentile(arr[:,col], 2.5)), float(np.nanpercentile(arr[:,col], 97.5)))
    return {"CAGR_CI": ci(0), "MDD_CI": ci(1), "Calmar_CI": ci(2)}

# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/processed/meta_train.parquet")
    ap.add_argument("--preds", type=str, default="reports/mae_q/test_preds.parquet")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--test-ratio", type=float, default=0.2)
    ap.add_argument("--cost", type=float, default=0.0005)  # per-trade cost
    ap.add_argument("--scheduler", choices=["entry_only","exclusive_duration"], default="exclusive_duration")
    # repeatable options (쉘 안전)
    ap.add_argument("--tau-q", dest="tau_qs", action="append", type=int,
                    help="repeatable: e.g. --tau-q 20 --tau-q 30 ... (percentiles of q_mae)")
    ap.add_argument("--theta", dest="thetas", action="append", type=float,
                    help="repeatable: e.g. --theta -0.001 --theta 0.0 --theta 0.001")
    ap.add_argument("--rv-cap-pct", type=float, default=None, help="exclude rows with mkt_rv_20 above this percentile")
    ap.add_argument("--B", type=int, default=200, help="bootstrap iterations")
    ap.add_argument("--block", type=int, default=20, help="bootstrap block length (candidate days)")
    ap.add_argument("--outdir", type=str, default="reports/mae_gate")
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    df_te = load_data(args.data, args.preds, test_ratio=args.test_ratio,
                      val_ratio=args.val_ratio, rv_cap_pct=args.rv_cap_pct)

    # grid
    tau_qs = args.tau_qs if args.tau_qs else [20,30,40,50]
    thetas = args.thetas if args.thetas else [-0.001, 0.0, 0.001]
    qs_vals = np.percentile(df_te["q_mae"].values, tau_qs)

    rows = []
    for tau, tau_q in zip(tau_qs, qs_vals):
        for th in thetas:
            res  = evaluate_gate(df_te, tau_q, th, args.cost, scheduler=args.scheduler)
            rnd  = evaluate_random(df_te, res["acc_rate"], args.cost, scheduler=args.scheduler, seed=42)
            alln = evaluate_all(df_te, args.cost, scheduler=args.scheduler)
            # CI on gate
            pick_gate = ((df_te["q_mae"] <= tau_q) & (df_te["E_R"] >= th)).values
            ci = block_bootstrap_metrics(df_te, pick_gate, cost=args.cost, scheduler=args.scheduler,
                                         block=args.block, B=args.B, seed=42)
            rows.append({
                "tau_pct": tau, "tau_val": float(tau_q), "theta": th,
                **res,
                "Gate_CAGR_CI_low": ci["CAGR_CI"][0], "Gate_CAGR_CI_high": ci["CAGR_CI"][1],
                "Gate_MDD_CI_low":  ci["MDD_CI"][0],  "Gate_MDD_CI_high":  ci["MDD_CI"][1],
                "Rand_CAGR": rnd["CAGR"], "Rand_MDD": rnd["MDD"], "Rand_Calmar": rnd["Calmar"],
                "All_CAGR": alln["CAGR"], "All_MDD": alln["MDD"], "All_Calmar": alln["Calmar"],
                "CAGR_minus_Rand": res["CAGR"] - rnd["CAGR"],
                "CAGR_minus_All":  res["CAGR"] - alln["CAGR"]
            })

    out = pd.DataFrame(rows).sort_values(["MDD","Calmar"], ascending=[True, False])
    out.to_csv(Path(args.outdir)/"grid_results.csv", index=False)

    # 요약 1: "CAGR ≥ Random" 제약 하 MDD 최소 상위 5
    out["meets_rand_cagr"] = out["CAGR"] >= out["Rand_CAGR"]
    top = out[out["meets_rand_cagr"]].sort_values(["MDD","Calmar"], ascending=[True, False]).head(5)

    print("\n=== Top (MDD min, CAGR ≥ Random) ===")
    if len(top)==0:
        print("No config meets the random CAGR constraint.")
    else:
        print(top[["tau_pct","tau_val","theta","acc_rate","n_trades","accepted","skipped","days",
                   "CAGR","MDD","Calmar",
                   "Gate_CAGR_CI_low","Gate_CAGR_CI_high",
                   "Gate_MDD_CI_low","Gate_MDD_CI_high",
                   "Rand_CAGR","Rand_MDD","Rand_Calmar",
                   "All_CAGR","All_MDD","All_Calmar",
                   "CAGR_minus_Rand","CAGR_minus_All"]])

    with open(Path(args.outdir)/"summary.json","w") as f:
        json.dump(top.to_dict(orient="records"), f, indent=2)

    # 요약 2: 제약 없이 MDD 최소 상위 5 (백업용)
    top2 = out.sort_values(["MDD","Calmar"], ascending=[True, False]).head(5)
    with open(Path(args.outdir)/"summary_unconstrained.json","w") as f:
        json.dump(top2.to_dict(orient="records"), f, indent=2)

    print(f"\n[saved] grid -> {args.outdir}/grid_results.csv")
    print(f"[saved] summary -> {args.outdir}/summary.json")
    print(f"[saved] summary_unconstrained -> {args.outdir}/summary_unconstrained.json")

if __name__ == "__main__":
    main()
