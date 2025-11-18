import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def size_riskaware(q_mae, e_r, B_t, theta, kappa=0.5, f_max=0.2, f_total=0.6,
                   scale_er=None, eps=1e-4):
    q_mae = np.asarray(q_mae, dtype=float)
    e_r   = np.asarray(e_r, dtype=float)

    if scale_er is None:
        # robust scale for E[R] (avoid division by zero)
        std = np.nanstd(e_r)
        scale_er = std if std > 1e-6 else 1e-3

    s_er = sigmoid((e_r - theta) / (scale_er + 1e-12))
    raw  = kappa * (B_t / (q_mae + eps)) * s_er
    f    = np.clip(raw, 0.0, f_max)
    return f


def mdd_from_equity(eq):
    peak = np.maximum.accumulate(eq)
    dd = 1.0 - eq / peak
    return float(np.max(dd))

def cagr_from_equity(eq, total_days):
    if len(eq) < 2 or total_days <= 0:
        return 0.0
    years = total_days / 252.0
    total = eq[-1] / eq[0]
    # guard: if eq[-1] <= 0 (numerical issue), return -1.0
    if total <= 0:
        return -1.0
    return float(total ** (1.0 / years) - 1.0)

def eval_from_calendar_sized(df_te, f_vec, cost, max_concurrent=2, f_total=0.6):
    assert len(df_te) == len(f_vec)
    df = df_te.copy()
    df = df.sort_index()
    f_vec = np.asarray(f_vec, dtype=float)

    if len(df) == 0:
        return {"CAGR": 0.0, "MDD": 1.0, "Calmar": 0.0,
                "accepted": 0, "skipped": 0, "days": 0}

    cal_start = pd.to_datetime(df.index.min()).normalize()
    cal_end   = pd.to_datetime(df.index.max()).normalize()
    cal = pd.bdate_range(cal_start, cal_end)
    T = len(cal)
    idx_map = {d.normalize(): i for i, d in enumerate(cal)}

    # Build entries list
    entries = []
    for (d, r, dur, f) in zip(df.index, df["y_R"].values,
                              df["y_dur"].round().clip(lower=1).astype(int).values,
                              f_vec):
        if f <= 0.0:
            continue
        pos = idx_map.get(pd.to_datetime(d).normalize(), None)
        if pos is None:
            continue
        entries.append((pos, float(r), int(dur), float(f)))

    eq = np.ones(T + 1, dtype=float)
    # active list: (endpos, daily_ret, size, started_bool)
    active = []
    accepted = len(entries)
    skipped  = 0

    for t in range(T):
        # Remove finished positions
        new_active = []
        for (e, dr, sz, started) in active:
            if e > t:
                new_active.append((e, dr, sz, started))
        active = new_active

        # Today's entries
        todays = [(e, rtot, d, f) for (e, rtot, d, f) in entries if e == t]

        # Enforce max_concurrent: if too many, keep highest size first
        if max_concurrent is not None and len(active) + len(todays) > max_concurrent:
            room = max(0, max_concurrent - len(active))
            if room < len(todays):
                # keep largest f first
                todays.sort(key=lambda x: x[3], reverse=True)
                kept, dropped = todays[:room], todays[room:]
                skipped += len(dropped)
                todays = kept

        # compute daily_ret for today's entries and append
        # daily_ret_i = (1+R)^(1/dur) - 1
        todays_active = []
        for (e, rtot, d, f) in todays:
            d = max(d, 1)
            dr = (1.0 + rtot) ** (1.0 / d) - 1.0
            todays_active.append((e, dr, f, False))  # not started (cost to apply)
        active.extend(todays_active)

        # Apply f_total cap by rescaling sizes for all active
        if len(active):
            sum_f = sum(sz for (_, _, sz, __) in active)
            if sum_f > f_total:
                scale = f_total / sum_f
                active = [(e, dr, sz * scale, started) for (e, dr, sz, started) in active]

        # Compute daily gross return
        if len(active):
            gross = 1.0
            cost_sum = 0.0
            updated = []
            for (e, dr, sz, started) in active:
                if not started:
                    # entry day: apply cost once
                    cost_sum += max(0.0, sz) * cost
                    started = True
                # limit daily_ret effect to avoid numerical explosion
                gross *= (1.0 + sz * dr)
                updated.append((e, dr, sz, started))
            active = updated
            gross = gross - 1.0
            # Equity update
            eq[t+1] = eq[t] * (1.0 + gross - cost_sum)
            # guard: avoid negative equity
            if eq[t+1] <= 1e-9:
                eq[t+1] = 1e-9
        else:
            eq[t+1] = eq[t]

    total_days = T
    mdd  = mdd_from_equity(eq)
    cagr = cagr_from_equity(eq, total_days)
    calmar = cagr / (mdd if mdd > 0 else np.inf)
    return {"CAGR": cagr, "MDD": mdd, "Calmar": calmar,
            "accepted": int(accepted), "skipped": int(skipped), "days": int(total_days)}


def add_common_args(ap: argparse.ArgumentParser):
    ap.add_argument("--data",  default="data/processed/meta_train.parquet")
    ap.add_argument("--wfdir", default="reports/wf_mae")
    ap.add_argument("--alpha", dest="alphas", action="append", type=float)
    ap.add_argument("--qmae-buf", dest="qbufs", action="append", type=float)
    ap.add_argument("--tau-q", dest="tau_qs", action="append", type=int,
                    help="percentiles for q_mae threshold on val (e.g., 40 50)")
    ap.add_argument("--theta", dest="thetas", action="append", type=float,
                    help="absolute θ (bp)")
    ap.add_argument("--theta-pctl", dest="theta_pctls", action="append", type=float,
                    help="relative θ percentile on val E[R] (e.g., 70 80)")
    ap.add_argument("--rv-cap-pct", type=float, default=None)
    ap.add_argument("--cost", type=float, default=0.0005)
    ap.add_argument("--max-concurrent", type=int, default=2)
    # sizing hyper-parameters
    ap.add_argument("--kappa", type=float, default=0.5)
    ap.add_argument("--f-max", type=float, default=0.2)
    ap.add_argument("--f-total", type=float, default=0.6)
    ap.add_argument("--B0", type=float, default=0.2, help="residual DD budget (simple constant)")
    # reporting
    ap.add_argument("--dd-eps", type=float, default=1e-3, help="epsilon for Calmar_eps")
    ap.add_argument("--min-trades", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="reports/wf_sizing")


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    alphas = args.alphas if args.alphas else [0.90]
    qbufs  = args.qbufs  if args.qbufs  else [0.0]
    tau_qs = args.tau_qs if args.tau_qs else [40, 50]
    thetas = args.thetas if args.thetas else []
    theta_pctls = args.theta_pctls if args.theta_pctls else []
    rng = np.random.RandomState(args.seed)

    # Load meta folds
    meta = json.loads(Path(args.wfdir, "wf_meta.json").read_text())
    rows = []

    for m in meta:
        # slice raw data for val/test windows
        def slice_period(s, e):
            df = pd.read_parquet(args.data).sort_index()
            return df.loc[(df.index >= pd.to_datetime(s)) & (df.index < pd.to_datetime(e))].copy()

        df_val = slice_period(m["val"][0],  m["val"][1])
        df_tst = slice_period(m["test"][0], m["test"][1])

        keep = ["y_R", "y_mae", "y_dur", "symbol"]
        if (args.rv_cap_pct is not None) and ("mkt_rv_20" in df_val.columns):
            capv = np.nanpercentile(df_val["mkt_rv_20"].values, args.rv_cap_pct)
            capt = np.nanpercentile(df_tst["mkt_rv_20"].values, args.rv_cap_pct)
            df_val = df_val[df_val["mkt_rv_20"] <= capv]
            df_tst = df_tst[df_tst["mkt_rv_20"] <= capt]

        df_val = df_val[keep].dropna()
        df_tst = df_tst[keep].dropna()
        if len(df_val) == 0 or len(df_tst) == 0:
            rows.append({"fold": m["fold"], "status": "empty_after_join"})
            continue

        # Load E_R preds for this fold
        E_va = pd.read_parquet(Path(args.wfdir, f"fold{m['fold']:02d}_val_E_R.parquet")).rename(columns={"E_R": "E_R"})
        E_te = pd.read_parquet(Path(args.wfdir, f"fold{m['fold']:02d}_test_E_R.parquet")).rename(columns={"E_R": "E_R"})
        df_val = df_val.join(E_va, how="inner").dropna()
        df_tst = df_tst.join(E_te, how="inner").dropna()
        df_val["y_dur"] = df_val["y_dur"].round().clip(lower=1).astype(int)
        df_tst["y_dur"] = df_tst["y_dur"].round().clip(lower=1).astype(int)

        if len(df_val) == 0 or len(df_tst) == 0:
            rows.append({"fold": m["fold"], "status": "empty_after_E_R_join"})
            continue

        for a in alphas:
            tag = f"a{int(round(a*100)):02d}"
            try:
                qv = pd.read_parquet(Path(args.wfdir, f"fold{m['fold']:02d}_val_q_{tag}.parquet")).rename(columns={"q_mae": "q_mae"})
                qt = pd.read_parquet(Path(args.wfdir, f"fold{m['fold']:02d}_test_q_{tag}.parquet")).rename(columns={"q_mae": "q_mae"})
            except Exception:
                rows.append({"fold": m["fold"], "alpha": a, "status": "missing_q"})
                continue

            v = df_val.join(qv, how="inner").dropna()
            t = df_tst.join(qt, how="inner").dropna()
            if len(v) == 0 or len(t) == 0:
                rows.append({"fold": m["fold"], "alpha": a, "status": "empty_after_q_join"})
                continue

            for qbuf in qbufs:
                vv = v.copy(); tt = t.copy()
                vv["q_mae_adj"] = vv["q_mae"] + qbuf
                tt["q_mae_adj"] = tt["q_mae"] + qbuf

                # thresholds on val
                # tau from percentile of q_mae_adj
                tau_vals = np.percentile(vv["q_mae_adj"].values, [int(x) for x in tau_qs])
                # theta from absolute list + relative percentile(s)
                theta_vals = list(thetas)
                if theta_pctls:
                    for pctl in theta_pctls:
                        theta_vals.append(float(np.nanpercentile(vv["E_R"].values, pctl)))
                theta_vals = sorted(set(theta_vals))

                for tau_val in tau_vals:
                    for theta in theta_vals if len(theta_vals) else [0.0]:
                        # Gate on test
                        pick_mask = ((tt["q_mae_adj"].values <= tau_val) &
                                     (tt["E_R"].values >= theta))
                        # sizes
                        q_arr  = tt["q_mae_adj"].values
                        er_arr = tt["E_R"].values
                        B0 = args.B0
                        f_raw = size_riskaware(q_arr, er_arr, B0, theta,
                                               kappa=args.kappa, f_max=args.f_max,
                                               f_total=args.f_total, eps=1e-4)
                        f_vec = np.where(pick_mask, f_raw, 0.0)

                        # Gate (sized)
                        gate = eval_from_calendar_sized(tt, f_vec, cost=args.cost,
                                                        max_concurrent=args.max_concurrent,
                                                        f_total=args.f_total)

                        # Random baseline: same acceptance ratio & average size
                        acc_rate = float((f_vec > 0).mean())
                        # choose random days
                        rand_mask = rng.rand(len(tt)) < acc_rate
                        # draw random sizes ~ U(0,f_max) scaled by sigmoid((E_R-theta)/scale)
                        scale_er = max(1e-4, np.nanstd(er_arr))
                        s_er = sigmoid((er_arr - theta) / (scale_er + 1e-8))
                        rand_sizes = rng.rand(len(tt)) * args.f_max * s_er
                        f_rand = np.where(rand_mask, rand_sizes, 0.0)
                        rnd = eval_from_calendar_sized(tt, f_rand, cost=args.cost,
                                                       max_concurrent=args.max_concurrent,
                                                       f_total=args.f_total)

                        # All-in baseline: accept all candidates with constant size f_max * s_er
                        s_er_all = sigmoid((er_arr - theta) / (scale_er + 1e-8))
                        f_all = args.f_max * s_er_all  # mild quality scaling
                        alln = eval_from_calendar_sized(tt, f_all, cost=args.cost,
                                                        max_concurrent=args.max_concurrent,
                                                        f_total=args.f_total)

                        row = {
                            "fold": m["fold"], "alpha": a, "qbuf": qbuf,
                            "tau_val": float(tau_val), "theta": float(theta),
                            "Gate_CAGR": gate["CAGR"], "Gate_MDD": gate["MDD"], "Gate_Calmar": gate["Calmar"],
                            "Rand_CAGR": rnd["CAGR"],  "Rand_MDD": rnd["MDD"],  "Rand_Calmar": rnd["Calmar"],
                            "All_CAGR":  alln["CAGR"], "All_MDD":  alln["MDD"], "All_Calmar":  alln["Calmar"],
                            "accepted": gate["accepted"], "skipped": gate["skipped"], "days": gate["days"],
                            "status": "ok"
                        }
                        rows.append(row)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows)

    # ε-Calmar & quality filter
    eps = args.dd_eps
    for who in ["Gate", "Rand", "All"]:
        out[f"{who}_Calmar_eps"] = out[f"{who}_CAGR"] / (out[f"{who}_MDD"] + eps)

    out.to_csv(outdir / "wf_results_per_fold.csv", index=False)

    mask = (out["status"] == "ok") & (out["accepted"] >= args.min_trades)
    grp = out[mask].groupby(["alpha", "qbuf"])

    # mean & median summaries
    cols = ["Gate_CAGR", "Gate_MDD", "Gate_Calmar_eps",
            "Rand_CAGR", "Rand_MDD", "Rand_Calmar_eps",
            "All_CAGR",  "All_MDD",  "All_Calmar_eps"]
    agg_mean   = grp[cols].mean().reset_index()
    agg_median = grp[cols].median().reset_index()

    agg_mean.to_csv(outdir / "wf_summary_sizing_mean.csv", index=False)
    agg_median.to_csv(outdir / "wf_summary_sizing_median.csv", index=False)

    print("\n=== WF sizing summary (median ε-Calmar) ===")
    if len(agg_median):
        print(agg_median.sort_values(["Gate_Calmar_eps"], ascending=False).head(10))
    else:
        print("no rows after filter")
    print(f"[saved] per-fold  -> {outdir/'wf_results_per_fold.csv'}")
    print(f"[saved] mean     -> {outdir/'wf_summary_sizing_mean.csv'}")
    print(f"[saved] median   -> {outdir/'wf_summary_sizing_median.csv'}")

if __name__ == "__main__":
    main()
