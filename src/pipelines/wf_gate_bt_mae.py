import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

try:
    from .gate_bt_mae import (
        evaluate_random, evaluate_all, block_bootstrap_metrics
    )
except ImportError:
    import sys, os
    sys.path.append(os.path.dirname(__file__))
    from gate_bt_mae import (
        evaluate_random, evaluate_all, block_bootstrap_metrics
    )

def mdd_from_equity(eq):
    peak = np.maximum.accumulate(eq)
    dd = 1.0 - eq/peak
    return float(np.max(dd))

def cagr_from_equity(eq, total_days):
    """연 252 영업일 가정, 전체 캘린더 일수로 연율화."""
    if len(eq) < 2 or total_days <= 0:
        return 0.0
    years = total_days / 252.0
    total = eq[-1] / eq[0]
    return float(total**(1/years) - 1.0)

def eval_from_mask_calendar(df_te, pick_mask, cost, max_concurrent=1):
    """
    캘린더(전체 영업일) 기준 누적
    - 동시 보유 시 동일가중 평균으로 하루 수익 합성
    - per-trade 비용은 당일 새로 진입한 포지션의 가중치만큼 1회 차감
    """
    # 1) 캘린더
    cal_start = pd.to_datetime(df_te.index.min()).normalize()
    cal_end   = pd.to_datetime(df_te.index.max()).normalize()
    cal = pd.date_range(cal_start, cal_end, freq="B")
    T = len(cal)
    idx_map = {d.normalize(): i for i, d in enumerate(cal)}

    # 2) 엔트리 빌드
    R = df_te["y_R"].values
    D = df_te["y_dur"].round().clip(lower=1).astype(int).values
    entries = []
    for d, use, r, dur in zip(df_te.index, pick_mask, R, D):
        if not use:
            continue
        pos = idx_map.get(pd.to_datetime(d).normalize(), None)
        if pos is None:
            continue
        entries.append((pos, float(r), int(dur)))

    # 3) 누적
    eq = np.ones(T + 1, dtype=float)
    # active: (endpos, daily_ret, started?)
    active = []
    accepted = skipped = 0

    for t in range(T):
        # 만기 제거
        if active:
            active = [(e, dr, st) for (e, dr, st) in active if e > t]

        # 오늘 엔트리
        todays = [e for e in entries if e[0] == t]
        # 동시보유 상한
        if max_concurrent is not None and len(active) + len(todays) > max_concurrent:
            room = max(0, max_concurrent - len(active))
            if room < len(todays):
                # 들어온 순서대로 컷(원한다면 크기 기준 정렬해도 됨)
                kept, dropped = todays[:room], todays[room:]
                skipped += len(dropped)
                todays = kept

        # 오늘 시작할 포지션 daily_ret 계산
        new_active = []
        for (_, rtot, d) in todays:
            d = max(d, 1)
            dr = (1.0 + rtot)**(1.0 / d) - 1.0
            new_active.append((t + d, dr, False))
        if new_active:
            accepted += len(new_active)
            active.extend(new_active)

        # 하루 수익 합성(동일가중) + 비용(새 진입 가중치만큼)
        if active:
            m = len(active)
            # 평균 수익
            avg_dr = sum(dr for (_, dr, __) in active) / m
            # 오늘 새 진입 수
            n_new = sum(1 for (_, __, st) in active if not st)
            # 비용은 새 진입 가중치만큼 차감
            cost_pen = (n_new / m) * cost if m > 0 else 0.0

            # started 플래그 갱신
            active = [(e, dr, True) for (e, dr, st) in active]

            # 자본 갱신 (과대 레버리지 방지: 더 이상 ∏ 사용 안 함)
            port_ret = avg_dr - cost_pen
            eq[t+1] = eq[t] * (1.0 + port_ret)
        else:
            eq[t+1] = eq[t]

    # 성과 지표
    total_days = T
    mdd  = mdd_from_equity(eq)
    cagr = cagr_from_equity(eq, total_days)   # 연 252일 가정
    calmar = (cagr / mdd) if mdd > 0 else np.inf
    return {"CAGR": cagr, "MDD": mdd, "Calmar": calmar,
            "accepted": accepted, "skipped": skipped, "days": total_days}

# 파라미터 대입
def pick_on_val_grid(vv, tau_qs, theta_bps, theta_pctls, cost, max_concurrent,
                     enforce_rand=True, select="calmar", mu=None, mu_eps=1e-4):
    qarr = vv["q_mae_adj"].to_numpy()
    qarr = qarr[np.isfinite(qarr)]
    if qarr.size == 0:
        return None
    qs_vals = np.nanpercentile(qarr, tau_qs)

    best = None
    best_key = None

    thetas_all = list(theta_bps or [])
    if theta_pctls:
        er = vv["E_R"].to_numpy()
        er = er[np.isfinite(er)]
        if er.size:
            for p in theta_pctls:
                try:
                    thetas_all.append(float(np.nanpercentile(er, p)))
                except Exception:
                    pass
    # 후보가 하나도 없으면 기본 0.0만 남기기
    if not thetas_all:
        thetas_all = [0.0]

    rng = np.random.RandomState(42)

    for tau, tau_q in zip(tau_qs, qs_vals):
        for th in thetas_all:
            pick = ((vv["q_mae_adj"] <= tau_q) & (vv["E_R"] >= th)).values
            if pick.size == 0:
                continue
            acc_rate = float(pick.mean())
            if not np.isfinite(acc_rate) or acc_rate == 0.0:
                continue

            gate = eval_from_mask_calendar(vv, pick, cost, max_concurrent=max_concurrent)
            rnd  = eval_from_mask_calendar(vv, rng.rand(len(vv)) < acc_rate, cost, max_concurrent=max_concurrent)

            if enforce_rand and (gate["CAGR"] < rnd["CAGR"]):
                continue

            if mu is None:
                key = (-gate["Calmar"], gate["MDD"]) if select == "calmar" else (gate["MDD"], -gate["Calmar"])
            else:
                J = gate["CAGR"] - mu * (gate["MDD"] + mu_eps)
                key = (-J,)

            if (best is None) or (key < best_key):
                best_key = key
                best = {"tau_pct": tau, "tau_val": float(tau_q), "theta": float(th),
                        **gate,
                        "Rand_CAGR": rnd["CAGR"], "Rand_MDD": rnd["MDD"], "Rand_Calmar": rnd["Calmar"]}
    return best

def validation(vv, tau_qs, theta_bps, theta_pctls, cost, max_concurrent,
                      mu_init=0.5, mu_eps=1e-4, mu_tol=1e-4, mu_max_iter=10,
                      enforce_rand=True):
    mu = float(mu_init); hist=[]
    best=None; best_key=None
    for it in range(1, mu_max_iter+1):
        cand = pick_on_val_grid(vv, tau_qs, theta_bps, theta_pctls, cost, max_concurrent,
                                enforce_rand=enforce_rand, select="calmar",
                                mu=mu, mu_eps=mu_eps)
        if cand is None:
            return None, hist
        f, g = cand["CAGR"], cand["MDD"] + mu_eps
        phi = f - mu*g
        mu_new = f/g if g > 0 else mu
        hist.append({"iter": it, "mu": mu, "CAGR": f, "MDD": g - mu_eps, "phi": phi,
                     "tau_val": cand["tau_val"], "theta": cand["theta"]})
        if abs(phi) <= mu_tol or abs(mu_new - mu) <= mu_tol:
            cand.update({"mu_final": mu_new, "phi_final": phi, "db_iters": it})
            return cand, hist
        mu = mu_new
        best = cand
    best.update({"mu_final": mu, "phi_final": best["CAGR"] - mu*(best["MDD"] + mu_eps), "db_iters": mu_max_iter})
    return best, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",  default="data/processed/meta_train.parquet")
    ap.add_argument("--wfdir", default="reports/wf_mae")
    ap.add_argument("--max-concurrent", type=int, default=1)
    ap.add_argument("--rv-cap-pct", type=float, default=None)
    ap.add_argument("--cost", type=float, default=0.0005)
    ap.add_argument("--tau-q", dest="tau_qs", action="append", type=int)
    ap.add_argument("--theta", dest="thetas", action="append", type=float, help="absolute θ(bps)")
    ap.add_argument("--theta-pctl", dest="theta_pctls", action="append", type=float, help="relative θ(percentile on val E[R])")
    ap.add_argument("--alpha", dest="alphas", action="append", type=float)
    ap.add_argument("--qmae-buf", dest="qbufs", action="append", type=float)
    ap.add_argument("--use-optimization", action="store_true")
    ap.add_argument("--mu-init", type=float, default=0.5)
    ap.add_argument("--mu-eps",  type=float, default=1e-4)
    ap.add_argument("--mu-tol",  type=float, default=1e-4)
    ap.add_argument("--mu-max-iter", type=int, default=10)
    ap.add_argument("--relax-rand-constraint", action="store_true")
    ap.add_argument("--dd-eps", type=float, default=1e-3, help="MDD floor for Calmar_eps (e.g., 0.001=10bp)")
    ap.add_argument("--min-trades", type=int, default=3, help="min accepted trades per fold included in summary")
    ap.add_argument("--use-conformal", action="store_true",
                    help="apply delta stored by wf_train_mae: q_mae_adj = q_mae + delta (+ qbuf)")
    ap.add_argument("--delta-from-json", action="store_true",
                    help="prefer per-fold delta JSON (fold##_val_q_aXX_delta.json) over wf_meta.json")
    ap.add_argument("--outdir", default="reports/wf_gate")
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    tau_qs = args.tau_qs if args.tau_qs else [20, 30, 40, 50]
    thetas = args.thetas if args.thetas else []             # 절대 θ
    theta_pctls = args.theta_pctls if args.theta_pctls else []  # 상대 θ
    alphas = args.alphas if args.alphas else [0.90]
    qbufs  = args.qbufs  if args.qbufs  else [0.0]
    enforce_rand = (not args.relax_rand_constraint)

    meta = json.loads(Path(args.wfdir, "wf_meta.json").read_text())

    rows = []; db_logs = []

    for m in meta:
        # 공통 E_R
        E_va = pd.read_parquet(Path(args.wfdir, f"fold{m['fold']:02d}_val_E_R.parquet")).rename(columns={"E_R": "E_R"})
        E_te = pd.read_parquet(Path(args.wfdir, f"fold{m['fold']:02d}_test_E_R.parquet")).rename(columns={"E_R": "E_R"})

        # 기간 슬라이스
        def _slice(s, e):
            df = pd.read_parquet(args.data).sort_index()
            return df.loc[(df.index >= pd.to_datetime(s)) & (df.index < pd.to_datetime(e))].copy()

        df_val = _slice(m["val"][0],  m["val"][1])
        df_tst = _slice(m["test"][0], m["test"][1])
        keep = ["y_R", "y_mae", "y_dur", "symbol"]

        # 레짐 컷
        if (args.rv_cap_pct is not None) and ("mkt_rv_20" in df_val.columns):
            capv = np.nanpercentile(df_val["mkt_rv_20"].values, args.rv_cap_pct); df_val = df_val[df_val["mkt_rv_20"] <= capv]
            capt = np.nanpercentile(df_tst["mkt_rv_20"].values, args.rv_cap_pct); df_tst = df_tst[df_tst["mkt_rv_20"] <= capt]

        df_val = df_val[keep].join(E_va, how="inner").dropna()
        df_tst = df_tst[keep].join(E_te, how="inner").dropna()
        df_val["y_dur"] = df_val["y_dur"].round().clip(lower=1).astype(int)
        df_tst["y_dur"] = df_tst["y_dur"].round().clip(lower=1).astype(int)
        if len(df_val) == 0 or len(df_tst) == 0:
            rows.append({"fold": m["fold"], "status": "empty_after_join"})
            continue

        for a in alphas:
            tag = f"a{int(round(a * 100)):02d}"
            qv = pd.read_parquet(Path(args.wfdir, f"fold{m['fold']:02d}_val_q_{tag}.parquet")).rename(columns={"q_mae": "q_mae"})
            qt = pd.read_parquet(Path(args.wfdir, f"fold{m['fold']:02d}_test_q_{tag}.parquet")).rename(columns={"q_mae": "q_mae"})
            v = df_val.join(qv, how="inner").dropna()
            t = df_tst.join(qt, how="inner").dropna()
            if len(v) == 0 or len(t) == 0:
                rows.append({"fold": m["fold"], "alpha": a, "status": "empty_after_q_join"})
                continue

            # delta 로드
            delta = 0.0
            if args.use_conformal:
                if args.delta_from_json:
                    jp = Path(args.wfdir)/f"fold{m['fold']:02d}_val_q_{tag}_delta.json"
                    if jp.exists():
                        try:
                            delta = float(json.loads(jp.read_text(encoding="utf-8")).get("delta", 0.0))
                        except Exception:
                            delta = 0.0
                if delta == 0.0 and "delta" in m and tag in m["delta"]:
                    try:
                        delta = float(m["delta"][tag])
                    except Exception:
                        delta = 0.0

            for eps in qbufs:
                vv = v.copy(); tt = t.copy()
                # delta + qbuf 적용
                vv["q_mae_adj"] = vv["q_mae"] + (delta if args.use_conformal else 0.0) + eps
                tt["q_mae_adj"] = tt["q_mae"] + (delta if args.use_conformal else 0.0) + eps

                # (1) val에서 선택
                if args.use_optimization:
                    best, hist = validation(
                        vv, tau_qs, thetas, theta_pctls, cost=args.cost,
                        max_concurrent=args.max_concurrent,
                        mu_init=args.mu_init, mu_eps=args.mu_eps,
                        mu_tol=args.mu_tol, mu_max_iter=args.mu_max_iter,
                        enforce_rand=enforce_rand
                    )
                    db_logs.append({"fold": m["fold"], "alpha": a, "qbuf": eps, "hist": hist})
                else:
                    best = pick_on_val_grid(
                        vv, tau_qs, thetas, theta_pctls, cost=args.cost,
                        max_concurrent=args.max_concurrent,
                        enforce_rand=enforce_rand, select="calmar", mu=None
                    )

                if best is None:
                    rows.append({"fold": m["fold"], "alpha": a, "qbuf": eps, "status": "no_feasible_on_val"})
                    continue

                tau_val = best["tau_val"]; theta = best["theta"]

                # (2) test 적용
                pick_te = ((tt["q_mae_adj"] <= tau_val) & (tt["E_R"] >= theta)).values
                gate = eval_from_mask_calendar(tt, pick_te, args.cost, max_concurrent=args.max_concurrent)
                rnd  = eval_from_mask_calendar(tt, np.random.RandomState(42).rand(len(tt)) < float(pick_te.mean()),
                                               args.cost, max_concurrent=args.max_concurrent)
                alln = eval_from_mask_calendar(tt, np.ones(len(tt), dtype=bool), args.cost, max_concurrent=args.max_concurrent)
                ci   = block_bootstrap_metrics(tt, pick_te, cost=args.cost, scheduler="exclusive_duration",
                                               block=20, B=200, seed=42)

                row = {"fold": m["fold"], "alpha": a, "qbuf": eps, "tau_val": float(tau_val), "theta": float(theta), "delta": float(delta),
                       "Gate_CAGR": gate["CAGR"], "Gate_MDD": gate["MDD"], "Gate_Calmar": gate["Calmar"],
                       "Rand_CAGR": rnd["CAGR"],  "Rand_MDD": rnd["MDD"],  "Rand_Calmar": rnd["Calmar"],
                       "All_CAGR":  alln["CAGR"], "All_MDD":  alln["MDD"], "All_Calmar":  alln["Calmar"],
                       "Gate_CAGR_CI_low": ci["CAGR_CI"][0], "Gate_CAGR_CI_high": ci["CAGR_CI"][1],
                       "Gate_MDD_CI_low":  ci["MDD_CI"][0],  "Gate_MDD_CI_high":  ci["MDD_CI"][1],
                       "accepted": gate["accepted"], "skipped": gate["skipped"], "days": gate["days"],
                       "status": "ok"}
                if args.use_optimization and ("mu_final" in best):
                    row.update({"mu_final": best["mu_final"], "phi_final": best["phi_final"], "db_iters": best["db_iters"]})
                rows.append(row)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows)
    # ε-Calmar (MDD=0 대비)
    eps = args.dd_eps
    for who in ["Gate", "Rand", "All"]:
        out[f"{who}_Calmar_eps"] = out[f"{who}_CAGR"] / (out[f"{who}_MDD"] + eps)
    out.to_csv(outdir / "wf_results_per_fold.csv", index=False)
    # 집계(품질 필터: 최소 체결수)
    mask = (out["status"] == "ok") & (out["accepted"] >= args.min_trades)
    grp = out[mask].groupby(["alpha", "qbuf"])

    # 평균 요약
    agg_mean = grp[["Gate_CAGR", "Gate_MDD", "Gate_Calmar_eps",
                    "Rand_CAGR", "Rand_MDD", "Rand_Calmar_eps",
                    "All_CAGR",  "All_MDD",  "All_Calmar_eps"]].mean().reset_index()
    agg_mean.to_csv(outdir / "wf_summary_by_alpha_qbuf_mean.csv", index=False)

    # 중앙값 요약(권장)
    agg_median = grp[["Gate_CAGR", "Gate_MDD", "Gate_Calmar_eps",
                      "Rand_CAGR", "Rand_MDD", "Rand_Calmar_eps",
                      "All_CAGR",  "All_MDD",  "All_Calmar_eps"]].median().reset_index()
    agg_median.to_csv(outdir / "wf_summary_by_alpha_qbuf_median.csv", index=False)

    # 중앙값 기준 Top-10 출력
    if len(agg_median):
        print("\n=== WF summary (median ε-Calmar; mean/median CSV 저장) ===")
        print(agg_median.sort_values(["Gate_Calmar_eps"], ascending=False).head(10))
    else:
        print("\n=== WF summary ===\n(no rows after min-trades filter)")

    # 최적화 로그
    if len([1 for _ in db_logs]):
        with open(outdir / "wf_db_logs.json", "w") as f:
            json.dump(db_logs, f, indent=2)

    print(f"[saved] per-fold -> {outdir}/wf_results_per_fold.csv")
    print(f"[saved] mean  -> {outdir}/wf_summary_by_alpha_qbuf_mean.csv")
    print(f"[saved] median-> {outdir}/wf_summary_by_alpha_qbuf_median.csv")
    if len([1 for _ in db_logs]):
        print(f"[saved] optimization logs -> {outdir}/wf_db_logs.json")

if __name__ == "__main__":
    main()
