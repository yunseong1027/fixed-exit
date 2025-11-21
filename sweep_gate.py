import itertools, subprocess, sys, shlex
from pathlib import Path
import pandas as pd
import numpy as np

PY = sys.executable
DATA  = "data/processed/meta_train.parquet"
WFDIR = "reports/wf_mae"
OUTROOT = Path("reports/sweep_gate_v2"); OUTROOT.mkdir(parents=True, exist_ok=True)

grid = {
    "tau":   [(50,)],           # τ 단독: 50
    "theta": [[0.000,0.001], [0.001]],  # 완화/중간
    "mc":    [1],               # max-concurrent
    "mintr": [10],              # min-trades 집계 필수
    "eps":   [0.002]            # dd-eps
}

def run(tag, tau, thetas, mc, mintr, eps):
    outdir = OUTROOT / f"{tag}_tau{','.join(map(str,tau))}_t{''.join(str(x).replace('.','') for x in thetas)}_mc{mc}_m{mintr}_eps{str(eps).replace('.','')}"
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = f"""{PY} -m src.pipelines.wf_gate_bt_mae \
      --data {DATA} --wfdir {WFDIR} --alpha 0.90 \
      {" ".join(f"--tau-q {t}" for t in tau)} \
      {" ".join(f"--theta {t:.3f}" for t in thetas)} \
      --max-concurrent {mc} --min-trades {mintr} --dd-eps {eps} \
      --outdir {outdir}"""
    print("[run]", cmd)
    return outdir, subprocess.call(shlex.split(cmd))

def collect(outdir):
    p = outdir/"wf_results_per_fold.csv"
    if not p.exists(): return None
    df = pd.read_csv(p); df["acc_rate"]=df["accepted"]/df["days"]; df["dCalmar"]=df["Gate_Calmar_eps"]-df["Rand_Calmar_eps"]
    return {
        "outdir": str(outdir),
        "acc_rate_med": df["acc_rate"].median(skipna=True),
        "win_rate": (df["dCalmar"]>0).mean(),
        "dCalmar_med": df["dCalmar"].median(skipna=True),
        "Gate_Calmar_med": df["Gate_Calmar_eps"].median(skipna=True),
        "Rand_Calmar_med": df["Rand_Calmar_eps"].median(skipna=True),
        "All_Calmar_med": df["All_Calmar_eps"].median(skipna=True),
    }

def main():
    rows=[]
    for thetas in grid["theta"]:
        outdir, code = run("a90", grid["tau"][0], thetas, grid["mc"][0], grid["mintr"][0], grid["eps"][0])
        if code==0:
            rows.append(collect(outdir))
    pd.DataFrame(rows).to_csv(OUTROOT/"summary.csv", index=False)
    print("\n[saved]", OUTROOT/"summary.csv")

if __name__ == "__main__":
    main()





exit(0)


import itertools, subprocess, sys, shlex, json, os
from pathlib import Path
import pandas as pd
import numpy as np

PY = sys.executable

DATA  = "data/processed/meta_train.parquet"
WFDIR = "reports/wf_mae"
OUTROOT = Path("reports/sweep_gate")

# 스윕 그리드 정의
alphas    = [0.90]
tau_pairs = [(40,50), (45,55), (50,60)]
thetas    = [[0.000,0.001,0.002], [0.001,0.002], [0.001]]
mc_list   = [1]               # max-concurrent
min_trs   = [0, 10]           # min-trades
eps_list  = [0.001, 0.002]    # dd-eps

def run_one(tag, tau_pair, thetas_, mc, min_tr, dd_eps):
    outdir = OUTROOT / f"{tag}_tau{tau_pair[0]}{tau_pair[1]}_t{''.join(str(x).replace('.','') for x in thetas_)}_mc{mc}_m{min_tr}_eps{str(dd_eps).replace('.','')}"
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = f"""{PY} -m src.pipelines.wf_gate_bt_mae \
      --data {DATA} \
      --wfdir {WFDIR} \
      --alpha 0.90 \
      --tau-q {tau_pair[0]} --tau-q {tau_pair[1]} \
      {" ".join(f"--theta {t:.3f}" for t in thetas_)} \
      --max-concurrent {mc} \
      {"--min-trades "+str(min_tr) if min_tr>0 else ""} \
      --dd-eps {dd_eps} \
      --outdir {outdir}"""
    print("\n[run]", cmd)
    code = subprocess.call(shlex.split(cmd))
    return outdir, code

def collect_one(outdir):
    p = Path(outdir)/"wf_results_per_fold.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["acc_rate"] = df["accepted"]/df["days"]
    df["dCalmar"]  = df["Gate_Calmar_eps"] - df["Rand_Calmar_eps"]
    out = {
        "outdir": str(outdir),
        "n_folds": df["fold"].nunique(),
        "acc_rate_med": df["acc_rate"].median(skipna=True),
        "Gate_Calmar_med": df["Gate_Calmar_eps"].median(skipna=True),
        "Rand_Calmar_med": df["Rand_Calmar_eps"].median(skipna=True),
        "All_Calmar_med":  df["All_Calmar_eps"].median(skipna=True),
        "dCalmar_med":     df["dCalmar"].median(skipna=True),
        "win_rate":        (df["dCalmar"]>0).mean()
    }
    return out

def main():
    OUTROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for tau in tau_pairs:
        for ths in thetas:
            for mc in mc_list:
                for mt in min_trs:
                    for eps in eps_list:
                        tag=f"a90"
                        outdir, code = run_one(tag, tau, ths, mc, mt, eps)
                        if code==0:
                            s = collect_one(outdir)
                            if s:
                                s.update({"tau_lo":tau[0],"tau_hi":tau[1],"thetas":str(ths),
                                          "mc":mc,"min_trades":mt,"dd_eps":eps})
                                rows.append(s)
    summary = pd.DataFrame(rows)
    csv_path = OUTROOT/"summary.csv"
    summary.to_csv(csv_path, index=False)
    print("\n[saved]", csv_path)
    # 상위 10개 보여주기(ΔCalmar median 기준)
    if len(summary):
        print(summary.sort_values("dCalmar_med", ascending=False).head(10).to_string(index=False))

if __name__ == "__main__":
    main()