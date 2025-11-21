import pandas as pd, numpy as np
df = pd.read_csv("reports/wf_gate_tau50_t1_m10_eps2/wf_results_per_fold.csv")
df["acc_rate"] = df["accepted"]/df["days"]
df["dCalmar"] = df["Gate_Calmar_eps"] - df["Rand_Calmar_eps"]
print("median ΔCalmar:", np.median(df["dCalmar"].dropna()))
print("win_rate(pairwise):", (df["dCalmar"]>0).mean())


exit(0)


import pandas as pd
df = pd.read_csv("reports/wf_gate_tau5060_t12_m10/wf_results_per_fold.csv")
df["win"] = (df["Gate_Calmar_eps"] > df["Rand_Calmar_eps"]).astype(int)
print("win_rate:", df["win"].mean())

exit(0)


import pandas as pd
df = pd.read_csv("reports/wf_gate_re/wf_results_per_fold.csv")
df["acc_rate"] = df["accepted"]/df["days"]
print("acc_rate median:", round(df["acc_rate"].median(),3))
print("Gate_MDD quantiles:", df["Gate_MDD"].quantile([0.1,0.5,0.9]).to_dict())


exit(0)

import pandas as pd

df = pd.read_csv("reports/wf_gate_re/wf_results_per_fold.csv")
df["acc_rate"] = df["accepted"] / df["days"]

cols = ["fold","tau_val","theta","accepted","days","acc_rate",
        "Gate_CAGR","Gate_MDD","Rand_CAGR","Rand_MDD"]

print(df[cols].head(12).to_string(index=False))
print("\nacc_rate median:", round(df["acc_rate"].median(),4))
print("Gate_CAGR median:", round(df["Gate_CAGR"].median(),4),
      "  Rand_CAGR median:", round(df["Rand_CAGR"].median(),4))
print("Gate_MDD  median:", round(df["Gate_MDD"].median(),4),
      "  Rand_MDD  median:", round(df["Rand_MDD"].median(),4))

bad = df[(df["Gate_MDD"] > df["Rand_MDD"]) | (df["Gate_CAGR"] < df["Rand_CAGR"])]
print("\n# folds where Gate worse than Random (either MDD↑ or CAGR↓):",
      bad["fold"].nunique(), "/", df["fold"].nunique())
print(bad[["fold","tau_val","theta","acc_rate","Gate_CAGR","Gate_MDD","Rand_CAGR","Rand_MDD"]]
      .head(12).to_string(index=False))