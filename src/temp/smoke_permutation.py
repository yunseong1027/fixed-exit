import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score, log_loss
from lightgbm import LGBMClassifier

d = pd.read_parquet("data/processed/train_aapl_qqq.parquet").sort_index()
X = d.drop(columns=["symbol","y"])
y = (d["y"]==1).astype(int).values  # binary(+1 vs rest)

i_test = int(len(X)*0.8); i_val = int(i_test*0.9)
Xtr,Xv,Xte = X.iloc[:i_val],X.iloc[i_val:i_test],X.iloc[i_test:]
ytr,yv,yte = y[:i_val],y[i_val:i_test],y[i_test:]

# 라벨 섞기
rng = np.random.default_rng(42)
ytr_perm = rng.permutation(ytr)

clf = LGBMClassifier(objective="binary", n_estimators=3000, num_leaves=255, min_data_in_leaf=20)
clf.fit(Xtr,ytr_perm, eval_set=[(Xv,yv)], eval_metric="binary_logloss")
p = clf.predict_proba(Xte)[:,1]
print("perm AUC :", roc_auc_score(yte,p))
print("perm LogL:", log_loss(yte, np.c_[1-p,p]))
