import pandas as pd
d = pd.read_parquet("data/processed/meta_aapl_qqq.parquet")
print(d.shape)                # (행, 열) — 이제 0행이면 안 됩니다
print(d["y"].value_counts())  # 0/1 라벨 분포
print(d.groupby("symbol")["y"].mean())  # 종목별 성공률 대략 확인


exit(0)




import pandas as pd
d = pd.read_parquet("data/processed/train_aapl_qqq.parquet")
print(d.head())
print(d.groupby("symbol")["y"].value_counts(normalize=True).unstack().round(3))
print(d.drop(columns=["symbol","y"]).isna().sum().sum())  # 0 이어야 함

exit(0)


import pandas as pd
df = pd.read_parquet("data/interim/AAPL_labeled.parquet")

# 1) ATR 준비된 구간만
valid = df["atr_14"].notna()
print("valid rows:", valid.sum(), "/", len(df))

# 2) 라벨 분포(유효 구간)
print(df.loc[valid, "label"].value_counts(normalize=True))

# 3) 이벤트 종료가 있는 레코드(±1) 몇 개 보기
print(df.loc[valid & df["label"].isin([1,-1]), ["close","tp","sl","label","event_end"]].head(10))

exit(0)

import pandas as pd
import pyarrow.parquet as pq

df = pd.read_parquet("data/raw/AAPL.parquet")
print(df.head())
print(df.dtypes)         # 각 컬럼 타입
print(df.index.min(), df.index.max(), df.shape)

# 메타데이터/스키마 확인(선택)
pf = pq.ParquetFile("data/raw/AAPL.parquet")
print(pf.metadata)       # row groups, compression 등
print(pf.schema)         # 컬럼별 타입

df = pd.read_parquet("data/interim/AAPL_labeled.parquet")
print(df.head())
print(df.dtypes)         # 각 컬럼 타입
print(df.index.min(), df.index.max(), df.shape)

# 메타데이터/스키마 확인(선택)
pf = pq.ParquetFile("data/interim/AAPL_labeled.parquet")
print(pf.metadata)       # row groups, compression 등
print(pf.schema)         # 컬럼별 타입