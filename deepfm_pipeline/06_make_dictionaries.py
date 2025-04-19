#!/usr/bin/env python
import gzip, pickle, yaml
import pyarrow.parquet as pq
import pandas as pd
from pathlib import Path

def read_cfg(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

cfg = read_cfg()
base = cfg["outputs"]["hdfs_base"]
out_dir = Path(cfg["outputs"]["data_dir"])
out_dir.mkdir(exist_ok=True)

def p2p(path, cols=None):
    return pq.read_table(path, columns=cols).to_pandas()

# categorical features --------------------------------------------------
kt_cols = ["inn_kt_idx"] + [f"{c}_idx" for c in
    ["inn_kt","okved_cd_kt","okato_cd_kt",
     "bic_kt_34","bic_kt_56","bic_kt_79",
     "num_kt_13","num_kt_45","num_kt_68","kt_buyers_count"]]
dt_cols = ["inn_dt_idx"] + [f"{c}_idx" for c in
    ["inn_dt","okved_cd_dt","okato_cd_dt",
     "bic_dt_34","bic_dt_56","bic_dt_79",
     "num_dt_13","num_dt_45","num_dt_68","dt_buyers_count"]]

kt_df = p2p(f"{base}/kt_pass_idx", kt_cols)
dt_df = p2p(f"{base}/dt_pass_idx", dt_cols)

kt_feat = {row["inn_kt_idx"]: [row[c] for c in kt_cols[1:]]
           for _, row in kt_df.iterrows()}
dt_feat = {row["inn_dt_idx"]: [row[c] for c in dt_cols[1:]]
           for _, row in dt_df.iterrows()}

with gzip.open(out_dir / "kt_features.pkl.gz", "wb") as f:
    pickle.dump(kt_feat, f, protocol=pickle.HIGHEST_PROTOCOL)
with gzip.open(out_dir / "dt_features.pkl.gz", "wb") as f:
    pickle.dump(dt_feat, f, protocol=pickle.HIGHEST_PROTOCOL)

# embeddings ------------------------------------------------------------
for side in ["kt", "dt"]:
    df = p2p(f"{base}/{side}_embeddings")
    key = f"inn_{side}_idx"
    emb_dict = {row[key]: row["embedding"] for _, row in df.iterrows()}
    with gzip.open(out_dir / f"{side}_embeddings.pkl.gz", "wb") as f:
        pickle.dump(emb_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

# test_dict >=15 --------------------------------------------------------
inter_test = p2p(f"{base}/interactions_test", ["inn_dt","inn_kt"])
inn_dt_idx = p2p(f"{base}/dt_pass_idx", ["inn_dt","inn_dt_idx"])
inn_kt_idx = p2p(f"{base}/kt_pass_idx", ["inn_kt","inn_kt_idx"])

merged = (inter_test.merge(inn_dt_idx, on="inn_dt")
                     .merge(inn_kt_idx, on="inn_kt"))
grp = merged.groupby("inn_dt_idx")["inn_kt_idx"].agg(list)
cnt = merged.groupby("inn_dt_idx")["inn_kt_idx"].agg(len)
test_dict = {k:v for k,v in grp.items() if cnt[k] >= 15}

with gzip.open(out_dir / "test_dict.pkl.gz", "wb") as f:
    pickle.dump(test_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

# unique indices --------------------------------------------------------
unique_dt = list(dt_df["inn_dt_idx"].unique())
unique_kt = list(kt_df["inn_kt_idx"].unique())

with gzip.open(out_dir / "unique_inn_dt_idx.pkl.gz", "wb") as f:
    pickle.dump(unique_dt, f, protocol=pickle.HIGHEST_PROTOCOL)
with gzip.open(out_dir / "unique_inn_kt_idx.pkl.gz", "wb") as f:
    pickle.dump(unique_kt, f, protocol=pickle.HIGHEST_PROTOCOL)
