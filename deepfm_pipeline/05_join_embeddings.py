#!/usr/bin/env python
from pyspark.sql import SparkSession, functions as F
from utils import read_cfg

cfg = read_cfg()
base = cfg["outputs"]["hdfs_base"]
spark = SparkSession.builder.appName("deepfm_join_embeddings").getOrCreate()

emb_cols = [f"embed_{i}" for i in range(256)]
emb = (spark.read.parquet(cfg["paths"]["embeddings"])
       .withColumn("embedding", F.array(*emb_cols))
       .select("inn", "embedding"))

kt_idx = spark.read.parquet(f"{base}/kt_pass_idx")
dt_idx = spark.read.parquet(f"{base}/dt_pass_idx")

kt_emb = (kt_idx.join(emb.withColumnRenamed("inn","inn_kt"), "inn_kt")
                 .select("inn_kt_idx", "embedding"))
dt_emb = (dt_idx.join(emb.withColumnRenamed("inn","inn_dt"), "inn_dt")
                 .select("inn_dt_idx", "embedding"))

kt_emb.write.mode("overwrite").parquet(f"{base}/kt_embeddings")
dt_emb.write.mode("overwrite").parquet(f"{base}/dt_embeddings")
spark.stop()
