#!/usr/bin/env python
from pyspark.sql import SparkSession, functions as F
from utils import read_cfg

cfg = read_cfg()
base = cfg["outputs"]["hdfs_base"]
spark = SparkSession.builder.appName("deepfm_cat_passport").getOrCreate()

train = spark.read.parquet(f"{base}/interactions_train")

kt_cat_cols = ["okved_cd_kt","okato_cd_kt",
               "bic_kt_34","bic_kt_56","bic_kt_79",
               "num_kt_13","num_kt_45","num_kt_68"]
dt_cat_cols = ["okved_cd_dt","okato_cd_dt",
               "bic_dt_34","bic_dt_56","bic_dt_79",
               "num_dt_13","num_dt_45","num_dt_68"]

kt_pass = (train.groupBy("inn_kt")
           .agg(*[F.first(c, True).alias(c) for c in kt_cat_cols]))
dt_pass = (train.groupBy("inn_dt")
           .agg(*[F.first(c, True).alias(c) for c in dt_cat_cols]))

kt_buyers = spark.read.parquet(f"{base}/kt_stats").select("inn_kt","kt_buyers_count")
dt_buyers = spark.read.parquet(f"{base}/dt_stats").select("inn_dt","dt_buyers_count")

kt_pass = kt_pass.join(kt_buyers, "inn_kt")
dt_pass = dt_pass.join(dt_buyers, "inn_dt")

kt_pass.write.mode("overwrite").parquet(f"{base}/kt_passport")
dt_pass.write.mode("overwrite").parquet(f"{base}/dt_passport")
spark.stop()
