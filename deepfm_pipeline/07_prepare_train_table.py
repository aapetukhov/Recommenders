#!/usr/bin/env python
from pyspark.sql import SparkSession, functions as F
from utils import read_cfg

cfg = read_cfg()
base = cfg["outputs"]["hdfs_base"]
spark = (SparkSession.builder
         .enableHiveSupport()
         .appName("deepfm_train_table")
         .getOrCreate())

inter = spark.read.parquet(f"{base}/interactions_train")
kt_idx = spark.read.parquet(f"{base}/kt_pass_idx").select("inn_kt","inn_kt_idx")
dt_idx = spark.read.parquet(f"{base}/dt_pass_idx").select("inn_dt","inn_dt_idx")

(inter.join(kt_idx, "inn_kt")
      .join(dt_idx, "inn_dt")
      .withColumn("label", F.lit(1))
      .write.mode("overwrite")
      .saveAsTable(f"{cfg['outputs']['save_schema']}.{cfg['outputs']['train_name']}"))

spark.stop()
