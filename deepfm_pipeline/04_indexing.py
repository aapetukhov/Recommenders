#!/usr/bin/env python
from pyspark.sql import SparkSession, functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from utils import read_cfg

cfg = read_cfg()
base = cfg["outputs"]["hdfs_base"]
spark = SparkSession.builder.appName("deepfm_indexing").getOrCreate()

kt_cat_full = ["inn_kt","okved_cd_kt","okato_cd_kt",
               "bic_kt_34","bic_kt_56","bic_kt_79",
               "num_kt_13","num_kt_45","num_kt_68","kt_buyers_count"]
dt_cat_full = ["inn_dt","okved_cd_dt","okato_cd_dt",
               "bic_dt_34","bic_dt_56","bic_dt_79",
               "num_dt_13","num_dt_45","num_dt_68","dt_buyers_count"]

def fit_transform(df, cols, tag):
    for c in cols:
        df = df.withColumn(c, F.col(c).cast("string"))
    idxrs = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
             for c in cols]
    model = Pipeline(stages=idxrs).fit(df)
    model.write().overwrite().save(f"{base}/models/{tag}_pipe")
    out = model.transform(df)
    out.write.mode("overwrite").parquet(f"{base}/{tag}_pass_idx")
    return out

fit_transform(spark.read.parquet(f"{base}/kt_passport"), kt_cat_full, "kt")
fit_transform(spark.read.parquet(f"{base}/dt_passport"), dt_cat_full, "dt")
spark.stop()
