#!/usr/bin/env python
from pyspark.sql import SparkSession, functions as F
from utils import read_cfg

cfg = read_cfg()
spark = SparkSession.builder.appName("deepfm_stats").getOrCreate()

base = cfg["outputs"]["hdfs_base"]
train = spark.read.parquet(f"{base}/interactions_train")

kt_stats = (train.groupBy("inn_kt")
            .agg(F.avg("c_sum_fin").alias("kt_avg_sum"),
                 F.stddev("c_sum_fin").alias("kt_stddev_sum"),
                 F.min("c_sum_fin").alias("kt_min_sum"),
                 F.max("c_sum_fin").alias("kt_max_sum"),
                 F.expr("percentile_approx(c_sum_fin,0.5)").alias("kt_median_sum"),
                 F.countDistinct("inn_dt").alias("kt_buyers_count"),
                 F.skewness("c_sum_fin").alias("kt_skewness_sum")))

dt_stats = (train.groupBy("inn_dt")
            .agg(F.avg("c_sum_fin").alias("dt_avg_sum"),
                 F.stddev("c_sum_fin").alias("dt_stddev_sum"),
                 F.min("c_sum_fin").alias("dt_min_sum"),
                 F.max("c_sum_fin").alias("dt_max_sum"),
                 F.expr("percentile_approx(c_sum_fin,0.5)").alias("dt_median_sum"),
                 F.countDistinct("inn_kt").alias("dt_buyers_count"),
                 F.skewness("c_sum_fin").alias("dt_skewness_sum")))

kt_stats.write.mode("overwrite").parquet(f"{base}/kt_stats")
dt_stats.write.mode("overwrite").parquet(f"{base}/dt_stats")
spark.stop()
