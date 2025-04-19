#!/usr/bin/env python
from pyspark.sql import SparkSession
from utils import read_cfg, filter_inn

cfg = read_cfg()
spark = (SparkSession.builder
         .appName("deepfm_extract_interactions")
         .enableHiveSupport()
         .getOrCreate())

train = filter_inn(spark,
                   cfg["dates"]["train_date_start"],
                   cfg["dates"]["train_date_end"])

test = filter_inn(spark,
                  cfg["dates"]["test_date_start"],
                  cfg["dates"]["test_date_end"])

base = cfg["outputs"]["hdfs_base"]
(train.write.mode("overwrite")
      .parquet(f"{base}/interactions_train"))
(test.write.mode("overwrite")
     .parquet(f"{base}/interactions_test"))
spark.stop()
