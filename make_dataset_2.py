import os, sys, pickle, random, math
from datetime import datetime, timedelta

import numpy as np
from pyspark import SparkConf
from pyspark.sql import SparkSession, functions as sf
from pyspark.sql.window import Window
from pyspark.sql.types import StringType, DoubleType
from pyspark.ml.feature import StringIndexer

# SPARK ENV
os.environ['SPARK_MAJOR_VERSION'] = '3'
os.environ['SPARK_HOME'] = '/usr/sdp/current/spark3-client/'
os.environ['PYSPARK_DRIVER_PYTHON'] = 'python'
os.environ['LD_LIBRARY_PATH'] = '/opt/python/virtualenv/jupyter/lib:/opt/cloudera/parcels/PYENV.AUTOML/bin/python'
os.environ['PYSPARK_PYTHON'] = '/data/sdp/mlpy3811v23/bin/python'
sys.path.insert(0, '/usr/sdp/current/spark3-client/python')
sys.path.insert(0, '/usr/sdp/current/spark3-client/python/lib/py4j_current')

conf = SparkConf()\
    .setAppName('Train-Feat-TestSplit')\
    .setMaster("yarn")\
    .set("spark.executor.instances", "12")\
    .set("spark.executor.cores", "16")\
    .set("spark.driver.memory", "30g")\
    .set("spark.executor.memory", "40g")\
    .set("spark.sql.parquet.int96RebaseModeInRead", "CORRECTED")\
    .set("spark.sql.parquet.int96RebaseMode", "CORRECTED")
spark = SparkSession.builder.config(conf=conf).getOrCreate()

# PARAMETERS
train_date_start = "2024-10-01"
train_date_end = "2024-10-07"
test_date_start = "2024-10-21"
test_date_end = "2024-10-22"
save_schema = "arnsdpsbx_t_team_fin_adviser"

# READ EMBEDDINGS
emb_cols = [f"embed_{i}" for i in range(256)]
embeddings = (spark.read.parquet(
    "hdfs:///hdfsgw/arnsdpcc360__Podpidka_na_produkty_Postnovoj-CUSTOM_CIB_ML360-MON_AI_UI_EMBEDDING_V2/data/custom/cib/ml360/pa/mon_ai_ui_embedding_v2/mon=2024-09-30")
    .withColumn("embedding", sf.array(*emb_cols))
    .select("inn", "embedding")
)

# FILTER FUNCTION
def filter_inn(sqlCtx, start_date, end_date, report_dt="2024-09-30", active_flag=True):
    base = sqlCtx.table("arnsdpsbx_t_team_apm.tmb_basis_client")\
        .filter((sf.col("report_dt") == report_dt) &
                (sf.col("org_segment_sber").isin("Микро", "Малые")) &
                (sf.col("type").isin("ИП", "ЮЛ")))
    if active_flag:
        base = base.filter(sf.col("active_flg") == 1)
    bas_col = sqlCtx.table("prx_baza_custom_cib_products_custom_cib_products.basis_transactions_coloured")
    f = (bas_col.filter((sf.col("short_dt") >= start_date) & (sf.col("short_dt") <= end_date))
         .filter(sf.col("predicted_value_last") == "оплата по договору")
         .join(
            base.select("inn", "okved_cd", "okato_cd", "org_segment_msp", "org_segment_sber")\
                .withColumnRenamed("inn", "inn_kt")\
                .withColumnRenamed("okved_cd", "okved_cd_kt")\
                .withColumnRenamed("okato_cd", "okato_cd_kt"),
            on="inn_kt", how="inner")
         .select("inn_dt", "inn_kt", "c_sum", "short_dt",
                 "okved_cd_kt", "okato_cd_kt", "c_bic_kt", "c_bic_dt", "c_num_kt", "c_num_dt")
         .withColumn("bic_kt_34", sf.substring("c_bic_kt", 3, 2))
         .withColumn("bic_kt_56", sf.substring("c_bic_kt", 5, 2))
         .withColumn("bic_kt_79", sf.substring("c_bic_kt", 7, 2))
         .withColumn("bic_dt_34", sf.substring("c_bic_dt", 3, 2))
         .withColumn("bic_dt_56", sf.substring("c_bic_dt", 5, 2))
         .withColumn("bic_dt_79", sf.substring("c_bic_dt", 7, 2))
         .withColumn("num_kt_13", sf.substring("c_num_kt", 1, 3))
         .withColumn("num_kt_45", sf.substring("c_num_kt", 4, 2))
         .withColumn("num_kt_68", sf.substring("c_num_kt", 6, 3))
         .withColumn("num_dt_13", sf.substring("c_num_dt", 1, 3))
         .withColumn("num_dt_45", sf.substring("c_num_dt", 4, 2))
         .withColumn("num_dt_68", sf.substring("c_num_dt", 6, 3))
    )
    f = f.join(
        base.select("inn", "okved_cd", "okato_cd")\
            .withColumnRenamed("inn", "inn_dt")\
            .withColumnRenamed("okved_cd", "okved_cd_dt")\
            .withColumnRenamed("okato_cd", "okato_cd_dt"),
        on="inn_dt", how="inner"
    )
    return (f.groupby("inn_dt", "inn_kt", "short_dt",
                      "okved_cd_kt", "okato_cd_kt", "okved_cd_dt", "okato_cd_dt",
                      "bic_kt_34", "bic_kt_56", "bic_kt_79",
                      "bic_dt_34", "bic_dt_56", "bic_dt_79",
                      "num_kt_13", "num_kt_45", "num_kt_68",
                      "num_dt_13", "num_dt_45", "num_dt_68")
            .agg(sf.sum("c_sum").alias("c_sum_fin"))
    )

# AGGREGATE FEATURES & SKEWNESS
df = filter_inn(spark, train_date_start, train_date_end)
kt_win = Window.partitionBy("inn_kt")
dt_win = Window.partitionBy("inn_dt")
agg_kt = df.groupBy("inn_kt").agg(
    sf.avg("c_sum_fin").alias("kt_avg_sum"),
    sf.stddev("c_sum_fin").alias("kt_stddev_sum"),
    sf.min("c_sum_fin").alias("kt_min_sum"),
    sf.max("c_sum_fin").alias("kt_max_sum"),
    sf.expr("percentile_approx(c_sum_fin, 0.5)").alias("kt_median_sum"),
    sf.countDistinct("inn_dt").alias("kt_buyers_count")
)
agg_dt = df.groupBy("inn_dt").agg(
    sf.avg("c_sum_fin").alias("dt_avg_sum"),
    sf.stddev("c_sum_fin").alias("dt_stddev_sum"),
    sf.min("c_sum_fin").alias("dt_min_sum"),
    sf.max("c_sum_fin").alias("dt_max_sum"),
    sf.expr("percentile_approx(c_sum_fin, 0.5)").alias("dt_median_sum"),
    sf.countDistinct("inn_kt").alias("dt_buyers_count")
)
df = df.join(agg_kt, on="inn_kt", how="left")\
       .join(agg_dt, on="inn_dt", how="left")
df = df.withColumn("kt_skewness_sum",
        (sf.avg(sf.pow(sf.col("c_sum_fin")-sf.col("kt_avg_sum"), 3)).over(kt_win))/
        (sf.pow(sf.col("kt_stddev_sum"), 3)+sf.lit(1e-9))
    ).withColumn("dt_skewness_sum",
        (sf.avg(sf.pow(sf.col("c_sum_fin")-sf.col("dt_avg_sum"), 3)).over(dt_win))/
        (sf.pow(sf.col("dt_stddev_sum"), 3)+sf.lit(1e-9))
    )

# SPLIT: ВИТРИНЫ
# Витрина взаимодействий
interactions_train = df.select("inn_kt", "inn_dt", "short_dt")\
    .withColumn("label", sf.lit(1))\
    .withColumn("short_dt", sf.to_date("short_dt", "yyyy-MM-dd"))

# Витрина с фичами kt
kt_cols = ["inn_kt", "okved_cd_kt", "okato_cd_kt", "bic_kt_34", "bic_kt_56", "bic_kt_79",
           "num_kt_13", "num_kt_45", "num_kt_68", "kt_buyers_count",
           "kt_avg_sum", "kt_stddev_sum", "kt_min_sum", "kt_max_sum",
           "kt_median_sum", "kt_skewness_sum"]
kt_features = df.select(kt_cols).distinct()\
    .join(embeddings.withColumnRenamed("inn", "inn_kt"), on="inn_kt", how="inner")\
    .withColumnRenamed("embedding", "kt_embedding")

# Витрина с фичами dt
dt_cols = ["inn_dt", "okved_cd_dt", "okato_cd_dt", "bic_dt_34", "bic_dt_56", "bic_dt_79",
           "num_dt_13", "num_dt_45", "num_dt_68", "dt_buyers_count",
           "dt_avg_sum", "dt_stddev_sum", "dt_min_sum", "dt_max_sum",
           "dt_median_sum", "dt_skewness_sum"]
dt_features = df.select(dt_cols).distinct()\
    .join(embeddings.withColumnRenamed("inn", "inn_dt"), on="inn_dt", how="inner")\
    .withColumnRenamed("embedding", "dt_embedding")

# КАТЕГОРИАЛЬНАЯ ИНДЕКСАЦИЯ
kt_cat = ["inn_kt", "okved_cd_kt", "okato_cd_kt", "bic_kt_34", "bic_kt_56", "bic_kt_79",
          "num_kt_13", "num_kt_45", "num_kt_68", "kt_buyers_count"]
dt_cat = ["inn_dt", "okved_cd_dt", "okato_cd_dt", "bic_dt_34", "bic_dt_56", "bic_dt_79",
          "num_dt_13", "num_dt_45", "num_dt_68", "dt_buyers_count"]

indexer_kt = StringIndexer(inputCols=kt_cat,
                           outputCols=[f"{c}_index" for c in kt_cat],
                           handleInvalid="keep")
model_kt = indexer_kt.fit(kt_features)
kt_features_indexed = model_kt.transform(kt_features)

indexer_dt = StringIndexer(inputCols=dt_cat,
                           outputCols=[f"{c}_index" for c in dt_cat],
                           handleInvalid="keep")
model_dt = indexer_dt.fit(dt_features)
dt_features_indexed = model_dt.transform(dt_features)

# СОХРАНЕНИЕ МАППИНГОВ
kt_mappings = {col: model_kt.labelsArray[i] for i, col in enumerate(kt_cat)}
dt_mappings = {col: model_dt.labelsArray[i] for i, col in enumerate(dt_cat)}
kt_num_embeddings = {col: len(kt_mappings[col]) for col in kt_cat}
dt_num_embeddings = {col: len(dt_mappings[col]) for col in dt_cat}
with open("mappings/kt_mappings.pkl", "wb") as f:
    pickle.dump(kt_mappings, f)
with open("mappings/dt_mappings.pkl", "wb") as f:
    pickle.dump(dt_mappings, f)
with open("mappings/kt_num_embeddings.pkl", "wb") as f:
    pickle.dump(kt_num_embeddings, f)
with open("mappings/dt_num_embeddings.pkl", "wb") as f:
    pickle.dump(dt_num_embeddings, f)

# TEST СЛОВАРЬ (только для пользователей с >=15 взаимодействиями)
test_df = filter_inn(spark, test_date_start, test_date_end)\
    .select("inn_kt", "inn_dt", "short_dt")\
    .withColumn("label", sf.lit(1))
test_df = test_df.join(
    dt_features_indexed.select("inn_dt", "inn_dt_index"),
    on="inn_dt", how="inner"
).join(
    kt_features_indexed.select("inn_kt", "inn_kt_index"),
    on="inn_kt", how="inner"
)
test_dict_df = test_df.groupBy("inn_dt_index")\
    .agg(sf.collect_set("inn_kt_index").alias("kt_set"), sf.count("*").alias("cnt"))\
    .filter(sf.col("cnt") >= 15)\
    .select("inn_dt_index", "kt_set")
test_dict = {row["inn_dt_index"]: row["kt_set"] for row in test_dict_df.collect()}
with open("mappings/test_dict.pkl", "wb") as f:
    pickle.dump(test_dict, f)

# SAVE ВИТРИН
interactions_train.coalesce(60)\
    .write.mode("overwrite").saveAsTable(f"{save_schema}.interactions_train")
kt_features_indexed.coalesce(60)\
    .write.mode("overwrite").saveAsTable(f"{save_schema}.kt_features")
dt_features_indexed.coalesce(60)\
    .write.mode("overwrite").saveAsTable(f"{save_schema}.dt_features")
