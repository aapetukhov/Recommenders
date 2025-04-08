import os
import sys
import random
import pickle
import math
from collections import Counter
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import functions as sf
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

# SPARK SESSION
conf = SparkConf().setAppName('Andrey Train-Test-12 Generation') \
    .setMaster("yarn") \
    .set("spark.executor.instances", "12") \
    .set("spark.executor.cores", "16") \
    .set("spark.driver.memory", "30g") \
    .set("spark.executor.memory", "40g") \
    .set("spark.sql.parquet.int96RebaseModeInRead", "CORRECTED") \
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
embeddings_array = (
    spark.read.parquet("hdfs:///hdfsgw/arnsdpcc360__Podpidka_na_produkty_Postnovoj-CUSTOM_CIB_ML360-MON_AI_UI_EMBEDDING_V2/data/custom/cib/ml360/pa/mon_ai_ui_embedding_v2/mon=2024-09-30")
    .withColumn("embedding", sf.array(*emb_cols))
    .select("inn", "embedding")
)

# FILTERED DATA
def filter_inn_4recs_extended(sqlContext, start_date, end_date, report_dt, active_flag=True):
    """
    Фильтрация ИНН для рекомендаций.

    Источники:
    - arnsdpsbx_t_team_apm.tmb_basis_client
    - prx_baza_custom_cib_products_custom_cib_products.basis_transactions_coloured
    """
    import pyspark.sql.functions as sf

    # Клиенты Микро и Малые, опционально только активные
    tmb_basis_client = sqlContext.table("arnsdpsbx_t_team_apm.tmb_basis_client")
    filters = [
        sf.col("report_dt") == report_dt,
        sf.col("org_segment_sber").isin("Микро", "Малые"),
        sf.col("type").isin("ИП", "ЮЛ")
    ]
    if active_flag:
        filters.append(sf.col("active_flg") == 1)

    inn_base_info = tmb_basis_client.filter(*filters)

    # Витрина с транзакциями
    bas_col = sqlContext.table("prx_baza_custom_cib_products_custom_cib_products.basis_transactions_coloured")
    inn_filters = (
        bas_col
        .filter((sf.col("short_dt") >= start_date) & (sf.col("short_dt") <= end_date))
        .filter(sf.col("predicted_value_last") == "оплата по договору")
        .join(
            inn_base_info.select(
                "inn", "okved_cd", "okato_cd", "org_segment_msp", "org_segment_sber"
            )
            .withColumnRenamed("inn", "inn_kt")
            .withColumnRenamed("okved_cd", "okved_cd_kt")
            .withColumnRenamed("okato_cd", "okato_cd_kt"),
            on="inn_kt", how="inner"
        )
        .select(
            "inn_dt", "inn_kt", "c_sum", "short_dt",
            "okved_cd_kt", "okato_cd_kt", "c_bic_kt", "c_bic_dt", "c_num_kt", "c_num_dt"
        )
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

    # Присоединяем доп. инфу по ИНН
    inn_filters = inn_filters.join(
        inn_base_info.select("inn", "okved_cd", "okato_cd")
        .withColumnRenamed("inn", "inn_dt")
        .withColumnRenamed("okved_cd", "okved_cd_dt")
        .withColumnRenamed("okato_cd", "okato_cd_dt"),
        on="inn_dt", how="inner"
    )

    # Агрегация
    inn_filters = (
        inn_filters
        .groupby(
            "inn_dt", "inn_kt", "short_dt",
            "okved_cd_kt", "okato_cd_kt", "okved_cd_dt", "okato_cd_dt",
            "bic_kt_34", "bic_kt_56", "bic_kt_79",
            "bic_dt_34", "bic_dt_56", "bic_dt_79",
            "num_kt_13", "num_kt_45", "num_kt_68",
            "num_dt_13", "num_dt_45", "num_dt_68"
        )
        .agg(sf.sum("c_sum").alias("c_sum_fin"))
    )

    return inn_filters

inn_filters = filter_inn_4recs_extended(spark, start_date=train_date_start, end_date=train_date_end)

# BASIC AGG FEATURES
kt_window = Window.partitionBy("inn_kt")
dt_window = Window.partitionBy("inn_dt")
agg_kt = inn_filters.groupBy("inn_kt").agg(
    sf.avg("c_sum_fin").alias("kt_avg_sum"),
    sf.stddev("c_sum_fin").alias("kt_stddev_sum"),
    sf.min("c_sum_fin").alias("kt_min_sum"),
    sf.max("c_sum_fin").alias("kt_max_sum"),
    sf.expr("percentile_approx(c_sum_fin, 0.5)").alias("kt_median_sum"),
    sf.countDistinct("inn_dt").alias("kt_buyers_count")
)
agg_dt = inn_filters.groupBy("inn_dt").agg(
    sf.avg("c_sum_fin").alias("dt_avg_sum"),
    sf.stddev("c_sum_fin").alias("dt_stddev_sum"),
    sf.min("c_sum_fin").alias("dt_min_sum"),
    sf.max("c_sum_fin").alias("dt_max_sum"),
    sf.expr("percentile_approx(c_sum_fin, 0.5)").alias("dt_median_sum"),
    sf.countDistinct("inn_kt").alias("dt_buyers_count")
)
df = inn_filters.join(agg_kt, on="inn_kt", how="left").join(agg_dt, on="inn_dt", how="left")

# SKEWNESS
df = df.withColumn("kt_skewness_sum",
    (sf.avg(sf.pow(sf.col("c_sum_fin") - sf.col("kt_avg_sum"), 3)).over(kt_window)) /
    (sf.pow(sf.col("kt_stddev_sum"), 3) + sf.lit(1e-9))
).withColumn("dt_skewness_sum",
    (sf.avg(sf.pow(sf.col("c_sum_fin") - sf.col("dt_avg_sum"), 3)).over(dt_window)) /
    (sf.pow(sf.col("dt_stddev_sum"), 3) + sf.lit(1e-9))
)

# ENTROPY
def calc_entropy(values):
    counts = Counter(values)
    total = sum(counts.values())
    probs = [c / total for c in counts.values()]
    return -sum(p * math.log2(p) for p in probs if p > 0)

calc_entropy_udf = sf.udf(calc_entropy, returnType=DoubleType())
kt_entropy = df.groupBy("inn_kt").agg(
    sf.collect_list("inn_dt").alias("all_interactions"),
    sf.countDistinct("inn_dt").alias("unique_count")
).withColumn("kt_entropy", calc_entropy_udf("all_interactions")
).withColumn("kt_normalized_entropy", sf.col("kt_entropy") / sf.log2(sf.col("unique_count"))
).select("inn_kt", "kt_entropy", "kt_normalized_entropy")

dt_entropy = df.groupBy("inn_dt").agg(
    sf.collect_list("inn_kt").alias("all_interactions"),
    sf.countDistinct("inn_kt").alias("unique_count")
).withColumn("dt_entropy", calc_entropy_udf("all_interactions")
).withColumn("dt_normalized_entropy", sf.col("dt_entropy") / sf.log2(sf.col("unique_count"))
).select("inn_dt", "dt_entropy", "dt_normalized_entropy")

df = df.join(kt_entropy, on="inn_kt", how="left").join(dt_entropy, on="inn_dt", how="left").fillna(0)

# FEATURES
dt_features = ["inn_dt", "dt_okved", "dt_okato", "dt_bic_34", "dt_bic_56", "dt_bic_79", "dt_num_13", "dt_num_45", "dt_num_68",
               "dt_avg_sum", "dt_stddev_sum", "dt_min_sum", "dt_max_sum", "dt_median_sum", "dt_buyers_count",
               "dt_skewness_sum", "dt_entropy", "dt_normalized_entropy"]
kt_features = ["inn_kt", "kt_okved", "kt_okato", "kt_bic_34", "kt_bic_56", "kt_bic_79", "kt_num_13", "kt_num_45", "kt_num_68",
               "kt_avg_sum", "kt_stddev_sum", "kt_min_sum", "kt_max_sum", "kt_median_sum", "kt_buyers_count",
               "kt_skewness_sum", "kt_entropy", "kt_normalized_entropy"]

interactions = df
dtDF = interactions.select(*dt_features).distinct()
ktDF = interactions.select(*kt_features).distinct()
emb_kt = embeddings_array.withColumnRenamed("inn", "inn_kt")
emb_dt = embeddings_array.withColumnRenamed("inn", "inn_dt")
ktDF = ktDF.join(emb_kt, on="inn_kt", how="inner").withColumnRenamed("embedding", "kt_embedding")
dtDF = dtDF.join(emb_dt, on="inn_dt", how="inner").withColumnRenamed("embedding", "dt_embedding")

# POSITIVES
positives = interactions.select("inn_kt", "inn_dt", "short_dt").withColumn("label", sf.lit(1))
dts = dtDF.select("inn_dt").rdd.flatMap(lambda x: x).collect()
dt_bc = spark.sparkContext.broadcast(dts)

@sf.udf(returnType=StringType())
def sample_neg():
    return random.choice(dt_bc.value)

start_dt = datetime.strptime(train_date_start, "%Y-%m-%d")
end_dt = datetime.strptime(test_date_end, "%Y-%m-%d")

@sf.udf(returnType=StringType())
def random_date():
    return (start_dt + timedelta(days=random.randint(0, (end_dt - start_dt).days))).strftime("%Y-%m-%d")

neg_idx_df = spark.createDataFrame([(i,) for i in range(15)], ["neg_idx"])
negatives = ktDF.crossJoin(neg_idx_df) \
    .withColumn("inn_dt", sample_neg()) \
    .withColumn("label", sf.lit(0)) \
    .withColumn("short_dt", random_date()) \
    .select("inn_dt", "inn_kt", "label", "short_dt")

interactions_train = positives.unionByName(negatives) \
    .join(dtDF, on="inn_dt", how="left") \
    .join(ktDF, on="inn_kt", how="left") \
    .withColumn("short_dt", sf.to_date("short_dt", "yyyy-MM-dd"))

# INDEXING
kt_cat_columns = ["inn_kt", "kt_okved", "kt_okato", "kt_bic_34", "kt_bic_56", "kt_bic_79", "kt_num_13", "kt_num_45", "kt_num_68", "kt_buyers_count"]
dt_cat_columns = ["inn_dt", "dt_okved", "dt_okato", "dt_bic_34", "dt_bic_56", "dt_bic_79", "dt_num_13", "dt_num_45", "dt_num_68", "dt_buyers_count"]
all_cat_columns = kt_cat_columns + dt_cat_columns

indexer = StringIndexer(
    inputCols=all_cat_columns,
    outputCols=[f"{col}_index" for col in all_cat_columns],
    handleInvalid="keep"
)
model = indexer.fit(interactions_train)
interactions_train_indexed = model.transform(interactions_train)

# TEST DATA
interactions_test = filter_inn_4recs_extended(spark, start_date=test_date_start, end_date=test_date_end) \
    .select("inn_dt", "inn_kt", "short_dt").withColumn("label", sf.lit(1)) \
    .join(ktDF, on="inn_kt", how="inner") \
    .join(dtDF, on="inn_dt", how="inner")
interactions_test_indexed = model.transform(interactions_test)
interactions_all_indexed = interactions_train_indexed.unionByName(interactions_test_indexed)

# TRAIN / TEST SPLIT
cols = [f"{col}_index" for col in all_cat_columns] + [
    "short_dt", "kt_avg_sum", "kt_stddev_sum", "kt_min_sum", "kt_max_sum", "kt_median_sum",
    "kt_skewness_sum", "kt_entropy", "kt_normalized_entropy",
    "dt_avg_sum", "dt_stddev_sum", "dt_min_sum", "dt_max_sum", "dt_median_sum",
    "dt_skewness_sum", "dt_entropy", "dt_normalized_entropy",
    "kt_embedding", "dt_embedding", "label"
]
train_df = interactions_all_indexed.select(cols).filter(sf.col("short_dt") <= train_date_end)
test_df = interactions_all_indexed.select(cols).filter(sf.col("short_dt") >= test_date_start)

# SAVE
train_df.coalesce(60).write.mode("overwrite").saveAsTable(f"{save_schema}.deepfm_train")
test_df.coalesce(60).write.mode("overwrite").saveAsTable(f"{save_schema}.deepfm_test")

# SAVE MAPPINGS
mappings = {col: model.labelsArray[i] for i, col in enumerate(all_cat_columns)}
with open("mappings/mappings.pkl", "wb") as f:
    pickle.dump(mappings, f)

# SAVE VALUE COUNTS FOR EMBEDDING LAYER INITIALIZATION
unique_values_count = {}
for i, (key, value) in enumerate(mappings.items()):
    unique_values_count[all_cat_columns[i]] = len(value)

with open("mappings/unique_values_count.pkl", "wb") as f:
    pickle.dump(unique_values_count, f)