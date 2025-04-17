# -*- coding: utf-8 -*-
"""
spark-submit \
  --master yarn \
  --conf spark.sql.adaptive.enabled=true \
  --conf spark.sql.shuffle.partitions=800 \
  --conf spark.executor.memoryOverhead=3g \
  ml_pipeline.py
"""
import os, sys, yaml, logging, pickle, glob, uuid
from pathlib import Path
from pyspark.sql import SparkSession, functions as F, Window
from pyspark.ml.feature import StringIndexer
from pyspark import StorageLevel

###### 1. utility ################################################################
def save_dict_partition(partition, base_path: str, key_col: str):
    import pickle, os, uuid
    os.makedirs(base_path, exist_ok=True)
    uid = uuid.uuid4().hex
    part_dict = {row[key_col]: row.asDict() for row in partition}
    with open(f"{base_path}/part_{uid}.pkl", "wb") as f:
        pickle.dump(part_dict, f)

def load_combined_dict(dir_path: str):
    combined = {}
    for fname in glob.glob(os.path.join(dir_path, "*.pkl")):
        with open(fname, "rb") as f:
            combined.update(pickle.load(f))
    return combined

###### 2. config & spark #########################################################
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

data_dir = Path(cfg["outputs"]["data_dir"])
data_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    filename=data_dir / cfg["outputs"]["log_file"],
    format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger(__name__)

spark = (
    SparkSession.builder.appName(cfg["spark"]["appName"])
    .enableHiveSupport()
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

###### 3. source tables & helpers #################################################
train_start, train_end = cfg["dates"]["train_date_start"], cfg["dates"]["train_date_end"]
test_start,  test_end  = cfg["dates"]["test_date_start"],  cfg["dates"]["test_date_end"]

@F.udf("array<double>")
def to_array(*cols):
    return list(cols)

emb_cols = [f"embed_{i}" for i in range(256)]
embeddings = (
    spark.read.parquet(cfg["paths"]["embeddings"])
    .withColumn("embedding", F.array(*emb_cols))
    .select("inn", "embedding")
).persist(StorageLevel.MEMORY_AND_DISK)

def filter_inn(sqlCtx, start_date, end_date, report_dt="2024-09-30", active_flag=True):
    base = (
        sqlCtx.table("arnsdpsbx_t_team_apm.tmb_basis_client")
        .filter(
            (F.col("report_dt") == report_dt)
            & (F.col("org_segment_sber").isin("Микро", "Малые"))
            & (F.col("type").isin("ИП", "ЮЛ"))
        )
    )
    if active_flag:
        base = base.filter(F.col("active_flg") == 0)

    bas_col = sqlCtx.table(
        "prx_baza_custom_cib_products_custom_cib_products.basis_transactions_coloured"
    ).filter(
        (F.col("short_dt").between(start_date, end_date))
        & (F.col("predicted_value_last") == "оплата по договору")
    )

    f = (
        bas_col.join(
            base.selectExpr(
                "inn            as inn_kt",
                "okved_cd       as okved_cd_kt",
                "okato_cd       as okato_cd_kt",
                "org_segment_msp",
                "org_segment_sber"
            ),
            "inn_kt",
            "inner",
        )
        .select(
            "inn_dt", "inn_kt", "c_sum", "short_dt",
            "okved_cd_kt", "okato_cd_kt", "c_bic_kt", "c_bic_dt", "c_num_kt", "c_num_dt",
        )
        # fast regexp‑substring вместо восьми вызовов
        .withColumn("bic_kt_parts", F.regexp_extract("c_bic_kt", r"^..(..)(..)(..)", 0))
        .withColumn("bic_dt_parts", F.regexp_extract("c_bic_dt", r"^..(..)(..)(..)", 0))
        .withColumn("num_kt_parts", F.regexp_extract("c_num_kt", r"^(...)(..)(...)", 0))
        .withColumn("num_dt_parts", F.regexp_extract("c_num_dt", r"^(...)(..)(...)", 0))
    )

    f = f.join(
        base.selectExpr(
            "inn            as inn_dt",
            "okved_cd       as okved_cd_dt",
            "okato_cd       as okato_cd_dt"
        ),
        "inn_dt",
        "inner"
    )

    return (
        f.groupBy(
            "inn_dt", "inn_kt", "short_dt",
            "okved_cd_kt", "okato_cd_kt",
            "okved_cd_dt", "okato_cd_dt",
            "bic_kt_parts", "bic_dt_parts",
            "num_kt_parts", "num_dt_parts"
        )
        .agg(F.sum("c_sum").alias("c_sum_fin"))
    )

###### 4. train features #########################################################
df = filter_inn(spark, train_start, train_end).persist(StorageLevel.MEMORY_AND_DISK)

# agg & skew (без окон на каждый row, снова агрегации)
kt_stats = (
    df.groupBy("inn_kt")
    .agg(
        F.avg("c_sum_fin").alias("kt_avg_sum"),
        F.stddev("c_sum_fin").alias("kt_stddev_sum"),
        F.min("c_sum_fin").alias("kt_min_sum"),
        F.max("c_sum_fin").alias("kt_max_sum"),
        F.expr("percentile_approx(c_sum_fin, 0.5)").alias("kt_median_sum"),
        F.countDistinct("inn_dt").alias("kt_buyers_count"),
        F.skewness("c_sum_fin").alias("kt_skewness_sum"),
    )
)

dt_stats = (
    df.groupBy("inn_dt")
    .agg(
        F.avg("c_sum_fin").alias("dt_avg_sum"),
        F.stddev("c_sum_fin").alias("dt_stddev_sum"),
        F.min("c_sum_fin").alias("dt_min_sum"),
        F.max("c_sum_fin").alias("dt_max_sum"),
        F.expr("percentile_approx(c_sum_fin, 0.5)").alias("dt_median_sum"),
        F.countDistinct("inn_kt").alias("dt_buyers_count"),
        F.skewness("c_sum_fin").alias("dt_skewness_sum"),
    )
)

df = (
    df.join(kt_stats, "inn_kt", "left")
      .join(dt_stats, "inn_dt", "left")
)

###### 5. feature stores #########################################################
interactions_train = df.select("inn_kt", "inn_dt", F.to_date("short_dt").alias("short_dt")).withColumn("label", F.lit(1))

kt_cols = [c for c in kt_stats.columns if c != "inn_kt"]
dt_cols = [c for c in dt_stats.columns if c != "inn_dt"]

kt_features = kt_stats
dt_features = dt_stats

###### 6. categorical indexing ###################################################
def index_features(df_, cat_cols, prefix):
    indexer = StringIndexer(
        inputCols=cat_cols,
        outputCols=[f"{c}_idx" for c in cat_cols],
        handleInvalid="keep"
    )
    model = indexer.fit(df_)
    model.write().overwrite().save(f"{data_dir}/{prefix}_indexer")      # <-- современное сохранение
    return model.transform(df_), model.labelsArray

kt_cat = ["inn_kt"]                       # остальные «категории» уже числовые
dt_cat = ["inn_dt"]

kt_features_idx, kt_labels = index_features(kt_features, kt_cat, "kt")
dt_features_idx, dt_labels = index_features(dt_features, dt_cat, "dt")

###### 7. distributed dict storage ###############################################
# inn_dt_index ➜ features
(
    dt_features_idx.select("*")            # все колонки нужны
    .foreachPartition(lambda part: save_dict_partition(part, f"{data_dir}/dt_features_dict_parts", "inn_dt_idx"))
)

(
    kt_features_idx.select("*")
    .foreachPartition(lambda part: save_dict_partition(part, f"{data_dir}/kt_features_dict_parts", "inn_kt_idx"))
)

# Embeddings ─ broadcast сначала, потом foreachPartition
dt_emb = (
    dt_features_idx.join(embeddings.withColumnRenamed("inn", "inn_dt"), "inn_dt")
    .select("inn_dt_idx", "embedding")
)
kt_emb = (
    kt_features_idx.join(embeddings.withColumnRenamed("inn", "inn_kt"), "inn_kt")
    .select("inn_kt_idx", "embedding")
)

dt_emb.foreachPartition(lambda p: save_dict_partition(p, f"{data_dir}/dt_embeddings_dict_parts", "inn_dt_idx"))
kt_emb.foreachPartition(lambda p: save_dict_partition(p, f"{data_dir}/kt_embeddings_dict_parts", "inn_kt_idx"))

###### 8. test dictionary (>=15 tx) #############################################
test_df = (
    filter_inn(spark, test_start, test_end)
    .select("inn_kt", "inn_dt")
    .join(dt_features_idx.select("inn_dt", "inn_dt_idx"), "inn_dt")
    .join(kt_features_idx.select("inn_kt", "inn_kt_idx"), "inn_kt")
)

test_dict_df = (
    test_df.groupBy("inn_dt_idx")
           .agg(F.collect_set("inn_kt_idx").alias("kt_set"), F.count("*").alias("cnt"))
           .filter("cnt >= 15")
           .select("inn_dt_idx", "kt_set")
)

test_dict_df.foreachPartition(lambda p: save_dict_partition(p, f"{data_dir}/test_dict_parts", "inn_dt_idx"))

###### 9. unique indices for negative sampling ###################################
dt_unique_df = dt_features_idx.select("inn_dt_idx")
kt_unique_df = kt_features_idx.select("inn_kt_idx")

dt_unique_df.foreachPartition(lambda p: save_dict_partition(p, f"{data_dir}/unique_inn_dt_indices_parts", "inn_dt_idx"))
kt_unique_df.foreachPartition(lambda p: save_dict_partition(p, f"{data_dir}/unique_inn_kt_indices_parts", "inn_kt_idx"))

###### 10. interaction store → Hive ##############################################
save_schema = cfg["outputs"]["save_schema"]
(
    interactions_train
    .join(dt_features_idx.select("inn_dt", "inn_dt_idx"), "inn_dt")
    .join(kt_features_idx.select("inn_kt", "inn_kt_idx"), "inn_kt")
    .repartition(800)
    .write.mode("overwrite").saveAsTable(f"{save_schema}.{cfg['outputs']['train_name']}")
)

###### 11. offline dict assembly (example local script) ###########################
# ─────────────────────────  run locally after spark job ─────────────────────────
"""
from utils import load_combined_dict
root = Path("data_18")

dt_features_dict    = load_combined_dict(root / "dt_features_dict_parts")
kt_features_dict    = load_combined_dict(root / "kt_features_dict_parts")
dt_embeddings_dict  = load_combined_dict(root / "dt_embeddings_dict_parts")
kt_embeddings_dict  = load_combined_dict(root / "kt_embeddings_dict_parts")
test_dict           = load_combined_dict(root / "test_dict_parts")
unique_inn_dt_idx   = list(load_combined_dict(root / "unique_inn_dt_indices_parts").keys())
unique_inn_kt_idx   = list(load_combined_dict(root / "unique_inn_kt_indices_parts").keys())
"""
log.info("Pipeline finished ok.")
spark.stop()
