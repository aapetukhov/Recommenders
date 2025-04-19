import yaml
import pyspark.sql.functions as F

def read_cfg(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def filter_inn(spark, start_date, end_date, report_dt="2024-09-30", active_flag=True):
    sf = F
    base = (spark.table("arnsdpsbx_t_team_apm.tmb_basis_client")
            .filter((sf.col("report_dt") == report_dt) &
                    (sf.col("org_segment_sber").isin("Микро", "Малые")) &
                    (sf.col("type").isin("ИП", "ЮЛ")))
           )
    if active_flag:
        base = base.filter(sf.col("active_flg") == 0)

    bas_col = spark.table("prx_baza_custom_cib_products_custom_cib_products.basis_transactions_coloured")

    f = (bas_col
         .filter((sf.col("short_dt") >= start_date) & (sf.col("short_dt") <= end_date))
         .filter(sf.col("predicted_value_last") == "оплата по договору")
         .join(base.select("inn", "okved_cd", "okato_cd", "org_segment_msp", "org_segment_sber")
                   .withColumnRenamed("inn", "inn_kt")
                   .withColumnRenamed("okved_cd", "okved_cd_kt")
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

    f = f.join(base.select("inn", "okved_cd", "okato_cd")
                 .withColumnRenamed("inn", "inn_dt")
                 .withColumnRenamed("okved_cd", "okved_cd_dt")
                 .withColumnRenamed("okato_cd", "okato_cd_dt"),
               on="inn_dt", how="inner")

    return (f.groupby("inn_dt", "inn_kt", "short_dt",
                      "okved_cd_kt", "okato_cd_kt", "okved_cd_dt", "okato_cd_dt",
                      "bic_kt_34", "bic_kt_56", "bic_kt_79",
                      "bic_dt_34", "bic_dt_56", "bic_dt_79",
                      "num_kt_13", "num_kt_45", "num_kt_68",
                      "num_dt_13", "num_dt_45", "num_dt_68")
              .agg(sf.sum("c_sum").alias("c_sum_fin")))
