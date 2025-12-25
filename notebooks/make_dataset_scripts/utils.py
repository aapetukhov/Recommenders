# -*- coding: utf-8 -*-

import os
import sys
import ast
import yaml
import json
import logging
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

os.environ['SPARK_MAJOR_VERSION'] = '3'
os.environ['SPARK_HOME'] = '/usr/sdp/current/spark3-client/'
os.environ['PYSPARK_DRIVER_PYTHON'] = 'python'
os.environ['LD_LIBRARY_PATH'] = '/opt/python/virtualenv/jupyter/lib'
os.environ['PYSPARK_PYTHON'] = '/data/sdp/mlpy3811v23/bin/python' #/opt/cloudera/parcels/PYENV.AUTOML/bin/python
 
sys.path.insert(0, '/usr/sdp/current/spark3-client/python/')
sys.path.insert(0, '/usr/sdp/current/spark3-client/python/lib/py4j_current')

import re
import pickle
import gzip
import calendar
import datetime
from tqdm import tqdm
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.options.display.float_format = '{:.2f}'.format

import warnings
warnings.filterwarnings("ignore")

import pyspark.sql.functions as sf
from pyspark.sql.functions import udf
from pyspark.sql.window import Window as W
from pyspark import SparkContext, SparkConf, HiveContext
import pyspark.sql.types as st
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer

from lightfm import LightFM
from lightfm.evaluation import auc_score, precision_at_k
from scipy.sparse import save_npz, load_npz
from scipy.sparse import coo_matrix, csr_matrix

import pandas as pd
import numpy as np
from tqdm import tqdm
from lightfm.evaluation import auc_score, precision_at_k
from pyspark.sql.window import Window as W
import pyspark.sql.functions as sf
import os

import pandas as pd
import numpy as np
from tqdm import tqdm
from lightfm.evaluation import auc_score, precision_at_k
from pyspark.sql.window import Window as W
import pyspark.sql.functions as sf
from pyspark.sql.types import DoubleType
import os
from collections import Counter
from math import log2

from pyspark.sql.window import Window
from pyspark.ml.feature import Bucketizer, QuantileDiscretizer

import os, sys, yaml, logging, pickle, glob, uuid
from pathlib import Path
from pyspark.sql import SparkSession, functions as F, Window
from pyspark.ml.feature import StringIndexer
from pyspark import StorageLevel






# @F.udf("array<double>")
# def to_array(*cols):
#     return list(cols)


def filter_inn(sqlCtx, start_date, end_date, report_dt="2024-09-30", active_flag=True):
    base = sqlCtx.table("arnsdpsbx_t_team_apm.tmb_basis_client")\
        .filter((sf.col("report_dt") == report_dt) &
                (sf.col("org_segment_sber").isin("Микро", "Малые")) &
                (sf.col("type").isin("ИП", "ЮЛ")))
    if active_flag:
        base = base.filter(sf.col("active_flg") == 0)
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

def read_cfg():
    with open("../config.yaml") as f:
        cfg = yaml.safe_load(f)
    return cfg
        