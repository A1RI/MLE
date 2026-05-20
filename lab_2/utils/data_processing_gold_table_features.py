import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import argparse

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

from pyspark.sql.functions import regexp_extract, when
from pyspark.ml.feature import VectorAssembler, StandardScaler 
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder


def process_gold_table(snapshot_date_str, silver_directory, gold_feature_store_directory, spark):
    
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    
    # connect to silver table
    partition_name = "attributes" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    # One-hot encode Occupation data
    indexer = StringIndexer(inputCol="Occupation", outputCol="occupation_index", handleInvalid="keep")

    encoder = OneHotEncoder(inputCols=["occupation_index"], outputCols=["occupation_ohe"], dropLast=False)

    pipeline = Pipeline(stages=[indexer, encoder])
    df_encoded = pipeline.fit(df).transform(df)
    df_encoded.select("occupation", "occupation_index", "occupation_ohe").show(20, truncate=False)

    # Calculate age_squared feature
    df_encoded = df_encoded.withColumn("age_squared", col("age") ** 2)

     # save gold table - IRL connect to database to write
    partition_name = "gold_feature_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_feature_store_directory + partition_name
    df_encoded.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df