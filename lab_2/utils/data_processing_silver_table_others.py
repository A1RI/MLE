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

from pyspark.ml.feature import VectorAssembler, StandardScaler


def process_silver_table(snapshot_date_str, bronze_directory, feature, silver_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_feature_" + feature + "_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: enforce schema / data type
    # Drop exact duplicate rows
    df = df.dropDuplicates()

    # Remove rows missing critical fields
    df = df.dropna(subset=["snapshot_date", "Annual_Income"])
    
    # Assemble the numeric feature into a vector
    assembler = VectorAssembler(inputCols=["annual_income"], outputCol="annual_income_vec")
    df = assembler.transform(df)

    # Apply standard scaling
    scaler = StandardScaler(inputCol="annual_income_vec", outputCol="annual_income_scaled", withMean=True, withStd=True)
    scaler_model = scaler.fit(df)
    df = scaler_model.transform(df)

    # save silver table - IRL connect to database to write
    partition_name = feature + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df