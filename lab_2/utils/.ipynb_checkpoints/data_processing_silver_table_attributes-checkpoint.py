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
    df = df.dropna(subset=["snapshot_date", "Monthly_Inhand_Salary"])

    exclude = {"Customer_ID", "snapshot_date"}
    cols_to_clean = ["Annual_Income", "Monthly_Inhand_Salary", "Interest_Rate", "Num_of_Loan", "Delay_from_due_date", "Num_of_Delayed_Payment", "Amount_invested_monthly", "Outstanding_Debt","Monthly_Balance"]

    for c in df.columns:
        if c in cols_to_clean:
            df = df.withColumn(c, when(col(c).rlike(r"^[0-9]+(?:\.[0-9]+)?_?$"), regexp_extract(col(c), r"^([0-9]+(?:\.[0-9]+)?)",1).cast("float")).otherwise(None))
            df.filter(col(c).rlike(r".*[^0-9\.\-].*")).count()
    # Validate after all columns are cleaned
    for c in cols_to_clean:
        invalid_count = df.filter(col(c).rlike(r".*[^0-9\.\-].*")).count()
        print(f"Column {c}: {invalid_count} invalid values")
    
    # Assemble the numeric feature into a vector
    assembler = VectorAssembler(inputCols=["Monthly_Inhand_Salary"], outputCol="monthly_inhand_salary_vec")
    df = assembler.transform(df)

    # Apply standard scaling
    scaler = StandardScaler(inputCol="monthly_inhand_salary_vec", outputCol="monthly_inhand_salary_vec_scaled", withMean=True, withStd=True)
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