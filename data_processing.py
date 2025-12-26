from pyspark.sql import functions as f
from pyspark.sql.functions import col, unix_timestamp, hour, dayofweek, mean, desc
from pyspark.sql.types import IntegerType


def load_data(spark, file_path):
    df = spark.read.parquet(file_path)
    return df


def clean_and_process_data(df):
    data = df.withColumn("duration_minutes",
                         (unix_timestamp(col("tpep_dropoff_datetime")) - unix_timestamp(
                             col("tpep_pickup_datetime"))) / 60
                         )
    data = data.withColumn("pickup_hour", hour(col("tpep_pickup_datetime"))) \
        .withColumn("pickup_day_of_week", dayofweek(col("tpep_pickup_datetime")))
    data = data.withColumn("passenger_count", df["passenger_count"].cast(IntegerType()))
    data = data.filter((col("fare_amount") > 0) & (col("trip_distance") > 0))
    cols_drop = ["VendorID", "store_and_fwd_flag"]
    data_processed = data.drop(*cols_drop)
    cols_dropna = [
        'tpep_pickup_datetime', 'tpep_dropoff_datetime', 'PULocationID',
        'DOLocationID', 'fare_amount', 'trip_distance',
        'duration_minutes', 'pickup_hour', 'pickup_day_of_week'
    ]
    data_processed = data_processed.dropna(subset=cols_dropna)
    col_fillna_mode = ['RatecodeID', 'payment_type', 'passenger_count']
    mode_dict = {}
    for c in col_fillna_mode:
        mode_value = df.groupBy(c).count().orderBy(desc("count")).first()[0]
        mode_dict[c] = mode_value
    data_processed = data_processed.fillna(mode_dict)
    exclude_columns = cols_dropna + col_fillna_mode
    col_fillna_mean = [col_name for col_name in data_processed.columns
                       if col_name not in exclude_columns and col_name not in ['tpep_pickup_datetime',
                                                                               'tpep_dropoff_datetime']]
    if col_fillna_mean:
        mean_val = df.select([mean(c).alias(c) for c in col_fillna_mean]).collect()[0].asDict()
        data_processed = data_processed.fillna(mean_val)
    return data_processed


def remove_outliers(df, columns):
    for column in columns:
        q1, q3 = df.approxQuantile(column, [0.25, 0.75], 0.01)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df = df.filter((col(column) >= lower) & (col(column) <= upper))
    return df