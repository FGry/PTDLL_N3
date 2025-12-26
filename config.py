import os
import sys

os.environ["JAVA_HOME"] = r"D:\Java\jdk-11"
os.environ["SPARK_HOME"] = r"D:\apache_spark\spark-3.5.7-bin-hadoop3"

sys.path.append(os.path.join(os.environ["SPARK_HOME"], "python"))
sys.path.append(os.path.join(os.environ["SPARK_HOME"], "python", "lib", "py4j-0.10.9.7-src.zip"))

from pyspark.sql import SparkSession

def create_spark_session(app_name="Spark_Taxi_Analysis"):
    spark = SparkSession.builder \
        .appName(app_name) \
        .master("local[*]") \
        .getOrCreate()
    return spark