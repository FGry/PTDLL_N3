import os
import sys

os.environ["JAVA_HOME"] = r"D:\Java\jdk-11"
os.environ["SPARK_HOME"] = r"D:\apache_spark\spark-3.5.7-bin-hadoop3"

sys.path.append(os.path.join(os.environ["SPARK_HOME"], "python"))
sys.path.append(os.path.join(os.environ["SPARK_HOME"], "python", "lib", "py4j-0.10.9.7-src.zip"))

import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql.functions import col, corr, count, avg


def plot_fare_distribution(df):
    fare_sample = df.select("fare_amount").sample(False, 0.1, seed=36).toPandas()
    plt.figure(figsize=(6, 4))
    plt.hist(fare_sample['fare_amount'], bins=50, color='coral', edgecolor='black', alpha=0.7)
    plt.xlabel('Fare Amount ($)')
    plt.ylabel('Tần suất')
    plt.title('Phân phối giá cước (10%)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()


def plot_fare_range_0_100(df):
    fare_sample_filtered = df.select("fare_amount") \
        .sample(False, 0.1, seed=36) \
        .filter(col("fare_amount").between(0, 100)) \
        .toPandas()
    median_val = fare_sample_filtered['fare_amount'].median()
    mode_val = fare_sample_filtered['fare_amount'].mode()[0]
    mean_val = fare_sample_filtered['fare_amount'].mean()
    plt.figure(figsize=(6, 4))
    plt.hist(fare_sample_filtered['fare_amount'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(mode_val, color='#e67e22', linestyle='-.', linewidth=2, label=f'Mode: {mode_val:.2f}')
    plt.axvline(median_val, color='#e74c3c', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
    plt.axvline(mean_val, color='#2ecc71', linestyle='-', linewidth=2, label=f'Mean: {mean_val:.2f}')
    plt.legend()
    plt.title('Phân phối giá cước (0-100$)')
    plt.tight_layout()
    return plt.gcf()


def plot_hourly_demand(df):
    hourly_stats = df.groupBy("pickup_hour") \
        .agg(count("*").alias("trip_count"), avg("fare_amount").alias("avg_fare")) \
        .orderBy("pickup_hour").toPandas()
    plt.figure(figsize=(8, 4))
    plt.bar(hourly_stats['pickup_hour'], hourly_stats['trip_count'], color='lightblue', label='Số lượng', alpha=0.7)
    plt.xlabel('Giờ (0-23h)')
    plt.ylabel('Số lượng chuyến đi')
    plt.twinx()
    plt.plot(hourly_stats['pickup_hour'], hourly_stats['avg_fare'], color='red', marker='o', linewidth=2,
             label='Giá TB')
    plt.ylabel('Giá cước trung bình ($)')
    plt.title('Biến động nhu cầu và giá cước theo giờ')
    plt.tight_layout()
    return plt.gcf()


def plot_distance_fare_relation(df):
    df_filtered = df.filter(
        (col("fare_amount") > 0) & (col("fare_amount") <= 100) &
        (col("trip_distance") > 0) & (col("trip_distance") < 50)
    )
    try:
        r_value = df_filtered.select(corr("trip_distance", "fare_amount")).collect()[0][0]
    except:
        r_value = 0
    sample_data = df_filtered.select("trip_distance", "fare_amount").sample(False, 0.1, seed=42).toPandas()
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=sample_data, x="trip_distance", y="fare_amount", alpha=0.1, color='blue', label='Dữ liệu')
    sns.regplot(data=sample_data, x="trip_distance", y="fare_amount", scatter=False, color='red',
                label=f'Hồi quy (r={r_value:.2f})')
    plt.title("Tương quan Quãng đường vs Giá cước")
    plt.xlabel("Quãng đường (miles)")
    plt.ylabel("Giá cước ($)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    return plt.gcf()