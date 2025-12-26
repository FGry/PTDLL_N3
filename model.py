import os
import sys

os.environ["JAVA_HOME"] = r"D:\Java\jdk-11"
os.environ["SPARK_HOME"] = r"D:\apache_spark\spark-3.5.7-bin-hadoop3"

sys.path.append(os.path.join(os.environ["SPARK_HOME"], "python"))
sys.path.append(os.path.join(os.environ["SPARK_HOME"], "python", "lib", "py4j-0.10.9.7-src.zip"))

# from pyspark.ml.feature import VectorAssembler
# from pyspark.ml.regression import LinearRegression
# from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
# from pyspark.ml.evaluation import RegressionEvaluator
#
#
# def train_linear_regression(df):
#     assembler = VectorAssembler(
#         inputCols=['trip_distance'],
#         outputCol="features"
#     )
#     df_final = assembler.transform(df).select("features", "fare_amount")
#     train_data, test_data = df_final.randomSplit([0.8, 0.2], seed=42)
#     lr = LinearRegression(featuresCol="features", labelCol="fare_amount")
#     paramGrid = ParamGridBuilder() \
#         .addGrid(lr.regParam, [0.01, 0.1]) \
#         .addGrid(lr.elasticNetParam, [0.0, 0.5]) \
#         .addGrid(lr.maxIter, [100]) \
#         .build()
#     evaluator_rmse = RegressionEvaluator(labelCol="fare_amount", predictionCol="prediction", metricName="rmse")
#     cross_val = CrossValidator(estimator=lr,
#                                estimatorParamMaps=paramGrid,
#                                evaluator=evaluator_rmse,
#                                numFolds=3)
#     cv_model = cross_val.fit(train_data)
#     best_model = cv_model.bestModel
#     predictions = best_model.transform(test_data)
#     rmse = evaluator_rmse.evaluate(predictions)
#     evaluator_r2 = RegressionEvaluator(labelCol="fare_amount", predictionCol="prediction", metricName="r2")
#     r2 = evaluator_r2.evaluate(predictions)
#     sample_preds = predictions.select("fare_amount", "prediction").limit(20).toPandas()
#     return rmse, r2, sample_preds

from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator

import data_processing


def train_linear_regression(df):
    print("Đang lọc RatecodeID == 1...")
    data_select = df.filter(col("RatecodeID") == 1)
    print("Đang loại bỏ ngoại lai (Outliers)...")
    data_clean = data_processing.remove_outliers(data_select, ["fare_amount", "trip_distance"])
    assembler = VectorAssembler(
        inputCols=['trip_distance'],
        outputCol="features"
    )
    df_final = assembler.transform(data_clean).select("features", "fare_amount")
    train_data, test_data = df_final.randomSplit([0.8, 0.2], seed=42)

    lr = LinearRegression(featuresCol="features", labelCol="fare_amount")
    paramGrid = ParamGridBuilder() \
        .addGrid(lr.regParam, [0.001, 0.01, 0.1]) \
        .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
        .addGrid(lr.maxIter, [100, 200]) \
        .build()

    evaluator_rmse = RegressionEvaluator(labelCol="fare_amount", predictionCol="prediction", metricName="rmse")

    print("Đang huấn luyện CrossValidator (có thể mất vài phút)...")
    cross_val = CrossValidator(estimator=lr,
                               estimatorParamMaps=paramGrid,
                               evaluator=evaluator_rmse,
                               numFolds=3)  # BTL dùng 3 folds

    cv_model = cross_val.fit(train_data)
    best_model = cv_model.bestModel

    # In ra hệ số để kiểm tra (trong console)
    print(f"Hệ số (Coefficients): {best_model.coefficients}")
    print(f"Sai số (Intercept): {best_model.intercept}")

    predictions = best_model.transform(test_data)

    # Đánh giá
    rmse = evaluator_rmse.evaluate(predictions)

    evaluator_r2 = RegressionEvaluator(labelCol="fare_amount", predictionCol="prediction", metricName="r2")
    r2 = evaluator_r2.evaluate(predictions)

    print(f"Kết quả: RMSE={rmse}, R2={r2}")

    sample_preds = predictions.select("fare_amount", "prediction").limit(20).toPandas()

    return rmse, r2, sample_preds