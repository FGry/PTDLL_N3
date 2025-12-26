import os
import sys


os.environ["JAVA_HOME"] = r"D:\Java\jdk-11"
os.environ["SPARK_HOME"] = r"D:\apache_spark\spark-3.5.7-bin-hadoop3"

sys.path.append(os.path.join(os.environ["SPARK_HOME"], "python"))
sys.path.append(os.path.join(os.environ["SPARK_HOME"], "python", "lib", "py4j-0.10.9.7-src.zip"))


# Đảm bảo các đường dẫn thư viện Spark giống như trong config.py
import config
import data_processing
import model
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def run_evaluation():
    # 1. Khởi tạo Spark Session
    print("--- Đang khởi tạo Spark Session ---")
    spark = config.create_spark_session("Taxi_Evaluation_Only")

    # 2. Kiểm tra và tải dữ liệu
    # Thay đổi tên file nếu file của bạn khác
    file_path = "yellow_tripdata_2022-01.parquet"

    if not os.path.exists(file_path):
        print(f"Lỗi: Không tìm thấy file '{file_path}'")
        return

    print(f"--- Đang tải dữ liệu từ {file_path} ---")
    df_raw = data_processing.load_data(spark, file_path)

    # 3. Làm sạch dữ liệu
    print("--- Đang làm sạch dữ liệu ---")
    df_cleaned = data_processing.clean_and_process_data(df_raw)

    # 4. Huấn luyện và lấy kết quả (Hàm này đã có sẵn trong model.py của bạn)
    print("--- Bắt đầu huấn luyện và tính toán RMSE, R2 ---")
    # Hàm model.train_linear_regression trả về 3 giá trị: rmse, r2, sample_preds
    rmse, r2, sample_preds = model.train_linear_regression(df_cleaned)

    print("\n" + "=" * 40)
    print("       KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH")
    print("=" * 40)

    # Hiển thị RMSE
    print(f"\n1. RMSE (Root Mean Squared Error): {rmse:.4f}")
    print(f"   -> Ý nghĩa: Sai số trung bình của mô hình là khoảng ${rmse:.2f}.")
    print("   -> (Càng thấp càng tốt)")

    # Hiển thị R2
    print(f"\n2. R2 Score (Hệ số xác định):      {r2:.4f}")
    print(f"   -> Ý nghĩa: Mô hình giải thích được {r2 * 100:.2f}% sự biến động của giá cước.")
    print("   -> (Càng gần 1 càng tốt)")

    # Đánh giá tổng quan bằng lời
    print("\n----------------------------------------")
    print("NHẬN XÉT TỰ ĐỘNG:")
    if r2 >= 0.7:
        print(" Mô hình TỐT. Độ chính xác cao.")
    elif r2 >= 0.5:
        print("  Mô hình CHẤP NHẬN ĐƯỢC. Có thể cải thiện thêm.")
    else:
        print(" Mô hình CHƯA TỐT. Cần xem lại dữ liệu hoặc tham số.")
    print("----------------------------------------")

    # Hiển thị một vài ví dụ thực tế để đối chiếu
    print("\nVí dụ 5 kết quả dự báo:")
    print(sample_preds.head(5).to_string(index=False))
    print("=" * 40)

    # Dừng Spark
    spark.stop()


def plot_model_evaluation(sample_preds):
    y_true = sample_preds["fare_amount"]
    y_pred = sample_preds["prediction"]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.6, color='#2980b9', edgecolor='w', label='Dữ liệu dự báo')
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Đường lý tưởng')
    plt.title('Độ khớp: Thực tế vs Dự đoán')
    plt.xlabel('Giá thực tế ($)')
    plt.ylabel('Giá dự đoán ($)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.subplot(1, 2, 2)
    residuals = y_true - y_pred
    sns.histplot(residuals, kde=True, color='#8e44ad', bins=20, edgecolor='black')
    plt.axvline(0, color='red', linestyle='--', lw=2, label="Mức 0 (Không sai số)")
    plt.title('Phân phối sai số (Residuals)')
    plt.xlabel('Độ lệch (Thực tế - Dự đoán)')
    plt.ylabel('Tần suất')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_evaluation()