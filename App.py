import os
import sys

os.environ["JAVA_HOME"] = r"D:\Java\jdk-11"
os.environ["SPARK_HOME"] = r"D:\apache_spark\spark-3.5.7-bin-hadoop3"

sys.path.append(os.path.join(os.environ["SPARK_HOME"], "python"))
sys.path.append(os.path.join(os.environ["SPARK_HOME"], "python", "lib", "py4j-0.10.9.7-src.zip"))

import customtkinter as ctk
from tkinter import filedialog, messagebox, ttk
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import os

import config
import data_processing
import visualization
import model

ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")


class TaxiAnalysisApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.is_running = True
        self.after_id = None

        self.title("Phân tích & Dự báo Giá cước Taxi (Spark)")
        self.geometry("1200x750")

        self.setup_table_style()

        self.spark = config.create_spark_session()
        self.df_raw = None
        self.df_cleaned = None

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.sidebar_frame = ctk.CTkFrame(self, width=220, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(7, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Taxi Analytics",
                                       font=ctk.CTkFont(size=22, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))


        self.btn_info = ctk.CTkButton(self.sidebar_frame, text="1. Thông tin Dữ liệu", command=self.show_info_event)
        self.btn_info.grid(row=1, column=0, padx=20, pady=10)

        self.btn_clean = ctk.CTkButton(self.sidebar_frame, text="2. Làm sạch Dữ liệu", command=self.clean_data_event)
        self.btn_clean.grid(row=2, column=0, padx=20, pady=10)

        self.btn_viz = ctk.CTkButton(self.sidebar_frame, text="3. Biểu đồ", command=self.visualization_event)
        self.btn_viz.grid(row=3, column=0, padx=20, pady=10)

        self.btn_train = ctk.CTkButton(self.sidebar_frame, text="4. Huấn luyện & Đánh giá",
                                       command=self.train_eval_event)
        self.btn_train.grid(row=4, column=0, padx=20, pady=10)

        self.main_frame = ctk.CTkFrame(self, fg_color="white")  # Nền chính màu trắng
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        self.show_welcome()

        default_file = "yellow_tripdata_2022-01.parquet"
        if os.path.exists(default_file):
            print(f"Phát hiện file mặc định: {default_file}. Đang tải...")
            self.after_id = self.after(100, lambda: self.load_data_from_path(default_file, silent=True))
        else:
            print(f"Không tìm thấy file mặc định: {default_file}")
            messagebox.showerror("Lỗi",
                                 f"Không tìm thấy file dữ liệu mặc định:\n{default_file}\nVui lòng copy file vào thư mục code.")

    def setup_table_style(self):
        style = ttk.Style()
        style.theme_use("default")

        style.configure("Treeview",
                        background="white",
                        foreground="black",
                        rowheight=25,
                        fieldbackground="white",
                        bordercolor="#d3d3d3",
                        borderwidth=1,
                        font=("Arial", 10))

        style.map('Treeview',
                  background=[('selected', '#0078d7')],
                  foreground=[('selected', 'white')])

        style.configure("Treeview.Heading",
                        background="#f0f0f0",
                        foreground="black",
                        relief="flat",
                        font=("Arial", 11, "bold"))

        style.map("Treeview.Heading",
                  background=[('active', '#e1e1e1')])

    def create_table(self, parent, dataframe):
        try:
            table_frame = ctk.CTkFrame(parent, fg_color="transparent")
            table_frame.pack(fill="both", expand=True, padx=5, pady=5)

            tree_scroll_y = ttk.Scrollbar(table_frame)
            tree_scroll_y.pack(side="right", fill="y")

            tree_scroll_x = ttk.Scrollbar(table_frame, orient="horizontal")
            tree_scroll_x.pack(side="bottom", fill="x")

            columns = list(dataframe.columns)
            tree = ttk.Treeview(table_frame, columns=columns, show="headings",
                                yscrollcommand=tree_scroll_y.set, xscrollcommand=tree_scroll_x.set)

            tree_scroll_y.config(command=tree.yview)
            tree_scroll_x.config(command=tree.xview)

            for col in columns:
                tree.heading(col, text=col)
                try:
                    max_width = max([len(str(x)) for x in dataframe[col].values] + [len(col)]) * 9
                    max_width = min(max_width, 300)
                    max_width = max(max_width, 80)
                except:
                    max_width = 100
                tree.column(col, width=max_width, anchor="center")

            for index, row in dataframe.iterrows():
                tree.insert("", "end", values=list(row))

            tree.pack(fill="both", expand=True)
            return tree
        except Exception:
            return None

    def clear_main_frame(self):
        try:
            for widget in self.main_frame.winfo_children():
                widget.destroy()
        except:
            pass

    def show_welcome(self):
        self.clear_main_frame()
        label = ctk.CTkLabel(self.main_frame, text="Đang khởi tạo...", font=ctk.CTkFont(size=24), text_color="black")
        label.place(relx=0.5, rely=0.5, anchor="center")

    def safe_update(self, widget):
        if self.is_running and self.winfo_exists():
            try:
                widget.update()
            except Exception:
                pass

    def load_data_from_path(self, file_path, silent=False):
        if not self.is_running: return
        try:
            self.df_raw = data_processing.load_data(self.spark, file_path)

            if not self.is_running: return
            self.clear_main_frame()
            label = ctk.CTkLabel(self.main_frame, text=f"Đã tải dữ liệu thành công:\n{os.path.basename(file_path)}",
                                 font=ctk.CTkFont(size=20), text_color="green")
            label.place(relx=0.5, rely=0.5, anchor="center")

            if not silent:
                messagebox.showinfo("Thành công", f"Đã tải dữ liệu: {os.path.basename(file_path)}")
        except Exception as e:
            if not silent:
                if self.is_running: messagebox.showerror("Lỗi", str(e))
            else:
                self.clear_main_frame()
                label = ctk.CTkLabel(self.main_frame, text=f"Lỗi tải file mặc định:\n{str(e)}",
                                     font=ctk.CTkFont(size=16), text_color="red")
                label.place(relx=0.5, rely=0.5, anchor="center")

    def show_info_event(self):
        if self.df_raw is None:
            messagebox.showwarning("Cảnh báo", "Chưa có dữ liệu nào được tải!")
            return

        self.clear_main_frame()

        top_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        top_frame.pack(fill="x", padx=10, pady=10)

        count = self.df_raw.count()
        ctk.CTkLabel(top_frame, text=f"Tổng số bản ghi: {count:,}", font=("Arial", 16, "bold"),
                     text_color="black").pack(anchor="w", padx=10, pady=5)

        schema_btn = ctk.CTkButton(top_frame, text="Xem chi tiết cấu trúc (Schema)",
                                   command=lambda: messagebox.showinfo("Schema",
                                                                       self.df_raw._jdf.schema().treeString()))
        schema_btn.pack(anchor="w", padx=10, pady=5)

        ctk.CTkLabel(self.main_frame, text="20 dòng đầu tiên (Dữ liệu gốc):", font=("Arial", 16, "bold"),
                     text_color="black").pack(anchor="w", padx=10, pady=(10, 0))

        df_head = self.df_raw.limit(20).toPandas()
        self.create_table(self.main_frame, df_head)

    def clean_data_event(self):
        if self.df_raw is None:
            messagebox.showwarning("Cảnh báo", "Chưa có dữ liệu nào được tải!")
            return

        try:
            self.df_cleaned = data_processing.clean_and_process_data(self.df_raw)
            self.df_cleaned.cache()

            self.clear_main_frame()

            ctk.CTkLabel(self.main_frame, text="Dữ liệu sau khi làm sạch (20 dòng đầu):",
                         font=("Arial", 16, "bold"), text_color="black").pack(anchor="w", padx=10, pady=10)

            df_head_clean = self.df_cleaned.limit(20).toPandas()
            self.create_table(self.main_frame, df_head_clean)

            try:
                save_path = "cleaned_data.csv"
                self.df_cleaned.limit(10000).toPandas().to_csv(save_path, index=False)
                messagebox.showinfo("Thành công", f"Đã làm sạch dữ liệu!\nĐã tự động lưu file: {save_path}")
            except Exception as e_save:
                messagebox.showerror("Lỗi Lưu file", f"Lỗi lưu file tự động: {e_save}")

        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi làm sạch dữ liệu: {str(e)}")

    def visualization_event(self):
        if self.df_cleaned is None:
            messagebox.showwarning("Cảnh báo", "Vui lòng làm sạch dữ liệu trước!")
            return

        self.clear_main_frame()
        control_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        control_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(control_frame, text="Chọn loại biểu đồ:", text_color="black").pack(side="left", padx=10)
        self.chart_option = ctk.CTkComboBox(control_frame, values=[
            "Phân phối giá cước (Toàn bộ)",
            "Phân phối giá cước (0-100$)",
            "Nhu cầu theo giờ",
            "Tương quan Quãng đường vs Giá"
        ], width=250)
        self.chart_option.set("Phân phối giá cước (Toàn bộ)")
        self.chart_option.pack(side="left", padx=10)

        btn_draw = ctk.CTkButton(control_frame, text="Vẽ biểu đồ", command=self.draw_chart)
        btn_draw.pack(side="left", padx=10)

        self.plot_frame = ctk.CTkFrame(self.main_frame, fg_color="white")
        self.plot_frame.pack(fill="both", expand=True, padx=10, pady=10)

    def draw_chart(self):
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        canvas_scroll = ctk.CTkScrollableFrame(self.plot_frame, fg_color="transparent")
        canvas_scroll.pack(fill="both", expand=True)

        chart_type = self.chart_option.get()
        fig = None
        try:
            if chart_type == "Phân phối giá cước (Toàn bộ)":
                fig = visualization.plot_fare_distribution(self.df_cleaned)
            elif chart_type == "Phân phối giá cước (0-100$)":
                fig = visualization.plot_fare_range_0_100(self.df_cleaned)
            elif chart_type == "Nhu cầu theo giờ":
                fig = visualization.plot_hourly_demand(self.df_cleaned)
            elif chart_type == "Tương quan Quãng đường vs Giá":
                fig = visualization.plot_distance_fare_relation(self.df_cleaned)

            if fig:
                canvas = FigureCanvasTkAgg(fig, master=canvas_scroll)
                canvas.draw()
                canvas.get_tk_widget().pack(fill="both", expand=True)
        except Exception as e:
            messagebox.showerror("Lỗi Vẽ biểu đồ", str(e))

    def train_eval_event(self):
        if self.df_cleaned is None:
            messagebox.showwarning("Cảnh báo", "Vui lòng làm sạch dữ liệu trước!")
            return

        self.clear_main_frame()

        control_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        control_frame.pack(fill="x", padx=10, pady=10)

        btn_start_train = ctk.CTkButton(control_frame, text="Bắt đầu Huấn luyện",
                                        command=lambda: self.run_training(control_frame),
                                        fg_color="green", hover_color="darkgreen")
        btn_start_train.pack(side="left", padx=20, pady=10)

        self.result_label = ctk.CTkLabel(control_frame, text="Trạng thái: Chưa huấn luyện", font=("Arial", 14),
                                         text_color="black")
        self.result_label.pack(side="left", padx=20)

        self.metrics_frame = ctk.CTkFrame(self.main_frame, height=50, fg_color="transparent")
        self.metrics_frame.pack(fill="x", padx=10, pady=5)

        self.table_container = ctk.CTkFrame(self.main_frame, fg_color="white")
        self.table_container.pack(fill="both", expand=True, padx=10, pady=10)

        ctk.CTkLabel(self.table_container, text="Bảng so sánh: Giá thực tế vs Dự đoán (20 dòng)",
                     font=("Arial", 14, "bold"), text_color="black").pack(anchor="w", pady=5)

    def run_training(self, parent_frame):
        if not self.is_running or not self.winfo_exists(): return

        self.result_label.configure(text="Trạng thái: Đang huấn luyện... (Vui lòng chờ)", text_color="orange")

        self.safe_update(parent_frame)

        try:
            rmse, r2, sample_preds = model.train_linear_regression(self.df_cleaned)

            if not self.is_running or not self.winfo_exists(): return

            self.result_label.configure(text="Trạng thái: Huấn luyện hoàn tất!", text_color="green")

            for widget in self.metrics_frame.winfo_children(): widget.destroy()

            m1 = ctk.CTkFrame(self.metrics_frame, fg_color="#f0f0f0")  # Màu nền card sáng
            m1.pack(side="left", fill="both", expand=True, padx=5, pady=5)
            ctk.CTkLabel(m1, text="RMSE (Sai số)", font=("Arial", 12), text_color="black").pack()
            ctk.CTkLabel(m1, text=f"{rmse:.4f}", font=("Arial", 20, "bold"), text_color="#2980b9").pack()

            m2 = ctk.CTkFrame(self.metrics_frame, fg_color="#f0f0f0")
            m2.pack(side="left", fill="both", expand=True, padx=5, pady=5)
            ctk.CTkLabel(m2, text="R2 Score (Độ chính xác)", font=("Arial", 12), text_color="black").pack()
            ctk.CTkLabel(m2, text=f"{r2:.4f}", font=("Arial", 20, "bold"), text_color="#27ae60").pack()

            for widget in self.table_container.winfo_children():
                if isinstance(widget, ctk.CTkFrame) or isinstance(widget, ttk.Treeview):
                    widget.destroy()

            ctk.CTkLabel(self.table_container, text="Bảng so sánh: Giá thực tế vs Dự đoán (20 dòng)",
                         font=("Arial", 14, "bold"), text_color="black").pack(anchor="w", pady=5)

            self.create_table(self.table_container, sample_preds)
            try:
                save_path = "predictions.csv"
                sample_preds.to_csv(save_path, index=False)
                messagebox.showinfo("Thành công", f"Huấn luyện xong!\nĐã tự động lưu kết quả dự báo vào: {save_path}")
            except Exception as e_save:
                messagebox.showerror("Lỗi Lưu file", f"Lỗi lưu file dự báo: {e_save}")

        except Exception as e:
            if self.is_running:
                self.result_label.configure(text="Trạng thái: Lỗi!", text_color="red")
                messagebox.showerror("Lỗi Huấn luyện", str(e))
                print("Chi tiết lỗi:", e)

    def on_close(self):
        self.is_running = False

        if self.after_id:
            try:
                self.after_cancel(self.after_id)
            except:
                pass

        if self.spark:
            try:
                self.spark.stop()
            except:
                pass

        try:
            self.withdraw()
        except:
            pass
        try:
            self.quit()
            self.destroy()
        except:
            pass


if __name__ == "__main__":
    app = TaxiAnalysisApp()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()