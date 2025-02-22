import pandas as pd

# 创建 Transformer-base 数据（带小数）
transformer_data = {
    "Training Data (M)": [0.2, 0.5, 1, 2, 5, 8, 10, 15],
    "Data Provider A BLEU": [7.35, 11.21, 16.12, 19.45, 22.78, 25.34, 26.05, 27.89],
    "Data Provider B BLEU": [29.42, 29.35, 28.97, 28.75, 27.83, 27.62, 27.55, 28.31]
}

# 创建 M2M-100 数据（带小数）
m2m100_data = {
    "Training Data (M)": [0.2, 0.5, 1, 2, 5, 8, 10, 15],
    "Data Provider A BLEU": [6.24, 9.14, 14.32, 18.45, 22.51, 25.76, 27.43, 29.12],
    "Data Provider B BLEU": [30.57, 30.42, 29.78, 29.63, 28.92, 28.75, 28.69, 30.25]
}

# 转换为 Pandas DataFrame
transformer_df = pd.DataFrame(transformer_data)
m2m100_df = pd.DataFrame(m2m100_data)

# 保存为 CSV 文件（保留两位小数）
transformer_df.to_csv("Transformer-base.csv", index=False, float_format="%.2f")
m2m100_df.to_csv("M2M-100.csv", index=False, float_format="%.2f")

print("CSV 文件已生成：Transformer-base.csv 和 M2M-100.csv")