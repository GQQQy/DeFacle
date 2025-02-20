import numpy as np
import matplotlib.pyplot as plt

# Transformer 数据
transformer_data = {
    "Training Data (M)": [0.2, 0.5, 1, 2, 5, 8, 10, 15],
    "Data Provider A BLEU": [7.35, 11.21, 16.12, 19.45, 22.78, 25.34, 26.05, 27.89],
    "Data Provider B BLEU": [29.42, 29.35, 28.97, 28.75, 27.83, 27.62, 27.55, 28.31]
}

# M2M-100 数据
m2m100_data = {
    "Training Data (M)": [0.2, 0.5, 1, 2, 5, 8, 10, 15],
    "Data Provider A BLEU": [6.24, 9.14, 14.32, 18.45, 22.51, 25.76, 27.43, 29.12],
    "Data Provider B BLEU": [30.57, 30.42, 29.78, 29.63, 28.92, 28.75, 28.69, 30.25]
}

def compute_shapley_values(data):
    training_sizes = np.array(data["Training Data (M)"])
    bleu_A = np.array(data["Data Provider A BLEU"])
    bleu_B = np.array(data["Data Provider B BLEU"])

    # 计算 Shapley Value
    shapley_A = 0.5 * (bleu_A - 0) + 0.5 * ((bleu_A + bleu_B) - bleu_B)
    shapley_B = 0.5 * (bleu_B - 0) + 0.5 * ((bleu_A + bleu_B) - bleu_A)

    return training_sizes, shapley_A, shapley_B

# 计算 Transformer 和 M2M-100 的 Shapley Value
train_sizes_t, shapley_A_t, shapley_B_t = compute_shapley_values(transformer_data)
train_sizes_m, shapley_A_m, shapley_B_m = compute_shapley_values(m2m100_data)

# 可视化 Transformer 和 M2M-100 Shapley Value
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Transformer Shapley Value 曲线
axes[0].plot(train_sizes_t, shapley_A_t, 'r-o', label="Shapley Value - Data Provider A")
axes[0].plot(train_sizes_t, shapley_B_t, 'b-s', label="Shapley Value - Data Provider B")
axes[0].set_xlabel("Training Data (M)")
axes[0].set_ylabel("Shapley Value")
axes[0].set_title("Shapley Value for Transformer Model")
axes[0].legend()
axes[0].grid(True)

# M2M-100 Shapley Value 曲线
axes[1].plot(train_sizes_m, shapley_A_m, 'r-o', label="Shapley Value - Data Provider A")
axes[1].plot(train_sizes_m, shapley_B_m, 'b-s', label="Shapley Value - Data Provider B")
axes[1].set_xlabel("Training Data (M)")
axes[1].set_ylabel("Shapley Value")
axes[1].set_title("Shapley Value for M2M-100 Model")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()