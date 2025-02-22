from itertools import combinations
from math import factorial
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义集合
D = {1, 2}  # 存储矿工
R = {3, 4, 5}  # 数据中继者
T = {6}  # 数据训练方
P = D | R | T  # 所有参与者
v_P = 1  # 总收益 v(P) = 1

# 收益函数 B(D')，包含 miner1_value 和 miner2_value
def B(D_subset, miner1_value=1.0, miner2_value=1.0):
    """收益函数 B(D')，矿工1 和 矿工2 的收益均可变"""
    value = 0
    if 1 in D_subset :
        value += miner1_value
    if 2 in D_subset:
        value += miner2_value
    return value

# 计算标准 Shapley 权重 w(x) = (|S|! (|N| - |S| - 1)! ) / |N|!
def shapley_weight(S_size, N_size):
    return (factorial(S_size) * factorial(N_size - S_size - 1)) / factorial(N_size)

# 计算 ϕ_i^D
def compute_shapley_D(i, D, R, T, miner1_value, miner2_value):
    """计算 ϕ_i^D（存储矿工）"""
    n = len(D) + len(R) + len(T)
    shapley_D = 0

    for D_prime in (set(combo) for r in range(len(D)) for combo in combinations(D - {i}, r)):
        weight = shapley_weight(len(D_prime) + len(R) + len(T), n)
        marginal_contribution = (B(D_prime | {i}, miner1_value, miner2_value) - B(D_prime, miner1_value, miner2_value)) / B(D, miner1_value, miner2_value)
        shapley_D += weight * marginal_contribution
    return shapley_D

# 计算 ϕ_T
def compute_shapley_T(D, R, T, miner1_value, miner2_value):
    """计算 ϕ_T（数据训练方）"""
    n = len(D) + len(R) + len(T)
    shapley_T = 0

    for D_prime in (set(combo) for r in range(len(D) + 1) for combo in combinations(D, r)):
        weight = shapley_weight(len(D_prime) + len(R), n)
        shapley_T += weight * (B(D_prime, miner1_value, miner2_value) / B(D, miner1_value, miner2_value))
    return shapley_T

# 定义矿工 1 和矿工 2 的收益范围
miner1_values = np.linspace(0.1, 1.0, 100)  # 矿工 1 收益从 0.1 到 1.0
miner2_values = np.linspace(0.5, 1.5, 100)  # 矿工 2 收益从 0.5 到 1.5

# 创建网格数据
X, Y = np.meshgrid(miner1_values, miner2_values)
Z_miner1 = np.zeros_like(X)
Z_miner2 = np.zeros_like(X)
Z_training = np.zeros_like(X)

# 遍历所有矿工 1 和矿工 2 的收益组合，计算 Shapley 值
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        miner1_value = X[i, j]
        miner2_value = Y[i, j]

        # 计算 Shapley 值
        shapley_D_values = {i: compute_shapley_D(i, D, R, T, miner1_value, miner2_value) for i in D}
        shapley_T = compute_shapley_T(D, R, T, miner1_value, miner2_value)

        # 计算最终收益占比
        final_D_values = {i: shapley_D_values[i] * v_P for i in D}
        final_T_value = shapley_T * v_P  # 数据训练方的收益

        # 记录数据
        Z_miner1[i, j] = final_D_values[1]
        Z_miner2[i, j] = final_D_values[2]
        Z_training[i, j] = final_T_value

# 绘制 3D 曲面图
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 绘制曲面
ax.plot_surface(X, Y, Z_miner1, cmap='Reds', alpha=0.8, edgecolor='k', label='Miner 1')
ax.plot_surface(X, Y, Z_miner2, cmap='Blues', alpha=0.8, edgecolor='k', label='Miner 2')
ax.plot_surface(X, Y, Z_training, cmap='Purples', alpha=0.8, edgecolor='k', label='Training Party')

# 设置坐标轴
ax.set_xlabel('Miner 1 B(D) Value')
ax.set_ylabel('Miner 2 B(D) Value')
ax.set_zlabel('Shapley Value Distribution')
ax.set_title('3D Visualization of Shapley Value Distribution')

plt.savefig('plot.png')

plt.show()