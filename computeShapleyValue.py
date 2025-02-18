from itertools import combinations
from math import factorial
import numpy as np
import matplotlib.pyplot as plt

# 定义集合
D = {1, 2}  # 存储矿工
R = {3, 4, 5}  # 数据中继者
T = {6}  # 数据训练方
P = D | R | T  # 所有参与者
v_P = 1  # 总收益 v(P) = 1

# 定义收益函数 B(D')
def B(D_subset, miner1_value=1.0):
    """收益函数 B(D')，矿工1的收益从0.1递增到1"""
    if 1 in D_subset and 2 in D_subset:
        return miner1_value + 1  # 矿工1的收益 + 矿工2的收益
    elif 1 in D_subset:
        return miner1_value
    elif 2 in D_subset:
        return 1  # 矿工2固定收益
    else:
        return 0  # 空集收益为0

# 计算标准 Shapley 权重 w(x) = (|S|! (|N| - |S| - 1)! ) / |N|!
def shapley_weight(S_size, N_size):
    return (factorial(S_size) * factorial(N_size - S_size - 1)) / factorial(N_size)

# 计算 ϕ_i^D
def compute_shapley_D(i, D, R, T, miner1_value):
    """计算 ϕ_i^D（存储矿工）"""
    n = len(D) + len(R) + len(T)
    shapley_D = 0

    for D_prime in (set(combo) for r in range(len(D)) for combo in combinations(D - {i}, r)):
        weight = shapley_weight(len(D_prime) + len(R) + len(T), n)
        marginal_contribution = (B(D_prime | {i}, miner1_value) - B(D_prime, miner1_value)) / B(D, miner1_value)
        shapley_D += weight * marginal_contribution
    return shapley_D

# 计算 ϕ_R
def compute_shapley_R(D, R, T, miner1_value):
    """计算 ϕ_R（数据中继者）"""
    n = len(D) + len(R) + len(T)
    shapley_R = 0

    for D_prime in (set(combo) for r in range(len(D) + 1) for combo in combinations(D, r)):
        weight = shapley_weight(len(D_prime) + len(T), n)
        shapley_R += weight * (B(D_prime, miner1_value) / B(D, miner1_value))
    return shapley_R

# 计算 ϕ_T
def compute_shapley_T(D, R, T, miner1_value):
    """计算 ϕ_T（数据训练方）"""
    n = len(D) + len(R) + len(T)
    shapley_T = 0

    for D_prime in (set(combo) for r in range(len(D) + 1) for combo in combinations(D, r)):
        weight = shapley_weight(len(D_prime) + len(R), n)
        shapley_T += weight * (B(D_prime, miner1_value) / B(D, miner1_value))
    return shapley_T

# 计算 Shapley 值在矿工 1 的收益从 0.1 递增到 1 的情况下的变化
miner1_values = np.linspace(0.1, 1.0, 10)  # 10 个递增值
shapley_D1_values = []
shapley_D2_values = []
shapley_R_values = []
shapley_T_values = []

for miner1_value in miner1_values:
    shapley_D_values = {i: compute_shapley_D(i, D, R, T, miner1_value) for i in D}
    shapley_R = compute_shapley_R(D, R, T, miner1_value)
    shapley_T = compute_shapley_T(D, R, T, miner1_value)

    # 计算最终收益分配
    final_D_values = {i: shapley_D_values[i] * v_P for i in D}
    final_R_value = shapley_R * v_P  # 数据中继者的总收益
    final_T_value = shapley_T * v_P  # 数据训练方的收益

    # 记录数据
    shapley_D1_values.append(final_D_values[1])
    shapley_D2_values.append(final_D_values[2])
    shapley_R_values.append(final_R_value)
    shapley_T_values.append(final_T_value)

# 绘制 Shapley 值曲线
plt.figure(figsize=(8, 5))
plt.plot(miner1_values, shapley_D1_values, 'r-o', label='Shapley Value of Miner 1')
plt.plot(miner1_values, shapley_D2_values, 'b-s', label='Shapley Value of Miner 2')
plt.plot(miner1_values, shapley_R_values, 'g-^', label='Shapley Value of Relayers (Total)')
plt.plot(miner1_values, shapley_T_values, 'm-d', label='Shapley Value of Training Party')

# 设置图表属性
plt.xlabel('Miner 1 B(D) Value')
plt.ylabel('Shapley Value')
plt.title('Shapley Value Changes with Miner 1 B(D)')
plt.legend()
plt.grid(True)

# 显示图表
plt.show()
