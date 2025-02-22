import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import factorial

# 计算 Shapley 权重 w(m)
def shapley_weight(m, n):
    return (factorial(m) * factorial(n - m - 1)) / factorial(n)

# 计算 Shapley 值
def compute_shapley(B_D1, B_D2, B_D):
    B_empty = 0  # 空集贡献值
    B_D1_D2 = B_D  # 两个数据存储者的总贡献值

    # 参与者数量
    num_D = 2  # 数据存储者
    num_R = 1  # 资源提供者
    num_T = 1  # 任务分发者
    n = num_D + num_R + num_T  # 总参与者数

    # 计算 ϕ^D_1 和 ϕ^D_2
    phi_D1 = (
        shapley_weight(2, n) * (B_D1 - B_empty) / B_D +
        shapley_weight(3, n) * (B_D1_D2 - B_D2) / B_D
    )

    phi_D2 = (
        shapley_weight(2, n) * (B_D2 - B_empty) / B_D +
        shapley_weight(3, n) * (B_D1_D2 - B_D1) / B_D
    )

    phi_R = (
        shapley_weight(2, n) * (B_D1 / B_D) +
        shapley_weight(2, n) * (B_D2 / B_D) +
        shapley_weight(3, n) * (B_D1_D2 / B_D)
    )

    phi_T = (
        shapley_weight(2, n) * (B_D1 / B_D) +
        shapley_weight(2, n) * (B_D2 / B_D) +
        shapley_weight(3, n) * (B_D1_D2 / B_D)
    )

    # 验证 Shapley 值总和
    total_sum = phi_D1 + phi_D2 + phi_R + phi_T
    if abs(total_sum - 1) > 1e-6:
        print(f"⚠️ Shapley 值总和错误: {total_sum:.6f} (B_D1={B_D1}, B_D2={B_D2})")
    # else :
    #     print("计算正确")

    return phi_D1 + phi_D2, phi_T, phi_D1, phi_D2

# 生成 X, Y 轴数据（B_D1 和 B_D2）
B_D = 10  # 总数据贡献值固定为 10
B_D1_values = np.linspace(1, 5, 100)  # 细化网格点
B_D2_values = np.linspace(5, 9, 100)
X, Y = np.meshgrid(B_D1_values, B_D2_values)

# 计算 Z 轴数据
Z_D_sum = np.zeros_like(X)
Z_T = np.zeros_like(X)
Z_D1 = np.zeros_like(X)
Z_D2 = np.zeros_like(X)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z_D_sum[i, j], Z_T[i, j], Z_D1[i,j], Z_D2[i,j] = compute_shapley(X[i, j], Y[i, j], B_D)

# 绘制 3D 曲面图
fig = plt.figure(figsize=(12, 6))

ax = fig.add_subplot(121, projection='3d')

# 绘制 ϕ^D_1 + ϕ^D_2 曲面
surf1 = ax.plot_surface(X, Y, Z_D_sum, cmap='viridis', edgecolor='none', alpha=0.8)
# 绘制 ϕ_T 曲面
surf2 = ax.plot_surface(X, Y, Z_T, cmap='plasma', edgecolor='none', alpha=0.8)

# 绘制 ϕ^D_1 + ϕ^D_2 曲面
surfD1 = ax.plot_surface(X, Y, Z_D1, cmap='BuPu', edgecolor='none', alpha=0.8)
# 绘制 ϕ_T 曲面
surfD2 = ax.plot_surface(X, Y, Z_D2, cmap='GnBu', edgecolor='none', alpha=0.8)



# # 添加颜色条
# fig.colorbar(surf1, ax=ax, shrink=0.5, aspect=10, label="ϕ^D_1 + ϕ^D_2")
# fig.colorbar(surf2, ax=ax, shrink=0.5, aspect=10, label="ϕ_T")

# 设置坐标轴标签
ax.set_xlabel('B_D1')
ax.set_ylabel('B_D2')
ax.set_zlabel('Shapley Value')

# 设置标题
ax.set_title('Data Miners and Train Value')

plt.show()