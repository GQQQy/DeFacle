from math import factorial

# 定义函数计算 Shapley 权重 w(m)
def shapley_weight(m, n):
    return (factorial(m) * factorial(n - m - 1)) / factorial(n)

# 设定参数
B_D = 10  # 总数据集贡献值
B_D1 = 1  # D1 贡献值
B_D2 = 9  # D2 贡献值
B_empty = 0  # 空集贡献值
B_D1_D2 = B_D  # D1 和 D2 贡献值

# 参与者数量
num_D = 2  # 数据存储者
num_R = 1  # 资源提供者
num_T = 1  # 任务分发者
n = num_D + num_R + num_T  # 总参与者数

# 计算 ϕ^D_1 和 ϕ^D_2
def compute_phi_D():
    phi_D1 = (
        shapley_weight(2, n) * (B_D1 - B_empty) / B_D +
        shapley_weight(3, n) * (B_D1_D2 - B_D2) / B_D
    )
    
    phi_D2 = (
        shapley_weight(2, n) * (B_D2 - B_empty) / B_D +
        shapley_weight(3, n) * (B_D1_D2 - B_D1) / B_D
    )
    
    return phi_D1, phi_D2

# 计算 ϕ_R
def compute_phi_R():
    return (
        shapley_weight(2, n) * (B_D1 / B_D) +
        shapley_weight(2, n) * (B_D2 / B_D) +
        shapley_weight(3, n) * (B_D1_D2 / B_D)
    )

# 计算 ϕ_T
def compute_phi_T():
    return (
        shapley_weight(2, n) * (B_D1 / B_D) +
        shapley_weight(2, n) * (B_D2 / B_D) +
        shapley_weight(3, n) * (B_D1_D2 / B_D)
    )

# 计算 Shapley 值
phi_D1, phi_D2 = compute_phi_D()
phi_R = compute_phi_R()
phi_T = compute_phi_T()

# 输出结果
print(f"ϕ^D_1 = {phi_D1:.4f}")
print(f"ϕ^D_2 = {phi_D2:.4f}")
print(f"ϕ_R = {phi_R:.4f}")
print(f"ϕ_T = {phi_T:.4f}")

# 验证总和是否等于 1
total_sum = phi_D1 + phi_D2 + phi_R + phi_T
print(f"Total Sum = {total_sum:.4f}")
print("✅ Shapley 值正确！" if abs(total_sum - 1) < 1e-6 else "❌ 计算有误！")