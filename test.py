import torch
import numpy as np
import matplotlib.pyplot as plt
from value_iteration.cost_functions import ArcTangent, SineQuadraticCost, BarrierCost
# 定义 Q 矩阵
Q = np.array([5., 5., 1.e-1, 1.e-1])

# 设置 x_target 和 x_penalty
x_target = torch.tensor([np.pi, 0, 0, 0])
x_penalty = torch.tensor([0.5 * np.pi, 0.5 * np.pi, 5, 5])

# 使用 SineQuadraticCost
cuda = False
q = SineQuadraticCost(np.diag(Q), np.array([1.0, 1.0, 0.0, 0.0]), cuda=cuda)

# 使用 BarrierCost
# # q = BarrierCost(_q, x_penalty, cuda)

# # 创建 x 从 -2π 到 2π
# x_values = torch.linspace(-np.pi,  np.pi, 100)
# cost_values = []

# for x_val in x_values:
#     x = torch.tensor([[x_val], [0], [0], [0]])  # 假设只有第一个变量变化
#     x = x - x_target.view(-1, 4, 1).to(x.device)  # 考虑偏移
#     cost = q(x)
#     cost_values.append(cost.item())

# # 绘制成本随 x 变化的曲线
# plt.plot(x_values, cost_values)
# plt.xlabel('x')
# plt.ylabel('Cost')
# plt.title('Cost vs x')
# plt.grid(True)
# plt.show()




alpha = 6
beta = 0.04 *1000

# 创建ArcTangent对象
arc_tangent = ArcTangent(alpha, beta)

# 生成输入数据
x = np.linspace(-alpha, alpha, 100)

# 计算对应的输出数据
y = arc_tangent(x)

# 绘制图像
plt.figure(figsize=(8, 6))
plt.plot(x, y, label='ArcTangent Function')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('ArcTangent Function')
plt.legend()
plt.grid(True)
plt.show()