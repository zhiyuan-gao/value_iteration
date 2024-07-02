import numpy as np
import matplotlib.pyplot as plt
import torch
# 定义奖励函数
def action_cost(u, umax, beta):
    return -2 * beta * umax / np.pi * np.log(np.cos(np.pi * u / (2 * umax)))


g = np.zeros([100,2,1])
g = np.sum(g, axis=1)
alpha = np.array([1.0, 0.5])
print((-np.abs(alpha), +np.abs(alpha)))

# 参数设置
# umax = 1.0  # 动作的最大值
# beta = 1.0  # 正的常数
# R = np.array([1.0, 0.5])
# R = np.diag(R).reshape((2, 2))

# u_lim =  torch.tensor([6., 0])

# beta = 4. * u_lim[0] ** 2 / np.pi * R

# print(beta)
# alpha=u_lim.numpy()
# print(alpha.shape[0],'o')
# beta=beta.numpy()[0, 0]
# print(beta)
# 生成动作范围
# u = np.linspace(-umax, umax, 500)

# # 计算动作成本
# cost = action_cost(u, umax, beta)
# print(action_cost(1.0, umax, beta))


# # 绘制图像
# plt.figure(figsize=(10, 6))
# plt.plot(u, cost)
# plt.xlabel('Action (u)')
# plt.ylabel('Cost')
# plt.title('Action Cost Function')
# plt.legend()
# plt.grid(True)
# plt.show()
