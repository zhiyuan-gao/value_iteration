import numpy as np
import torch
import matplotlib.pyplot as plt




# # 假设 x 是一个 3D tensor
# x = torch.tensor([[[1.0], [2.0], [3.0]],
#                   [[4.0], [5.0], [6.0]],
#                   [[7.0], [8.0], [9.0]]])

# # 索引列表 indices
# indices = [0, 2]  # 例如，我们想替换第 0 行和第 2 行

# # 计算 z 的大小，大小是 (x.shape[0], x.shape[1] + len(indices), x.shape[2])
# z = torch.zeros(x.shape[0], x.shape[1] + len(indices), x.shape[2])

# # 当前 z 的行索引
# z_row = 0

# for i in range(x.shape[1]):
#     if i in indices:
#         # 插入 sin(x[:,i,0]) 和 cos(x[:,i,0])
#         z[:, z_row, 0] = torch.sin(x[:, i, 0])
#         z[:, z_row + 1, 0] = torch.cos(x[:, i, 0])
#         z_row += 2
#     else:
#         # 插入原始 x[:,i,0]
#         z[:, z_row, 0] = x[:, i, 0]
#         z_row += 1

# print(z)






# tensor = torch.tensor([0, 1, 0, 1])

# # 找到值为1的索引
# indices = torch.nonzero(tensor).squeeze()
# print(indices)  # 输出: tensor([1, 3])



n_input = 4

# # feature = [1,0,0,0]
# feature = [0,0,0,1]
# feature = [0,0,1,0]
# # set up the feature mask:
# feature = feature if feature is not None else torch.cat([torch.ones(1), torch.zeros(n_input-1)], dim=0)
# feature = np.clip(feature, 0., 1.0)
# # assert feature.size()[0] == n_input and torch.sum(feature) == 1.
# # idx = feature.argmax()
# idx = feature.argmax()

# m = int(((n_input + 2) ** 2 + (n_input + 2)) / 2)

# print('m',m)

# # # Calculate the indices of the diagonal elements of L:
diag_idx = np.arange(n_input) + 1
print('diag_idx',diag_idx)
diag_idx = diag_idx * (diag_idx + 1) / 2 - 1  #diag of x?
print('diag_idx',diag_idx)
# tri_idx = np.extract([x not in diag_idx for x in np.arange(m)], np.arange(m))
# tril_idx = np.tril_indices(n_input)

# print('tri_idx',tri_idx)

# print('tril_idx',tril_idx)
# n_feature = 4
# eye_f_input = torch.eye(n_feature - idx - 1)

# eye_idx = torch.eye(idx)

# dzdx = torch.zeros(1, 5, 4)
# dzdx[:, :idx, :idx] = eye_idx

# dzdx[:, idx + 2:, idx + 1:] = eye_f_input
# print('eye_f_input',eye_f_input)
# print('dzdx',dzdx)




# u_lim =  torch.tensor([[6.], [0.00001]])
u_lim =  torch.tensor([2.5, ])
R  = np.array([5.e-1 ])


R = np.diag(R).reshape((1,1))
beta = (4. * u_lim[0] ** 2 / np.pi * R)[0, 0].item()

print(beta)



u_lim =  torch.tensor([[6.], [0.00001]])
R  = np.array([0.01,0.01 ])

R = np.diag(R).reshape((2,2))
beta = 4. * u_lim ** 2 / torch.pi * R

print(beta)

from value_iteration.cost_functions import ArcTangent, SineQuadraticCost, BarrierCost


alpha = 6
beta = 0.4

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
