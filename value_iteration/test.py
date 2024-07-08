import torch

# 假设 theta 的维度是 (1, 10, 1)
theta = torch.randn(1, 10, 1, requires_grad=True)

# 假设 x 的维度是 (100, 4, 1)
x = torch.randn(100, 4, 1, requires_grad=True)

# 示例运算：假设 a 是 x 和 theta 的一些运算结果，维度是 (100, 4, 1)
# 这里我们假设 a 是通过简单的加法得到的，只是为了示例
# 实际中，你会根据具体运算来计算 a
a = x + theta  # (100, 4, 1)
print(a.shape)
# 创建一个与 a 形状相同的单位张量，用于 grad_outputs
grad_outputs = torch.ones_like(a)

# 计算 Jacobian
jacobian = torch.zeros(a.size(0), a.size(1), theta.size(1))

for i in range(a.size(1)):
    for j in range(a.size(0)):
        grad = torch.autograd.grad(a[j, i], theta, grad_outputs=grad_outputs[j, i], retain_graph=True)[0]
        jacobian[j, i, :] = grad.view(-1)

print(jacobian)
