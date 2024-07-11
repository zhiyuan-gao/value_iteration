import numpy as np
import torch
import matplotlib.pyplot as plt







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




class ArcTangent:
    def __init__(self, alpha=+1., beta=+1.0):
        if isinstance(alpha, np.ndarray):
            self.n = alpha.shape[0]
        elif isinstance(alpha, torch.Tensor):
            self.n = alpha.shape[0]
        else:
            self.n = 1

        self.a = alpha
        self.b = beta
        self.convex_domain = (-10.0, +10.0)
        self.a += 1.e-3

    def __call__(self, x):
        if isinstance(x, np.ndarray):
            assert np.all(np.abs(x) <= self.a)
            g = -2.0 * self.b * self.a / np.pi * np.log(np.clip(np.cos(np.pi / (2. * self.a) * x), 0.0, 1.0))
            if self.n > 1:
                g = np.sum(g, axis=1, keepdims=True)

        elif isinstance(x, torch.Tensor):
            x = torch.clamp(x, -self.a, self.a)
            assert torch.all(torch.abs(x) <= self.a)
            g = -2.0 * self.b * self.a / np.pi * torch.log(torch.clamp(torch.cos(np.pi / (2. * self.a) * x), 0.0, 1.0))
            if self.n > 1:
                g = torch.sum(g, axis=1, keepdim=True)
        else:
            raise ValueError("x must be either an numpy.ndarray or torch.Tensor, but is type {0}.".format(type(x)))
        return g


    def grad_convex_conjugate(self, x):
        if isinstance(x, np.ndarray):
            g_star_grad = 2. * self.a / np.pi * np.arctan(x / self.b)

        elif isinstance(x, torch.Tensor):
            # print(self.a / np.pi )
            # print('---')
            # print(x.device)
            # print(self.b.device)
            # print(torch.atan(x / self.b).device)
            g_star_grad = 2. * self.a / np.pi * torch.atan(x / self.b)

        else:
            raise ValueError("x must be either an numpy.ndarray or torch.Tensor, but is type {0}.".format(type(x)))
        
        return g_star_grad.to(torch.float32)
# 设置alpha和beta
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
