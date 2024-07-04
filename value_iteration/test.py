import torch
import sympy as smp
from datetime import datetime
import numpy as np
# symbol_theta = smp.symbols("Ir r1 I1 b1 cf1 r2 m2 I2 b2 cf2")

# # m1, m2, l1, l2, r1, r2, b1, b2, cf1, cf2, g, I1, I2, Ir, gr =self.symbol_theta
# Ir, r1, I1, b1, cf1, r2, m2, I2, b2, cf2 = symbol_theta

# # print(r2)

# x = torch.tensor([1., 2., 3.])
# print(x)
# print(x.shape)

# print(np.linspace(0.0, 1e-4, 21))


def get_par_list(x0, min_rel, max_rel, n):
    if x0 != 0:
        if n % 2 == 0:
            n = n+1
        li = np.linspace(min_rel, max_rel, n)
    else:
        li = np.linspace(0, max_rel, n)
    par_list = li*x0
    return par_list


print(get_par_list(0.608 * 0.3, 0.75, 1.25, 21))