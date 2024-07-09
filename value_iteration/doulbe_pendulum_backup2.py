import numpy as np
import torch
import time
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..'))

import sympy as smp
from sympy.utilities import lambdify
from deep_differential_network.utils import jacobian
from value_iteration.pendulum import BaseSystem
from value_iteration.cost_functions import ArcTangent, SineQuadraticCost, BarrierCost
from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
CUDA_AVAILABLE = torch.cuda.is_available()

# model parameters to dict
def theta_to_dict(theta_tensor):
    tensor_flattened = theta_tensor.view(-1)
    keys = ['Ir', 'r1', 'I1', 'b1', 'cf1', 'r2', 'm2', 'I2', 'b2', 'cf2']
    change_para = {key: value.item() for key, value in zip(keys, tensor_flattened)}

    return change_para


def compute_dadp_x_matrix(x):
    # 将输入张量移动到 GPU
    x = x.cuda()

    # 从 x 中提取 q1, q2, dot_q1, dot_q2
    q1 = x[:, 0, :].view(-1, 1)
    q2 = x[:, 1, :].view(-1, 1)
    dot_q1 = x[:, 2, :].view(-1, 1)
    dot_q2 = x[:, 3, :].view(-1, 1)

    # 定义 cos 和 sin 的函数
    cos_q2 = torch.cos(q2)
    sin_q2 = torch.sin(q2)
    # cos_q1 = torch.cos(q1)
    sin_q1 = torch.sin(q1)
    sin_q1_plus_q2 = torch.sin(q1 + q2)

    # 定义 atan 函数
    atan_100_dot_q1 = torch.atan(100 * dot_q1)
    atan_100_dot_q2 = torch.atan(100 * dot_q2)

    # 计算 self.dadp_x 矩阵的每个元素
    def element_0_2():
        term1 = (-5.64732*cos_q2 - 9.5766) * (0.15687*dot_q1*dot_q2*sin_q2 - 0.005*dot_q1 + 0.078435*dot_q2**2*sin_q2 - 3.494322*sin_q1 - 2.5648245*sin_q1_plus_q2 - 0.093*atan_100_dot_q1)
        denom1 = (0.00096507196192575*cos_q2**3 + 0.0011319770574*cos_q2**2 - 0.0013043991492*cos_q2 - 0.00152998944)
        term1 /= denom1

        term2 = 427192.021899475 * (-0.012518226*cos_q2 - 0.0146832) * (0.805918448475*cos_q2**2 + 2.864791314*cos_q2 + 2.24595372)
        term2 *= (0.15687*dot_q1*dot_q2*sin_q2 - 0.005*dot_q1 + 0.078435*dot_q2**2*sin_q2 - 3.494322*sin_q1 - 2.5648245*sin_q1_plus_q2 - 0.093*atan_100_dot_q1)
        term2 /= (0.630770341738927*cos_q2**3 + 0.739859392362865*cos_q2**2 - 0.852554347826087*cos_q2 - 1)**2

        term3 = (1.960875*cos_q2 + 1.8486) * (-0.078435*dot_q1**2*sin_q2 - 0.005*dot_q2 - 2.5648245*sin_q1_plus_q2 - 0.14*atan_100_dot_q2)
        term3 /= denom1

        term4 = 427192.021899475 * (0.01230409845*cos_q2**2 + 0.026950266*cos_q2 + 0.0146832) * (0.805918448475*cos_q2**2 + 2.864791314*cos_q2 + 2.24595372)
        term4 *= (-0.078435*dot_q1**2*sin_q2 - 0.005*dot_q2 - 2.5648245*sin_q1_plus_q2 - 0.14*atan_100_dot_q2)
        term4 /= (0.630770341738927*cos_q2**3 + 0.739859392362865*cos_q2**2 - 0.852554347826087*cos_q2 - 1)**2

        return (term1 + term2 + term3 + term4).view(-1, 1)

    def element_0_3():
        term1 = 14463.0130934286 * (-0.15687*cos_q2 - 0.184) * (6.58854*cos_q2 + 10.5342)
        term1 *= (-0.078435*dot_q1**2*sin_q2 - 0.005*dot_q2 - 2.5648245*sin_q1_plus_q2 - 0.14*atan_100_dot_q2)
        term1 /= (0.739859392362865*cos_q2**2 - 1)**2

        term2 = 14463.0130934286 * (0.078435*cos_q2 + 0.0798) * (6.58854*cos_q2 + 10.5342)
        term2 *= (0.15687*dot_q1*dot_q2*sin_q2 - 0.005*dot_q1 + 0.078435*dot_q2**2*sin_q2 - 3.494322*sin_q1 - 2.5648245*sin_q1_plus_q2 - 0.093*atan_100_dot_q1)
        term2 /= (0.739859392362865*cos_q2**2 - 1)**2

        term3 = -37.0 * (-0.078435*dot_q1**2*sin_q2 - 0.005*dot_q2 - 2.5648245*sin_q1_plus_q2 - 0.14*atan_100_dot_q2)
        term3 /= (0.006152049225*cos_q2**2 - 0.00831516)

        term4 = -6.0 * (0.15687*dot_q1*dot_q2*sin_q2 - 0.005*dot_q1 + 0.078435*dot_q2**2*sin_q2 - 3.494322*sin_q1 - 2.5648245*sin_q1_plus_q2 - 0.093*atan_100_dot_q1)
        term4 /= (0.006152049225*cos_q2**2 - 0.00831516)

        return (term1 + term2 + term3 + term4).view(-1, 1)

    def element_1_2():
        term = -5.96448 * (-0.012518226*cos_q2 - 0.0146832) * sin_q1
        term /= (0.00096507196192575*cos_q2**3 + 0.0011319770574*cos_q2**2 - 0.0013043991492*cos_q2 - 0.00152998944)
        return term.view(-1, 1)

    def element_1_3():
        term = -5.96448 * (0.078435*cos_q2 + 0.0798) * sin_q1
        term /= (0.006152049225*cos_q2**2 - 0.00831516)
        return term.view(-1, 1)

    def element_2_2():
        term1 = 427192.021899475 * (-0.012518226*cos_q2 - 0.0146832) * (-0.006152049225*cos_q2**2 + 0.012518226*cos_q2 + 0.02299836)
        term1 *= (0.15687*dot_q1*dot_q2*sin_q2 - 0.005*dot_q1 + 0.078435*dot_q2**2*sin_q2 - 3.494322*sin_q1 - 2.5648245*sin_q1_plus_q2 - 0.093*atan_100_dot_q1)
        term1 /= (0.630770341738927*cos_q2**3 + 0.739859392362865*cos_q2**2 - 0.852554347826087*cos_q2 - 1)**2

        term2 = (0.078435*cos_q2 + 0.0798) * (-0.078435*dot_q1**2*sin_q2 - 0.005*dot_q2 - 2.5648245*sin_q1_plus_q2 - 0.14*atan_100_dot_q2)
        term2 /= (0.00096507196192575*cos_q2**3 + 0.0011319770574*cos_q2**2 - 0.0013043991492*cos_q2 - 0.00152998944)

        term3 = 427192.021899475 * (-0.006152049225*cos_q2**2 + 0.012518226*cos_q2 + 0.02299836) * (0.01230409845*cos_q2**2 + 0.026950266*cos_q2 + 0.0146832)
        term3 *= (-0.078435*dot_q1**2*sin_q2 - 0.005*dot_q2 - 2.5648245*sin_q1_plus_q2 - 0.14*atan_100_dot_q2)
        term3 /= (0.630770341738927*cos_q2**3 + 0.739859392362865*cos_q2**2 - 0.852554347826087*cos_q2 - 1)**2

        term4 = -0.0798 * (0.15687*dot_q1*dot_q2*sin_q2 - 0.005*dot_q1 + 0.078435*dot_q2**2*sin_q2 - 3.494322*sin_q1 - 2.5648245*sin_q1_plus_q2 - 0.093*atan_100_dot_q1)
        term4 /= (0.00096507196192575*cos_q2**3 + 0.0011319770574*cos_q2**2 - 0.0013043991492*cos_q2 - 0.00152998944)

        return (term1 + term2 + term3 + term4).view(-1, 1)

    def element_2_3():
        term1 = 1154.1484448556 * (-0.15687*cos_q2 - 0.184) * (-0.078435*dot_q1**2*sin_q2 - 0.005*dot_q2 - 2.5648245*sin_q1_plus_q2 - 0.14*atan_100_dot_q2)
        term1 /= (0.739859392362865*cos_q2**2 - 1)**2

        term2 = 1154.1484448556 * (0.078435*cos_q2 + 0.0798) * (0.15687*dot_q1*dot_q2*sin_q2 - 0.005*dot_q1 + 0.078435*dot_q2**2*sin_q2 - 3.494322*sin_q1 - 2.5648245*sin_q1_plus_q2 - 0.093*atan_100_dot_q1)
        term2 /= (0.739859392362865*cos_q2**2 - 1)**2

        term3 = (-0.078435*dot_q1**2*sin_q2 - 0.005*dot_q2 - 2.5648245*sin_q1_plus_q2 - 0.14*atan_100_dot_q2)
        term3 /= (0.006152049225*cos_q2**2 - 0.00831516)

        return (term1 + term2 - term3).view(-1, 1)

    def element_3_2():
        term = -dot_q1 * (-0.012518226*cos_q2 - 0.0146832)
        term /= (0.00096507196192575*cos_q2**3 + 0.0011319770574*cos_q2**2 - 0.0013043991492*cos_q2 - 0.00152998944)
        return term.view(-1, 1)

    def element_3_3():
        term = -dot_q1 * (0.078435*cos_q2 + 0.0798)
        term /= (0.006152049225*cos_q2**2 - 0.00831516)
        return term.view(-1, 1)

    def element_4_2():
        term = -(-0.012518226*cos_q2 - 0.0146832) * atan_100_dot_q1
        term /= (0.00096507196192575*cos_q2**3 + 0.0011319770574*cos_q2**2 - 0.0013043991492*cos_q2 - 0.00152998944)
        return term.view(-1, 1)

    def element_4_3():
        term = -(0.078435*cos_q2 + 0.0798) * atan_100_dot_q1
        term /= (0.006152049225*cos_q2**2 - 0.00831516)
        return term.view(-1, 1)

    def element_5_2():
        term1 = (-0.189*dot_q1**2*sin_q2 - 6.1803*sin_q1_plus_q2) * (0.01230409845*cos_q2**2 + 0.026950266*cos_q2 + 0.0146832)
        term1 /= (0.00096507196192575*cos_q2**3 + 0.0011319770574*cos_q2**2 - 0.0013043991492*cos_q2 - 0.00152998944)

        term2 = (-0.012518226*cos_q2 - 0.0146832) * (0.378*dot_q1*dot_q2*sin_q2 + 0.189*dot_q2**2*sin_q2 - 6.1803*sin_q1_plus_q2)
        term2 /= (0.00096507196192575*cos_q2**3 + 0.0011319770574*cos_q2**2 - 0.0013043991492*cos_q2 - 0.00152998944)

        term3 = 427192.021899475 * (-0.012518226*cos_q2 - 0.0146832) * (-0.00697642382115*cos_q2**3 - 0.00545531112*cos_q2**2 + 0.00314313048*cos_q2)
        term3 *= (0.15687*dot_q1*dot_q2*sin_q2 - 0.005*dot_q1 + 0.078435*dot_q2**2*sin_q2 - 3.494322*sin_q1 - 2.5648245*sin_q1_plus_q2 - 0.093*atan_100_dot_q1)
        term3 /= (0.630770341738927*cos_q2**3 + 0.739859392362865*cos_q2**2 - 0.852554347826087*cos_q2 - 1)**2

        term4 = (0.05929686*cos_q2**2 + 0.0649404*cos_q2) * (-0.078435*dot_q1**2*sin_q2 - 0.005*dot_q2 - 2.5648245*sin_q1_plus_q2 - 0.14*atan_100_dot_q2)
        term4 /= (0.00096507196192575*cos_q2**3 + 0.0011319770574*cos_q2**2 - 0.0013043991492*cos_q2 - 0.00152998944)

        term5 = 427192.021899475 * (0.01230409845*cos_q2**2 + 0.026950266*cos_q2 + 0.0146832) * (-0.00697642382115*cos_q2**3 - 0.00545531112*cos_q2**2 + 0.00314313048*cos_q2)
        term5 *= (-0.078435*dot_q1**2*sin_q2 - 0.005*dot_q2 - 2.5648245*sin_q1_plus_q2 - 0.14*atan_100_dot_q2)
        term5 /= (0.630770341738927*cos_q2**3 + 0.739859392362865*cos_q2**2 - 0.852554347826087*cos_q2 - 1)**2

        term6 = -0.0301644 * (0.15687*dot_q1*dot_q2*sin_q2 - 0.005*dot_q1 + 0.078435*dot_q2**2*sin_q2 - 3.494322*sin_q1 - 2.5648245*sin_q1_plus_q2 - 0.093*atan_100_dot_q1) * cos_q2
        term6 /= (0.00096507196192575*cos_q2**3 + 0.0011319770574*cos_q2**2 - 0.0013043991492*cos_q2 - 0.00152998944)

        return (term1 + term2 + term3 + term4 + term5 + term6).view(-1, 1)

    def element_5_3():
        term1 = (-0.189*dot_q1**2*sin_q2 - 6.1803*sin_q1_plus_q2) * (-0.15687*cos_q2 - 0.184)
        term1 /= (0.006152049225*cos_q2**2 - 0.00831516)

        term2 = -428.805631289602 * (-0.15687*cos_q2 - 0.184) * (-0.078435*dot_q1**2*sin_q2 - 0.005*dot_q2 - 2.5648245*sin_q1_plus_q2 - 0.14*atan_100_dot_q2) * cos_q2**2
        term2 /= (0.739859392362865*cos_q2**2 - 1)**2

        term3 = -428.805631289602 * (0.078435*cos_q2 + 0.0798) * (0.15687*dot_q1*dot_q2*sin_q2 - 0.005*dot_q1 + 0.078435*dot_q2**2*sin_q2 - 3.494322*sin_q1 - 2.5648245*sin_q1_plus_q2 - 0.093*atan_100_dot_q1) * cos_q2**2
        term3 /= (0.739859392362865*cos_q2**2 - 1)**2

        term4 = (0.078435*cos_q2 + 0.0798) * (0.378*dot_q1*dot_q2*sin_q2 + 0.189*dot_q2**2*sin_q2 - 6.1803*sin_q1_plus_q2)
        term4 /= (0.006152049225*cos_q2**2 - 0.00831516)

        term5 = -0.378 * (-0.078435*dot_q1**2*sin_q2 - 0.005*dot_q2 - 2.5648245*sin_q1_plus_q2 - 0.14*atan_100_dot_q2) * cos_q2
        term5 /= (0.006152049225*cos_q2**2 - 0.00831516)

        term6 = 0.189 * (0.15687*dot_q1*dot_q2*sin_q2 - 0.005*dot_q1 + 0.078435*dot_q2**2*sin_q2 - 3.494322*sin_q1 - 2.5648245*sin_q1_plus_q2 - 0.093*atan_100_dot_q1) * cos_q2
        term6 /= (0.006152049225*cos_q2**2 - 0.00831516)

        return (term1 + term2 + term3 + term4 + term5 + term6).view(-1, 1)

    def element_6_2():
        term1 = (-0.1245*dot_q1**2*sin_q2 - 4.07115*sin_q1_plus_q2) * (0.01230409845*cos_q2**2 + 0.026950266*cos_q2 + 0.0146832)
        term1 /= (0.00096507196192575*cos_q2**3 + 0.0011319770574*cos_q2**2 - 0.0013043991492*cos_q2 - 0.00152998944)

        term2 = (-0.0198702*cos_q2 - 0.007182) * (0.15687*dot_q1*dot_q2*sin_q2 - 0.005*dot_q1 + 0.078435*dot_q2**2*sin_q2 - 3.494322*sin_q1 - 2.5648245*sin_q1_plus_q2 - 0.093*atan_100_dot_q1)
        term2 /= (0.00096507196192575*cos_q2**3 + 0.0011319770574*cos_q2**2 - 0.0013043991492*cos_q2 - 0.00152998944)

        term3 = (-0.012518226*cos_q2 - 0.0146832) * (0.249*dot_q1*dot_q2*sin_q2 + 0.1245*dot_q2**2*sin_q2 - 2.943*sin_q1 - 4.07115*sin_q1_plus_q2)
        term3 /= (0.00096507196192575*cos_q2**3 + 0.0011319770574*cos_q2**2 - 0.0013043991492*cos_q2 - 0.00152998944)

        term4 = 427192.021899475 * (-0.012518226*cos_q2 - 0.0146832) * (-0.004595580771075*cos_q2**3 - 0.00414726239025*cos_q2**2 + 0.00319711518*cos_q2 + 0.0020698524)
        term4 *= (0.15687*dot_q1*dot_q2*sin_q2 - 0.005*dot_q1 + 0.078435*dot_q2**2*sin_q2 - 3.494322*sin_q1 - 2.5648245*sin_q1_plus_q2 - 0.093*atan_100_dot_q1)
        term4 /= (0.630770341738927*cos_q2**3 + 0.739859392362865*cos_q2**2 - 0.852554347826087*cos_q2 - 1)**2

        term5 = 427192.021899475 * (0.01230409845*cos_q2**2 + 0.026950266*cos_q2 + 0.0146832) * (-0.078435*dot_q1**2*sin_q2 - 0.005*dot_q2 - 2.5648245*sin_q1_plus_q2 - 0.14*atan_100_dot_q2)
        term5 *= (-0.004595580771075*cos_q2**3 - 0.00414726239025*cos_q2**2 + 0.00319711518*cos_q2 + 0.0020698524)
        term5 /= (0.630770341738927*cos_q2**3 + 0.739859392362865*cos_q2**2 - 0.852554347826087*cos_q2 - 1)**2

        term6 = (0.03906063*cos_q2**2 + 0.04983735*cos_q2 + 0.007182) * (-0.078435*dot_q1**2*sin_q2 - 0.005*dot_q2 - 2.5648245*sin_q1_plus_q2 - 0.14*atan_100_dot_q2)
        term6 /= (0.00096507196192575*cos_q2**3 + 0.0011319770574*cos_q2**2 - 0.0013043991492*cos_q2 - 0.00152998944)

        return (term1 + term2 + term3 + term4 + term5 + term6).view(-1, 1)

    def element_6_3():
        term1 = 14463.0130934286 * (0.007182 - 0.019530315*cos_q2**2) * (-0.15687*cos_q2 - 0.184) * (-0.078435*dot_q1**2*sin_q2 - 0.005*dot_q2 - 2.5648245*sin_q1_plus_q2 - 0.14*atan_100_dot_q2)
        term1 /= (0.739859392362865*cos_q2**2 - 1)**2

        term2 = 14463.0130934286 * (0.007182 - 0.019530315*cos_q2**2) * (0.078435*cos_q2 + 0.0798)
        term2 *= (0.15687*dot_q1*dot_q2*sin_q2 - 0.005*dot_q1 + 0.078435*dot_q2**2*sin_q2 - 3.494322*sin_q1 - 2.5648245*sin_q1_plus_q2 - 0.093*atan_100_dot_q1)
        term2 /= (0.739859392362865*cos_q2**2 - 1)**2

        term3 = (-0.1245*dot_q1**2*sin_q2 - 4.07115*sin_q1_plus_q2) * (-0.15687*cos_q2 - 0.184)
        term3 /= (0.006152049225*cos_q2**2 - 0.00831516)

        term4 = (-0.249*cos_q2 - 0.09) * (-0.078435*dot_q1**2*sin_q2 - 0.005*dot_q2 - 2.5648245*sin_q1_plus_q2 - 0.14*atan_100_dot_q2)
        term4 /= (0.006152049225*cos_q2**2 - 0.00831516)

        term5 = (0.078435*cos_q2 + 0.0798) * (0.249*dot_q1*dot_q2*sin_q2 + 0.1245*dot_q2**2*sin_q2 - 2.943*sin_q1 - 4.07115*sin_q1_plus_q2)
        term5 /= (0.006152049225*cos_q2**2 - 0.00831516)

        term6 = 0.1245 * (0.15687*dot_q1*dot_q2*sin_q2 - 0.005*dot_q1 + 0.078435*dot_q2**2*sin_q2 - 3.494322*sin_q1 - 2.5648245*sin_q1_plus_q2 - 0.093*atan_100_dot_q1) * cos_q2
        term6 /= (0.006152049225*cos_q2**2 - 0.00831516)

        return (term1 + term2 + term3 + term4 + term5 + term6).view(-1, 1)

    def element_7_2():
        term1 = (-0.15687*cos_q2 - 0.2638) * (0.15687*dot_q1*dot_q2*sin_q2 - 0.005*dot_q1 + 0.078435*dot_q2**2*sin_q2 - 3.494322*sin_q1 - 2.5648245*sin_q1_plus_q2 - 0.093*atan_100_dot_q1)
        term1 /= (0.00096507196192575*cos_q2**3 + 0.0011319770574*cos_q2**2 - 0.0013043991492*cos_q2 - 0.00152998944)

        term2 = 427192.021899475 * (-0.012518226*cos_q2 - 0.0146832) * (-0.006152049225*cos_q2**2 + 0.016345854*cos_q2 + 0.02748796)
        term2 *= (0.15687*dot_q1*dot_q2*sin_q2 - 0.005*dot_q1 + 0.078435*dot_q2**2*sin_q2 - 3.494322*sin_q1 - 2.5648245*sin_q1_plus_q2 - 0.093*atan_100_dot_q1)
        term2 /= (0.630770341738927*cos_q2**3 + 0.739859392362865*cos_q2**2 - 0.852554347826087*cos_q2 - 1)**2

        term3 = (0.235305*cos_q2 + 0.2638) * (-0.078435*dot_q1**2*sin_q2 - 0.005*dot_q2 - 2.5648245*sin_q1_plus_q2 - 0.14*atan_100_dot_q2)
        term3 /= (0.00096507196192575*cos_q2**3 + 0.0011319770574*cos_q2**2 - 0.0013043991492*cos_q2 - 0.00152998944)

        term4 = 427192.021899475 * (-0.006152049225*cos_q2**2 + 0.016345854*cos_q2 + 0.02748796) * (0.01230409845*cos_q2**2 + 0.026950266*cos_q2 + 0.0146832)
        term4 *= (-0.078435*dot_q1**2*sin_q2 - 0.005*dot_q2 - 2.5648245*sin_q1_plus_q2 - 0.14*atan_100_dot_q2)
        term4 /= (0.630770341738927*cos_q2**3 + 0.739859392362865*cos_q2**2 - 0.852554347826087*cos_q2 - 1)**2

        return (term1 + term2 + term3 + term4).view(-1, 1)

    def element_7_3():
        term1 = 1507.04596433526 * (-0.15687*cos_q2 - 0.184) * (-0.078435*dot_q1**2*sin_q2 - 0.005*dot_q2 - 2.5648245*sin_q1_plus_q2 - 0.14*atan_100_dot_q2)
        term1 /= (0.739859392362865*cos_q2**2 - 1)**2

        term2 = 1507.04596433526 * (0.078435*cos_q2 + 0.0798) * (0.15687*dot_q1*dot_q2*sin_q2 - 0.005*dot_q1 + 0.078435*dot_q2**2*sin_q2 - 3.494322*sin_q1 - 2.5648245*sin_q1_plus_q2 - 0.093*atan_100_dot_q1)
        term2 /= (0.739859392362865*cos_q2**2 - 1)**2

        term3 = (-0.078435*dot_q1**2*sin_q2 - 0.005*dot_q2 - 2.5648245*sin_q1_plus_q2 - 0.14*atan_100_dot_q2)
        term3 /= (0.006152049225*cos_q2**2 - 0.00831516)

        term4 = (0.15687*dot_q1*dot_q2*sin_q2 - 0.005*dot_q1 + 0.078435*dot_q2**2*sin_q2 - 3.494322*sin_q1 - 2.5648245*sin_q1_plus_q2 - 0.093*atan_100_dot_q1)
        term4 /= (0.006152049225*cos_q2**2 - 0.00831516)

        return (term1 + term2 - term3 + term4).view(-1, 1)

    def element_8_2():
        term = -dot_q2 * (0.01230409845*cos_q2**2 + 0.026950266*cos_q2 + 0.0146832)
        term /= (0.00096507196192575*cos_q2**3 + 0.0011319770574*cos_q2**2 - 0.0013043991492*cos_q2 - 0.00152998944)
        return term.view(-1, 1)

    def element_8_3():
        term = -dot_q2 * (-0.15687*cos_q2 - 0.184)
        term /= (0.006152049225*cos_q2**2 - 0.00831516)
        return term.view(-1, 1)

    def element_9_2():
        term = -(0.01230409845*cos_q2**2 + 0.026950266*cos_q2 + 0.0146832) * torch.atan(100*dot_q2)
        term /= (0.00096507196192575*cos_q2**3 + 0.0011319770574*cos_q2**2 - 0.0013043991492*cos_q2 - 0.00152998944)
        return term.view(-1, 1)

    def element_9_3():
        term = -(-0.15687*cos_q2 - 0.184) * torch.atan(100*dot_q2)
        term /= (0.006152049225*cos_q2**2 - 0.00831516)
        return term.view(-1, 1)

    # 使用这些函数计算矩阵 self.dadp_x
    dadp_x = torch.zeros((10, 4, x.size(0)), device=x.device)

    # 填充矩阵元素
    dadp_x[0, 2, :] = element_0_2().view(-1)
    dadp_x[0, 3, :] = element_0_3().view(-1)
    dadp_x[1, 2, :] = element_1_2().view(-1)
    dadp_x[1, 3, :] = element_1_3().view(-1)
    dadp_x[2, 2, :] = element_2_2().view(-1)
    dadp_x[2, 3, :] = element_2_3().view(-1)
    dadp_x[3, 2, :] = element_3_2().view(-1)
    dadp_x[3, 3, :] = element_3_3().view(-1)
    dadp_x[4, 2, :] = element_4_2().view(-1)
    dadp_x[4, 3, :] = element_4_3().view(-1)
    dadp_x[5, 2, :] = element_5_2().view(-1)
    dadp_x[5, 3, :] = element_5_3().view(-1)
    dadp_x[6, 2, :] = element_6_2().view(-1)
    dadp_x[6, 3, :] = element_6_3().view(-1)
    dadp_x[7, 2, :] = element_7_2().view(-1)
    dadp_x[7, 3, :] = element_7_3().view(-1)
    dadp_x[8, 2, :] = element_8_2().view(-1)
    dadp_x[8, 3, :] = element_8_3().view(-1)
    dadp_x[9, 2, :] = element_9_2().view(-1)
    dadp_x[9, 3, :] = element_9_3().view(-1)


    # 继续计算并填充 dadp_x 的其他元素
    return dadp_x.permute(2, 0, 1)


def compute_dBdp_x_matrix(x):
    device = x.device
    q2 = x[:, 0, :].view(-1, 1, 1), x[:, 1, :].view(-1, 1, 1)
    cos_q2 = torch.cos(q2).to(device)
    # sin_q2 = torch.sin(q2).to(device)
    # sin_q1 = torch.sin(q1).to(device)
    # sin_q1_plus_q2 = torch.sin(q1 + q2).to(device)
    # atan_100_dot_q1 = torch.atan(100 * dot_q1).to(device)
    # atan_100_dot_q2 = torch.atan(100 * dot_q2).to(device)

    dBdp_x = torch.zeros((10, 4, 2, x.shape[0]), device=device)

    def element_0_2_0():
        term1 = (-5.64732 * cos_q2 - 9.5766)
        term1 /= (0.00096507196192575 * cos_q2 ** 3 + 0.0011319770574 * cos_q2 ** 2 - 0.0013043991492 * cos_q2 - 0.00152998944)
        term2 = 427192.021899475 * (-0.012518226 * cos_q2 - 0.0146832)
        term2 *= (0.805918448475 * cos_q2 ** 2 + 2.864791314 * cos_q2 + 2.24595372)
        term2 /= (0.630770341738927 * cos_q2 ** 3 + 0.739859392362865 * cos_q2 ** 2 - 0.852554347826087 * cos_q2 - 1) ** 2
        return term1 + term2

    def element_0_2_1():
        term1 = (1.960875 * cos_q2 + 1.8486)
        term1 /= (0.00096507196192575 * cos_q2 ** 3 + 0.0011319770574 * cos_q2 ** 2 - 0.0013043991492 * cos_q2 - 0.00152998944)
        term2 = 427192.021899475 * (0.01230409845 * cos_q2 ** 2 + 0.026950266 * cos_q2 + 0.0146832)
        term2 *= (0.805918448475 * cos_q2 ** 2 + 2.864791314 * cos_q2 + 2.24595372)
        term2 /= (0.630770341738927 * cos_q2 ** 3 + 0.739859392362865 * cos_q2 ** 2 - 0.852554347826087 * cos_q2 - 1) ** 2
        return term1 + term2

    def element_0_3_0():
        term1 = 14463.0130934286 * (0.078435 * cos_q2 + 0.0798)
        term1 *= (6.58854 * cos_q2 + 10.5342)
        term1 /= (0.739859392362865 * cos_q2 ** 2 - 1) ** 2
        term2 = -6.0 / (0.006152049225 * cos_q2 ** 2 - 0.00831516)
        return term1 + term2

    def element_0_3_1():
        term1 = 14463.0130934286 * (-0.15687 * cos_q2 - 0.184)
        term1 *= (6.58854 * cos_q2 + 10.5342)
        term1 /= (0.739859392362865 * cos_q2 ** 2 - 1) ** 2
        term2 = -37.0 / (0.006152049225 * cos_q2 ** 2 - 0.00831516)
        return term1 + term2

    def element_2_2_0():
        term1 = 427192.021899475 * (-0.012518226 * cos_q2 - 0.0146832)
        term1 *= (-0.006152049225 * cos_q2 ** 2 + 0.012518226 * cos_q2 + 0.02299836)
        term1 /= (0.630770341738927 * cos_q2 ** 3 + 0.739859392362865 * cos_q2 ** 2 - 0.852554347826087 * cos_q2 - 1) ** 2
        term2 = -0.0798 / (0.00096507196192575 * cos_q2 ** 3 + 0.0011319770574 * cos_q2 ** 2 - 0.0013043991492 * cos_q2 - 0.00152998944)
        return term1 + term2

    def element_2_2_1():
        term1 = (0.078435 * cos_q2 + 0.0798)
        term1 /= (0.00096507196192575 * cos_q2 ** 3 + 0.0011319770574 * cos_q2 ** 2 - 0.0013043991492 * cos_q2 - 0.00152998944)
        term2 = 427192.021899475 * (-0.006152049225 * cos_q2 ** 2 + 0.012518226 * cos_q2 + 0.02299836)
        term2 *= (0.01230409845 * cos_q2 ** 2 + 0.026950266 * cos_q2 + 0.0146832)
        term2 /= (0.630770341738927 * cos_q2 ** 3 + 0.739859392362865 * cos_q2 ** 2 - 0.852554347826087 * cos_q2 - 1) ** 2
        return term1 + term2

    def element_2_3_0():
        term1 = 1154.1484448556 * (0.078435 * cos_q2 + 0.0798)
        term1 /= (0.739859392362865 * cos_q2 ** 2 - 1) ** 2
        return term1

    def element_2_3_1():
        term1 = 1154.1484448556 * (-0.15687 * cos_q2 - 0.184)
        term1 /= (0.739859392362865 * cos_q2 ** 2 - 1) ** 2
        term2 = -1 / (0.006152049225 * cos_q2 ** 2 - 0.00831516)
        return term1 + term2

    def element_5_2_0():
        term1 = 427192.021899475 * (-0.012518226 * cos_q2 - 0.0146832)
        term1 *= (-0.00697642382115 * cos_q2 ** 3 - 0.00545531112 * cos_q2 ** 2 + 0.00314313048 * cos_q2)
        term1 /= (0.630770341738927 * cos_q2 ** 3 + 0.739859392362865 * cos_q2 ** 2 - 0.852554347826087 * cos_q2 - 1) ** 2
        term2 = -0.0301644 * cos_q2
        term2 /= (0.00096507196192575 * cos_q2 ** 3 + 0.0011319770574 * cos_q2 ** 2 - 0.0013043991492 * cos_q2 - 0.00152998944)
        return term1 + term2

    def element_5_2_1():
        term1 = (0.05929686 * cos_q2 ** 2 + 0.0649404 * cos_q2)
        term1 /= (0.00096507196192575 * cos_q2 ** 3 + 0.0011319770574 * cos_q2 ** 2 - 0.0013043991492 * cos_q2 - 0.00152998944)
        term2 = 427192.021899475 * (0.01230409845 * cos_q2 ** 2 + 0.026950266 * cos_q2 + 0.0146832)
        term2 *= (-0.00697642382115 * cos_q2 ** 3 - 0.00545531112 * cos_q2 ** 2 + 0.00314313048 * cos_q2)
        term2 /= (0.630770341738927 * cos_q2 ** 3 + 0.739859392362865 * cos_q2 ** 2 - 0.852554347826087 * cos_q2 - 1) ** 2
        return term1 + term2

    def element_5_3_0():
        term1 = -428.805631289602 * (0.078435 * cos_q2 + 0.0798) * cos_q2 ** 2
        term1 /= (0.739859392362865 * cos_q2 ** 2 - 1) ** 2
        term2 = 0.189 * cos_q2
        term2 /= (0.006152049225 * cos_q2 ** 2 - 0.00831516)
        return term1 + term2

    def element_5_3_1():
        term1 = -428.805631289602 * (-0.15687 * cos_q2 - 0.184) * cos_q2 ** 2
        term1 /= (0.739859392362865 * cos_q2 ** 2 - 1) ** 2
        term2 = -0.378 * cos_q2
        term2 /= (0.006152049225 * cos_q2 ** 2 - 0.00831516)
        return term1 + term2

    def element_6_2_0():
        term1 = (-0.0198702 * cos_q2 - 0.007182)
        term1 /= (0.00096507196192575 * cos_q2 ** 3 + 0.0011319770574 * cos_q2 ** 2 - 0.0013043991492 * cos_q2 - 0.00152998944)
        term2 = 427192.021899475 * (-0.012518226 * cos_q2 - 0.0146832)
        term2 *= (-0.004595580771075 * cos_q2 ** 3 - 0.00414726239025 * cos_q2 ** 2 + 0.00319711518 * cos_q2 + 0.0020698524)
        term2 /= (0.630770341738927 * cos_q2 ** 3 + 0.739859392362865 * cos_q2 ** 2 - 0.852554347826087 * cos_q2 - 1) ** 2
        return term1 + term2

    def element_6_2_1():
        term1 = 427192.021899475 * (0.01230409845 * cos_q2 ** 2 + 0.026950266 * cos_q2 + 0.0146832)
        term1 *= (-0.004595580771075 * cos_q2 ** 3 - 0.00414726239025 * cos_q2 ** 2 + 0.00319711518 * cos_q2 + 0.0020698524)
        term1 /= (0.630770341738927 * cos_q2 ** 3 + 0.739859392362865 * cos_q2 ** 2 - 0.852554347826087 * cos_q2 - 1) ** 2
        term2 = (0.03906063 * cos_q2 ** 2 + 0.04983735 * cos_q2 + 0.007182)
        term2 /= (0.00096507196192575 * cos_q2 ** 3 + 0.0011319770574 * cos_q2 ** 2 - 0.0013043991492 * cos_q2 - 0.00152998944)
        return term1 + term2

    def element_6_3_0():
        term1 = 14463.0130934286 * (0.007182 - 0.019530315 * cos_q2 ** 2)
        term1 *= (0.078435 * cos_q2 + 0.0798)
        term1 /= (0.739859392362865 * cos_q2 ** 2 - 1) ** 2
        term2 = 0.1245 * cos_q2
        term2 /= (0.006152049225 * cos_q2 ** 2 - 0.00831516)
        return term1 + term2

    def element_6_3_1():
        term1 = 14463.0130934286 * (0.007182 - 0.019530315 * cos_q2 ** 2)
        term1 *= (-0.15687 * cos_q2 - 0.184)
        term1 /= (0.739859392362865 * cos_q2 ** 2 - 1) ** 2
        term2 = (-0.249 * cos_q2 - 0.09)
        term2 /= (0.006152049225 * cos_q2 ** 2 - 0.00831516)
        return term1 + term2

    def element_7_2_0():
        term1 = (-0.15687 * cos_q2 - 0.2638)
        term1 /= (0.00096507196192575 * cos_q2 ** 3 + 0.0011319770574 * cos_q2 ** 2 - 0.0013043991492 * cos_q2 - 0.00152998944)
        term2 = 427192.021899475 * (-0.012518226 * cos_q2 - 0.0146832)
        term2 *= (-0.006152049225 * cos_q2 ** 2 + 0.016345854 * cos_q2 + 0.02748796)
        term2 /= (0.630770341738927 * cos_q2 ** 3 + 0.739859392362865 * cos_q2 ** 2 - 0.852554347826087 * cos_q2 - 1) ** 2
        return term1 + term2

    def element_7_2_1():
        term1 = (0.235305 * cos_q2 + 0.2638)
        term1 /= (0.00096507196192575 * cos_q2 ** 3 + 0.0011319770574 * cos_q2 ** 2 - 0.0013043991492 * cos_q2 - 0.00152998944)
        term2 = 427192.021899475 * (-0.006152049225 * cos_q2 ** 2 + 0.016345854 * cos_q2 + 0.02748796)
        term2 *= (0.01230409845 * cos_q2 ** 2 + 0.026950266 * cos_q2 + 0.0146832)
        term2 /= (0.630770341738927 * cos_q2 ** 3 + 0.739859392362865 * cos_q2 ** 2 - 0.852554347826087 * cos_q2 - 1) ** 2
        return term1 + term2

    def element_7_3_0():
        term1 = 1507.04596433526 * (0.078435 * cos_q2 + 0.0798)
        term1 /= (0.739859392362865 * cos_q2 ** 2 - 1) ** 2
        term2 = 1 / (0.006152049225 * cos_q2 ** 2 - 0.00831516)
        return term1 + term2

    def element_7_3_1():
        term1 = 1507.04596433526 * (-0.15687 * cos_q2 - 0.184)
        term1 /= (0.739859392362865 * cos_q2 ** 2 - 1) ** 2
        term2 = -1 / (0.006152049225 * cos_q2 ** 2 - 0.00831516)
        return term1 + term2

    dBdp_x[0, 2, 0, :] = element_0_2_0().view(-1)
    dBdp_x[0, 2, 1, :] = element_0_2_1().view(-1)
    dBdp_x[0, 3, 0, :] = element_0_3_0().view(-1)
    dBdp_x[0, 3, 1, :] = element_0_3_1().view(-1)
    dBdp_x[2, 2, 0, :] = element_2_2_0().view(-1)
    dBdp_x[2, 2, 1, :] = element_2_2_1().view(-1)
    dBdp_x[2, 3, 0, :] = element_2_3_0().view(-1)
    dBdp_x[2, 3, 1, :] = element_2_3_1().view(-1)
    dBdp_x[5, 2, 0, :] = element_5_2_0().view(-1)
    dBdp_x[5, 2, 1, :] = element_5_2_1().view(-1)
    dBdp_x[5, 3, 0, :] = element_5_3_0().view(-1)
    dBdp_x[5, 3, 1, :] = element_5_3_1().view(-1)
    dBdp_x[6, 2, 0, :] = element_6_2_0().view(-1)
    dBdp_x[6, 2, 1, :] = element_6_2_1().view(-1)
    dBdp_x[6, 3, 0, :] = element_6_3_0().view(-1)
    dBdp_x[6, 3, 1, :] = element_6_3_1().view(-1)
    dBdp_x[7, 2, 0, :] = element_7_2_0().view(-1)
    dBdp_x[7, 2, 1, :] = element_7_2_1().view(-1)
    dBdp_x[7, 3, 0, :] = element_7_3_0().view(-1)
    dBdp_x[7, 3, 1, :] = element_7_3_1().view(-1)

    return dBdp_x.permute(3, 0, 1, 2)


def compute_dadxT_matrix(x, theta):
    device = x.device
    q1, q2 = x[:, 0, :].view(-1, 1, 1), x[:, 1, :].view(-1, 1, 1)
    dot_q1, dot_q2 = x[:, 2, :].view(-1, 1, 1), x[:, 3, :].view(-1, 1, 1)
    Ir = theta[:,0,0].view(-1, 1, 1)
    r1 = theta[:,1,0].view(-1, 1, 1)
    I1 = theta[:,2,0].view(-1, 1, 1)
    b1 = theta[:,3,0].view(-1, 1, 1)
    cf1 = theta[:,4,0].view(-1, 1, 1)
    r2 = theta[:,5,0].view(-1, 1, 1)
    m2 = theta[:,6,0].view(-1, 1, 1)
    I2 = theta[:,7,0].view(-1, 1, 1)
    b2 = theta[:,8,0].view(-1, 1, 1)
    cf2 = theta[:,9,0].view(-1, 1, 1)

    sin_q1 = torch.sin(q1)
    cos_q1 = torch.cos(q1)
    sin_q2 = torch.sin(q2)
    cos_q2 = torch.cos(q2)
    sin_q1_q2 = torch.sin(q1 + q2)
    cos_q1_q2 = torch.cos(q1 + q2)
    atan_100_dot_q1 = torch.atan(100 * dot_q1)
    atan_100_dot_q2 = torch.atan(100 * dot_q2)

    dadxT = torch.zeros((4, 4, x.shape[0]), device=device)

    # Define each element of the matrix according to the given symbolic expressions
    def dadxT_element_0_2():
        term1 = -9.81 * m2 * r2 * (I1 * I2 - 6.0 * I1 * Ir + 0.3 * I1 * m2 * r2 * cos_q2 + I2**2 + 31.0 * I2 * Ir + 0.9 * I2 * m2 * r2 * cos_q2 + 0.09 * I2 * m2 - 222.0 * Ir**2 + 7.5 * Ir * m2 * r2 * cos_q2 - 0.54 * Ir * m2 + 0.18 * m2**2 * r2**2 * cos_q2**2 + 0.027 * m2**2 * r2 * cos_q2) * cos_q1_q2
        term1 /= (-I1**2 * I2 - 36.0 * I1**2 * Ir - I1 * I2**2 - 158.0 * I1 * I2 * Ir - 0.6 * I1 * I2 * m2 * r2 * cos_q2 - 0.18 * I1 * I2 * m2 - 2628.0 * I1 * Ir**2 - 46.8 * I1 * Ir * m2 * r2 * cos_q2 - 6.48 * I1 * Ir * m2 + 0.09 * I1 * m2**2 * r2**2 * cos_q2**2 - 85.0 * I2**2 * Ir - 0.09 * I2**2 * m2 - 4441.0 * I2 * Ir**2 - 76.2 * I2 * Ir * m2 * r2 * cos_q2 - 14.22 * I2 * Ir * m2 + 0.09 * I2 * m2**2 * r2**2 * cos_q2**2 - 0.054 * I2 * m2**2 * r2 * cos_q2 - 0.0081 * I2 * m2**2 - 47952.0 * Ir**3 - 1710.0 * Ir**2 * m2 * r2 * cos_q2 - 236.52 * Ir**2 * m2 - 11.79 * Ir * m2**2 * r2**2 * cos_q2**2 - 4.212 * Ir * m2**2 * r2 * cos_q2 - 0.2916 * Ir * m2**2 + 0.054 * m2**3 * r2**3 * cos_q2**3 + 0.0081 * m2**3 * r2**2 * cos_q2**2)
        term2 = (-9.81 * m2 * (r2 * cos_q1_q2 + 0.3 * cos_q1) - 5.96448 * r1 * cos_q1) * (-I1 * I2 - 36.0 * I1 * Ir - I2**2 - 73.0 * I2 * Ir - 0.6 * I2 * m2 * r2 * cos_q2 - 0.09 * I2 * m2 - 1332.0 * Ir**2 - 21.6 * Ir * m2 * r2 * cos_q2 - 3.24 * Ir * m2)
        term2 /= (-I1**2 * I2 - 36.0 * I1**2 * Ir - I1 * I2**2 - 158.0 * I1 * I2 * Ir - 0.6 * I1 * I2 * m2 * r2 * cos_q2 - 0.18 * I1 * I2 * m2 - 2628.0 * I1 * Ir**2 - 46.8 * I1 * Ir * m2 * r2 * cos_q2 - 6.48 * I1 * Ir * m2 + 0.09 * I1 * m2**2 * r2**2 * cos_q2**2 - 85.0 * I2**2 * Ir - 0.09 * I2**2 * m2 - 4441.0 * I2 * Ir**2 - 76.2 * I2 * Ir * m2 * r2 * cos_q2 - 14.22 * I2 * Ir * m2 + 0.09 * I2 * m2**2 * r2**2 * cos_q2**2 - 0.054 * I2 * m2**2 * r2 * cos_q2 - 0.0081 * I2 * m2**2 - 47952.0 * Ir**3 - 1710.0 * Ir**2 * m2 * r2 * cos_q2 - 236.52 * Ir**2 * m2 - 11.79 * Ir * m2**2 * r2**2 * cos_q2**2 - 4.212 * Ir * m2**2 * r2 * cos_q2 - 0.2916 * Ir * m2**2 + 0.054 * m2**3 * r2**3 * cos_q2**3 + 0.0081 * m2**3 * r2**2 * cos_q2**2)
        return term1 + term2

    def dadxT_element_0_3():
        term1 = -9.81 * m2 * r2 * (-I1 - I2 - 37.0 * Ir - 0.6 * m2 * r2 * cos_q2 - 0.09 * m2) * cos_q1_q2
        term1 /= (-I1 * I2 - 36.0 * I1 * Ir - 85.0 * I2 * Ir - 0.09 * I2 * m2 - 1296.0 * Ir**2 - 25.2 * Ir * m2 * r2 * cos_q2 - 3.24 * Ir * m2 + 0.09 * m2**2 * r2**2 * cos_q2**2)
        term2 = (-9.81 * m2 * (r2 * cos_q1_q2 + 0.3 * cos_q1) - 5.96448 * r1 * cos_q1) * (I2 - 6.0 * Ir + 0.3 * m2 * r2 * cos_q2)
        term2 /= (-I1 * I2 - 36.0 * I1 * Ir - 85.0 * I2 * Ir - 0.09 * I2 * m2 - 1296.0 * Ir**2 - 25.2 * Ir * m2 * r2 * cos_q2 - 3.24 * Ir * m2 + 0.09 * m2**2 * r2**2 * cos_q2**2)
        return term1 + term2

    def dadxT_element_1_2():
        term1 = (0.6 * I2 * m2 * r2 * sin_q2 + 21.6 * Ir * m2 * r2 * sin_q2) * (0.6 * dot_q1 * dot_q2 * m2 * r2 * sin_q2 - dot_q1 * b1 + 0.3 * dot_q2**2 * m2 * r2 * sin_q2 - cf1 * atan_100_dot_q1 - 9.81 * m2 * (r2 * sin_q1_q2 + 0.3 * sin_q1) - 5.96448 * r1 * sin_q1)
        term1 /= (-I1**2 * I2 - 36.0 * I1**2 * Ir - I1 * I2**2 - 158.0 * I1 * I2 * Ir - 0.6 * I1 * I2 * m2 * r2 * cos_q2 - 0.18 * I1 * I2 * m2 - 2628.0 * I1 * Ir**2 - 46.8 * I1 * Ir * m2 * r2 * cos_q2 - 6.48 * I1 * Ir * m2 + 0.09 * I1 * m2**2 * r2**2 * cos_q2**2 - 85.0 * I2**2 * Ir - 0.09 * I2**2 * m2 - 4441.0 * I2 * Ir**2 - 76.2 * I2 * Ir * m2 * r2 * cos_q2 - 14.22 * I2 * Ir * m2 + 0.09 * I2 * m2**2 * r2**2 * cos_q2**2 - 0.054 * I2 * m2**2 * r2 * cos_q2 - 0.0081 * I2 * m2**2 - 47952.0 * Ir**3 - 1710.0 * Ir**2 * m2 * r2 * cos_q2 - 236.52 * Ir**2 * m2 - 11.79 * Ir * m2**2 * r2**2 * cos_q2**2 - 4.212 * Ir * m2**2 * r2 * cos_q2 - 0.2916 * Ir * m2**2 + 0.054 * m2**3 * r2**3 * cos_q2**3 + 0.0081 * m2**3 * r2**2 * cos_q2**2)
        term2 = (-0.3 * dot_q1**2 * m2 * r2 * cos_q2 - 9.81 * m2 * r2 * cos_q1_q2) * (I1 * I2 - 6.0 * I1 * Ir + 0.3 * I1 * m2 * r2 * cos_q2 + I2**2 + 31.0 * I2 * Ir + 0.9 * I2 * m2 * r2 * cos_q2 + 0.09 * I2 * m2 - 222.0 * Ir**2 + 7.5 * Ir * m2 * r2 * cos_q2 - 0.54 * Ir * m2 + 0.18 * m2**2 * r2**2 * cos_q2**2 + 0.027 * m2**2 * r2 * cos_q2)
        term2 /= (-I1**2 * I2 - 36.0 * I1**2 * Ir - I1 * I2**2 - 158.0 * I1 * I2 * Ir - 0.6 * I1 * I2 * m2 * r2 * cos_q2 - 0.18 * I1 * I2 * m2 - 2628.0 * I1 * Ir**2 - 46.8 * I1 * Ir * m2 * r2 * cos_q2 - 6.48 * I1 * Ir * m2 + 0.09 * I1 * m2**2 * r2**2 * cos_q2**2 - 85.0 * I2**2 * Ir - 0.09 * I2**2 * m2 - 4441.0 * I2 * Ir**2 - 76.2 * I2 * Ir * m2 * r2 * cos_q2 - 14.22 * I2 * Ir * m2 + 0.09 * I2 * m2**2 * r2**2 * cos_q2**2 - 0.054 * I2 * m2**2 * r2 * cos_q2 - 0.0081 * I2 * m2**2 - 47952.0 * Ir**3 - 1710.0 * Ir**2 * m2 * r2 * cos_q2 - 236.52 * Ir**2 * m2 - 11.79 * Ir * m2**2 * r2**2 * cos_q2**2 - 4.212 * Ir * m2**2 * r2 * cos_q2 - 0.2916 * Ir * m2**2 + 0.054 * m2**3 * r2**3 * cos_q2**3 + 0.0081 * m2**3 * r2**2 * cos_q2**2)
        term3 = (0.6 * dot_q1 * dot_q2 * m2 * r2 * cos_q2 + 0.3 * dot_q2**2 * m2 * r2 * cos_q2 - 9.81 * m2 * r2 * cos_q1_q2) * (-I1 * I2 - 36.0 * I1 * Ir - I2**2 - 73.0 * I2 * Ir - 0.6 * I2 * m2 * r2 * cos_q2 - 0.09 * I2 * m2 - 1332.0 * Ir**2 - 21.6 * Ir * m2 * r2 * cos_q2 - 3.24 * Ir * m2)
        term3 /= (-I1**2 * I2 - 36.0 * I1**2 * Ir - I1 * I2**2 - 158.0 * I1 * I2 * Ir - 0.6 * I1 * I2 * m2 * r2 * cos_q2 - 0.18 * I1 * I2 * m2 - 2628.0 * I1 * Ir**2 - 46.8 * I1 * Ir * m2 * r2 * cos_q2 - 6.48 * I1 * Ir * m2 + 0.09 * I1 * m2**2 * r2**2 * cos_q2**2 - 85.0 * I2**2 * Ir - 0.09 * I2**2 * m2 - 4441.0 * I2 * Ir**2 - 76.2 * I2 * Ir * m2 * r2 * cos_q2 - 14.22 * I2 * Ir * m2 + 0.09 * I2 * m2**2 * r2**2 * cos_q2**2 - 0.054 * I2 * m2**2 * r2 * cos_q2 - 0.0081 * I2 * m2**2 - 47952.0 * Ir**3 - 1710.0 * Ir**2 * m2 * r2 * cos_q2 - 236.52 * Ir**2 * m2 - 11.79 * Ir * m2**2 * r2**2 * cos_q2**2 - 4.212 * Ir * m2**2 * r2 * cos_q2 - 0.2916 * Ir * m2**2 + 0.054 * m2**3 * r2**3 * cos_q2**3 + 0.0081 * m2**3 * r2**2 * cos_q2**2)
        term4 = (-0.3 * dot_q1**2 * m2 * r2 * sin_q2 - dot_q2 * b2 - cf2 * atan_100_dot_q2 - 9.81 * m2 * r2 * sin_q1_q2) * (-0.3 * I1 * m2 * r2 * sin_q2 - 0.9 * I2 * m2 * r2 * sin_q2 - 7.5 * Ir * m2 * r2 * sin_q2 - 0.36 * m2**2 * r2**2 * sin_q2 * cos_q2 - 0.027 * m2**2 * r2 * sin_q2)
        term4 /= (-I1**2 * I2 - 36.0 * I1**2 * Ir - I1 * I2**2 - 158.0 * I1 * I2 * Ir - 0.6 * I1 * I2 * m2 * r2 * cos_q2 - 0.18 * I1 * I2 * m2 - 2628.0 * I1 * Ir**2 - 46.8 * I1 * Ir * m2 * r2 * cos_q2 - 6.48 * I1 * Ir * m2 + 0.09 * I1 * m2**2 * r2**2 * cos_q2**2 - 85.0 * I2**2 * Ir - 0.09 * I2**2 * m2 - 4441.0 * I2 * Ir**2 - 76.2 * I2 * Ir * m2 * r2 * cos_q2 - 14.22 * I2 * Ir * m2 + 0.09 * I2 * m2**2 * r2**2 * cos_q2**2 - 0.054 * I2 * m2**2 * r2 * cos_q2 - 0.0081 * I2 * m2**2 - 47952.0 * Ir**3 - 1710.0 * Ir**2 * m2 * r2 * cos_q2 - 236.52 * Ir**2 * m2 - 11.79 * Ir * m2**2 * r2**2 * cos_q2**2 - 4.212 * Ir * m2**2 * r2 * cos_q2 - 0.2916 * Ir * m2**2 + 0.054 * m2**3 * r2**3 * cos_q2**3 + 0.0081 * m2**3 * r2**2 * cos_q2**2)
        term5 = 4.34897137154951e-10 * (-0.3 * dot_q1**2 * m2 * r2 * sin_q2 - dot_q2 * b2 - cf2 * atan_100_dot_q2 - 9.81 * m2 * r2 * sin_q1_q2) * (-0.6 * I1 * I2 * m2 * r2 * sin_q2 - 46.8 * I1 * Ir * m2 * r2 * sin_q2 + 0.18 * I1 * m2**2 * r2**2 * sin_q2 * cos_q2 - 76.2 * I2 * Ir * m2 * r2 * sin_q2 + 0.18 * I2 * m2**2 * r2**2 * sin_q2 * cos_q2 - 0.054 * I2 * m2**2 * r2 * sin_q2 - 1710.0 * Ir**2 * m2 * r2 * sin_q2 - 23.58 * Ir * m2**2 * r2**2 * sin_q2 * cos_q2 - 4.212 * Ir * m2**2 * r2 * sin_q2 + 0.162 * m2**3 * r2**3 * sin_q2 * cos_q2**2 + 0.0162 * m2**3 * r2**2 * sin_q2 * cos_q2)
        term5 *= (I1 * I2 - 6.0 * I1 * Ir + 0.3 * I1 * m2 * r2 * cos_q2 + I2**2 + 31.0 * I2 * Ir + 0.9 * I2 * m2 * r2 * cos_q2 + 0.09 * I2 * m2 - 222.0 * Ir**2 + 7.5 * Ir * m2 * r2 * cos_q2 - 0.54 * Ir * m2 + 0.18 * m2**2 * r2**2 * cos_q2**2 + 0.027 * m2**2 * r2 * cos_q2)
        term5 /= (-2.08541875208542e-5 * I1**2 * I2 - 0.000750750750750751 * I1**2 * Ir - 2.08541875208542e-5 * I1 * I2**2 - 0.00329496162829496 * I1 * I2 * Ir - 1.25125125125125e-5 * I1 * I2 * m2 * r2 * cos_q2 - 3.75375375375375e-6 * I1 * I2 * m2 - 0.0548048048048048 * I1 * Ir**2 - 0.000975975975975976 * I1 * Ir * m2 * r2 * cos_q2 - 0.000135135135135135 * I1 * Ir * m2 + 1.87687687687688e-6 * I1 * m2**2 * r2**2 * cos_q2**2 - 0.00177260593927261 * I2**2 * Ir - 1.87687687687688e-6 * I2**2 * m2 - 0.0926134467801134 * I2 * Ir**2 - 0.00158908908908909 * I2 * Ir * m2 * r2 * cos_q2 - 0.000296546546546547 * I2 * Ir * m2 + 1.87687687687688e-6 * I2 * m2**2 * r2**2 * cos_q2**2 - 1.12612612612613e-6 * I2 * m2**2 * r2 * cos_q2 - 1.68918918918919e-7 * I2 * m2**2 - Ir**3 - 0.0356606606606607 * Ir**2 * m2 * r2 * cos_q2 - 0.00493243243243243 * Ir**2 * m2 - 0.000245870870870871 * Ir * m2**2 * r2**2 * cos_q2**2 - 8.78378378378378e-5 * Ir * m2**2 * r2 * cos_q2 - 6.08108108108108e-6 * Ir * m2**2 + 1.12612612612613e-6 * m2**3 * r2**3 * cos_q2**3 + 1.68918918918919e-7 * m2**3 * r2**2 * cos_q2**2)**2
        term6 = 4.34897137154951e-10 * (0.6 * dot_q1 * dot_q2 * m2 * r2 * sin_q2 - dot_q1 * b1 + 0.3 * dot_q2**2 * m2 * r2 * sin_q2 - cf1 * atan_100_dot_q1 - 9.81 * m2 * (r2 * sin_q1_q2 + 0.3 * sin_q1) - 5.96448 * r1 * sin_q1) * (-I1 * I2 - 36.0 * I1 * Ir - I2**2 - 73.0 * I2 * Ir - 0.6 * I2 * m2 * r2 * cos_q2 - 0.09 * I2 * m2 - 1332.0 * Ir**2 - 21.6 * Ir * m2 * r2 * cos_q2 - 3.24 * Ir * m2)
        term6 *= (-0.6 * I1 * I2 * m2 * r2 * sin_q2 - 46.8 * I1 * Ir * m2 * r2 * sin_q2 + 0.18 * I1 * m2**2 * r2**2 * sin_q2 * cos_q2 - 76.2 * I2 * Ir * m2 * r2 * sin_q2 + 0.18 * I2 * m2**2 * r2**2 * sin_q2 * cos_q2 - 0.054 * I2 * m2**2 * r2 * sin_q2 - 1710.0 * Ir**2 * m2 * r2 * sin_q2 - 23.58 * Ir * m2**2 * r2**2 * sin_q2 * cos_q2 - 4.212 * Ir * m2**2 * r2 * sin_q2 + 0.162 * m2**3 * r2**3 * sin_q2 * cos_q2**2 + 0.0162 * m2**3 * r2**2 * sin_q2 * cos_q2)
        term6 /= (-2.08541875208542e-5 * I1**2 * I2 - 0.000750750750750751 * I1**2 * Ir - 2.08541875208542e-5 * I1 * I2**2 - 0.00329496162829496 * I1 * I2 * Ir - 1.25125125125125e-5 * I1 * I2 * m2 * r2 * cos_q2 - 3.75375375375375e-6 * I1 * I2 * m2 - 0.0548048048048048 * I1 * Ir**2 - 0.000975975975975976 * I1 * Ir * m2 * r2 * cos_q2 - 0.000135135135135135 * I1 * Ir * m2 + 1.87687687687688e-6 * I1 * m2**2 * r2**2 * cos_q2**2 - 0.00177260593927261 * I2**2 * Ir - 1.87687687687688e-6 * I2**2 * m2 - 0.0926134467801134 * I2 * Ir**2 - 0.00158908908908909 * I2 * Ir * m2 * r2 * cos_q2 - 0.000296546546546547 * I2 * Ir * m2 + 1.87687687687688e-6 * I2 * m2**2 * r2**2 * cos_q2**2 - 1.12612612612613e-6 * I2 * m2**2 * r2 * cos_q2 - 1.68918918918919e-7 * I2 * m2**2 - Ir**3 - 0.0356606606606607 * Ir**2 * m2 * r2 * cos_q2 - 0.00493243243243243 * Ir**2 * m2 - 0.000245870870870871 * Ir * m2**2 * r2**2 * cos_q2**2 - 8.78378378378378e-5 * Ir * m2**2 * r2 * cos_q2 - 6.08108108108108e-6 * Ir * m2**2 + 1.12612612612613e-6 * m2**3 * r2**3 * cos_q2**3 + 1.68918918918919e-7 * m2**3 * r2**2 * cos_q2**2)**2
        return term1 + term2 + term3 + term4 + term5 + term6

    def dadxT_element_1_3():
        term1 = 0.6 * m2 * r2 * (-0.3 * dot_q1**2 * m2 * r2 * sin_q2 - dot_q2 * b2 - cf2 * atan_100_dot_q2 - 9.81 * m2 * r2 * sin_q1_q2) * sin_q2
        term1 /= (-I1 * I2 - 36.0 * I1 * Ir - 85.0 * I2 * Ir - 0.09 * I2 * m2 - 1296.0 * Ir**2 - 25.2 * Ir * m2 * r2 * cos_q2 - 3.24 * Ir * m2 + 0.09 * m2**2 * r2**2 * cos_q2**2)
        term2 = -0.3 * m2 * r2 * (0.6 * dot_q1 * dot_q2 * m2 * r2 * sin_q2 - dot_q1 * b1 + 0.3 * dot_q2**2 * m2 * r2 * sin_q2 - cf1 * atan_100_dot_q1 - 9.81 * m2 * (r2 * sin_q1_q2 + 0.3 * sin_q1) - 5.96448 * r1 * sin_q1) * sin_q2
        term2 /= (-I1 * I2 - 36.0 * I1 * Ir - 85.0 * I2 * Ir - 0.09 * I2 * m2 - 1296.0 * Ir**2 - 25.2 * Ir * m2 * r2 * cos_q2 - 3.24 * Ir * m2 + 0.09 * m2**2 * r2**2 * cos_q2**2)
        term3 = 5.95374180765127e-7 * (-25.2 * Ir * m2 * r2 * sin_q2 + 0.18 * m2**2 * r2**2 * sin_q2 * cos_q2) * (I2 - 6.0 * Ir + 0.3 * m2 * r2 * cos_q2) * (0.6 * dot_q1 * dot_q2 * m2 * r2 * sin_q2 - dot_q1 * b1 + 0.3 * dot_q2**2 * m2 * r2 * sin_q2 - cf1 * atan_100_dot_q1 - 9.81 * m2 * (r2 * sin_q1_q2 + 0.3 * sin_q1) - 5.96448 * r1 * sin_q1)
        term3 /= (-0.000771604938271605 * I1 * I2 - 0.0277777777777778 * I1 * Ir - 0.0655864197530864 * I2 * Ir - 6.94444444444444e-5 * I2 * m2 - Ir**2 - 0.0194444444444444 * Ir * m2 * r2 * cos_q2 - 0.0025 * Ir * m2 + 6.94444444444444e-5 * m2**2 * r2**2 * cos_q2**2)**2
        term4 = 5.95374180765127e-7 * (-25.2 * Ir * m2 * r2 * sin_q2 + 0.18 * m2**2 * r2**2 * sin_q2 * cos_q2) * (-0.3 * dot_q1**2 * m2 * r2 * sin_q2 - dot_q2 * b2 - cf2 * atan_100_dot_q2 - 9.81 * m2 * r2 * sin_q1_q2) * (-I1 - I2 - 37.0 * Ir - 0.6 * m2 * r2 * cos_q2 - 0.09 * m2)
        term4 /= (-0.000771604938271605 * I1 * I2 - 0.0277777777777778 * I1 * Ir - 0.0655864197530864 * I2 * Ir - 6.94444444444444e-5 * I2 * m2 - Ir**2 - 0.0194444444444444 * Ir * m2 * r2 * cos_q2 - 0.0025 * Ir * m2 + 6.94444444444444e-5 * m2**2 * r2**2 * cos_q2**2)**2
        term5 = (-0.3 * dot_q1**2 * m2 * r2 * cos_q2 - 9.81 * m2 * r2 * cos_q1_q2) * (-I1 - I2 - 37.0 * Ir - 0.6 * m2 * r2 * cos_q2 - 0.09 * m2)
        term5 /= (-I1 * I2 - 36.0 * I1 * Ir - 85.0 * I2 * Ir - 0.09 * I2 * m2 - 1296.0 * Ir**2 - 25.2 * Ir * m2 * r2 * cos_q2 - 3.24 * Ir * m2 + 0.09 * m2**2 * r2**2 * cos_q2**2)
        term6 = (I2 - 6.0 * Ir + 0.3 * m2 * r2 * cos_q2) * (0.6 * dot_q1 * dot_q2 * m2 * r2 * cos_q2 + 0.3 * dot_q2**2 * m2 * r2 * cos_q2 - 9.81 * m2 * r2 * cos_q1_q2)
        term6 /= (-I1 * I2 - 36.0 * I1 * Ir - 85.0 * I2 * Ir - 0.09 * I2 * m2 - 1296.0 * Ir**2 - 25.2 * Ir * m2 * r2 * cos_q2 - 3.24 * Ir * m2 + 0.09 * m2**2 * r2**2 * cos_q2**2)
        return term1 + term2 + term3 + term4 + term5 + term6

    def dadxT_element_2_2():
        term1 = -0.6 * dot_q1 * m2 * r2 * (
            I1 * I2 - 6.0 * I1 * Ir + 0.3 * I1 * m2 * r2 * torch.cos(q2) + I2**2 + 31.0 * I2 * Ir + 0.9 * I2 * m2 * r2 * torch.cos(q2) + 0.09 * I2 * m2 - 222.0 * Ir**2 + 7.5 * Ir * m2 * r2 * torch.cos(q2) - 0.54 * Ir * m2 + 0.18 * m2**2 * r2**2 * torch.cos(q2)**2 + 0.027 * m2**2 * r2 * torch.cos(q2)
        ) * torch.sin(q2)

        term2 = (0.6 * dot_q2 * m2 * r2 * torch.sin(q2) - b1 - 100 * cf1 / (10000 * dot_q1**2 + 1)) * (
            -I1 * I2 - 36.0 * I1 * Ir - I2**2 - 73.0 * I2 * Ir - 0.6 * I2 * m2 * r2 * torch.cos(q2) - 0.09 * I2 * m2 - 1332.0 * Ir**2 - 21.6 * Ir * m2 * r2 * torch.cos(q2) - 3.24 * Ir * m2
        )

        denom1 = -I1**2 * I2 - 36.0 * I1**2 * Ir - I1 * I2**2 - 158.0 * I1 * I2 * Ir - 0.6 * I1 * I2 * m2 * r2 * torch.cos(q2) - 0.18 * I1 * I2 * m2 - 2628.0 * I1 * Ir**2 - 46.8 * I1 * Ir * m2 * r2 * torch.cos(q2) - 6.48 * I1 * Ir * m2 + 0.09 * I1 * m2**2 * r2**2 * torch.cos(q2)**2 - 85.0 * I2**2 * Ir - 0.09 * I2**2 * m2 - 4441.0 * I2 * Ir**2 - 76.2 * I2 * Ir * m2 * r2 * torch.cos(q2) - 14.22 * I2 * Ir * m2 + 0.09 * I2 * m2**2 * r2**2 * torch.cos(q2)**2 - 0.054 * I2 * m2**2 * r2 * torch.cos(q2) - 0.0081 * I2 * m2**2 - 47952.0 * Ir**3 - 1710.0 * Ir**2 * m2 * r2 * torch.cos(q2) - 236.52 * Ir**2 * m2 - 11.79 * Ir * m2**2 * r2**2 * torch.cos(q2)**2 - 4.212 * Ir * m2**2 * r2 * torch.cos(q2) - 0.2916 * Ir * m2**2 + 0.054 * m2**3 * r2**3 * torch.cos(q2)**3 + 0.0081 * m2**3 * r2**2 * torch.cos(q2)**2

        return (term1 + term2) / denom1

    def dadxT_element_2_3():
        term1 = -0.6 * dot_q1 * m2 * r2 * (-I1 - I2 - 37.0 * Ir - 0.6 * m2 * r2 * torch.cos(q2) - 0.09 * m2) * torch.sin(q2)

        term2 = (I2 - 6.0 * Ir + 0.3 * m2 * r2 * torch.cos(q2)) * (0.6 * dot_q2 * m2 * r2 * torch.sin(q2) - b1 - 100 * cf1 / (10000 * dot_q1**2 + 1))

        denom1 = -I1 * I2 - 36.0 * I1 * Ir - 85.0 * I2 * Ir - 0.09 * I2 * m2 - 1296.0 * Ir**2 - 25.2 * Ir * m2 * r2 * torch.cos(q2) - 3.24 * Ir * m2 + 0.09 * m2**2 * r2**2 * torch.cos(q2)**2

        return (term1 + term2) / denom1

    def dadxT_element_3_2():
        term1 = (-b2 - 100 * cf2 / (10000 * dot_q2**2 + 1)) * (
            I1 * I2 - 6.0 * I1 * Ir + 0.3 * I1 * m2 * r2 * torch.cos(q2) + I2**2 + 31.0 * I2 * Ir + 0.9 * I2 * m2 * r2 * torch.cos(q2) + 0.09 * I2 * m2 - 222.0 * Ir**2 + 7.5 * Ir * m2 * r2 * torch.cos(q2) - 0.54 * Ir * m2 + 0.18 * m2**2 * r2**2 * torch.cos(q2)**2 + 0.027 * m2**2 * r2 * torch.cos(q2)
        )

        term2 = (0.6 * dot_q1 * m2 * r2 * torch.sin(q2) + 0.6 * dot_q2 * m2 * r2 * torch.sin(q2)) * (
            -I1 * I2 - 36.0 * I1 * Ir - I2**2 - 73.0 * I2 * Ir - 0.6 * I2 * m2 * r2 * torch.cos(q2) - 0.09 * I2 * m2 - 1332.0 * Ir**2 - 21.6 * Ir * m2 * r2 * torch.cos(q2) - 3.24 * Ir * m2
        )

        denom1 = -I1**2 * I2 - 36.0 * I1**2 * Ir - I1 * I2**2 - 158.0 * I1 * I2 * Ir - 0.6 * I1 * I2 * m2 * r2 * torch.cos(q2) - 0.18 * I1 * I2 * m2 - 2628.0 * I1 * Ir**2 - 46.8 * I1 * Ir * m2 * r2 * torch.cos(q2) - 6.48 * I1 * Ir * m2 + 0.09 * I1 * m2**2 * r2**2 * torch.cos(q2)**2 - 85.0 * I2**2 * Ir - 0.09 * I2**2 * m2 - 4441.0 * I2 * Ir**2 - 76.2 * I2 * Ir * m2 * r2 * torch.cos(q2) - 14.22 * I2 * Ir * m2 + 0.09 * I2 * m2**2 * r2**2 * torch.cos(q2)**2 - 0.054 * I2 * m2**2 * r2 * torch.cos(q2) - 0.0081 * I2 * m2**2 - 47952.0 * Ir**3 - 1710.0 * Ir**2 * m2 * r2 * torch.cos(q2) - 236.52 * Ir**2 * m2 - 11.79 * Ir * m2**2 * r2**2 * torch.cos(q2)**2 - 4.212 * Ir * m2**2 * r2 * torch.cos(q2) - 0.2916 * Ir * m2**2 + 0.054 * m2**3 * r2**3 * torch.cos(q2)**3 + 0.0081 * m2**3 * r2**2 * torch.cos(q2)**2

        return term1 / denom1 + term2 / denom1

    def dadxT_element_3_3():
        term1 = (-b2 - 100 * cf2 / (10000 * dot_q2**2 + 1)) * (-I1 - I2 - 37.0 * Ir - 0.6 * m2 * r2 * torch.cos(q2) - 0.09 * m2)

        term2 = (0.6 * dot_q1 * m2 * r2 * torch.sin(q2) + 0.6 * dot_q2 * m2 * r2 * torch.sin(q2)) * (I2 - 6.0 * Ir + 0.3 * m2 * r2 * torch.cos(q2))

        denom1 = -I1 * I2 - 36.0 * I1 * Ir - 85.0 * I2 * Ir - 0.09 * I2 * m2 - 1296.0 * Ir**2 - 25.2 * Ir * m2 * r2 * torch.cos(q2) - 3.24 * Ir * m2 + 0.09 * m2**2 * r2**2 * torch.cos(q2)**2

        return term1 / denom1 + term2 / denom1
    
    dadxT[0, 2, :] = dadxT_element_0_2().view(-1)
    dadxT[0, 3, :] = dadxT_element_0_3().view(-1)
    dadxT[1, 2, :] = dadxT_element_1_2().view(-1)
    dadxT[1, 3, :] = dadxT_element_1_3().view(-1)

    dadxT[2, 2, :] = dadxT_element_2_2().view(-1)
    dadxT[2, 3, :] = dadxT_element_2_3().view(-1)
    dadxT[3, 2, :] = dadxT_element_3_2().view(-1)
    dadxT[3, 3, :] = dadxT_element_3_3().view(-1)

    dadxT[2, 0, :] = torch.ones_like(x[:, 0, 0],device=device)
    dadxT[3, 1, :] = torch.ones_like(x[:, 0, 0],device=device)


    return dadxT.permute(2, 0, 1)


def compute_dBdxT_matrix(x, theta):
    device = x.device
    cos_q2 = torch.cos(x[:, 1, 0])
    sin_q2 = torch.sin(x[:, 1, 0])
    
    Ir = theta[:, 0, 0].view(-1, 1, 1)
    I1 = theta[:, 2, 0].view(-1, 1, 1)
    r2 = theta[:, 5, 0].view(-1, 1, 1)
    m2 = theta[:, 6, 0].view(-1, 1, 1)
    I2 = theta[:, 7, 0].view(-1, 1, 1)

    dBdxT = torch.zeros((4, 4, 2, x.shape[0]), device=device)

    def dBdxT_element_1_2_0():
        term1 = (0.6 * I2 * m2 * r2 * sin_q2 + 21.6 * Ir * m2 * r2 * sin_q2) / (-I1**2 * I2 - 36.0 * I1**2 * Ir - I1 * I2**2 - 158.0 * I1 * I2 * Ir - 0.6 * I1 * I2 * m2 * r2 * cos_q2 - 0.18 * I1 * I2 * m2 - 2628.0 * I1 * Ir**2 - 46.8 * I1 * Ir * m2 * r2 * cos_q2 - 6.48 * I1 * Ir * m2 + 0.09 * I1 * m2**2 * r2**2 * cos_q2**2 - 85.0 * I2**2 * Ir - 0.09 * I2**2 * m2 - 4441.0 * I2 * Ir**2 - 76.2 * I2 * Ir * m2 * r2 * cos_q2 - 14.22 * I2 * Ir * m2 + 0.09 * I2 * m2**2 * r2**2 * cos_q2**2 - 0.054 * I2 * m2**2 * r2 * cos_q2 - 0.0081 * I2 * m2**2 - 47952.0 * Ir**3 - 1710.0 * Ir**2 * m2 * r2 * cos_q2 - 236.52 * Ir**2 * m2 - 11.79 * Ir * m2**2 * r2**2 * cos_q2**2 - 4.212 * Ir * m2**2 * r2 * cos_q2 - 0.2916 * Ir * m2**2 + 0.054 * m2**3 * r2**3 * cos_q2**3 + 0.0081 * m2**3 * r2**2 * cos_q2**2)
        term2 = 4.34897137154951e-10 * (-I1 * I2 - 36.0 * I1 * Ir - I2**2 - 73.0 * I2 * Ir - 0.6 * I2 * m2 * r2 * cos_q2 - 0.09 * I2 * m2 - 1332.0 * Ir**2 - 21.6 * Ir * m2 * r2 * cos_q2 - 3.24 * Ir * m2) * (-0.6 * I1 * I2 * m2 * r2 * sin_q2 - 46.8 * I1 * Ir * m2 * r2 * sin_q2 + 0.18 * I1 * m2**2 * r2**2 * sin_q2 * cos_q2 - 76.2 * I2 * Ir * m2 * r2 * sin_q2 + 0.18 * I2 * m2**2 * r2**2 * sin_q2 * cos_q2 - 0.054 * I2 * m2**2 * r2 * sin_q2 - 1710.0 * Ir**2 * m2 * r2 * sin_q2 - 23.58 * Ir * m2**2 * r2**2 * sin_q2 * cos_q2 - 4.212 * Ir * m2**2 * r2 * sin_q2 + 0.162 * m2**3 * r2**3 * sin_q2 * cos_q2**2 + 0.0162 * m2**3 * r2**2 * sin_q2 * cos_q2) / (-2.08541875208542e-5 * I1**2 * I2 - 0.000750750750750751 * I1**2 * Ir - 2.08541875208542e-5 * I1 * I2**2 - 0.00329496162829496 * I1 * I2 * Ir - 1.25125125125125e-5 * I1 * I2 * m2 * r2 * cos_q2 - 3.75375375375375e-6 * I1 * I2 * m2 - 0.0548048048048048 * I1 * Ir**2 - 0.000975975975975976 * I1 * Ir * m2 * r2 * cos_q2 - 0.000135135135135135 * I1 * Ir * m2 + 1.87687687687688e-6 * I1 * m2**2 * r2**2 * cos_q2**2 - 0.00177260593927261 * I2**2 * Ir - 1.87687687687688e-6 * I2**2 * m2 - 0.0926134467801134 * I2 * Ir**2 - 0.00158908908908909 * I2 * Ir * m2 * r2 * cos_q2 - 0.000296546546546547 * I2 * Ir * m2 + 1.87687687687688e-6 * I2 * m2**2 * r2**2 * cos_q2**2 - 1.12612612612613e-6 * I2 * m2**2 * r2 * cos_q2 - 1.68918918918919e-7 * I2 * m2**2 - Ir**3 - 0.0356606606606607 * Ir**2 * m2 * r2 * cos_q2 - 0.00493243243243243 * Ir**2 * m2 - 0.000245870870870871 * Ir * m2**2 * r2**2 * cos_q2**2 - 8.78378378378378e-5 * Ir * m2**2 * r2 * cos_q2 - 6.08108108108108e-6 * Ir * m2**2 + 1.12612612612613e-6 * m2**3 * r2**3 * cos_q2**3 + 1.68918918918919e-7 * m2**3 * r2**2 * cos_q2**2)**2
        return term1 + term2

    def dBdxT_element_1_2_1():
        term1 = (-0.3 * I1 * m2 * r2 * sin_q2 - 0.9 * I2 * m2 * r2 * sin_q2 - 7.5 * Ir * m2 * r2 * sin_q2 - 0.36 * m2**2 * r2**2 * sin_q2 * cos_q2 - 0.027 * m2**2 * r2 * sin_q2) / (-I1**2 * I2 - 36.0 * I1**2 * Ir - I1 * I2**2 - 158.0 * I1 * I2 * Ir - 0.6 * I1 * I2 * m2 * r2 * cos_q2 - 0.18 * I1 * I2 * m2 - 2628.0 * I1 * Ir**2 - 46.8 * I1 * Ir * m2 * r2 * cos_q2 - 6.48 * I1 * Ir * m2 + 0.09 * I1 * m2**2 * r2**2 * cos_q2**2 - 85.0 * I2**2 * Ir - 0.09 * I2**2 * m2 - 4441.0 * I2 * Ir**2 - 76.2 * I2 * Ir * m2 * r2 * cos_q2 - 14.22 * I2 * Ir * m2 + 0.09 * I2 * m2**2 * r2**2 * cos_q2**2 - 0.054 * I2 * m2**2 * r2 * cos_q2 - 0.0081 * I2 * m2**2 - 47952.0 * Ir**3 - 1710.0 * Ir**2 * m2 * r2 * cos_q2 - 236.52 * Ir**2 * m2 - 11.79 * Ir * m2**2 * r2**2 * cos_q2**2 - 4.212 * Ir * m2**2 * r2 * cos_q2 - 0.2916 * Ir * m2**2 + 0.054 * m2**3 * r2**3 * cos_q2**3 + 0.0081 * m2**3 * r2**2 * cos_q2**2)
        term2 = 4.34897137154951e-10 * (-0.6 * I1 * I2 * m2 * r2 * sin_q2 - 46.8 * I1 * Ir * m2 * r2 * sin_q2 + 0.18 * I1 * m2**2 * r2**2 * sin_q2 * cos_q2 - 76.2 * I2 * Ir * m2 * r2 * sin_q2 + 0.18 * I2 * m2**2 * r2**2 * sin_q2 * cos_q2 - 0.054 * I2 * m2**2 * r2 * sin_q2 - 1710.0 * Ir**2 * m2 * r2 * sin_q2 - 23.58 * Ir * m2**2 * r2**2 * sin_q2 * cos_q2 - 4.212 * Ir * m2**2 * r2 * sin_q2 + 0.162 * m2**3 * r2**3 * sin_q2 * cos_q2**2 + 0.0162 * m2**3 * r2**2 * sin_q2 * cos_q2) * (I1 * I2 - 6.0 * I1 * Ir + 0.3 * I1 * m2 * r2 * cos_q2 + I2**2 + 31.0 * I2 * Ir + 0.9 * I2 * m2 * r2 * cos_q2 + 0.09 * I2 * m2 - 222.0 * Ir**2 + 7.5 * Ir * m2 * r2 * cos_q2 - 0.54 * Ir * m2 + 0.18 * m2**2 * r2**2 * cos_q2**2 + 0.027 * m2**2 * r2 * cos_q2) / (-2.08541875208542e-5 * I1**2 * I2 - 0.000750750750750751 * I1**2 * Ir - 2.08541875208542e-5 * I1 * I2**2 - 0.00329496162829496 * I1 * I2 * Ir - 1.25125125125125e-5 * I1 * I2 * m2 * r2 * cos_q2 - 3.75375375375375e-6 * I1 * I2 * m2 - 0.0548048048048048 * I1 * Ir**2 - 0.000975975975975976 * I1 * Ir * m2 * r2 * cos_q2 - 0.000135135135135135 * I1 * Ir * m2 + 1.87687687687688e-6 * I1 * m2**2 * r2**2 * cos_q2**2 - 0.00177260593927261 * I2**2 * Ir - 1.87687687687688e-6 * I2**2 * m2 - 0.0926134467801134 * I2 * Ir**2 - 0.00158908908908909 * I2 * Ir * m2 * r2 * cos_q2 - 0.000296546546546547 * I2 * Ir * m2 + 1.87687687687688e-6 * I2 * m2**2 * r2**2 * cos_q2**2 - 1.12612612612613e-6 * I2 * m2**2 * r2 * cos_q2 - 1.68918918918919e-7 * I2 * m2**2 - Ir**3 - 0.0356606606606607 * Ir**2 * m2 * r2 * cos_q2 - 0.00493243243243243 * Ir**2 * m2 - 0.000245870870870871 * Ir * m2**2 * r2**2 * cos_q2**2 - 8.78378378378378e-5 * Ir * m2**2 * r2 * cos_q2 - 6.08108108108108e-6 * Ir * m2**2 + 1.12612612612613e-6 * m2**3 * r2**3 * cos_q2**3 + 1.68918918918919e-7 * m2**3 * r2**2 * cos_q2**2)**2
        return term1 + term2

    def dBdxT_element_1_3_0():
        term1 = -0.3 * m2 * r2 * sin_q2 / (-I1 * I2 - 36.0 * I1 * Ir - 85.0 * I2 * Ir - 0.09 * I2 * m2 - 1296.0 * Ir**2 - 25.2 * Ir * m2 * r2 * cos_q2 - 3.24 * Ir * m2 + 0.09 * m2**2 * r2**2 * cos_q2**2)
        term2 = 5.95374180765127e-7 * (-25.2 * Ir * m2 * r2 * sin_q2 + 0.18 * m2**2 * r2**2 * sin_q2 * cos_q2) * (I2 - 6.0 * Ir + 0.3 * m2 * r2 * cos_q2) / (-0.000771604938271605 * I1 * I2 - 0.0277777777777778 * I1 * Ir - 0.0655864197530864 * I2 * Ir - 6.94444444444444e-5 * I2 * m2 - Ir**2 - 0.0194444444444444 * Ir * m2 * r2 * cos_q2 - 0.0025 * Ir * m2 + 6.94444444444444e-5 * m2**2 * r2**2 * cos_q2**2)**2
        return term1 + term2

    def dBdxT_element_1_3_1():
        term1 = 0.6 * m2 * r2 * sin_q2 / (-I1 * I2 - 36.0 * I1 * Ir - 85.0 * I2 * Ir - 0.09 * I2 * m2 - 1296.0 * Ir**2 - 25.2 * Ir * m2 * r2 * cos_q2 - 3.24 * Ir * m2 + 0.09 * m2**2 * r2**2 * cos_q2**2)
        term2 = 5.95374180765127e-7 * (-25.2 * Ir * m2 * r2 * sin_q2 + 0.18 * m2**2 * r2**2 * sin_q2 * cos_q2) * (-I1 - I2 - 37.0 * Ir - 0.6 * m2 * r2 * cos_q2 - 0.09 * m2) / (-0.000771604938271605 * I1 * I2 - 0.0277777777777778 * I1 * Ir - 0.0655864197530864 * I2 * Ir - 6.94444444444444e-5 * I2 * m2 - Ir**2 - 0.0194444444444444 * Ir * m2 * r2 * cos_q2 - 0.0025 * Ir * m2 + 6.94444444444444e-5 * m2**2 * r2**2 * cos_q2**2)**2
        return term1 + term2

    dBdxT[1, 2, 0, :] = dBdxT_element_1_2_0().view(-1)
    dBdxT[1, 2, 1, :] = dBdxT_element_1_2_1().view(-1)
    dBdxT[1, 3, 0, :] = dBdxT_element_1_3_0().view(-1)
    dBdxT[1, 3, 1, :] = dBdxT_element_1_3_1().view(-1)

    return dBdxT.permute(3, 0, 1, 2)

class DoulbePendulum(BaseSystem):
    name = "DoulbePendulum"
    labels = ('q1', 'q2', 'q1_dot', 'q2_dot')

    def __init__(self, cuda=CUDA_AVAILABLE, **kwargs):
        super(DoulbePendulum, self).__init__()

        # Define Duration:
        self.T = kwargs.get("T", 5.0)
        self.dt = kwargs.get("dt", 1./100.)

        # Define the System:
        self.n_state = 4
        self.n_dof = 2
        self.n_act = 2
        self.n_parameter = 10

        # Continuous Joints:
        # Right now only one continuous joint is supported
        self.wrap, self.wrap_i = True, 1

        # State Constraints:
        # theta = 0, means the pendulum is pointing upward
        self.x_target = torch.tensor([np.pi, 0.0, 0.0, 0.0])
        self.x_start = torch.tensor([0.0, 0., 0.0, 0.0])
        self.x_start_var = torch.tensor([1.e-2, 1.e-2, 1.e-6, 1.e-6])
        self.x_lim = torch.tensor([np.pi, np.pi, 15., 15.])
        self.x_penalty = torch.tensor([10, 5., 1., 1])

        # 10 degree angle error for initial sampling
        self.x_init = torch.tensor([0.17, 0.17, 1.e-3, 1.e-3])
        self.u_lim = torch.tensor([6., 1.e-6])

        """
        Parameters
        ----------
        mass : array_like, optional
            shape=(2,), dtype=float, default=[1.0, 1.0]
            masses of the double pendulum,
            [m1, m2], units=[kg]
        length : array_like, optional
            shape=(2,), dtype=float, default=[0.5, 0.5]
            link lengths of the double pendulum,
            [l1, l2], units=[m]
        com : array_like, optional
            shape=(2,), dtype=float, default=[0.5, 0.5]
            center of mass lengths of the double pendulum links
            [r1, r2], units=[m]
        damping : array_like, optional
            shape=(2,), dtype=float, default=[0.5, 0.5]
            damping coefficients of the double pendulum actuators
            [b1, b2], units=[kg*m/s]
        gravity : float, optional
            default=9.81
            gravity acceleration (pointing downwards),
            units=[m/s²]
        coulomb_fric : array_like, optional
            shape=(2,), dtype=float, default=[0.0, 0.0]
            coulomb friction coefficients for the double pendulum actuators
            [cf1, cf2], units=[Nm]
        inertia : array_like, optional
            shape=(2,), dtype=float, default=[None, None]
            inertia of the double pendulum links
            [I1, I2], units=[kg*m²]
            if entry is None defaults to point mass m*l² inertia for the entry
        motor_inertia : float, optional
            default=0.0
            inertia of the actuators/motors
            [Ir1, Ir2], units=[kg*m²]
        gear_ratio : int, optional
            gear ratio of the motors, default=6
        torque_limit : array_like, optional
            shape=(2,), dtype=float, default=[np.inf, np.inf]
            torque limit of the motors
            [tl1, tl2], units=[Nm, Nm]
        model_pars : model_parameters object, optional
            object of the model_parameters class, default=None
            Can be used to set all model parameters above
            If provided, the model_pars parameters overwrite
            the other provided parameters
        """

        mass=[0.608, 0.630]
        length=[0.3, 0.4]
        com=[0.275, 0.415]
        damping=[0.005, 0.005]
        cfric=[0.093, 0.14]
        gravity=9.81
        inertia=[0.0475, 0.0798]
        motor_inertia=0.
        gear_ratio=6
        torque_limit=[10.0, 10.0] #this is to determine the D matrix. not the actual torque limit

        fixed_para = {'g':9.81, 'm1':0.608, 'gr':6.0, 'l1':0.3, 'l2':0.4}
        change_para = {'Ir':0.0, 'r1':0.275, 'I1':0.0475, 'b1':0.005, 'cf1':0.093, 'r2':0.415, 'm2':0.630, 'I2':0.0798, 'b2':0.005, 'cf2':0.14}

        self.symbol_theta = smp.symbols("Ir r1 I1 b1 cf1 r2 m2 I2 b2 cf2")
        Ir, r1, I1, b1, cf1, r2, m2, I2, b2, cf2 =self.symbol_theta
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Model parameters
        # self.theta = torch.tensor(list(change_para.values())).view(1, self.n_parameter, 1)
        self.theta = torch.tensor([0.0, 0.275, 0.0475, 0.005, 0.093, 0.415, 0.630, 0.0798, 0.005, 0.14],requires_grad=True, device=self.device).view(1, self.n_parameter, 1)

        theta_min = [0, 0.275*0.25, 0.75*0.0475, -0.1, -0.2, 0.75*0.415, 0.75*0.630, 0.75*0.0798, -0.1, 0.2]
        theta_max = [1e-4, 0.275*1.25, 1.25*0.0475, 0.1, 0.2, 1.25*0.415, 1.25*0.630, 1.25*0.0798, -0.1, 0.2]
        self.theta_min = torch.tensor(theta_min, device=self.device).view(1, self.n_parameter, 1)
        self.theta_max = torch.tensor(theta_max, device=self.device).view(1, self.n_parameter, 1)

        self.plant = SymbolicDoublePendulum(mass=mass,
                                    length=length,
                                    com=com,
                                    damping=damping,
                                    gravity=gravity,
                                    coulomb_fric=cfric,
                                    inertia=inertia,
                                    motor_inertia=motor_inertia,
                                    gear_ratio=gear_ratio,
                                    torque_limit=torque_limit)
        
        q1, q2, qd1, qd2, qdd1, qdd2  = self.plant.q1, self.plant.q2, self.plant.qd1, \
        self.plant.qd2, self.plant.qdd1, self.plant.qdd2
        self.symbol_x = smp.Matrix([q1, q2, qd1, qd2])

        # symbolic matrix
        invM  = self.plant.M.inv()
        q_dot = smp.Matrix([qd1, qd2])

        self.symbolic_a = q_dot.col_join(invM * (-self.plant.C*q_dot +self.plant.G - self.plant.F))
        self.symbolic_B = smp.Matrix([[0., 0.], [0., 0.]]).col_join(invM* self.plant.B)

        # substitute the fixed parameters
        self.symbolic_a = self.symbolic_a.subs(fixed_para)
        self.symbolic_B = self.symbolic_B.subs(fixed_para)

        assert self.symbolic_a.shape == (self.n_state, 1)
        assert self.symbolic_B.shape == (self.n_state, self.n_act)

        self.symbolic_dadxT = self.symbolic_a.jacobian(self.symbol_x).T

        assert self.symbolic_dadxT.shape == (self.n_state, self.n_state)

        self.symbolic_dBdxT = smp.MutableDenseNDimArray.zeros(self.n_state, self.n_state, self.n_act)
        for i in range(self.n_state):
            for j in range(self.n_state):
                for k in range(self.n_act):
                    self.symbolic_dBdxT[i, j, k] = smp.diff(self.symbolic_B[j, k], self.symbol_x[i])

        self.symbolic_dadpT = self.symbolic_a.jacobian(self.symbol_theta).T
        assert self.symbolic_dadpT.shape == (self.n_parameter, self.n_state)

        self.symbolic_dBdpT = smp.MutableDenseNDimArray.zeros(self.n_parameter, self.n_state, self.n_act)
        for i in range(self.n_parameter):
            for j in range(self.n_state):
                for k in range(self.n_act):
                    self.symbolic_dBdpT[i, j, k] = smp.diff(self.symbolic_B[j, k], self.symbol_theta[i])

        self.dadp_x = self.symbolic_dadpT.subs(change_para)

        # bring theta to symbolic_dBdpT, since MutableDenseNDimArray can not be used in subs
        self.dBdp_x = smp.MutableDenseNDimArray.zeros(self.n_parameter, self.n_state, self.n_act)
        for i in range(self.n_parameter):
            for j in range(self.n_state):
                for k in range(self.n_act):
                    self.dBdp_x[i, j, k] = self.symbolic_dBdpT[i, j, k].subs(change_para)

        self.dadp_la = lambdify(self.symbol_x, self.dadp_x)
        self.dBdp_la = lambdify(self.symbol_x, self.dBdp_x)


        # print('self.dadp_x:',self.dadp_x)
        # Compute Linearized System:
        # out = self.dyn(self.x_target, gradient=True)
        # self.A = out[2].view(1, self.n_state, self.n_state).transpose(dim0=1, dim1=2).numpy()
        # self.B = out[1].view(1, self.n_state, self.n_act).numpy()

        # Test Dynamics:
        self.check_dynamics()

        self.device = None
        DoulbePendulum.cuda(self) if cuda else DoulbePendulum.cpu(self)
        print("Double Pendulum System Initialized!")

    def dyn(self, x, dtheta=None, gradient=False):
        # print("dyn____________________________________________________")
        

        is_numpy = True if isinstance(x, np.ndarray) else False
        x = torch.from_numpy(x) if isinstance(x, np.ndarray) else x
        x = x.view(-1, self.n_state, 1).to(self.device)
        n_samples = x.shape[0]

        
        # Update the dynamics parameters with disturbance:
        if dtheta is not None:

            dtheta = torch.from_numpy(dtheta).float() if isinstance(dtheta, np.ndarray) else dtheta
            dtheta = dtheta.view(n_samples, self.n_parameter, 1)
            theta = self.theta + dtheta
            theta = torch.min(torch.max(theta, self.theta_min), self.theta_max)
            theta = theta.to(x.device)

        else:
            theta = self.theta

        # method 1 
        # change_para = theta_to_dict(theta)
        # a_x = self.symbolic_a.subs(change_para)
        # B_x = self.symbolic_B.subs(change_para)

        # a_la = lambdify(self.symbol_x, a_x)
        # B_la = lambdify(self.symbol_x, B_x)

        # a_test = torch.stack([
        #     torch.tensor(a_la(*sample.squeeze().detach().cpu().numpy())).reshape(self.n_state, 1).to(x.device)
        #     for sample in x
        # ]).float()

        # B_test = torch.stack([
        #     torch.tensor(B_la(*sample.squeeze().detach().cpu().numpy())).reshape(self.n_state, self.n_act).to(x.device)
        #     for sample in x
        # ]).float()

        # assert a_test.shape == (n_samples, self.n_state, 1)
        # assert B_test.shape == (n_samples, self.n_state, self.n_act)

        # end_time = time.time()
        # execution_time = end_time - start_time
        # print(f"bring x execution time: {execution_time} seconds")

        # 定义常数和符号变量
        # method 2
        theta.requires_grad_(True)
        x.requires_grad_(True)

        Ir = theta[:,0,0].view(-1, 1, 1)
        r1 = theta[:,1,0].view(-1, 1, 1)
        I1 = theta[:,2,0].view(-1, 1, 1)
        b1 = theta[:,3,0].view(-1, 1, 1)
        cf1 = theta[:,4,0].view(-1, 1, 1)
        r2 = theta[:,5,0].view(-1, 1, 1)
        m2 = theta[:,6,0].view(-1, 1, 1)
        I2 = theta[:,7,0].view(-1, 1, 1)
        b2 = theta[:,8,0].view(-1, 1, 1)
        cf2 = theta[:,9,0].view(-1, 1, 1)
        # x = x.view(100, 4, 1)
        q1, q2 = x[:, 0,:].view(-1, 1, 1), x[:, 1, :].view(-1, 1, 1)
        dot_q1, dot_q2 = x[:, 2, :].view(-1, 1, 1), x[:, 3, :].view(-1, 1, 1)

        # 定义矩阵计算中的中间变量
        cos_q2 = torch.cos(q2)
        sin_q1 = torch.sin(q1)
        sin_q2 = torch.sin(q2)
        sin_q1_q2 = torch.sin(q1 + q2)

        # 定义矩阵a中的元素
        a1 = dot_q1
        a2 = dot_q2

        numerator3 = (-0.3*dot_q1**2*m2*r2*sin_q2 - dot_q2*b2 - cf2*torch.atan(100*dot_q2) - 9.81*m2*r2*sin_q1_q2)*(I1*I2 - 6.0*I1*Ir + 0.3*I1*m2*r2*cos_q2 + I2**2 + 31.0*I2*Ir + 0.9*I2*m2*r2*cos_q2 + 0.09*I2*m2 - 222.0*Ir**2 + 7.5*Ir*m2*r2*cos_q2 - 0.54*Ir*m2 + 0.18*m2**2*r2**2*cos_q2**2 + 0.027*m2**2*r2*cos_q2)
        denominator3 = (-I1**2*I2 - 36.0*I1**2*Ir - I1*I2**2 - 158.0*I1*I2*Ir - 0.6*I1*I2*m2*r2*cos_q2 - 0.18*I1*I2*m2 - 2628.0*I1*Ir**2 - 46.8*I1*Ir*m2*r2*cos_q2 - 6.48*I1*Ir*m2 + 0.09*I1*m2**2*r2**2*cos_q2**2 - 85.0*I2**2*Ir - 0.09*I2**2*m2 - 4441.0*I2*Ir**2 - 76.2*I2*Ir*m2*r2*cos_q2 - 14.22*I2*Ir*m2 + 0.09*I2*m2**2*r2**2*cos_q2**2 - 0.054*I2*m2**2*r2*cos_q2 - 0.0081*I2*m2**2 - 47952.0*Ir**3 - 1710.0*Ir**2*m2*r2*cos_q2 - 236.52*Ir**2*m2 - 11.79*Ir*m2**2*r2**2*cos_q2**2 - 4.212*Ir*m2**2*r2*cos_q2 - 0.2916*Ir*m2**2 + 0.054*m2**3*r2**3*cos_q2**3 + 0.0081*m2**3*r2**2*cos_q2**2)
        numerator4 = (0.6*dot_q1*dot_q2*m2*r2*sin_q2 - dot_q1*b1 + 0.3*dot_q2**2*m2*r2*sin_q2 - cf1*torch.atan(100*dot_q1) - 9.81*m2*(r2*sin_q1_q2 + 0.3*sin_q1) - 5.96448*r1*sin_q1)*(-I1*I2 - 36.0*I1*Ir - I2**2 - 73.0*I2*Ir - 0.6*I2*m2*r2*cos_q2 - 0.09*I2*m2 - 1332.0*Ir**2 - 21.6*Ir*m2*r2*cos_q2 - 3.24*Ir*m2)
        denominator4 = (-I1**2*I2 - 36.0*I1**2*Ir - I1*I2**2 - 158.0*I1*I2*Ir - 0.6*I1*I2*m2*r2*cos_q2 - 0.18*I1*I2*m2 - 2628.0*I1*Ir**2 - 46.8*I1*Ir*m2*r2*cos_q2 - 6.48*I1*Ir*m2 + 0.09*I1*m2**2*r2**2*cos_q2**2 - 85.0*I2**2*Ir - 0.09*I2**2*m2 - 4441.0*I2*Ir**2 - 76.2*I2*Ir*m2*r2*cos_q2 - 14.22*I2*Ir*m2 + 0.09*I2*m2**2*r2**2*cos_q2**2 - 0.054*I2*m2**2*r2*cos_q2 - 0.0081*I2*m2**2 - 47952.0*Ir**3 - 1710.0*Ir**2*m2*r2*cos_q2 - 236.52*Ir**2*m2 - 11.79*Ir*m2**2*r2**2*cos_q2**2 - 4.212*Ir*m2**2*r2*cos_q2 - 0.2916*Ir*m2**2 + 0.054*m2**3*r2**3*cos_q2**3 + 0.0081*m2**3*r2**2*cos_q2**2)

        a3 = (numerator3 / denominator3) + (numerator4 / denominator4)

        numerator5 = (I2 - 6.0*Ir + 0.3*m2*r2*cos_q2)*(0.6*dot_q1*dot_q2*m2*r2*sin_q2 - dot_q1*b1 + 0.3*dot_q2**2*m2*r2*sin_q2 - cf1*torch.atan(100*dot_q1) - 9.81*m2*(r2*sin_q1_q2 + 0.3*sin_q1) - 5.96448*r1*sin_q1)
        denominator5 = (-I1*I2 - 36.0*I1*Ir - 85.0*I2*Ir - 0.09*I2*m2 - 1296.0*Ir**2 - 25.2*Ir*m2*r2*cos_q2 - 3.24*Ir*m2 + 0.09*m2**2*r2**2*cos_q2**2)
        numerator6 = (-0.3*dot_q1**2*m2*r2*sin_q2 - dot_q2*b2 - cf2*torch.atan(100*dot_q2) - 9.81*m2*r2*sin_q1_q2)*(-I1 - I2 - 37.0*Ir - 0.6*m2*r2*cos_q2 - 0.09*m2)
        denominator6 = (-I1*I2 - 36.0*I1*Ir - 85.0*I2*Ir - 0.09*I2*m2 - 1296.0*Ir**2 - 25.2*Ir*m2*r2*cos_q2 - 3.24*Ir*m2 + 0.09*m2**2*r2**2*cos_q2**2)

        a4 = (numerator5 / denominator5) + (numerator6 / denominator6)
        a = torch.cat((a1, a2, a3, a4), dim=1)
        a.requires_grad_(True)

        B11 = torch.zeros(n_samples, 1, 1, device=self.device,requires_grad=True)
        B12 = torch.zeros(n_samples, 1, 1, device=self.device,requires_grad=True)
        B21 = torch.zeros(n_samples, 1, 1, device=self.device,requires_grad=True)
        B22 = torch.zeros(n_samples, 1, 1, device=self.device,requires_grad=True)

        B31 = (-I1*I2 - 36.0*I1*Ir - I2**2 - 73.0*I2*Ir - 0.6*I2*m2*r2*cos_q2 - 0.09*I2*m2 - 1332.0*Ir**2 - 21.6*Ir*m2*r2*cos_q2 - 3.24*Ir*m2) / \
            (-I1**2*I2 - 36.0*I1**2*Ir - I1*I2**2 - 158.0*I1*I2*Ir - 0.6*I1*I2*m2*r2*cos_q2 - 0.18*I1*I2*m2 - 2628.0*I1*Ir**2 - 46.8*I1*Ir*m2*r2*cos_q2 - 6.48*I1*Ir*m2 + 0.09*I1*m2**2*r2**2*cos_q2**2 - 85.0*I2**2*Ir - 0.09*I2**2*m2 - 4441.0*I2*Ir**2 - 76.2*I2*Ir*m2*r2*cos_q2 - 14.22*I2*Ir*m2 + 0.09*I2*m2**2*r2**2*cos_q2**2 - 0.054*I2*m2**2*r2*cos_q2 - 0.0081*I2*m2**2 - 47952.0*Ir**3 - 1710.0*Ir**2*m2*r2*cos_q2 - 236.52*Ir**2*m2 - 11.79*Ir*m2**2*r2**2*cos_q2**2 - 4.212*Ir*m2**2*r2*cos_q2 - 0.2916*Ir*m2**2 + 0.054*m2**3*r2**3*cos_q2**3 + 0.0081*m2**3*r2**2*cos_q2**2)

        B32 = (I1*I2 - 6.0*I1*Ir + 0.3*I1*m2*r2*cos_q2 + I2**2 + 31.0*I2*Ir + 0.9*I2*m2*r2*cos_q2 + 0.09*I2*m2 - 222.0*Ir**2 + 7.5*Ir*m2*r2*cos_q2 - 0.54*Ir*m2 + 0.18*m2**2*r2**2*cos_q2**2 + 0.027*m2**2*r2*cos_q2) / \
            (-I1**2*I2 - 36.0*I1**2*Ir - I1*I2**2 - 158.0*I1*I2*Ir - 0.6*I1*I2*m2*r2*cos_q2 - 0.18*I1*I2*m2 - 2628.0*I1*Ir**2 - 46.8*I1*Ir*m2*r2*cos_q2 - 6.48*I1*Ir*m2 + 0.09*I1*m2**2*r2**2*cos_q2**2 - 85.0*I2**2*Ir - 0.09*I2**2*m2 - 4441.0*I2*Ir**2 - 76.2*I2*Ir*m2*r2*cos_q2 - 14.22*I2*Ir*m2 + 0.09*I2*m2**2*r2**2*cos_q2**2 - 0.054*I2*m2**2*r2*cos_q2 - 0.0081*I2*m2**2 - 47952.0*Ir**3 - 1710.0*Ir**2*m2*r2*cos_q2 - 236.52*Ir**2*m2 - 11.79*Ir*m2**2*r2**2*cos_q2**2 - 4.212*Ir*m2**2*r2*cos_q2 - 0.2916*Ir*m2**2 + 0.054*m2**3*r2**3*cos_q2**3 + 0.0081*m2**3*r2**2*cos_q2**2)

        B41 = (I2 - 6.0*Ir + 0.3*m2*r2*cos_q2) / \
            (-I1*I2 - 36.0*I1*Ir - 85.0*I2*Ir - 0.09*I2*m2 - 1296.0*Ir**2 - 25.2*Ir*m2*r2*cos_q2 - 3.24*Ir*m2 + 0.09*m2**2*r2**2*cos_q2**2)

        B42 = (-I1 - I2 - 37.0*Ir - 0.6*m2*r2*cos_q2 - 0.09*m2) / \
            (-I1*I2 - 36.0*I1*Ir - 85.0*I2*Ir - 0.09*I2*m2 - 1296.0*Ir**2 - 25.2*Ir*m2*r2*cos_q2 - 3.24*Ir*m2 + 0.09*m2**2*r2**2*cos_q2**2)

        row1 = torch.cat((B11, B12), dim=2)  # shape (n_samples, 1, 2)
        row2 = torch.cat((B21, B22), dim=2)  # shape (n_samples, 1, 2)
        row3 = torch.cat((B31, B32), dim=2)  # shape (n_samples, 1, 2)
        row4 = torch.cat((B41, B42), dim=2)  # shape (n_samples, 1, 2)

        # B shape is (n_samples, 4, 2)
        B = torch.cat((row1, row2, row3, row4), dim=1)  
        B.requires_grad_(True)

        assert a.shape == (n_samples, self.n_state, 1)
        assert B.shape == (n_samples, self.n_state, self.n_act)

        # assert torch.allclose(a_test, a, atol=1.e-3)
        # assert torch.allclose(B_test, B, atol=1.e-4)
        # print('a B correct')
        # result = torch.allclose(tensor1, tensor2, atol=1e-02)

        out = (a, B)
        
        if gradient:
            # # method 1
            
            # dadxT_x = self.symbolic_dadxT.subs(change_para)
            # dadxT_la = lambdify(self.symbol_x, dadxT_x)

            # dadx_test = torch.stack([
            #     torch.tensor(dadxT_la(*sample.squeeze().detach().cpu().numpy())).reshape(self.n_state, self.n_state).to(x.device)
            #     for sample in x
            # ]).float()

            # # bring theta to symbolic_dBdxT, since MutableDenseNDimArray can not be used in subs
            # dBdxT_x = smp.MutableDenseNDimArray.zeros(self.n_state, self.n_state, self.n_act)
            # for i in range(self.n_state):
            #     for j in range(self.n_state):
            #         for k in range(self.n_act):
            #             dBdxT_x[i, j, k] = self.symbolic_dBdxT[i, j, k].subs(change_para)

            # # end_time = time.time()
            # # execution_time = end_time - grad_start_time
            # # print(f"dyn subs execution time: {execution_time} seconds")

            # dBdxT_la = lambdify(self.symbol_x, dBdxT_x)

            # dBdx_test = torch.stack([
            #     torch.tensor(dBdxT_la(*sample.squeeze().detach().cpu().numpy())).reshape(self.n_state, self.n_state, self.n_act).to(x.device)
            #     for sample in x
            # ]).float()
            # assert dadx_test.shape == (n_samples, self.n_state, self.n_state,)
            # assert dBdx_test.shape == (n_samples, self.n_state, self.n_state, self.n_act)

            # # method 2
            # start_time = time.time()
            # dadx = torch.zeros(n_samples, self.n_state, self.n_state, device='cuda')
            # # 计算梯度
            # for i in range(self.n_state):
            #     # 计算 a[:, i, 0] 对 x 的梯度
            #     grad = torch.autograd.grad(a[:, i, 0], x, torch.ones_like(a[:, i, 0]), retain_graph=True, create_graph=True)[0]
            #     # 将结果放入 dadx 中对应的位置
            #     dadx[:, i, :] = grad[:, :, 0]
            # dadx = dadx.permute(0, 2, 1)

            # dBdx = torch.zeros(n_samples, self.n_state, self.n_state, self.n_act, device='cuda')
            # # 计算梯度

            # for i in range(self.n_state):
            #     for j in range(self.n_act):
            #         # 计算 B[:, i, j] 对 x 的梯度
            #         grad = torch.autograd.grad(outputs=B[:, i, j], inputs=x, grad_outputs=torch.ones_like(B[:, i, j]), retain_graph=True, create_graph=True)[0]
            #         dBdx[:, i, :, j] = grad[:, :, 0]

            # dBdx = dBdx.permute(0, 2, 1, 3)
            # assert dadx.shape == (n_samples, self.n_state, self.n_state,)
            # assert dBdx.shape == (n_samples, self.n_state, self.n_state, self.n_act)
            # out = (a, B, dadx, dBdx)
            # end_time = time.time()
            # execution_time = end_time - start_time
            # print(f"method 2 time: {execution_time} seconds")

            # metheod 3
            start_time = time.time()

            cos_q1 = torch.cos(q1)
            cos_q1_q2 = torch.cos(q1 + q2)
            atan_100_dot_q1 = torch.atan(100 * dot_q1)
            atan_100_dot_q2 = torch.atan(100 * dot_q2)

            dadxT = torch.zeros((4, 4, x.shape[0]), device=self.device)

            # Define each element of the matrix according to the given symbolic expressions
            def dadxT_element_0_2():
                term1 = -9.81 * m2 * r2 * (I1 * I2 - 6.0 * I1 * Ir + 0.3 * I1 * m2 * r2 * cos_q2 + I2**2 + 31.0 * I2 * Ir + 0.9 * I2 * m2 * r2 * cos_q2 + 0.09 * I2 * m2 - 222.0 * Ir**2 + 7.5 * Ir * m2 * r2 * cos_q2 - 0.54 * Ir * m2 + 0.18 * m2**2 * r2**2 * cos_q2**2 + 0.027 * m2**2 * r2 * cos_q2) * cos_q1_q2
                term1 /= (-I1**2 * I2 - 36.0 * I1**2 * Ir - I1 * I2**2 - 158.0 * I1 * I2 * Ir - 0.6 * I1 * I2 * m2 * r2 * cos_q2 - 0.18 * I1 * I2 * m2 - 2628.0 * I1 * Ir**2 - 46.8 * I1 * Ir * m2 * r2 * cos_q2 - 6.48 * I1 * Ir * m2 + 0.09 * I1 * m2**2 * r2**2 * cos_q2**2 - 85.0 * I2**2 * Ir - 0.09 * I2**2 * m2 - 4441.0 * I2 * Ir**2 - 76.2 * I2 * Ir * m2 * r2 * cos_q2 - 14.22 * I2 * Ir * m2 + 0.09 * I2 * m2**2 * r2**2 * cos_q2**2 - 0.054 * I2 * m2**2 * r2 * cos_q2 - 0.0081 * I2 * m2**2 - 47952.0 * Ir**3 - 1710.0 * Ir**2 * m2 * r2 * cos_q2 - 236.52 * Ir**2 * m2 - 11.79 * Ir * m2**2 * r2**2 * cos_q2**2 - 4.212 * Ir * m2**2 * r2 * cos_q2 - 0.2916 * Ir * m2**2 + 0.054 * m2**3 * r2**3 * cos_q2**3 + 0.0081 * m2**3 * r2**2 * cos_q2**2)
                term2 = (-9.81 * m2 * (r2 * cos_q1_q2 + 0.3 * cos_q1) - 5.96448 * r1 * cos_q1) * (-I1 * I2 - 36.0 * I1 * Ir - I2**2 - 73.0 * I2 * Ir - 0.6 * I2 * m2 * r2 * cos_q2 - 0.09 * I2 * m2 - 1332.0 * Ir**2 - 21.6 * Ir * m2 * r2 * cos_q2 - 3.24 * Ir * m2)
                term2 /= (-I1**2 * I2 - 36.0 * I1**2 * Ir - I1 * I2**2 - 158.0 * I1 * I2 * Ir - 0.6 * I1 * I2 * m2 * r2 * cos_q2 - 0.18 * I1 * I2 * m2 - 2628.0 * I1 * Ir**2 - 46.8 * I1 * Ir * m2 * r2 * cos_q2 - 6.48 * I1 * Ir * m2 + 0.09 * I1 * m2**2 * r2**2 * cos_q2**2 - 85.0 * I2**2 * Ir - 0.09 * I2**2 * m2 - 4441.0 * I2 * Ir**2 - 76.2 * I2 * Ir * m2 * r2 * cos_q2 - 14.22 * I2 * Ir * m2 + 0.09 * I2 * m2**2 * r2**2 * cos_q2**2 - 0.054 * I2 * m2**2 * r2 * cos_q2 - 0.0081 * I2 * m2**2 - 47952.0 * Ir**3 - 1710.0 * Ir**2 * m2 * r2 * cos_q2 - 236.52 * Ir**2 * m2 - 11.79 * Ir * m2**2 * r2**2 * cos_q2**2 - 4.212 * Ir * m2**2 * r2 * cos_q2 - 0.2916 * Ir * m2**2 + 0.054 * m2**3 * r2**3 * cos_q2**3 + 0.0081 * m2**3 * r2**2 * cos_q2**2)
                return term1 + term2

            def dadxT_element_0_3():
                term1 = -9.81 * m2 * r2 * (-I1 - I2 - 37.0 * Ir - 0.6 * m2 * r2 * cos_q2 - 0.09 * m2) * cos_q1_q2
                term1 /= (-I1 * I2 - 36.0 * I1 * Ir - 85.0 * I2 * Ir - 0.09 * I2 * m2 - 1296.0 * Ir**2 - 25.2 * Ir * m2 * r2 * cos_q2 - 3.24 * Ir * m2 + 0.09 * m2**2 * r2**2 * cos_q2**2)
                term2 = (-9.81 * m2 * (r2 * cos_q1_q2 + 0.3 * cos_q1) - 5.96448 * r1 * cos_q1) * (I2 - 6.0 * Ir + 0.3 * m2 * r2 * cos_q2)
                term2 /= (-I1 * I2 - 36.0 * I1 * Ir - 85.0 * I2 * Ir - 0.09 * I2 * m2 - 1296.0 * Ir**2 - 25.2 * Ir * m2 * r2 * cos_q2 - 3.24 * Ir * m2 + 0.09 * m2**2 * r2**2 * cos_q2**2)
                return term1 + term2

            def dadxT_element_1_2():
                term1 = (0.6 * I2 * m2 * r2 * sin_q2 + 21.6 * Ir * m2 * r2 * sin_q2) * (0.6 * dot_q1 * dot_q2 * m2 * r2 * sin_q2 - dot_q1 * b1 + 0.3 * dot_q2**2 * m2 * r2 * sin_q2 - cf1 * atan_100_dot_q1 - 9.81 * m2 * (r2 * sin_q1_q2 + 0.3 * sin_q1) - 5.96448 * r1 * sin_q1)
                term1 /= (-I1**2 * I2 - 36.0 * I1**2 * Ir - I1 * I2**2 - 158.0 * I1 * I2 * Ir - 0.6 * I1 * I2 * m2 * r2 * cos_q2 - 0.18 * I1 * I2 * m2 - 2628.0 * I1 * Ir**2 - 46.8 * I1 * Ir * m2 * r2 * cos_q2 - 6.48 * I1 * Ir * m2 + 0.09 * I1 * m2**2 * r2**2 * cos_q2**2 - 85.0 * I2**2 * Ir - 0.09 * I2**2 * m2 - 4441.0 * I2 * Ir**2 - 76.2 * I2 * Ir * m2 * r2 * cos_q2 - 14.22 * I2 * Ir * m2 + 0.09 * I2 * m2**2 * r2**2 * cos_q2**2 - 0.054 * I2 * m2**2 * r2 * cos_q2 - 0.0081 * I2 * m2**2 - 47952.0 * Ir**3 - 1710.0 * Ir**2 * m2 * r2 * cos_q2 - 236.52 * Ir**2 * m2 - 11.79 * Ir * m2**2 * r2**2 * cos_q2**2 - 4.212 * Ir * m2**2 * r2 * cos_q2 - 0.2916 * Ir * m2**2 + 0.054 * m2**3 * r2**3 * cos_q2**3 + 0.0081 * m2**3 * r2**2 * cos_q2**2)
                term2 = (-0.3 * dot_q1**2 * m2 * r2 * cos_q2 - 9.81 * m2 * r2 * cos_q1_q2) * (I1 * I2 - 6.0 * I1 * Ir + 0.3 * I1 * m2 * r2 * cos_q2 + I2**2 + 31.0 * I2 * Ir + 0.9 * I2 * m2 * r2 * cos_q2 + 0.09 * I2 * m2 - 222.0 * Ir**2 + 7.5 * Ir * m2 * r2 * cos_q2 - 0.54 * Ir * m2 + 0.18 * m2**2 * r2**2 * cos_q2**2 + 0.027 * m2**2 * r2 * cos_q2)
                term2 /= (-I1**2 * I2 - 36.0 * I1**2 * Ir - I1 * I2**2 - 158.0 * I1 * I2 * Ir - 0.6 * I1 * I2 * m2 * r2 * cos_q2 - 0.18 * I1 * I2 * m2 - 2628.0 * I1 * Ir**2 - 46.8 * I1 * Ir * m2 * r2 * cos_q2 - 6.48 * I1 * Ir * m2 + 0.09 * I1 * m2**2 * r2**2 * cos_q2**2 - 85.0 * I2**2 * Ir - 0.09 * I2**2 * m2 - 4441.0 * I2 * Ir**2 - 76.2 * I2 * Ir * m2 * r2 * cos_q2 - 14.22 * I2 * Ir * m2 + 0.09 * I2 * m2**2 * r2**2 * cos_q2**2 - 0.054 * I2 * m2**2 * r2 * cos_q2 - 0.0081 * I2 * m2**2 - 47952.0 * Ir**3 - 1710.0 * Ir**2 * m2 * r2 * cos_q2 - 236.52 * Ir**2 * m2 - 11.79 * Ir * m2**2 * r2**2 * cos_q2**2 - 4.212 * Ir * m2**2 * r2 * cos_q2 - 0.2916 * Ir * m2**2 + 0.054 * m2**3 * r2**3 * cos_q2**3 + 0.0081 * m2**3 * r2**2 * cos_q2**2)
                term3 = (0.6 * dot_q1 * dot_q2 * m2 * r2 * cos_q2 + 0.3 * dot_q2**2 * m2 * r2 * cos_q2 - 9.81 * m2 * r2 * cos_q1_q2) * (-I1 * I2 - 36.0 * I1 * Ir - I2**2 - 73.0 * I2 * Ir - 0.6 * I2 * m2 * r2 * cos_q2 - 0.09 * I2 * m2 - 1332.0 * Ir**2 - 21.6 * Ir * m2 * r2 * cos_q2 - 3.24 * Ir * m2)
                term3 /= (-I1**2 * I2 - 36.0 * I1**2 * Ir - I1 * I2**2 - 158.0 * I1 * I2 * Ir - 0.6 * I1 * I2 * m2 * r2 * cos_q2 - 0.18 * I1 * I2 * m2 - 2628.0 * I1 * Ir**2 - 46.8 * I1 * Ir * m2 * r2 * cos_q2 - 6.48 * I1 * Ir * m2 + 0.09 * I1 * m2**2 * r2**2 * cos_q2**2 - 85.0 * I2**2 * Ir - 0.09 * I2**2 * m2 - 4441.0 * I2 * Ir**2 - 76.2 * I2 * Ir * m2 * r2 * cos_q2 - 14.22 * I2 * Ir * m2 + 0.09 * I2 * m2**2 * r2**2 * cos_q2**2 - 0.054 * I2 * m2**2 * r2 * cos_q2 - 0.0081 * I2 * m2**2 - 47952.0 * Ir**3 - 1710.0 * Ir**2 * m2 * r2 * cos_q2 - 236.52 * Ir**2 * m2 - 11.79 * Ir * m2**2 * r2**2 * cos_q2**2 - 4.212 * Ir * m2**2 * r2 * cos_q2 - 0.2916 * Ir * m2**2 + 0.054 * m2**3 * r2**3 * cos_q2**3 + 0.0081 * m2**3 * r2**2 * cos_q2**2)
                term4 = (-0.3 * dot_q1**2 * m2 * r2 * sin_q2 - dot_q2 * b2 - cf2 * atan_100_dot_q2 - 9.81 * m2 * r2 * sin_q1_q2) * (-0.3 * I1 * m2 * r2 * sin_q2 - 0.9 * I2 * m2 * r2 * sin_q2 - 7.5 * Ir * m2 * r2 * sin_q2 - 0.36 * m2**2 * r2**2 * sin_q2 * cos_q2 - 0.027 * m2**2 * r2 * sin_q2)
                term4 /= (-I1**2 * I2 - 36.0 * I1**2 * Ir - I1 * I2**2 - 158.0 * I1 * I2 * Ir - 0.6 * I1 * I2 * m2 * r2 * cos_q2 - 0.18 * I1 * I2 * m2 - 2628.0 * I1 * Ir**2 - 46.8 * I1 * Ir * m2 * r2 * cos_q2 - 6.48 * I1 * Ir * m2 + 0.09 * I1 * m2**2 * r2**2 * cos_q2**2 - 85.0 * I2**2 * Ir - 0.09 * I2**2 * m2 - 4441.0 * I2 * Ir**2 - 76.2 * I2 * Ir * m2 * r2 * cos_q2 - 14.22 * I2 * Ir * m2 + 0.09 * I2 * m2**2 * r2**2 * cos_q2**2 - 0.054 * I2 * m2**2 * r2 * cos_q2 - 0.0081 * I2 * m2**2 - 47952.0 * Ir**3 - 1710.0 * Ir**2 * m2 * r2 * cos_q2 - 236.52 * Ir**2 * m2 - 11.79 * Ir * m2**2 * r2**2 * cos_q2**2 - 4.212 * Ir * m2**2 * r2 * cos_q2 - 0.2916 * Ir * m2**2 + 0.054 * m2**3 * r2**3 * cos_q2**3 + 0.0081 * m2**3 * r2**2 * cos_q2**2)
                term5 = 4.34897137154951e-10 * (-0.3 * dot_q1**2 * m2 * r2 * sin_q2 - dot_q2 * b2 - cf2 * atan_100_dot_q2 - 9.81 * m2 * r2 * sin_q1_q2) * (-0.6 * I1 * I2 * m2 * r2 * sin_q2 - 46.8 * I1 * Ir * m2 * r2 * sin_q2 + 0.18 * I1 * m2**2 * r2**2 * sin_q2 * cos_q2 - 76.2 * I2 * Ir * m2 * r2 * sin_q2 + 0.18 * I2 * m2**2 * r2**2 * sin_q2 * cos_q2 - 0.054 * I2 * m2**2 * r2 * sin_q2 - 1710.0 * Ir**2 * m2 * r2 * sin_q2 - 23.58 * Ir * m2**2 * r2**2 * sin_q2 * cos_q2 - 4.212 * Ir * m2**2 * r2 * sin_q2 + 0.162 * m2**3 * r2**3 * sin_q2 * cos_q2**2 + 0.0162 * m2**3 * r2**2 * sin_q2 * cos_q2)
                term5 *= (I1 * I2 - 6.0 * I1 * Ir + 0.3 * I1 * m2 * r2 * cos_q2 + I2**2 + 31.0 * I2 * Ir + 0.9 * I2 * m2 * r2 * cos_q2 + 0.09 * I2 * m2 - 222.0 * Ir**2 + 7.5 * Ir * m2 * r2 * cos_q2 - 0.54 * Ir * m2 + 0.18 * m2**2 * r2**2 * cos_q2**2 + 0.027 * m2**2 * r2 * cos_q2)
                term5 /= (-2.08541875208542e-5 * I1**2 * I2 - 0.000750750750750751 * I1**2 * Ir - 2.08541875208542e-5 * I1 * I2**2 - 0.00329496162829496 * I1 * I2 * Ir - 1.25125125125125e-5 * I1 * I2 * m2 * r2 * cos_q2 - 3.75375375375375e-6 * I1 * I2 * m2 - 0.0548048048048048 * I1 * Ir**2 - 0.000975975975975976 * I1 * Ir * m2 * r2 * cos_q2 - 0.000135135135135135 * I1 * Ir * m2 + 1.87687687687688e-6 * I1 * m2**2 * r2**2 * cos_q2**2 - 0.00177260593927261 * I2**2 * Ir - 1.87687687687688e-6 * I2**2 * m2 - 0.0926134467801134 * I2 * Ir**2 - 0.00158908908908909 * I2 * Ir * m2 * r2 * cos_q2 - 0.000296546546546547 * I2 * Ir * m2 + 1.87687687687688e-6 * I2 * m2**2 * r2**2 * cos_q2**2 - 1.12612612612613e-6 * I2 * m2**2 * r2 * cos_q2 - 1.68918918918919e-7 * I2 * m2**2 - Ir**3 - 0.0356606606606607 * Ir**2 * m2 * r2 * cos_q2 - 0.00493243243243243 * Ir**2 * m2 - 0.000245870870870871 * Ir * m2**2 * r2**2 * cos_q2**2 - 8.78378378378378e-5 * Ir * m2**2 * r2 * cos_q2 - 6.08108108108108e-6 * Ir * m2**2 + 1.12612612612613e-6 * m2**3 * r2**3 * cos_q2**3 + 1.68918918918919e-7 * m2**3 * r2**2 * cos_q2**2)**2
                term6 = 4.34897137154951e-10 * (0.6 * dot_q1 * dot_q2 * m2 * r2 * sin_q2 - dot_q1 * b1 + 0.3 * dot_q2**2 * m2 * r2 * sin_q2 - cf1 * atan_100_dot_q1 - 9.81 * m2 * (r2 * sin_q1_q2 + 0.3 * sin_q1) - 5.96448 * r1 * sin_q1) * (-I1 * I2 - 36.0 * I1 * Ir - I2**2 - 73.0 * I2 * Ir - 0.6 * I2 * m2 * r2 * cos_q2 - 0.09 * I2 * m2 - 1332.0 * Ir**2 - 21.6 * Ir * m2 * r2 * cos_q2 - 3.24 * Ir * m2)
                term6 *= (-0.6 * I1 * I2 * m2 * r2 * sin_q2 - 46.8 * I1 * Ir * m2 * r2 * sin_q2 + 0.18 * I1 * m2**2 * r2**2 * sin_q2 * cos_q2 - 76.2 * I2 * Ir * m2 * r2 * sin_q2 + 0.18 * I2 * m2**2 * r2**2 * sin_q2 * cos_q2 - 0.054 * I2 * m2**2 * r2 * sin_q2 - 1710.0 * Ir**2 * m2 * r2 * sin_q2 - 23.58 * Ir * m2**2 * r2**2 * sin_q2 * cos_q2 - 4.212 * Ir * m2**2 * r2 * sin_q2 + 0.162 * m2**3 * r2**3 * sin_q2 * cos_q2**2 + 0.0162 * m2**3 * r2**2 * sin_q2 * cos_q2)
                term6 /= (-2.08541875208542e-5 * I1**2 * I2 - 0.000750750750750751 * I1**2 * Ir - 2.08541875208542e-5 * I1 * I2**2 - 0.00329496162829496 * I1 * I2 * Ir - 1.25125125125125e-5 * I1 * I2 * m2 * r2 * cos_q2 - 3.75375375375375e-6 * I1 * I2 * m2 - 0.0548048048048048 * I1 * Ir**2 - 0.000975975975975976 * I1 * Ir * m2 * r2 * cos_q2 - 0.000135135135135135 * I1 * Ir * m2 + 1.87687687687688e-6 * I1 * m2**2 * r2**2 * cos_q2**2 - 0.00177260593927261 * I2**2 * Ir - 1.87687687687688e-6 * I2**2 * m2 - 0.0926134467801134 * I2 * Ir**2 - 0.00158908908908909 * I2 * Ir * m2 * r2 * cos_q2 - 0.000296546546546547 * I2 * Ir * m2 + 1.87687687687688e-6 * I2 * m2**2 * r2**2 * cos_q2**2 - 1.12612612612613e-6 * I2 * m2**2 * r2 * cos_q2 - 1.68918918918919e-7 * I2 * m2**2 - Ir**3 - 0.0356606606606607 * Ir**2 * m2 * r2 * cos_q2 - 0.00493243243243243 * Ir**2 * m2 - 0.000245870870870871 * Ir * m2**2 * r2**2 * cos_q2**2 - 8.78378378378378e-5 * Ir * m2**2 * r2 * cos_q2 - 6.08108108108108e-6 * Ir * m2**2 + 1.12612612612613e-6 * m2**3 * r2**3 * cos_q2**3 + 1.68918918918919e-7 * m2**3 * r2**2 * cos_q2**2)**2
                return term1 + term2 + term3 + term4 + term5 + term6

            def dadxT_element_1_3():
                term1 = 0.6 * m2 * r2 * (-0.3 * dot_q1**2 * m2 * r2 * sin_q2 - dot_q2 * b2 - cf2 * atan_100_dot_q2 - 9.81 * m2 * r2 * sin_q1_q2) * sin_q2
                term1 /= (-I1 * I2 - 36.0 * I1 * Ir - 85.0 * I2 * Ir - 0.09 * I2 * m2 - 1296.0 * Ir**2 - 25.2 * Ir * m2 * r2 * cos_q2 - 3.24 * Ir * m2 + 0.09 * m2**2 * r2**2 * cos_q2**2)
                term2 = -0.3 * m2 * r2 * (0.6 * dot_q1 * dot_q2 * m2 * r2 * sin_q2 - dot_q1 * b1 + 0.3 * dot_q2**2 * m2 * r2 * sin_q2 - cf1 * atan_100_dot_q1 - 9.81 * m2 * (r2 * sin_q1_q2 + 0.3 * sin_q1) - 5.96448 * r1 * sin_q1) * sin_q2
                term2 /= (-I1 * I2 - 36.0 * I1 * Ir - 85.0 * I2 * Ir - 0.09 * I2 * m2 - 1296.0 * Ir**2 - 25.2 * Ir * m2 * r2 * cos_q2 - 3.24 * Ir * m2 + 0.09 * m2**2 * r2**2 * cos_q2**2)
                term3 = 5.95374180765127e-7 * (-25.2 * Ir * m2 * r2 * sin_q2 + 0.18 * m2**2 * r2**2 * sin_q2 * cos_q2) * (I2 - 6.0 * Ir + 0.3 * m2 * r2 * cos_q2) * (0.6 * dot_q1 * dot_q2 * m2 * r2 * sin_q2 - dot_q1 * b1 + 0.3 * dot_q2**2 * m2 * r2 * sin_q2 - cf1 * atan_100_dot_q1 - 9.81 * m2 * (r2 * sin_q1_q2 + 0.3 * sin_q1) - 5.96448 * r1 * sin_q1)
                term3 /= (-0.000771604938271605 * I1 * I2 - 0.0277777777777778 * I1 * Ir - 0.0655864197530864 * I2 * Ir - 6.94444444444444e-5 * I2 * m2 - Ir**2 - 0.0194444444444444 * Ir * m2 * r2 * cos_q2 - 0.0025 * Ir * m2 + 6.94444444444444e-5 * m2**2 * r2**2 * cos_q2**2)**2
                term4 = 5.95374180765127e-7 * (-25.2 * Ir * m2 * r2 * sin_q2 + 0.18 * m2**2 * r2**2 * sin_q2 * cos_q2) * (-0.3 * dot_q1**2 * m2 * r2 * sin_q2 - dot_q2 * b2 - cf2 * atan_100_dot_q2 - 9.81 * m2 * r2 * sin_q1_q2) * (-I1 - I2 - 37.0 * Ir - 0.6 * m2 * r2 * cos_q2 - 0.09 * m2)
                term4 /= (-0.000771604938271605 * I1 * I2 - 0.0277777777777778 * I1 * Ir - 0.0655864197530864 * I2 * Ir - 6.94444444444444e-5 * I2 * m2 - Ir**2 - 0.0194444444444444 * Ir * m2 * r2 * cos_q2 - 0.0025 * Ir * m2 + 6.94444444444444e-5 * m2**2 * r2**2 * cos_q2**2)**2
                term5 = (-0.3 * dot_q1**2 * m2 * r2 * cos_q2 - 9.81 * m2 * r2 * cos_q1_q2) * (-I1 - I2 - 37.0 * Ir - 0.6 * m2 * r2 * cos_q2 - 0.09 * m2)
                term5 /= (-I1 * I2 - 36.0 * I1 * Ir - 85.0 * I2 * Ir - 0.09 * I2 * m2 - 1296.0 * Ir**2 - 25.2 * Ir * m2 * r2 * cos_q2 - 3.24 * Ir * m2 + 0.09 * m2**2 * r2**2 * cos_q2**2)
                term6 = (I2 - 6.0 * Ir + 0.3 * m2 * r2 * cos_q2) * (0.6 * dot_q1 * dot_q2 * m2 * r2 * cos_q2 + 0.3 * dot_q2**2 * m2 * r2 * cos_q2 - 9.81 * m2 * r2 * cos_q1_q2)
                term6 /= (-I1 * I2 - 36.0 * I1 * Ir - 85.0 * I2 * Ir - 0.09 * I2 * m2 - 1296.0 * Ir**2 - 25.2 * Ir * m2 * r2 * cos_q2 - 3.24 * Ir * m2 + 0.09 * m2**2 * r2**2 * cos_q2**2)
                return term1 + term2 + term3 + term4 + term5 + term6

            def dadxT_element_2_2():
                term1 = -0.6 * dot_q1 * m2 * r2 * (
                    I1 * I2 - 6.0 * I1 * Ir + 0.3 * I1 * m2 * r2 * torch.cos(q2) + I2**2 + 31.0 * I2 * Ir + 0.9 * I2 * m2 * r2 * torch.cos(q2) + 0.09 * I2 * m2 - 222.0 * Ir**2 + 7.5 * Ir * m2 * r2 * torch.cos(q2) - 0.54 * Ir * m2 + 0.18 * m2**2 * r2**2 * torch.cos(q2)**2 + 0.027 * m2**2 * r2 * torch.cos(q2)
                ) * torch.sin(q2)

                term2 = (0.6 * dot_q2 * m2 * r2 * torch.sin(q2) - b1 - 100 * cf1 / (10000 * dot_q1**2 + 1)) * (
                    -I1 * I2 - 36.0 * I1 * Ir - I2**2 - 73.0 * I2 * Ir - 0.6 * I2 * m2 * r2 * torch.cos(q2) - 0.09 * I2 * m2 - 1332.0 * Ir**2 - 21.6 * Ir * m2 * r2 * torch.cos(q2) - 3.24 * Ir * m2
                )

                denom1 = -I1**2 * I2 - 36.0 * I1**2 * Ir - I1 * I2**2 - 158.0 * I1 * I2 * Ir - 0.6 * I1 * I2 * m2 * r2 * torch.cos(q2) - 0.18 * I1 * I2 * m2 - 2628.0 * I1 * Ir**2 - 46.8 * I1 * Ir * m2 * r2 * torch.cos(q2) - 6.48 * I1 * Ir * m2 + 0.09 * I1 * m2**2 * r2**2 * torch.cos(q2)**2 - 85.0 * I2**2 * Ir - 0.09 * I2**2 * m2 - 4441.0 * I2 * Ir**2 - 76.2 * I2 * Ir * m2 * r2 * torch.cos(q2) - 14.22 * I2 * Ir * m2 + 0.09 * I2 * m2**2 * r2**2 * torch.cos(q2)**2 - 0.054 * I2 * m2**2 * r2 * torch.cos(q2) - 0.0081 * I2 * m2**2 - 47952.0 * Ir**3 - 1710.0 * Ir**2 * m2 * r2 * torch.cos(q2) - 236.52 * Ir**2 * m2 - 11.79 * Ir * m2**2 * r2**2 * torch.cos(q2)**2 - 4.212 * Ir * m2**2 * r2 * torch.cos(q2) - 0.2916 * Ir * m2**2 + 0.054 * m2**3 * r2**3 * torch.cos(q2)**3 + 0.0081 * m2**3 * r2**2 * torch.cos(q2)**2

                return (term1 + term2) / denom1

            def dadxT_element_2_3():
                term1 = -0.6 * dot_q1 * m2 * r2 * (-I1 - I2 - 37.0 * Ir - 0.6 * m2 * r2 * torch.cos(q2) - 0.09 * m2) * torch.sin(q2)

                term2 = (I2 - 6.0 * Ir + 0.3 * m2 * r2 * torch.cos(q2)) * (0.6 * dot_q2 * m2 * r2 * torch.sin(q2) - b1 - 100 * cf1 / (10000 * dot_q1**2 + 1))

                denom1 = -I1 * I2 - 36.0 * I1 * Ir - 85.0 * I2 * Ir - 0.09 * I2 * m2 - 1296.0 * Ir**2 - 25.2 * Ir * m2 * r2 * torch.cos(q2) - 3.24 * Ir * m2 + 0.09 * m2**2 * r2**2 * torch.cos(q2)**2

                return (term1 + term2) / denom1

            def dadxT_element_3_2():
                term1 = (-b2 - 100 * cf2 / (10000 * dot_q2**2 + 1)) * (
                    I1 * I2 - 6.0 * I1 * Ir + 0.3 * I1 * m2 * r2 * torch.cos(q2) + I2**2 + 31.0 * I2 * Ir + 0.9 * I2 * m2 * r2 * torch.cos(q2) + 0.09 * I2 * m2 - 222.0 * Ir**2 + 7.5 * Ir * m2 * r2 * torch.cos(q2) - 0.54 * Ir * m2 + 0.18 * m2**2 * r2**2 * torch.cos(q2)**2 + 0.027 * m2**2 * r2 * torch.cos(q2)
                )

                term2 = (0.6 * dot_q1 * m2 * r2 * torch.sin(q2) + 0.6 * dot_q2 * m2 * r2 * torch.sin(q2)) * (
                    -I1 * I2 - 36.0 * I1 * Ir - I2**2 - 73.0 * I2 * Ir - 0.6 * I2 * m2 * r2 * torch.cos(q2) - 0.09 * I2 * m2 - 1332.0 * Ir**2 - 21.6 * Ir * m2 * r2 * torch.cos(q2) - 3.24 * Ir * m2
                )

                denom1 = -I1**2 * I2 - 36.0 * I1**2 * Ir - I1 * I2**2 - 158.0 * I1 * I2 * Ir - 0.6 * I1 * I2 * m2 * r2 * torch.cos(q2) - 0.18 * I1 * I2 * m2 - 2628.0 * I1 * Ir**2 - 46.8 * I1 * Ir * m2 * r2 * torch.cos(q2) - 6.48 * I1 * Ir * m2 + 0.09 * I1 * m2**2 * r2**2 * torch.cos(q2)**2 - 85.0 * I2**2 * Ir - 0.09 * I2**2 * m2 - 4441.0 * I2 * Ir**2 - 76.2 * I2 * Ir * m2 * r2 * torch.cos(q2) - 14.22 * I2 * Ir * m2 + 0.09 * I2 * m2**2 * r2**2 * torch.cos(q2)**2 - 0.054 * I2 * m2**2 * r2 * torch.cos(q2) - 0.0081 * I2 * m2**2 - 47952.0 * Ir**3 - 1710.0 * Ir**2 * m2 * r2 * torch.cos(q2) - 236.52 * Ir**2 * m2 - 11.79 * Ir * m2**2 * r2**2 * torch.cos(q2)**2 - 4.212 * Ir * m2**2 * r2 * torch.cos(q2) - 0.2916 * Ir * m2**2 + 0.054 * m2**3 * r2**3 * torch.cos(q2)**3 + 0.0081 * m2**3 * r2**2 * torch.cos(q2)**2

                return term1 / denom1 + term2 / denom1

            def dadxT_element_3_3():
                term1 = (-b2 - 100 * cf2 / (10000 * dot_q2**2 + 1)) * (-I1 - I2 - 37.0 * Ir - 0.6 * m2 * r2 * torch.cos(q2) - 0.09 * m2)

                term2 = (0.6 * dot_q1 * m2 * r2 * torch.sin(q2) + 0.6 * dot_q2 * m2 * r2 * torch.sin(q2)) * (I2 - 6.0 * Ir + 0.3 * m2 * r2 * torch.cos(q2))

                denom1 = -I1 * I2 - 36.0 * I1 * Ir - 85.0 * I2 * Ir - 0.09 * I2 * m2 - 1296.0 * Ir**2 - 25.2 * Ir * m2 * r2 * torch.cos(q2) - 3.24 * Ir * m2 + 0.09 * m2**2 * r2**2 * torch.cos(q2)**2

                return term1 / denom1 + term2 / denom1
            
            dadxT[0, 2, :] = dadxT_element_0_2().view(-1)
            dadxT[0, 3, :] = dadxT_element_0_3().view(-1)
            dadxT[1, 2, :] = dadxT_element_1_2().view(-1)
            dadxT[1, 3, :] = dadxT_element_1_3().view(-1)

            dadxT[2, 2, :] = dadxT_element_2_2().view(-1)
            dadxT[2, 3, :] = dadxT_element_2_3().view(-1)
            dadxT[3, 2, :] = dadxT_element_3_2().view(-1)
            dadxT[3, 3, :] = dadxT_element_3_3().view(-1)

            dadxT[2, 0, :] = torch.ones_like(x[:, 0, 0],device=self.device)
            dadxT[3, 1, :] = torch.ones_like(x[:, 0, 0],device=self.device)

            dadxT  = dadxT.permute(2, 0, 1)

            dBdxT = torch.zeros((4, 4, 2, x.shape[0]), device=self.device)

            def element_1_2_0():
                term1 = (0.6 * I2 * m2 * r2 * sin_q2 + 21.6 * Ir * m2 * r2 * sin_q2) / (-I1**2 * I2 - 36.0 * I1**2 * Ir - I1 * I2**2 - 158.0 * I1 * I2 * Ir - 0.6 * I1 * I2 * m2 * r2 * cos_q2 - 0.18 * I1 * I2 * m2 - 2628.0 * I1 * Ir**2 - 46.8 * I1 * Ir * m2 * r2 * cos_q2 - 6.48 * I1 * Ir * m2 + 0.09 * I1 * m2**2 * r2**2 * cos_q2**2 - 85.0 * I2**2 * Ir - 0.09 * I2**2 * m2 - 4441.0 * I2 * Ir**2 - 76.2 * I2 * Ir * m2 * r2 * cos_q2 - 14.22 * I2 * Ir * m2 + 0.09 * I2 * m2**2 * r2**2 * cos_q2**2 - 0.054 * I2 * m2**2 * r2 * cos_q2 - 0.0081 * I2 * m2**2 - 47952.0 * Ir**3 - 1710.0 * Ir**2 * m2 * r2 * cos_q2 - 236.52 * Ir**2 * m2 - 11.79 * Ir * m2**2 * r2**2 * cos_q2**2 - 4.212 * Ir * m2**2 * r2 * cos_q2 - 0.2916 * Ir * m2**2 + 0.054 * m2**3 * r2**3 * cos_q2**3 + 0.0081 * m2**3 * r2**2 * cos_q2**2)
                term2 = 4.34897137154951e-10 * (-I1 * I2 - 36.0 * I1 * Ir - I2**2 - 73.0 * I2 * Ir - 0.6 * I2 * m2 * r2 * cos_q2 - 0.09 * I2 * m2 - 1332.0 * Ir**2 - 21.6 * Ir * m2 * r2 * cos_q2 - 3.24 * Ir * m2) * (-0.6 * I1 * I2 * m2 * r2 * sin_q2 - 46.8 * I1 * Ir * m2 * r2 * sin_q2 + 0.18 * I1 * m2**2 * r2**2 * sin_q2 * cos_q2 - 76.2 * I2 * Ir * m2 * r2 * sin_q2 + 0.18 * I2 * m2**2 * r2**2 * sin_q2 * cos_q2 - 0.054 * I2 * m2**2 * r2 * sin_q2 - 1710.0 * Ir**2 * m2 * r2 * sin_q2 - 23.58 * Ir * m2**2 * r2**2 * sin_q2 * cos_q2 - 4.212 * Ir * m2**2 * r2 * sin_q2 + 0.162 * m2**3 * r2**3 * sin_q2 * cos_q2**2 + 0.0162 * m2**3 * r2**2 * sin_q2 * cos_q2) / (-2.08541875208542e-5 * I1**2 * I2 - 0.000750750750750751 * I1**2 * Ir - 2.08541875208542e-5 * I1 * I2**2 - 0.00329496162829496 * I1 * I2 * Ir - 1.25125125125125e-5 * I1 * I2 * m2 * r2 * cos_q2 - 3.75375375375375e-6 * I1 * I2 * m2 - 0.0548048048048048 * I1 * Ir**2 - 0.000975975975975976 * I1 * Ir * m2 * r2 * cos_q2 - 0.000135135135135135 * I1 * Ir * m2 + 1.87687687687688e-6 * I1 * m2**2 * r2**2 * cos_q2**2 - 0.00177260593927261 * I2**2 * Ir - 1.87687687687688e-6 * I2**2 * m2 - 0.0926134467801134 * I2 * Ir**2 - 0.00158908908908909 * I2 * Ir * m2 * r2 * cos_q2 - 0.000296546546546547 * I2 * Ir * m2 + 1.87687687687688e-6 * I2 * m2**2 * r2**2 * cos_q2**2 - 1.12612612612613e-6 * I2 * m2**2 * r2 * cos_q2 - 1.68918918918919e-7 * I2 * m2**2 - Ir**3 - 0.0356606606606607 * Ir**2 * m2 * r2 * cos_q2 - 0.00493243243243243 * Ir**2 * m2 - 0.000245870870870871 * Ir * m2**2 * r2**2 * cos_q2**2 - 8.78378378378378e-5 * Ir * m2**2 * r2 * cos_q2 - 6.08108108108108e-6 * Ir * m2**2 + 1.12612612612613e-6 * m2**3 * r2**3 * cos_q2**3 + 1.68918918918919e-7 * m2**3 * r2**2 * cos_q2**2)**2
                return term1 + term2

            def element_1_2_1():
                term1 = (-0.3 * I1 * m2 * r2 * sin_q2 - 0.9 * I2 * m2 * r2 * sin_q2 - 7.5 * Ir * m2 * r2 * sin_q2 - 0.36 * m2**2 * r2**2 * sin_q2 * cos_q2 - 0.027 * m2**2 * r2 * sin_q2) / (-I1**2 * I2 - 36.0 * I1**2 * Ir - I1 * I2**2 - 158.0 * I1 * I2 * Ir - 0.6 * I1 * I2 * m2 * r2 * cos_q2 - 0.18 * I1 * I2 * m2 - 2628.0 * I1 * Ir**2 - 46.8 * I1 * Ir * m2 * r2 * cos_q2 - 6.48 * I1 * Ir * m2 + 0.09 * I1 * m2**2 * r2**2 * cos_q2**2 - 85.0 * I2**2 * Ir - 0.09 * I2**2 * m2 - 4441.0 * I2 * Ir**2 - 76.2 * I2 * Ir * m2 * r2 * cos_q2 - 14.22 * I2 * Ir * m2 + 0.09 * I2 * m2**2 * r2**2 * cos_q2**2 - 0.054 * I2 * m2**2 * r2 * cos_q2 - 0.0081 * I2 * m2**2 - 47952.0 * Ir**3 - 1710.0 * Ir**2 * m2 * r2 * cos_q2 - 236.52 * Ir**2 * m2 - 11.79 * Ir * m2**2 * r2**2 * cos_q2**2 - 4.212 * Ir * m2**2 * r2 * cos_q2 - 0.2916 * Ir * m2**2 + 0.054 * m2**3 * r2**3 * cos_q2**3 + 0.0081 * m2**3 * r2**2 * cos_q2**2)
                term2 = 4.34897137154951e-10 * (-0.6 * I1 * I2 * m2 * r2 * sin_q2 - 46.8 * I1 * Ir * m2 * r2 * sin_q2 + 0.18 * I1 * m2**2 * r2**2 * sin_q2 * cos_q2 - 76.2 * I2 * Ir * m2 * r2 * sin_q2 + 0.18 * I2 * m2**2 * r2**2 * sin_q2 * cos_q2 - 0.054 * I2 * m2**2 * r2 * sin_q2 - 1710.0 * Ir**2 * m2 * r2 * sin_q2 - 23.58 * Ir * m2**2 * r2**2 * sin_q2 * cos_q2 - 4.212 * Ir * m2**2 * r2 * sin_q2 + 0.162 * m2**3 * r2**3 * sin_q2 * cos_q2**2 + 0.0162 * m2**3 * r2**2 * sin_q2 * cos_q2) * (I1 * I2 - 6.0 * I1 * Ir + 0.3 * I1 * m2 * r2 * cos_q2 + I2**2 + 31.0 * I2 * Ir + 0.9 * I2 * m2 * r2 * cos_q2 + 0.09 * I2 * m2 - 222.0 * Ir**2 + 7.5 * Ir * m2 * r2 * cos_q2 - 0.54 * Ir * m2 + 0.18 * m2**2 * r2**2 * cos_q2**2 + 0.027 * m2**2 * r2 * cos_q2) / (-2.08541875208542e-5 * I1**2 * I2 - 0.000750750750750751 * I1**2 * Ir - 2.08541875208542e-5 * I1 * I2**2 - 0.00329496162829496 * I1 * I2 * Ir - 1.25125125125125e-5 * I1 * I2 * m2 * r2 * cos_q2 - 3.75375375375375e-6 * I1 * I2 * m2 - 0.0548048048048048 * I1 * Ir**2 - 0.000975975975975976 * I1 * Ir * m2 * r2 * cos_q2 - 0.000135135135135135 * I1 * Ir * m2 + 1.87687687687688e-6 * I1 * m2**2 * r2**2 * cos_q2**2 - 0.00177260593927261 * I2**2 * Ir - 1.87687687687688e-6 * I2**2 * m2 - 0.0926134467801134 * I2 * Ir**2 - 0.00158908908908909 * I2 * Ir * m2 * r2 * cos_q2 - 0.000296546546546547 * I2 * Ir * m2 + 1.87687687687688e-6 * I2 * m2**2 * r2**2 * cos_q2**2 - 1.12612612612613e-6 * I2 * m2**2 * r2 * cos_q2 - 1.68918918918919e-7 * I2 * m2**2 - Ir**3 - 0.0356606606606607 * Ir**2 * m2 * r2 * cos_q2 - 0.00493243243243243 * Ir**2 * m2 - 0.000245870870870871 * Ir * m2**2 * r2**2 * cos_q2**2 - 8.78378378378378e-5 * Ir * m2**2 * r2 * cos_q2 - 6.08108108108108e-6 * Ir * m2**2 + 1.12612612612613e-6 * m2**3 * r2**3 * cos_q2**3 + 1.68918918918919e-7 * m2**3 * r2**2 * cos_q2**2)**2
                return term1 + term2

            def element_1_3_0():
                term1 = -0.3 * m2 * r2 * sin_q2 / (-I1 * I2 - 36.0 * I1 * Ir - 85.0 * I2 * Ir - 0.09 * I2 * m2 - 1296.0 * Ir**2 - 25.2 * Ir * m2 * r2 * cos_q2 - 3.24 * Ir * m2 + 0.09 * m2**2 * r2**2 * cos_q2**2)
                term2 = 5.95374180765127e-7 * (-25.2 * Ir * m2 * r2 * sin_q2 + 0.18 * m2**2 * r2**2 * sin_q2 * cos_q2) * (I2 - 6.0 * Ir + 0.3 * m2 * r2 * cos_q2) / (-0.000771604938271605 * I1 * I2 - 0.0277777777777778 * I1 * Ir - 0.0655864197530864 * I2 * Ir - 6.94444444444444e-5 * I2 * m2 - Ir**2 - 0.0194444444444444 * Ir * m2 * r2 * cos_q2 - 0.0025 * Ir * m2 + 6.94444444444444e-5 * m2**2 * r2**2 * cos_q2**2)**2
                return term1 + term2

            def element_1_3_1():
                term1 = 0.6 * m2 * r2 * sin_q2 / (-I1 * I2 - 36.0 * I1 * Ir - 85.0 * I2 * Ir - 0.09 * I2 * m2 - 1296.0 * Ir**2 - 25.2 * Ir * m2 * r2 * cos_q2 - 3.24 * Ir * m2 + 0.09 * m2**2 * r2**2 * cos_q2**2)
                term2 = 5.95374180765127e-7 * (-25.2 * Ir * m2 * r2 * sin_q2 + 0.18 * m2**2 * r2**2 * sin_q2 * cos_q2) * (-I1 - I2 - 37.0 * Ir - 0.6 * m2 * r2 * cos_q2 - 0.09 * m2) / (-0.000771604938271605 * I1 * I2 - 0.0277777777777778 * I1 * Ir - 0.0655864197530864 * I2 * Ir - 6.94444444444444e-5 * I2 * m2 - Ir**2 - 0.0194444444444444 * Ir * m2 * r2 * cos_q2 - 0.0025 * Ir * m2 + 6.94444444444444e-5 * m2**2 * r2**2 * cos_q2**2)**2
                return term1 + term2

            dBdxT[1, 2, 0, :] = element_1_2_0().view(-1)
            dBdxT[1, 2, 1, :] = element_1_2_1().view(-1)
            dBdxT[1, 3, 0, :] = element_1_3_0().view(-1)
            dBdxT[1, 3, 1, :] = element_1_3_1().view(-1)
            dBdxT = dBdxT.permute(3, 0, 1, 2)

            out = (a, B, dadxT, dBdxT)
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"method 3 time: {execution_time} seconds")

        # if is_numpy:
        #     out = [array.numpy() for array in out]


        # end_time = time.time()
        # execution_time = end_time - start_time
        # print(f"dyn time: {execution_time} seconds")

        return out

    def grad_dyn_theta(self, x):

        print("grad_dyn_theta____________________________________________________")
        # print(f"x is on device: {x.device}")
        # print('x',x.shape)
        

        is_numpy = True if isinstance(x, np.ndarray) else False
        x = torch.from_numpy(x) if isinstance(x, np.ndarray) else x
        x = x.view(-1, self.n_state, 1)
        # print(f"x is on device: {x.device}")
        n_samples = x.shape[0]

        start_time = time.time()
        dadp = compute_dadp_x_matrix(x)
        dBdp = compute_dBdp_x_matrix(x)
        out = (dadp, dBdp)

        assert dadp.shape == (n_samples, self.n_parameter, self.n_state)
        assert dBdp.shape == (n_samples, self.n_parameter, self.n_state, self.n_act)
        
        # if is_numpy:
        #     out = [array.cpu().detach().numpy() for array in out]
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"grad_dyn_theta execution time: {execution_time} seconds")
        return out

    def cuda(self, device=None):
        self.u_lim = self.u_lim.cuda(device=device)
        self.theta_min = self.theta_min.cuda(device=device)
        self.theta = self.theta.cuda(device=device)
        self.theta_max = self.theta_max.cuda(device=device)
        self.device = self.theta.device
        return self

    def cpu(self):
        self.u_lim = self.u_lim.cpu()
        self.theta_min = self.theta_min.cpu()
        self.theta = self.theta.cpu()
        self.theta_max = self.theta_max.cpu()
        self.device = self.theta.device
        return self


class DoulbePendulumLogCos(DoulbePendulum):
    name = "DoulbePendulum_LogCosCost"

    def __init__(self, Q, R, cuda=False, **kwargs):

        # Create the dynamics:
        super(DoulbePendulumLogCos, self).__init__(cuda=cuda, **kwargs)
        self.u_lim = torch.tensor([[6.], [0.00001]])
        device = torch.device("cuda" if cuda else "cpu")

        # Create the Reward Function:
        assert Q.size == self.n_state and np.all(Q > 0.0)
        self.Q = np.diag(Q).reshape((self.n_state, self.n_state))

        assert R.size == self.n_act and np.all(R > 0.0)
        self.R = np.diag(R).reshape((self.n_act, self.n_act))

        self._q = SineQuadraticCost(self.Q, np.array([1.0, 1.0, 0.0, 0.0]), cuda=cuda)
        self.q = BarrierCost(self._q,  self.x_penalty, cuda)
        # print('self.R:',self.u_lim.view(-1,1)*self.R)
        # Determine beta s.t. the curvature at u = 0 is identical to 2R
        beta = 4. * self.u_lim ** 2 / np.pi * self.R
        beta = torch.diag(beta).view(self.n_act, 1).to(device)
        # beta = torch.tensor([40,40]) 
        self.u_lim = self.u_lim.to(device)
        self.r = ArcTangent(alpha=self.u_lim, beta=beta)

    def rwd(self, x, u):
        return self.q(x-self.x_target.view(-1,1).to(x.device)) + self.r(u)

    def cuda(self, device=None):
        super(DoulbePendulumLogCos, self).cuda(device=device)
        self.q.cuda(device=device)
        return self

    def cpu(self):
        super(DoulbePendulumLogCos, self).cpu()
        self.q.cpu()
        return self


if __name__ == "__main__":
    from deep_differential_network.utils import jacobian
    sys = DoulbePendulum(cuda=True)

    # # GPU vs. CPU:
    # cuda = True

    # # Seed the test:
    # seed = 42
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)

    # # Create system:
    # sys = DoulbePendulum()

    # n_samples = n_samples
    # x_lim = torch.from_numpy(sys.x_lim).float() if isinstance(sys.x_lim, np.ndarray) else sys.x_lim
    # x_test = torch.distributions.uniform.Uniform(-x_lim, x_lim).sample((n_samples,))
    # # x_test = torch.tensor([np.pi / 2., 0.5]).view(1, sys.n_state, 1)

    # dtheta = torch.zeros(1, sys.n_parameter, 1)

    # if cuda:
    #     sys, x_test, dtheta = sys.cuda(), x_test.cuda(), dtheta.cuda()

    # ###################################################################################################################
    # # Test dynamics gradient w.r.t. state:
    # dadx_shape = (n_samples, sys.n_state, sys.n_state)
    # dBdx_shape = (n_samples, sys.n_state, sys.n_state, sys.n_act)

    # a, B, dadx, dBdx = sys.dyn(x_test, gradient=True)

    # dadx_auto = torch.cat([jacobian(lambda x: sys.dyn(x)[0], x_test[i:i+1]) for i in range(n_samples)], dim=0)
    # dBdx_auto = torch.cat([jacobian(lambda x: sys.dyn(x)[1], x_test[i:i+1]) for i in range(n_samples)], dim=0)

    # err_a = (dadx_auto.view(dadx_shape) - dadx).abs().sum() / n_samples
    # err_B = (dBdx_auto.view(dBdx_shape) - dBdx).abs().sum() / n_samples
    # assert err_a <= 1.e-5 and err_B <= 1.e-6

    # ###################################################################################################################
    # # Test dynamics gradient w.r.t. model parameter:
    # dadp_shape = (n_samples, sys.n_parameter, sys.n_state)
    # dBdp_shape = (n_samples, sys.n_parameter, sys.n_state, sys.n_act)

    # dadp, dBdp = sys.grad_dyn_theta(x_test)

    # dadp_auto = torch.cat([jacobian(lambda x: sys.dyn(x_test[i], dtheta=x)[0], dtheta) for i in range(n_samples)], dim=0)
    # dBdp_auto = torch.cat([jacobian(lambda x: sys.dyn(x_test[i], dtheta=x)[1], dtheta) for i in range(n_samples)], dim=0)

    # err_a = (dadp_auto.view(dadp_shape) - dadp).abs().sum() / n_samples
    # err_B = (dBdp_auto.view(dBdp_shape) - dBdp).abs().sum() / n_samples
    # assert err_a <= 2.e-4 and err_B <= 2.e-4