import numpy as np
import torch
import time
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..'))
import sympytorch
import sympy as smp
from sympy.utilities import lambdify

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

        # print('self.symbolic_a:',self.symbolic_a)
        # print('self.symbolic_B:',self.symbolic_B)

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



        # self.dadp_x_torch = sympytorch.SymPyModule(expressions=self.dadp_x.tolist())
        self.expressions = [sympytorch.SymPyModule(expressions=[expr]) for expr in self.dadp_x]

        # expressions = []
        # for i in range(dadp_x.shape[0]):
        #     row = []
        #     for j in range(dadp_x.shape[1]):
        #         expr = dadp_x[i, j]
        #         row.append(spt.SymPyModule(expressions=[expr]))
        #     expressions.append(row)





        # bring theta to symbolic_dBdpT, since MutableDenseNDimArray can not be used in subs
        self.dBdp_x = smp.MutableDenseNDimArray.zeros(self.n_parameter, self.n_state, self.n_act)
        for i in range(self.n_parameter):
            for j in range(self.n_state):
                for k in range(self.n_act):
                    self.dBdp_x[i, j, k] = self.symbolic_dBdpT[i, j, k].subs(change_para)


        self.dadp_la = lambdify(self.symbol_x, self.dadp_x)
        self.dBdp_la = lambdify(self.symbol_x, self.dBdp_x)
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
        start_time = time.time()

        is_numpy = True if isinstance(x, np.ndarray) else False
        x = torch.from_numpy(x) if isinstance(x, np.ndarray) else x
        x = x.view(-1, self.n_state, 1).to(self.device)
        n_samples = x.shape[0]

        # requires_grad=True

        # Update the dynamics parameters with disturbance:
        if dtheta is not None:
            # raise NotImplementedError
            dtheta = torch.from_numpy(dtheta).float() if isinstance(dtheta, np.ndarray) else dtheta
            dtheta = dtheta.view(n_samples, self.n_parameter, 1)
            theta = self.theta + dtheta
            theta = torch.min(torch.max(theta, self.theta_min), self.theta_max)
            theta = theta.to(x.device)

        else:
            theta = self.theta

        theta.requires_grad_(True)
        x.requires_grad_(True)
        

        


        # 定义常数和符号变量
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

        # print("B31:",B31.shape)
        # print("B32:",B32.shape)
        # print("B41:",B41.shape)
        # print("B42:",B42.shape)


        # 将每一行的两个元素拼接成一个张量，形状为 (10000, 1, 2)
        row1 = torch.cat((B11, B12), dim=2)  # 形状 (10000, 1, 2)
        row2 = torch.cat((B21, B22), dim=2)  # 形状 (10000, 1, 2)
        row3 = torch.cat((B31, B32), dim=2)  # 形状 (10000, 1, 2)
        row4 = torch.cat((B41, B42), dim=2)  # 形状 (10000, 1, 2)

        # 将每一行拼接成最终的 B 矩阵，形状为 (10000, 4, 2)
        B = torch.cat((row1, row2, row3, row4), dim=1)  
        B.requires_grad_(True)

        # # 对 a 和 B 分别计算对 x_state 的梯度
        # grad_a = torch.autograd.grad(a, x_state, grad_outputs=torch.ones_like(a), create_graph=True)
        # grad_B = torch.autograd.grad(B, x_state, grad_outputs=torch.ones_like(B), create_graph=True)



        # theta 
        # change_para = theta_to_dict(theta)

        # a_x = self.symbolic_a.subs(change_para)
        # B_x = self.symbolic_B.subs(change_para)

        # a_la = lambdify(self.symbol_x, a_x)
        # B_la = lambdify(self.symbol_x, B_x)



        # # a = torch.stack([
        # #     torch.tensor(a_la(*sample.squeeze().detach().cpu().numpy())).reshape(self.n_state, 1).to(x.device)
        # #     for sample in x
        # # ]).float()

        # # B = torch.stack([
        # #     torch.tensor(B_la(*sample.squeeze().detach().cpu().numpy())).reshape(self.n_state, self.n_act).to(x.device)
        # #     for sample in x
        # # ]).float()


        # assert a.shape == (n_samples, self.n_state, 1)
        # assert B.shape == (n_samples, self.n_state, self.n_act)

        # end_time = time.time()
        # execution_time = end_time - start_time
        # # print(f"bring x execution time: {execution_time} seconds")
        
        out = (a, B)

        if gradient:
        #     grad_start_time = time.time()

        #     dadxT_x = self.symbolic_dadxT.subs(change_para)
        #     dadxT_la = lambdify(self.symbol_x, dadxT_x)

        #     dadx = torch.stack([
        #         torch.tensor(dadxT_la(*sample.squeeze().detach().cpu().numpy())).reshape(self.n_state, self.n_state).to(x.device)
        #         for sample in x
        #     ]).float()

        #     # bring theta to symbolic_dBdxT, since MutableDenseNDimArray can not be used in subs
        #     dBdxT_x = smp.MutableDenseNDimArray.zeros(self.n_state, self.n_state, self.n_act)
        #     for i in range(self.n_state):
        #         for j in range(self.n_state):
        #             for k in range(self.n_act):
        #                 dBdxT_x[i, j, k] = self.symbolic_dBdpT[i, j, k].subs(change_para)


        #     end_time = time.time()
        #     execution_time = end_time - grad_start_time
        #     # print(f"dyn subs execution time: {execution_time} seconds")


        #     dBdxT_la = lambdify(self.symbol_x, dBdxT_x)

        #     dBdx = torch.stack([
        #         torch.tensor(dBdxT_la(*sample.squeeze().detach().cpu().numpy())).reshape(self.n_state, self.n_state, self.n_act).to(x.device)
        #         for sample in x
        #     ]).float()
        #     assert dadx.shape == (n_samples, self.n_state, self.n_state,)
        #     assert dBdx.shape == (n_samples, self.n_state, self.n_state, self.n_act)
                # 创建一个张量与 a 形状相同，用于梯度计算
            grad_outputs_a = torch.ones_like(a)
            
            grad_outputs_B = torch.ones_like(B)

            grad_outputs_a.requires_grad_(True)
            grad_outputs_B.requires_grad_(True)


            # # 计算 a 对 x 的梯度
            # grad_a = torch.autograd.grad(a, x, grad_outputs=grad_outputs_a, create_graph=True, retain_graph=True)

            # # 计算 B 对 x 的梯度
            # grad_B = torch.autograd.grad(B, x, grad_outputs=grad_outputs_B, create_graph=True)
            dadx = torch.autograd.grad(a, x, grad_outputs=grad_outputs_a, create_graph=True, retain_graph=True)
            # print("grad_a:",grad_a)

            # 计算 B 对 x 的梯度
            dBdx = torch.autograd.grad(B, x, grad_outputs=grad_outputs_B, create_graph=True, retain_graph=True)

            out = (a, B, dadx, dBdx)

        # if is_numpy:
        #     out = [array.numpy() for array in out]


        end_time = time.time()
        execution_time = end_time - start_time
        # print(f"a time: {execution_time} seconds")

        return out

    def grad_dyn_theta(self, x):

        
        print("grad_dyn_theta____________________________________________________")
        print(f"x is on device: {x.device}")
        # print('x',x.shape)
        start_time = time.time()

        is_numpy = True if isinstance(x, np.ndarray) else False
        x = torch.from_numpy(x) if isinstance(x, np.ndarray) else x
        x = x.view(-1, self.n_state, 1)
        print(f"x is on device: {x.device}")
        n_samples = x.shape[0]

        # dadp_la = lambdify(self.symbol_x, self.dadp_x)
        # dBdp_la = lambdify(self.symbol_x, self.dBdp_x)
        # 使用 cupy 而不是 numpy
        # dadp_la = lambdify(self.symbol_x, self.dadp_x, 'cupy')
        # dBdp_la = lambdify(self.symbol_x, self.dBdp_x, 'cupy')

        # x = x.view(100, 4, 1)

        a,B = self.dyn(x, gradient=False)



        grad_outputs_a = torch.ones_like(a)
        
        grad_outputs_B = torch.ones_like(B)

        grad_outputs_a.requires_grad_(True)
        grad_outputs_B.requires_grad_(True)


        # # 计算 a 对 x 的梯度
        # grad_a = torch.autograd.grad(a, x, grad_outputs=grad_outputs_a, create_graph=True, retain_graph=True)

        # # 计算 B 对 x 的梯度
        # grad_B = torch.autograd.grad(B, x, grad_outputs=grad_outputs_B, create_graph=True)
        dadp = torch.autograd.grad(a, self.theta, grad_outputs=grad_outputs_a, create_graph=True, retain_graph=True)
        print("dadp:",dadp)
        # print("grad_a:",grad_a)

        # 计算 B 对 x 的梯度
        dBdp = torch.autograd.grad(B, self.theta, grad_outputs=grad_outputs_B, create_graph=True, retain_graph=True)




        end_time = time.time()
        execution_time = end_time - start_time
        # print(f"grad lambdify execution time: {execution_time} seconds")



        # # 将数据传递给 dadp 和 dBdp，并确保使用 GPU
        # dadp = torch.stack([
        #     torch.tensor(cp.asnumpy(dadp_la(*sample.squeeze().detach().cpu().numpy()))).reshape(self.n_parameter, self.n_state).to(x.device)
        #     for sample in x
        # ]).float()
        # dBdp = torch.stack([
        #     torch.tensor(cp.asnumpy(dBdp_la(*sample.squeeze().detach().cpu().numpy()))).reshape(self.n_parameter, self.n_state, self.n_act).to(x.device)
        #     for sample in x
        # ]).float()

        # device = x.device

        # # 确保 xj 在 GPU 上，并转换为 cupy 数组
        # xj_cupy = [cp.asarray(sample.squeeze().detach().cpu().numpy()) for sample in x]

        # dadp = torch.stack([
        #     torch.tensor(cp.asnumpy(dadp_la(*sample))).reshape(self.n_parameter, self.n_state).to(device)
        #     for sample in xj_cupy
        # ]).float()

        # dBdp = torch.stack([
        #     torch.tensor(cp.asnumpy(dBdp_la(*sample))).reshape(self.n_parameter, self.n_state, self.n_act).to(device)
        #     for sample in xj_cupy
        # ]).float()

        # dadp = torch.zeros(n_samples, self.n_parameter, self.n_state).to(x.device)

        # # bring x to dadp and dBdp
        # dadp = torch.stack([
        #     torch.tensor(self.dadp_la(*sample.squeeze().detach().cpu().numpy())).reshape(self.n_parameter, self.n_state).to(x.device)
        #     for sample in x
        # ]).float()

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"bring x to dadp and dBdp execution time: {execution_time} seconds")

        # dBdp = torch.zeros(n_samples, self.n_parameter, self.n_state, self.n_act).to(x.device)

        # dBdp = torch.stack([
        #     torch.tensor(self.dBdp_la(*sample.squeeze().detach().cpu().numpy())).reshape(self.n_parameter, self.n_state, self.n_act).to(x.device)
        #     for sample in x
        # # ]).float()




        # start_time = time.time()
        # # 创建用于存储结果的张量

        # x_flat = x.view(-1, 4).to('cuda')

        # # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # # print('device:',device) 
        # print(f"x is on device: {x_flat.device}")

        # # 创建用于存储结果的张量
        # results = []


        # # q1, q2, qd1, qd2, qdd1, qdd2 = smp.symbols(
        # #     "q1 q2 \dotq_1 \dotq_2 \ddotq_1 \ddotq_2"
        # # )




        # # for expr in self.expressions:
        # #     print('expr:',expr)
        # # 逐点计算符号矩阵的值
        # for point in x_flat:
        #     q1_val, q2_val, qd1_val, qd2_val = point
        #     output_row = []
        #     for expr in self.expressions:

        #         output = expr(q1=q1_val, q2=q2_val, **{"\\dot{q}_1": qd1_val, "\\dot{q}_2": qd2_val}).to('cuda')
        #         output_row.append(output)
        #     results.append(torch.cat(output_row))

        # end_time = time.time()
        # execution_time = end_time - start_time
        # print(f"using symtorch execution time: {execution_time} seconds")



        # assert dadp.shape == (n_samples, self.n_parameter, self.n_state)
        # assert dBdp.shape == (n_samples, self.n_parameter, self.n_state, self.n_act)
        out = (dadp, dBdp)
        # if is_numpy:
        #     out = [array.cpu().detach().numpy() for array in out]

        # end_time = time.time()
        # execution_time = end_time - start_time
        # print(f"grad_dyn_theta execution time: {execution_time} seconds")
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

    # n_samples = 10000
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