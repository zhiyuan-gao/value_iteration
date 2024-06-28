import numpy as np
import torch

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..'))

import sympy as smp
from sympy.utilities import lambdify

from value_iteration.pendulum import BaseSystem
from value_iteration.cost_functions import ArcTangent, SineQuadraticCost, BarrierCost
from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
CUDA_AVAILABLE = torch.cuda.is_available()

# model parameters to dict
def theta_to_dict(theta_tensor):

    tensor_flattened = theta_tensor.view(-1)

    keys = ['m1', 'm2', 'l1', 'l2', 'r1', 'r2', 'b1', 'b2', 'cf1', 'cf2', 'g', 'I1', 'I2', 'Ir', 'gr']

    para_dict = {key: value.item() for key, value in zip(keys, tensor_flattened)}

    return para_dict



class DoulbePendulum(BaseSystem):
    name = "DoulbePendulum"
    labels = ('q1', 'q2', 'q1_dot', 'q2_dot')

    def __init__(self, cuda=CUDA_AVAILABLE, **kwargs):
        super(DoulbePendulum, self).__init__()

        # Define Duration:
        self.T = kwargs.get("T", 10.0)
        self.dt = kwargs.get("dt", 1./500.)

        # Define the System:
        self.n_state = 4
        self.n_dof = 2
        self.n_act = 2
        self.n_parameter = 15

        # Continuous Joints:
        # Right now only one continuous joint is supported
        self.wrap, self.wrap_i = True, 2

        # State Constraints:
        # theta = 0, means the pendulum is pointing upward
        self.x_target = torch.tensor([np.pi, 0.0, 0.0, 0.0])
        self.x_start = torch.tensor([0.0, 0., 0.0, 0.0])
        self.x_start_var = torch.tensor([1.e-3, 5.e-2, 1.e-6, 1.e-6])
        self.x_lim = torch.tensor([2*np.pi, 2*np.pi, 15., 15.])
        self.x_penalty = torch.tensor([10, 5., 1., 1])

        # 10 degree angle error for initial sampling
        self.x_init = torch.tensor([0.17, 0.17, 0.01, 0.01])
        self.u_lim = torch.tensor([6., 0.5])

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
        torque_limit=[6.0, 0.0]

        self.symbol_theta = smp.symbols("m1 m2 l1 l2 r1 r2 b1 b2 cf1 cf2 g I1 I2 Ir gr")

        m1, m2, l1, l2, r1, r2, b1, b2, cf1, cf2, g, I1, I2, Ir, gr =self.symbol_theta

        parameters = mass + length + com + damping + cfric + [gravity] + inertia + [motor_inertia] + [gear_ratio]

        # Dynamics parameter:
        self.theta = torch.tensor(parameters).view(1, self.n_parameter, 1)

        #TODO change to the var later
        self.theta_min = 0.5 * self.theta
        self.theta_max = 1.5 * self.theta

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

        invM  = self.plant.M.inv()
        # print(self.plant.B)
        q_dot = smp.Matrix([qd1, qd2])
        # q = smp.Matrix([q1, q1])
        # x = smp.Matrix([q1, q2, qd1, qd2])
        self.symbolic_a = q_dot.col_join(invM * (-self.plant.C*q_dot +self.plant.G - self.plant.F))

        # plant.B is Actuator selection Matrix, same as D in the formulars.

        self.symbolic_B = smp.Matrix([[0., 0.], [0., 0.]]).col_join(invM* self.plant.B)


        # Compute Linearized System:
        out = self.dyn(self.x_target, gradient=True)
        self.A = out[2].view(1, self.n_state, self.n_state).transpose(dim0=1, dim1=2).numpy()
        self.B = out[1].view(1, self.n_state, self.n_act).numpy()

        # Test Dynamics:
        self.check_dynamics()

        # to_x_test = torch.distributions.uniform.Uniform(-self.x_lim, self.x_lim).sample((10,))
        # to_x_test = to_x_test.view(-1, self.n_state, 1).float().to(self.theta.device)
        # np_x_test = to_x_test.cpu().numpy()
        # out2 = self.grad_dyn_theta(np_x_test)

        self.device = None
        DoulbePendulum.cuda(self) if cuda else DoulbePendulum.cpu(self)

    def dyn(self, x, dtheta=None, gradient=False):

        is_numpy = True if isinstance(x, np.ndarray) else False
        x = torch.from_numpy(x) if isinstance(x, np.ndarray) else x
        x = x.view(-1, self.n_state, 1)
        n_samples = x.shape[0]

        # Update the dynamics parameters with disturbance:
        if dtheta is not None:
            # raise NotImplementedError
            dtheta = torch.from_numpy(dtheta).float() if isinstance(dtheta, np.ndarray) else dtheta
            dtheta = dtheta.view(n_samples, self.n_parameter, 1)
            theta = self.theta + dtheta
            theta = torch.min(torch.max(theta, self.theta_min), self.theta_max)

        else:
            theta = self.theta

        para_dict = theta_to_dict(theta)

        a_x = self.symbolic_a.subs(para_dict)
        B_x = self.symbolic_B.subs(para_dict)

        a_la = lambdify(self.symbol_x, a_x)
        B_la = lambdify(self.symbol_x, B_x)

        # a = torch.stack([torch.tensor(a_la(*sample.squeeze().detach().numpy())).reshape(self.n_state, 1) for sample in x])

        # B = torch.stack([torch.tensor(B_la(*sample.squeeze().detach().numpy())).reshape(self.n_state, self.n_act) for sample in x])

        a = torch.stack([
            torch.tensor(a_la(*sample.squeeze().detach().cpu().numpy())).reshape(self.n_state, 1).to(x.device)
            for sample in x
        ]).float()

        B = torch.stack([
            torch.tensor(B_la(*sample.squeeze().detach().cpu().numpy())).reshape(self.n_state, self.n_act).to(x.device)
            for sample in x
        ]).float()


        assert a.shape == (n_samples, self.n_state, 1)
        assert B.shape == (n_samples, self.n_state, self.n_act)
        out = (a, B)

        if gradient:

            dadx_x = a_x.jacobian(self.symbol_x)
            dadx_la = lambdify(self.symbol_x, dadx_x)

            # dadx = torch.stack([torch.tensor(dadx_la(*sample.squeeze().detach().numpy())).reshape(self.n_state, self.n_state) for sample in x])
            dadx = torch.stack([
                torch.tensor(dadx_la(*sample.squeeze().detach().cpu().numpy())).reshape(self.n_state, self.n_state).to(x.device)
                for sample in x
            ]).float()




            dBdx_x = smp.MutableDenseNDimArray.zeros(self.n_state, self.n_state, self.n_act)
            for i in range(4):
                for j in range(2):
                    for k in range(4):
                        dBdx_x[i, k, j] = smp.diff(B_x[i, j], self.symbol_x[k])
            dBdx_la = lambdify(self.symbol_x, dBdx_x)

            # dBdx = torch.stack([torch.tensor(dBdx_la(*sample.squeeze().detach().numpy())).reshape(self.n_state, self.n_state, self.n_act) for sample in x])
            dBdx = torch.stack([
                torch.tensor(dBdx_la(*sample.squeeze().detach().cpu().numpy())).reshape(self.n_state, self.n_state, self.n_act).to(x.device)
                for sample in x
            ]).float()
            assert dadx.shape == (n_samples, self.n_state, self.n_state,)
            assert dBdx.shape == (n_samples, self.n_state, self.n_state, self.n_act)
        
            out = (a, B, dadx, dBdx)

        if is_numpy:
            out = [array.numpy() for array in out]

        return out

    def grad_dyn_theta(self, x):

        is_numpy = True if isinstance(x, np.ndarray) else False
        x = torch.from_numpy(x) if isinstance(x, np.ndarray) else x
        x = x.view(-1, self.n_state, 1)
        n_samples = x.shape[0]

        symbol_dadp = self.symbolic_a.jacobian(self.symbol_theta)
        
        # torch.zeros(n_samples, self.n_parameter, self.n_state, self.n_act).to(x.device)
        symbol_dBdp = smp.MutableDenseNDimArray.zeros(self.n_parameter, self.n_state, self.n_act)
        for i in range(self.n_state):
            for j in range(self.n_act):
                for k in range(self.n_parameter):
                    symbol_dBdp[k, i, j] = smp.diff(self.symbolic_B[i, j], self.symbol_theta[k])


        para_dict = theta_to_dict(self.theta)

        # bring theta to symbol_dadp
        dadp_x = symbol_dadp.subs(para_dict)

        # bring theta to symbol_dBdp, since MutableDenseNDimArray can not be used in subs
        dBdp_x = smp.MutableDenseNDimArray.zeros(self.n_parameter, self.n_state, self.n_act)
        for k in range(self.n_parameter):
            for i in range(self.n_state):
                for j in range(self.n_act):
                    dBdp_x[k, i, j] = symbol_dBdp[k, i, j].subs(para_dict)
        

        dadp_la = lambdify(self.symbol_x, dadp_x)
        dBdp_la = lambdify(self.symbol_x, dBdp_x)

        # dadp = torch.zeros(n_samples, self.n_parameter, self.n_state).to(x.device)
        # dadp[:, :, self.n_dof:, ] = (torch.matmul(dinvHdp, f - n) + torch.matmul(invH_4d, dfdp - dndp)).view(-1, self.n_parameter, self.n_dof)

        # dBdp = torch.zeros(n_samples, self.n_parameter, self.n_state, self.n_act).to(x.device)
        # dBdp[:, :, self.n_dof:, ] = dinvHdp[:, :, :, :self.n_act]

        # dBdx = np.zeros((n_samples, self.n_state, self.n_state, self.n_act))
        # sample_test = x[0]
        # test_result = torch.tensor(dBdp_la(*sample_test.squeeze().detach().numpy()))
        dadp = torch.stack([
            torch.tensor(dadp_la(*sample.squeeze().detach().cpu().numpy())).reshape(self.n_parameter, self.n_state).to(x.device)
            for sample in x
        ]).float()
        dBdp = torch.stack([
            torch.tensor(dBdp_la(*sample.squeeze().detach().cpu().numpy())).reshape(self.n_parameter, self.n_state, self.n_act).to(x.device)
            for sample in x
        ]).float()

        # dadp = torch.stack([torch.tensor(dadp_la(*sample.squeeze().detach().numpy())).reshape(self.n_parameter, self.n_state) for sample in x])
        # dBdp = torch.stack([torch.tensor(dBdp_la(*sample.squeeze().detach().numpy())).reshape(self.n_parameter, self.n_state, self.n_act) for sample in x])
        assert dadp.shape == (n_samples, self.n_parameter, self.n_state)
        assert dBdp.shape == (n_samples, self.n_parameter, self.n_state, self.n_act)
        out = (dadp, dBdp)
        if is_numpy:
            out = [array.cpu().detach().numpy() for array in out]

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
        self.u_lim = torch.tensor([6., 0])

        # Create the Reward Function:
        assert Q.size == self.n_state and np.all(Q > 0.0)
        self.Q = np.diag(Q).reshape((self.n_state, self.n_state))

        assert R.size == self.n_act and np.all(R > 0.0)
        self.R = np.diag(R).reshape((self.n_act, self.n_act))

        self._q = SineQuadraticCost(self.Q, np.array([0.0, 1.0, 0.0, 0.0]), cuda=cuda)
        self.q = BarrierCost(self._q,  self.x_penalty, cuda)

        # Determine beta s.t. the curvature at u = 0 is identical to 2R
        beta = 4. * self.u_lim[0] ** 2 / np.pi * self.R
        self.r = ArcTangent(alpha=self.u_lim.numpy()[0], beta=beta.numpy()[0, 0])

    def rwd(self, x, u):
        return self.q(x) + self.r(u)

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