import numpy as np
import torch
from double_pendulum.controller.abstract_controller import AbstractController
import os
from datetime import datetime

import matplotlib.pyplot
from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.simulation.simulation import Simulator

from double_pendulum.utils.plotting import plot_timeseries
from double_pendulum.utils.wrap_angles import wrap_angles_top,wrap_angles_diff

from double_pendulum.simulation.gym_env import (
    double_pendulum_dynamics_func,
)
import argparse


try:
    import matplotlib as mpl
    mpl.use("Qt5Agg")

except ImportError as e:
    pass


from value_iteration.value_function import QuadraticNetwork, TrigonometricQuadraticNetwork
from value_iteration.run_experiment import run_experiment
from value_iteration.doulbe_pendulum import DoulbePendulumLogCos

from deep_differential_network.replay_memory import PyTorchReplayMemory, PyTorchTestMemory
from value_iteration.update_value_function import update_value_function, eval_memory
from value_iteration.value_function import ValueFunctionMixture
from value_iteration.sample_rollouts import sample_data
from value_iteration.utils import linspace, add_nan

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


plant = SymbolicDoublePendulum(mass=mass,
                            length=length,
                            com=com,
                            damping=damping,
                            gravity=gravity,
                            coulomb_fric=cfric,
                            inertia=inertia,
                            motor_inertia=motor_inertia,
                            gear_ratio=gear_ratio,
                            torque_limit=torque_limit)

sim = Simulator(plant=plant)

# Define Hyper-parameters:
hyper = {
    # Learning Mode:
    'mode': 'DP',
    'robust': False,

    # Value Function:
    'val_class': TrigonometricQuadraticNetwork,
    'checkpoint': None,
    'plot': True,

    # System Specification:
    'system_class': DoulbePendulumLogCos,
    'state_cost': '5.e+0, 5.e+0, 1.e-1, 1.e-1',
    'action_cost': '1.e-2,1.e-2',
    'eps': 0.80,  # eps = 1 => \gamma = 1
    'dt': 1. / 500.,
    'T': 10.,

    # Network:
    'n_network': 4,
    'activation': 'Tanh',
    'n_width': 64,
    'n_depth': 4,
    'n_output': 1,
    'g_hidden': 1.41,
    'g_output': 1.,
    'b_output': -0.1,

    # Samples
    'n_iter': 250,
    'eval_minibatch': 256 * 200,
    'test_minibatch': 256 * 20,
    'n_minibatch': 512,
    'n_batches': 200,

    # Network Optimization
    'max_epoch': 20,
    'lr_SGD': 1.0e-4,
    'weight_decay': 1.e-6,
    'exp': 1.,

    # Lambda Traces
    'trace_weight_n': 1.e-4,
    'trace_lambda': 0.90,

    # Exploration:
    'x_noise': 1.e-6,
    'u_noise': 1.e-6,
}

# Select the admissible set of the adversary:
hyper['xi_x_alpha'] = 1.e-6 if hyper["robust"] else 1.e-6
hyper['xi_u_alpha'] = 0.100 if hyper["robust"] else 1.e-6
hyper['xi_o_alpha'] = 0.025 if hyper["robust"] else 1.e-6
hyper['xi_m_alpha'] = 1. if hyper["robust"] else 1.e-6

alg_name = "rFVI" if hyper['robust'] else "cFVI"

# Configuration for sampling trajectories from the system
run_config = {"verbose": False, 'mode': 'init', 'fs_return': 10.,
                'x_noise': hyper['x_noise'], 'u_noise': hyper['u_noise']}


cuda = torch.cuda.is_available()
# Configuration for sampling trajectories from the system
run_config = {"verbose": False, 'mode': 'init', 'fs_return': 10.,
                'x_noise': hyper['x_noise'], 'u_noise': hyper['u_noise']}

# Build the dynamical system:
Q = np.array([float(x) for x in hyper['state_cost'].split(',')])
R = np.array([float(x) for x in hyper['action_cost'].split(',')])
system = hyper['system_class'](Q, R, cuda=cuda, **hyper)




# Construct Value Function:
feature = torch.zeros(system.n_state)
if system.wrap:
    feature[system.wrap_i] = 1.0


val_fun_kwargs = {'feature': feature}
value_fun = ValueFunctionMixture(system.n_state, **val_fun_kwargs, **hyper)

if hyper['checkpoint'] is not None:
    data = torch.load(hyper['checkpoint'], map_location=torch.device('cpu'))

    hyper = data['hyper']
    hyper['n_iter'] = 0

    value_fun = ValueFunctionMixture(system.n_state, **val_fun_kwargs, **data['hyper'])
    value_fun.load_state_dict(data["state_dict"])

value_fun = value_fun.cuda() if cuda else value_fun.cpu()



class ValueFunPolicy:
    def __init__(self, sys, val_fun):
        self.v = val_fun
        self.sys = sys

    def __call__(self, x, B):
        if B is None:
            _, B = self.sys.dyn(x)

        Vi, dVidx = self.v(x)  # negative_definite(*val_fun(x[-1]))
        dVidx = dVidx.transpose(dim0=1, dim1=2)

        BT_dVdx = torch.matmul(B.transpose(dim0=1, dim1=2), dVidx)
        ui = self.sys.r.grad_convex_conjugate(BT_dVdx)
        return Vi, dVidx, ui
    
    
# args = (value_fun, hyper, system, run_config)

class RfviController(AbstractController):
    def __init__(self, value_fun, system, scaling=True):
        super().__init__()

        self.system = system
        self.value_fun = value_fun
        self.scaling = scaling
        self.pi = ValueFunPolicy(system, value_fun)

    def get_control_output_(self, x, t=None):

        B = None
        
        _, _, ui = self.pi(x, B)

        return ui
    


# initialize combined controller
controller = RfviController(value_fun, system)
controller.init()

# # start simulation
# T, X, U = sim.simulate_and_animate(
#     t0=0.0,
#     x0=[0.0, 0.0, 0.0, 0.0],
#     tf=t_final,
#     dt=dt,
#     controller=controller,
#     integrator=integrator,
#     # save_video=False,
# )

# # plot timeseries
# plot_timeseries(
#     T,
#     X,
#     U,
#     X_meas=sim.meas_x_values,
#     pos_y_lines=[np.pi],
#     tau_y_lines=[-torque_limit[active_act], torque_limit[active_act]],
#     # save_to="/home/chi/Github/double_pendulum/src/python/double_pendulum/controller/SAC/plots/acrobot.png"
# )
