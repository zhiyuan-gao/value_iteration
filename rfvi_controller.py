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
from value_iteration.sample_rollouts import sample_data, ValueFunPolicy
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
torque_limit=[10.0, 0.0] #this is to determine the D matrix. not the actual torque limit


active_act = 1
# # model parameters
# design = "design_A.0"
# model = "model_2.0"
# robot = "acrobot"

# if robot == "pendubot":
#     torque_limit = [5.0, 0.0]
#     active_act = 0
# elif robot == "acrobot":
#     torque_limit = [0.0, 5.0]
#     active_act = 1

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


# data = torch.load('/home/zgao/AIOly/data_tar/cFVI_DoulbePendulum_LogCosCost_step_225.torch')
data = torch.load('/home/zgao/AIOly/data_old/cFVI_DoulbePendulum_LogCosCost_step_300.torch',map_location=torch.device('cpu'))
# print(data.keys())
hyper = data['hyper']
hyper['n_iter'] = 0


cuda = torch.cuda.is_available()
# Build the dynamical system:
Q = np.array([float(x) for x in hyper['state_cost'].split(',')])
R = np.array([float(x) for x in hyper['action_cost'].split(',')])
system = hyper['system_class'](Q, R, cuda=cuda, **hyper)


# Construct Value Function:
feature = torch.zeros(system.n_state)
if system.wrap:
    feature[system.wrap_i] = 1.0


val_fun_kwargs = {'feature': feature}
value_fun = ValueFunctionMixture(system.n_state, **val_fun_kwargs, **data['hyper'])
value_fun.load_state_dict(data["state_dict"])
# args = (value_fun, hyper, system, run_config)
value_fun = value_fun.cuda() if cuda else value_fun.cpu()

class RfviController(AbstractController):
    def __init__(self, value_fun, system, scaling=True):
        super().__init__()

        self.system = system
        self.value_fun = value_fun
        self.scaling = scaling
        self.pi = ValueFunPolicy(system, value_fun)

    def get_control_output_(self, x, t=None):
        # print(x)
        # print(type(x))
        # print(x.shape)

        x = torch.from_numpy(x).float().to('cuda')
        _, _, ui = self.pi(x, B = None)
        print(x)
        # ui = torch.randn(2)*5
        # ui[1] = 0
        # print(ui)

        return ui.cpu().detach().numpy().reshape(-1)
    
# initialize combined controller
controller = RfviController(value_fun, system)


controller.init()
# torch.tensor([0.17, 0.17, 1.e-3, 1.e-3])
# start simulation
T, X, U = sim.simulate_and_animate(
    t0=0.0,
    x0=[0.0, 0.0, 0.00, 0.00],
    # x0=[0.17, 0.17, 1.e-3, 1.e-3],
    tf= 15.0,
    dt= 0.01,
    controller=controller,
    integrator="runge_kutta",
    # save_video=False,
)

# plot timeseries
plot_timeseries(
    T,
    X,
    U,
    X_meas=sim.meas_x_values,
    pos_y_lines=[np.pi],
    tau_y_lines=[-torque_limit[active_act], torque_limit[active_act]],
    # save_to="/home/chi/Github/double_pendulum/src/python/double_pendulum/controller/SAC/plots/acrobot.png"
)
