import time
import torch

from diffsqp.problems import Problem
from diffsqp.costs import LqrCost, TerminalCost
from diffsqp.dynamics import AcrobotDynamics, AcrobotInverseDynamics
from diffsqp.solvers import Lqr
from diffsqp.solvers import Admm
from diffsqp.solvers import Ssqp

# torch.set_default_dtype(torch.double)
# torch.set_default_device("cuda")

dyn = AcrobotDynamics(m1=0.1, m2=0.1, l1=0.3, l2=0.3, grav=9.81)
# dyn = AcrobotInverseDynamics(m1=0.1, m2=0.1, l1=0.3, l2=0.3, grav=9.81)

dt = 0.01
tf = 1.0
horizon = int(tf / dt)
n_batch = 4
n_state = dyn.nx
n_ctrl = dyn.nu

# x_init = torch.tensor(
#     [
#         [0.0, 0.0, 0.0, 0.0],
#         [0.0, torch.pi, 0.0, 0.0],
#         [0.0, torch.pi, 0.0, 0.0],
#         [-4.6296e-02, 2.8597e00, 2.8562e-01, 2.3995e00],
#     ]
# )
x_init = 0.0 * torch.randn((n_batch, n_state))
x_des = torch.tensor([torch.pi, torch.pi, 0.0, 0.0]).repeat(n_batch, 1)

prob = Problem(horizon, dt, n_state, n_ctrl)

Q = 1e-6 * torch.eye(n_state).repeat(n_batch, 1, 1)
R = 1e-3 * torch.eye(n_ctrl).repeat(n_batch, 1, 1)
Qf = 1e5 * torch.eye(n_state).repeat(n_batch, 1, 1)

# Set stage cost and constraints
for i in range(horizon - 1):
    prob.states.append(x_init.clone())
    prob.controls.append(torch.zeros((n_batch, n_ctrl)))
    prob.costs.append(LqrCost(Q, R))
    prob.stage_dynamics.append(dyn)
# Set terminal cost
# prob.states.append(torch.zeros((n_batch, n_state)))
prob.states.append(x_des)
prob.costs.append(TerminalCost(Qf, x_des))

# Create solver object
qp_solver = Lqr(prob)
solver = Ssqp(prob, qp_solver)

start = time.time()

solver.solve()

end = time.time()
print("Time elapsed: ", end - start, " s.")

import matplotlib.pyplot as plt


def plot_states(states_list):
    # 1. Stack the list of tensors into one tensor: (horizon, n_batch, n_x)
    states_tensor = torch.stack(states_list)

    # 2. Extract the first batch (index 0) and convert to numpy
    # Shape becomes: (horizon, n_x)
    first_batch = states_tensor[:, 0, :].detach().cpu().numpy()

    horizon, n_x = first_batch.shape
    time = range(horizon)

    # 3. Plot each dimension of the state
    for i in range(n_x):
        plt.plot(time, first_batch[:, i], label=f"State $x_{{{i}}}$")

    plt.xlabel("Time Step $k$")
    plt.ylabel("Value")
    plt.title("State Trajectory (First Batch)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.savefig("state_trajectory.png")
    plt.show()


plot_states(prob.states)

print(solver.terminated)
