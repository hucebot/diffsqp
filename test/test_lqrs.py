##############################################################################
## THIS TEST IS FOR AN OLDER VERSION OF THE CODE AND DOES NOT WORK PROPERLY ##
##############################################################################

import copy
import torch

from diffsqp.problems import Problem
from diffsqp.costs import LqrCost, TerminalCost
from diffsqp.dynamics import PendulumDynamics
from diffsqp.solvers import Lqr
from diffsqp.solvers import NewLqr

# torch.set_default_device("cuda")


def generate_problem(horizon, dt, n_batch, n_state, n_ctrl, x_des):
    prob = Problem(horizon, dt, n_state, n_ctrl)
    dyn = PendulumDynamics(m=1.0, l=1.0, b=0.2, grav=9.81)
    Q = 1e-5 * torch.eye(n_state).repeat(n_batch, 1, 1)
    R = 1e-3 * torch.eye(n_ctrl).repeat(n_batch, 1, 1)
    cost = LqrCost(Q, R)

    Qf = 1e6 * torch.eye(n_state).repeat(n_batch, 1, 1)
    final_cost = TerminalCost(Qf, x_des)

    # Set stage cost and constraints
    for i in range(horizon - 1):
        prob.states.append(torch.zeros((n_batch, n_state)))
        prob.controls.append(torch.zeros((n_batch, n_ctrl)))
        prob.costs.append(cost)
        prob.stage_dynamics.append(dyn)
    # Set terminal cost
    # prob.states.append(torch.zeros((n_batch, n_state)))
    prob.states.append(x_des.clone())
    prob.costs.append(final_cost)

    return prob


horizon = 6
dt = 0.05
n_batch = 3
n_state = 2
n_ctrl = 1
x_des = torch.tensor([torch.pi, 0.0]).repeat(n_batch, 1)

prob = generate_problem(horizon, dt, n_batch, n_state, n_ctrl, x_des)

# Create solver object
solver = Lqr(prob)
dx_old, du_old = solver.solve()

new_solver = NewLqr(copy.deepcopy(prob))
dx_new, du_new = new_solver.solve()

iter = 0
for old, new in zip(solver.H, new_solver.R_bar):
    # print(iter)
    # print("----------------")
    # iter += 1
    # print(torch.inverse(old))
    # print("----------------")
    # print(torch.linalg.pinv(new))
    # print("################")
    # assert torch.allclose(old, new, atol=1e-1)
    assert torch.allclose(torch.inverse(old), torch.linalg.pinv(new), atol=1e-0)

for old, new in zip(solver.delta_u, new_solver.u):
    # print(iter)
    # print("----------------")
    # iter += 1
    # print(old)
    # print("----------------")
    # print(new)
    # print("################")
    assert torch.allclose(old, new, atol=1e-0)

# for x_old, x_new, u_old, u_new in zip(dx_old, dx_new, du_old, du_new):
#     print(x_old, x_new)
#     print("----------------")
#     print(u_old, u_new)
#     print("################")
#     assert torch.allclose(x_old, x_new, atol=1e-6)
#     assert torch.allclose(u_old, u_new, atol=1e-6)

print("All tests passed")
