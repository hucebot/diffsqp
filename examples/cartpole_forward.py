"""Batched cart-pole swing-up with forward dynamics.

Optimizes over states and controls, with x_{k+1} = f(x_k, u_k) as an equality
constraint, plus box bounds on the cart position, velocities and force. Every
batch element starts from a randomly perturbed initial state.

    uv run examples/cartpole_forward.py --batch_size 16 --save out/cartpole_forward
"""

import argparse
import torch

from diffsqp.problems import Problem, ProblemParameters
from diffsqp.costs import LqrCost
from diffsqp.solvers import sqp_solve, SqpParameters
from diffsqp.dynamics import CartPoleDynamics, CartPoleParameters
from diffsqp.constraints import StateBounds, ControlBounds
from diffsqp.types import SqpSolution
from diffsqp.utils.load_save import save_solution


def main(args):
    torch.set_default_device(args.device)
    batch_size = args.batch_size

    sqp_parameters = SqpParameters(
        **{
            ## ADMM ##
            "admm_max_iter": 30,
            "admm_alpha": 1.6,
            "admm_sigma": 1e-6,
            # Rho related
            "admm_reset_rho": False,
            "admm_update_rho": False,
            "admm_rho_init": 0.1,
            "admm_rho_min": 1e-6,
            "admm_rho_max": 1e8,
            "admm_adaptive_rho_tolerance": 10.0,
            "admm_rho_update_iter_freq": 30,
            # Warm starting
            "admm_warm_start_unconstrained": True,
            "admm_reset_ksi": False,
            # Tolerances
            "admm_abs_tolerance": 0.001,
            "admm_abs_tolerance_final": -1.0,
            "admm_rel_tolerance": 0.0001,
            "admm_rel_tolerance_final": -1.0,
            "admm_tolerance_update_steps": 0,
            ## SQP ##
            "sqp_max_iter": 50,
            "lqr_reg_init": 1e-5,
            "merit_mu": 1e7,
            "armijo_beta": 1e-3,
            "ls_max_iter": 10,
            "sqp_cost_eps": 1e-1,
            "sqp_viol_eps": 1e-4,
            "check_complementarity": False,
            "qp_solver": "lqr",
            "ls_function": "filter",
        }
    )

    # State: [cart position, pole angle, cart velocity, pole angular velocity]
    problem_parameters = ProblemParameters(
        **{
            "inverse_dynamics": False,
            "n_h": 0,
            "batch_size": batch_size,
            "dt": 0.01,
            "tf": 1.0,
            "x_init": [0.0, 0.0, 0.0, 0.0],
            "x_des": [0.0, 3.14159, 0.0, 0.0],
            "noise_std": [0.01, 0.01, 0.001, 0.001],
            # State-control bounds
            "x_lb": [-2.5, -1e6, -5.0, -20.0],
            "x_ub": [2.5, 1e6, 5.0, 20.0],
            "u_lb": [-20.0],
            "u_ub": [20.0],
            # Cost weights
            "q_w": [1e-1, 1e-1, 1e-1, 1e-1],
            "r_w": [1e0],
            "qf_w": [1e6, 1e6, 1e6, 1e6],
        }
    )

    system_parameters = CartPoleParameters(
        **{
            "name": "cartpole",
            "n_x": 4,
            "n_q": 2,
            "n_v": 2,
            "n_j": 2,
            "n_u": 1,
            "mc": 0.5,
            "mp": 0.3,
            "lp": 0.2,
            "grav": 9.81,
        }
    )

    dynamics = CartPoleDynamics(system_parameters)
    problem = Problem(problem_parameters, system_parameters)
    problem.dynamics = dynamics

    # Batched quadratic cost weights: (batch_size, n, n)
    Q = problem_parameters.q_w * torch.eye(dynamics.nx).repeat(batch_size, 1, 1)
    R = problem_parameters.r_w * torch.eye(dynamics.nu).repeat(batch_size, 1, 1)
    Qf = problem_parameters.qf_w * torch.eye(dynamics.nx).repeat(batch_size, 1, 1)

    state_bounds = StateBounds(
        problem.n_x, problem.n_u, problem_parameters.x_lb, problem_parameters.x_ub
    )
    control_bounds = ControlBounds(
        problem.n_x, problem.n_u, problem_parameters.u_lb, problem_parameters.u_ub
    )

    # Stage costs and constraints
    for k in range(problem.horizon - 1):
        problem.costs.append([LqrCost(Q=Q, R=R)])
        problem.constraints[k] = [state_bounds, control_bounds]
    # Terminal stage: drive the pole upright
    problem.costs.append([LqrCost(Q=Qf, x_des=problem_parameters.x_des.clone())])
    problem.constraints[-1] = [state_bounds]

    # Zero initial guess, starting from a perturbed initial state per batch element
    initial_guess = SqpSolution(
        x=torch.zeros((batch_size, problem.horizon, problem.n_x)),
        u=torch.zeros((batch_size, problem.horizon - 1, problem.n_u)),
        mu=torch.zeros((batch_size, problem.horizon, problem.n_x)),
        nu=torch.zeros((batch_size, problem.horizon - 1, problem.n_h)),
        ksi=[None] * problem.horizon,
    )
    noise_std = torch.tensor(problem_parameters.noise_std)
    initial_guess.x[:, 0] = problem_parameters.x_init + noise_std * torch.randn(
        (batch_size, len(noise_std))
    )

    print(f"Solving {batch_size} cart-pole swing-up problems (forward dynamics)...")
    solution, log = sqp_solve(problem, sqp_parameters, initial_guess)
    print(log)

    final_error = torch.norm(solution.x[:, -1] - problem_parameters.x_des, dim=-1)
    print(f"Final state error  (max over batch): {final_error.max().item():.2e}")
    print(f"Constraint violation (max over batch): {max(log.constraint_violation):.2e}")

    if args.save:
        save_solution(solution, args.save, x_des=problem_parameters.x_des)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="Number of problems")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--save", type=str, help="Save path (without .pt)")
    main(parser.parse_args())
