"""Batched quadrotor point-to-point trajectory optimization on SE(3).

The state is [position(3), quaternion wxyz(4), linear velocity(3), angular
velocity(3)] and the controls are [thrust, torque(3)]. `QuadrotorTrackingCost`
measures the orientation error on SO(3) (quaternion error vector), so the cost
stays well defined despite the 4D quaternion parameterization. The initial guess
interpolates the position linearly from start to goal, at hover thrust.

    uv run examples/quadrotor_se3_trajopt.py --batch_size 16 --save out/quadrotor
"""

import argparse
import torch

from diffsqp.problems import Problem, ProblemParameters
from diffsqp.costs import LqrCost, QuadrotorTrackingCost
from diffsqp.solvers import sqp_solve, SqpParameters
from diffsqp.dynamics import QuadrotorDynamics, QuadrotorParameters
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
            "admm_rho_init": 0.4,
            "admm_rho_min": 1e-6,
            "admm_rho_max": 1e8,
            "admm_adaptive_rho_tolerance": 10.0,
            "admm_rho_update_iter_freq": 10,
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
            "sqp_max_iter": 100,
            "lqr_reg_init": 1e-5,
            "merit_mu": 1e6,
            "armijo_beta": 1e-4,
            "ls_max_iter": 10,
            "sqp_cost_eps": 1e-2,
            "sqp_viol_eps": 1e-3,
            "check_complementarity": False,
            "qp_solver": "lqr",
            "ls_function": "merit",
        }
    )

    big = 1e6  # Effectively unbounded
    problem_parameters = ProblemParameters(
        **{
            "inverse_dynamics": False,
            "n_h": 0,
            "batch_size": batch_size,
            "dt": 0.01,
            "tf": 1.0,
            # Hover at the origin -> hover 1 m forward and 0.5 m up
            "x_init": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0] + [0.0] * 6,
            "x_des": [1.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0] + [0.0] * 6,
            "noise_std": [0.01, 0.01, 0.01],  # Perturb the start position only
            # Keep the position inside a 100 m box
            "x_lb": [-100.0] * 3 + [-big] * 10,
            "x_ub": [100.0] * 3 + [big] * 10,
            "u_lb": [-big] * 4,
            "u_ub": [big] * 4,
            # Tracking weights on the 12D error: [pos(3), ori(3), vel(3), omega(3)]
            "q_w": [1e-6] * 3 + [1e-8] * 4 + [1e-6] * 5,
            "r_w": [1e-1] * 4,
            "qf_w": [1e5] * 12,
        }
    )

    system_parameters = QuadrotorParameters(
        **{
            "name": "quadrotor",
            "n_x": 13,
            "n_q": 7,
            "n_v": 6,
            "n_j": 0,
            "n_u": 4,
            "mass": 0.1,
            "inertia": [0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1],
            "grav": 9.81,
        }
    )

    dynamics = QuadrotorDynamics(system_parameters)
    problem = Problem(problem_parameters, system_parameters)
    problem.dynamics = dynamics

    # Control effort only; the state is handled by the SE(3) tracking cost
    Q = torch.zeros(dynamics.nx, dynamics.nx).repeat(batch_size, 1, 1)
    R = problem_parameters.r_w * torch.eye(dynamics.nu).repeat(batch_size, 1, 1)

    state_bounds = StateBounds(
        problem.n_x, problem.n_u, problem_parameters.x_lb, problem_parameters.x_ub
    )
    control_bounds = ControlBounds(
        problem.n_x, problem.n_u, problem_parameters.u_lb, problem_parameters.u_ub
    )

    # Stage costs: effort + weak regularization towards the start pose
    for k in range(problem.horizon - 1):
        problem.costs.append(
            [
                LqrCost(Q=Q, R=R),
                QuadrotorTrackingCost(
                    Q_diag=problem_parameters.q_w,
                    x_des=problem_parameters.x_init.clone(),
                ),
            ]
        )
        problem.constraints[k] = [state_bounds, control_bounds]
    # Terminal stage: reach the goal pose at rest
    problem.costs.append(
        [
            QuadrotorTrackingCost(
                Q_diag=problem_parameters.qf_w,
                x_des=problem_parameters.x_des.clone(),
            )
        ]
    )
    problem.constraints[-1] = [state_bounds]

    # Initial guess: straight-line positions at hover thrust
    x_init = problem_parameters.x_init
    x_des = problem_parameters.x_des
    alphas = torch.linspace(0, 1, problem.horizon).unsqueeze(1)
    x = x_init.repeat(batch_size, problem.horizon, 1)
    x[:, :, 0:3] = (1.0 - alphas) * x_init[0:3] + alphas * x_des[0:3]
    x[:, -1] = x_des
    hover = torch.tensor([system_parameters.mass * system_parameters.grav, 0, 0, 0])
    u = hover.repeat(batch_size, problem.horizon - 1, 1)

    initial_guess = SqpSolution(
        x=x,
        u=u,
        mu=torch.zeros((batch_size, problem.horizon, problem.n_x)),
        nu=torch.zeros((batch_size, problem.horizon - 1, problem.n_h)),
        ksi=[None] * problem.horizon,
    )
    noise_std = torch.tensor(problem_parameters.noise_std)
    initial_guess.x[:, 0] = x_init
    initial_guess.x[:, 0, : len(noise_std)] += noise_std * torch.randn(
        (batch_size, len(noise_std))
    )

    print(f"Solving {batch_size} quadrotor SE(3) trajectory problems...")
    solution, log = sqp_solve(problem, sqp_parameters, initial_guess)
    print(log)

    pos_error = torch.norm(solution.x[:, -1, 0:3] - x_des[0:3], dim=-1)
    print(f"Final position error (max over batch): {pos_error.max().item():.2e}")
    print(f"Constraint violation (max over batch): {max(log.constraint_violation):.2e}")

    if args.save:
        save_solution(solution, args.save, x_des=x_des)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="Number of problems")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--save", type=str, help="Save path (without .pt)")
    main(parser.parse_args())
