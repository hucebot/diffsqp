import argparse
import time

import numpy as np
import matplotlib.pyplot as plt
from jax import config

config.update("jax_enable_x64", True)  # use double precision

import jax
import jax.numpy as jnp
from jax import jit, vmap

from diffmpc.dynamics.integrators import DiscretizationScheme, predict_next_state
from diffmpc.dynamics.cartpole_dynamics import CartpoleDynamics
from diffmpc.problems.optimal_control_problem import OptimalControlProblem
from diffmpc.solvers.sqp import SQPSolver
from diffmpc.utils.load_params import (
    load_problem_params,
    load_solver_params,
)


# def generate_problem_data(n_batch, seed):
#     """Generate lipm problem data for benchmarking"""
#     np.random.seed(seed)
#
#     # Randomise initial states
#     min_state = -0.05
#     max_state = 0.05
#     initial_states = min_state + np.random.rand(n_batch, 4) * (max_state - min_state)
#
#     return jnp.array(initial_states)


def load_trajectory(p):
    # +1 because diffmpc horizon is N+1
    p["reference_state_trajectory"] = jnp.zeros((p["horizon"] + 1, 4))
    p["reference_control_trajectory"] = jnp.zeros((p["horizon"] + 1, 1))


problem_params = {
    "nx": 4,
    "nu": 1,
    "horizon": 100,
    "discretization_resolution": jnp.array([0.01]),  # dt
    "discretization_scheme": DiscretizationScheme.RUNGEKUTTA4,
    "penalize_control_reference": True,
    "initial_state": jnp.array([0.0, 0.0, 0.0, 0.0]),
    "final_state": jnp.array([0.0, 0.0, jnp.pi, 0.0]),
    "weights_penalization_reference_state_trajectory": jnp.array(
        [1e-2, 1e-2, 1e-2, 1e-2]
    ),
    "weights_penalization_control_squared": jnp.array([1e-1]),
    "weights_penalization_final_state": jnp.array([1e6, 1e6, 1e6, 1e6]),
    # "weights_penalization_reference_state_trajectory": jnp.array([1e0, 1e0, 1e0, 1e0]),
    # "weights_penalization_control_squared": jnp.array([1e1]),
    # "weights_penalization_final_state": jnp.array([1e4, 1e4, 1e4, 1e4]),
}

load_trajectory(problem_params)

dynamics = CartpoleDynamics()
problem = OptimalControlProblem(dynamics=dynamics, params=problem_params)

# Load solver
solver_params = load_solver_params("sqp.yaml")
solver_params["tol_convergence"] = 1.0e-4
solver_params["num_scp_iteration_max"] = 100
solver_params["pcg"]["tol_epsilon"] = 1.0e-6
solver_params["linesearch"] = True
solver_params["warm_start_backward"] = False
solver_params["linesearch_alphas"] = [1.0]
solver = SQPSolver(program=problem, params=solver_params)

weights = {
    k: problem_params[k]
    for k in [
        "weights_penalization_reference_state_trajectory",
        "weights_penalization_control_squared",
    ]
}


def generate_problem_data(n_batch, seed):
    """Generate initial states for benchmarking"""
    key = jax.random.PRNGKey(seed)
    std = 0.05
    # initial_states = std * jax.random.normal(key, (n_batch, 4))
    initial_states = jnp.zeros((n_batch, 4))
    return initial_states


n_batch = 1  # Adjust based on your GPU memory
initial_states = generate_problem_data(n_batch, seed=int(time.time()))


def solve_single_instance(init_state):
    local_params = problem_params.copy()
    local_params["initial_state"] = init_state
    init_guess = solver.initial_guess(local_params)
    return solver.solve(init_guess, local_params, weights)


parallel_solver = jit(vmap(solve_single_instance))

print(f"Starting parallel solve for {n_batch} instances...")

# Warm-up (JIT compilation happens here)
_ = parallel_solver(initial_states[:1])

start = time.time()
solutions = parallel_solver(initial_states)
# Ensure GPU finishes before stopping timer
jax.block_until_ready(solutions)
end = time.time()

print(f"Total time for {n_batch} problems: {end - start:.4f}s")
print(f"Average time per problem: {(end - start)/n_batch:.4f}s")

# control = solution.controls[0]
# sim_dt = 0.01
# state = predict_next_state(
#     dynamics,
#     sim_dt,
#     DiscretizationScheme.RUNGEKUTTA4,
#     problem_params,
#     state,
#     control,
# )
#
# problem_params = {
#     **problem_params,
#     "initial_state": state,
# }

# import matplotlib.pyplot as plt
#
# # print(solution.states)
# plt.plot(solution.states[:, 1])
# plt.plot(solution.states[:, 2])
# # plt.plot(solution.states[:, 4])
# # plt.plot(solution.controls[:, 2])
# plt.show()
