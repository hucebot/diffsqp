# diffsqp

**diffsqp** is a batchable Sequential Quadratic Programming (SQP) solver built with PyTorch. It is designed to solve trajectory optimization and optimal control problems, natively supporting both forward and inverse dynamics formulations.

## Features

* **PyTorch-Native:** Leverages PyTorch for tensor operations and GPU acceleration, allowing you to solve batches of optimization problems in parallel.
* **Modular Dynamics & Costs:** Includes built-in models for the CartPole, Acrobot and quadrotor, and makes it easy to plug in your own dynamics, costs and constraints (e.g. a Franka arm using [bard](https://github.com/YueWang996/bard-pytorch-dynamics) kinematics).
* **Forward & Inverse Dynamics:** Configure the solver to optimize over states and controls directly, or use inverse dynamics constraints depending on your problem setup.
* **Visualization:** Play back batches of optimized trajectories in the browser with [viser](https://viser.studio).

## Prerequisites

This project requires **Python 3.13 or newer**.

We recommend using [uv](https://github.com/astral-sh/uv), an extremely fast Python package and project manager written in Rust, to manage dependencies and virtual environments.

## Installation

0. **Install `uv`** (if you haven't already):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   (For Windows or alternative installation methods, refer to the [uv documentation](https://github.com/astral-sh/uv).)

1. **Clone the repository**:
   ```bash
    git clone https://github.com/hucebot/diffsqp
    cd diffsqp
   ```

2. **Install dependencies and setup the environment**:
Because the project uses a pyproject.toml, you can use uv sync to automatically create a virtual environment and install all required dependencies (like torch, cvxpylayers, and matplotlib):
   ```bash
    uv sync
   ```
## Examples

The `examples/` folder contains small, self-contained scripts. Each one solves a batch of problems from randomly perturbed initial states, then prints the solver log, the final error and the maximum constraint violation:

| Script | Problem |
|---|---|
| `examples/cartpole_forward.py` | Cart-pole swing-up with forward dynamics and state/control bounds |
| `examples/cartpole_inverse.py` | Cart-pole swing-up with inverse dynamics and an underactuation constraint |
| `examples/quadrotor_se3_trajopt.py` | Quadrotor point-to-point flight on SE(3) (quaternion orientation) |
| `examples/franka_constrained.py` | Franka FP3 end-effector reaching with joint position and velocity limits, using user-defined dynamics and cost classes |

All examples take the same arguments:

```bash
uv run examples/cartpole_forward.py --batch_size 16 --device cpu --save out/cartpole_forward
```

* `--batch_size`: number of problems solved in parallel (default 8).
* `--device`: `cpu` or `cuda`.
* `--save`: optional path (without extension). The trajectories are written to `<path>.pt` for playback.

On CPU with small batches, limiting PyTorch to one thread is much faster (about 4× in our tests), because the per-step tensors are tiny:

```bash
OMP_NUM_THREADS=1 uv run examples/cartpole_forward.py --batch_size 4
```

## Visualization

`visuals/visualize_trajectory.py` plays back a saved batch of trajectories, drawing one robot per batch element on a grid:

```bash
# Cart-pole (joint states)
uv run visuals/visualize_trajectory.py --urdf cartpole.urdf --file out/cartpole_forward.pt --batch_size 16

# Quadrotor (floating base: position + wxyz quaternion)
uv run visuals/visualize_trajectory.py --urdf quadrotor.urdf --floating_base --file out/quadrotor.pt --batch_size 16

# Franka, with the target configuration drawn as translucent ghosts
uv run visuals/visualize_trajectory.py --urdf fp3.urdf --show_target --spacing_x 1.5 --spacing_y 1.5 --file out/franka.pt --batch_size 4
```

Open http://localhost:8080 and press **Play Trajectory**. To download the playback as a GIF, add `--record` (use `--record_skip N` to keep every N-th frame). **Take Snapshot** saves a PNG. Run with `--help` to see all options (grid layout, camera, playback speed).

Two more specialized scripts are kept:

* `visuals/quadrotor_single/video.py`: a single quadrotor flight with start/goal markers and a ghost trail.
* `visuals/franka_tf.py`: joint sliders for the FP3 that print the end-effector transform.

## Reproducing the paper

The scripts used for the paper's experiments are in `paper_experiments/`. See [`paper_experiments/README.md`](paper_experiments/README.md) for details. In short:

```bash
# Cart-pole, Franka and quadrotor MPC experiments
./paper_experiments/run_experiments.sh

# Batch-size sweep with peak-VRAM logging for diffsqp, mpc.pytorch and TurboMPC
./paper_experiments/benchmarks/run_ours.sh
./paper_experiments/benchmarks/run_mpc_pytorch.sh
./paper_experiments/benchmarks/run_turbo_mpc.sh
```

The scripts can be launched from any directory. Results are written to `results/` next to each script.

## Tests

```bash
uv run python test/test_lqr_cost.py
```

## Project Structure

- `src/diffsqp/`: Core library containing the SQP solver (with LQR and ADMM QP subsolvers), constraints, costs, dynamics definitions and types.
- `examples/`: Proof-of-concept scripts showing how to set up and solve problems.
- `paper_experiments/`: Experiment and benchmark scripts used for the paper.
- `visuals/`: Trajectory playback and figure-generation tools based on viser.
- `resources/robots/`: URDFs and meshes (cart-pole, acrobot, quadrotor, Franka FP3).
- `test/`: Derivative checks.
