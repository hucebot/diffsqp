# Paper experiments

These are the scripts used to produce the paper's results. They are kept unchanged, apart from making the shell scripts runnable from any directory and fixing two script paths in the benchmark runners. For readable starting points, see `../examples/` instead.

All experiments assume a CUDA GPU (`DEVICE="cuda"` in the shell scripts). Each `.sh` script `cd`s into its own directory, so every output path below is relative to that directory.

## Main experiments: `run_experiments.sh`

```bash
./paper_experiments/run_experiments.sh
```

| Script | Task | Batch size | Runs |
|---|---|---|---|
| `cartpole/trajopt_forward.py` | Cart-pole swing-up, forward dynamics | 32768 | 5 |
| `cartpole/trajopt_inverse_lqr.py` | Cart-pole swing-up, inverse dynamics, underactuation constraint handled inside the LQR (`n_h=1`) | 32768 | 5 |
| `cartpole/trajopt_inverse.py` | Cart-pole swing-up, inverse dynamics, underactuation as a general constraint in the ADMM QP | 32768 | 5 |
| `franka/ee_task_unconstrained.py` | Franka FP3 end-effector reaching, loose velocity bounds (±10 rad/s) | 32768 | 5 |
| `franka/ee_task_constrained.py` | Franka FP3 end-effector reaching, joint position and velocity limits | 32768 | 5 |
| `quadrotor/mpc_forward.py` | Quadrotor MPC (`-n_mpc` steps of `-dt_mpc`) | 1024 | 2 |

Outputs go to `results/<script>/run_<i>/solution.pt` (trajectories `x`, `u`, `x_des`) and, for the cart-pole scripts, `solution.json` (the SQP log: iterations, cost, constraint violation, wall time, CUDA memory). The franka scripts open a matplotlib window after solving, so close it to continue the sweep.

Every script also runs on its own:

```bash
cd paper_experiments
uv run cartpole/trajopt_forward.py -batch_size 1024 -device cuda -save results/cartpole_forward
```

## Benchmarks: `benchmarks/`

These compare diffsqp against [mpc.pytorch](https://github.com/locuslab/mpc.pytorch) and TurboMPC on the same quadrotor problem. The sweep covers batch sizes 1, 32, 1024 and 32768, with 5 runs each, and logs peak GPU memory with `track_vram.sh` (polls `nvidia-smi`).

```bash
./paper_experiments/benchmarks/run_ours.sh         # diffsqp
./paper_experiments/benchmarks/run_mpc_pytorch.sh  # needs the `mpc` package
./paper_experiments/benchmarks/run_turbo_mpc.sh    # needs jax and turbompc
```

`mpc` (mpc.pytorch), `jax` and `turbompc` are not project dependencies. Install them separately into the environment you run these scripts in.

Outputs go to `benchmarks/results/quadrotor/<solver>/batch_<B>/run_<i>/` (`solution.json`, `vram_peak.txt` in MiB, and `solution.pt` for diffsqp).

`benchmarks/double_integrator/` holds the same comparison for a double integrator (`ours.py`, `mpc_pytorch.py`). No shell runner covers it, so run those scripts directly.
