import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt

from plot_utils import *


def print_results_table(data_dict, title):
    header = f"{'Implementation':<12} | {'5th %':<10} | {'Median (s)':<10} | {'95th %':<10} | {'dyn_viol':<8} | {'uact_viol':<8}"
    divider = "-" * len(header)

    print(f"\n--- {title} ---")
    print(header)
    print(divider)

    for label, data in data_dict.items():
        # Calculations
        i_sqp = data["sqp_iters"]
        dyn_viols = data["dyn_viols"]
        uact_viols = data["dyn_viols"]
        median = np.median(i_sqp)
        p5 = np.percentile(i_sqp, 0)
        p95 = np.percentile(i_sqp, 100)
        dyn = np.max(dyn_viols)
        uact = np.max(uact_viols)

        # Row formatting
        # :<12 means left-aligned with 12 spaces
        # :.4f means float with 4 decimal places
        print(
            f"{label:<12} | {p5:<10.6f} | {median:<10.6f} | {p95:<10.6f} | {dyn:.2e} | {uact:<.2e}"
        )


def load(dir):
    f_list = glob.glob(dir + "*/*/log.json")

    sqp_iter_hist = []
    dyn_viol_hist = []
    uact_viol_hist = []
    for file in f_list:
        with open(file, "r") as f:
            data = json.load(f)
            sqp_iter_hist.append(data["ssqp_iterations"])
            dyn_viol_hist.append(data["max_dyn_viol"])
            uact_viol_hist.append(data["max_uact_viol"])

    return {
        "sqp_iters": np.array(sqp_iter_hist),
        "dyn_viols": np.array(dyn_viol_hist),
        "uact_viols": np.array(uact_viol_hist),
    }


root = "../results/execution_time/"
cpu_for = load(root + "cpu_parallel/forward_dynamics/")
gpu_for = load(root + "cuda_parallel/forward_dynamics/")

cpu_inv = load(root + "cpu_parallel/inverse_dynamics/")
gpu_inv = load(root + "cuda_parallel/inverse_dynamics/")

forward_data = {
    "CPU": cpu_for,
    "GPU": gpu_for,
}
inverse_data = {
    "CPU": cpu_inv,
    "GPU": gpu_inv,
}

print_results_table(forward_data, "FORWARD DYNAMICS")
print_results_table(inverse_data, "INVERSE DYNAMICS")
