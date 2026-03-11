import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt

from plot_utils import *


def print_results_table(data_dict, title):
    header = f"{'Solver':<12} & {'5%':<5} & {'50%':<5} & {'95%':<5} & {'dyn_viol':<8} & {'uact_viol':<8} & {'MB/Env':<10}"
    divider = "-" * len(header)

    print(f"\n--- {title} ---")
    print(header)
    print(divider)

    for label, data in data_dict.items():
        # Calculations
        mb_per_env = data["bytes_per_env"] / 1e6
        t_qp_solve = data["t_qp_solve"]
        dyn_viols = data["dyn_viols"]
        uact_viols = data["uact_viols"]
        median = np.median(t_qp_solve)
        p5 = np.percentile(t_qp_solve, 5)
        p95 = np.percentile(t_qp_solve, 95)
        dyn = np.max(dyn_viols)
        uact = np.max(uact_viols)

        # Row formatting
        # :<12 means left-aligned with 12 spaces
        # :.4f means float with 4 decimal places
        print(
            f"{label:<12} & {p5:<5.2f} & {median:<5.2f} & {p95:<5.2f} & {dyn:.2e} & {uact:<.2e} & {mb_per_env:<10.2f} \\\\"
        )


def load(dir, dict):
    f_list = glob.glob(dir + "*/log.json")
    t_qp_solve = []
    dyn_viols = []
    uact_viols = []
    bytes_per_env = None
    for file in f_list:
        with open(file, "r") as f:
            data = json.load(f)
            t_qp_solve.append(data["t_qp_solve"])
            bytes_per_env = float(data["cuda_reserved_bytes"]) / float(data["n_batch"])
            dyn_viols.append(data["max_dyn_viol"])
            uact_viols.append(data["max_uact_viol"])

    t_qp_solve = np.array(t_qp_solve).flatten()
    dict["t_qp_solve"] = t_qp_solve[t_qp_solve > 0.0]  # Keep only non-zero times
    dict["dyn_viols"] = np.array(dyn_viols)
    dict["uact_viols"] = np.array(uact_viols)
    dict["bytes_per_env"] = bytes_per_env


lqr_for = {}
kkt_for = {}
qpt_for = {}
qir_for = {}
lqr_inv = {}
kkt_inv = {}
qpt_inv = {}
qir_inv = {}

root = "../results/qp_comparison/"
load(root + "lqr/forward_dynamics/", lqr_for)
load(root + "kkt/forward_dynamics/", kkt_for)
load(root + "qpth/forward_dynamics/", qpt_for)
load(root + "qpth_ir/forward_dynamics/", qir_for)

load(root + "lqr/inverse_dynamics/", lqr_inv)
load(root + "kkt/inverse_dynamics/", kkt_inv)
load(root + "qpth/inverse_dynamics/", qpt_inv)
load(root + "qpth_ir/inverse_dynamics/", qir_inv)

forward_data = {
    "LQR (Ours)": lqr_for,
    "KKT": kkt_for,
    "QPTH": qpt_for,
    "QPTH-IR": qir_for,
}
inverse_data = {
    "LQR (Ours)": lqr_inv,
    "KKT": kkt_inv,
    "QPTH": qpt_inv,
    "QPTH-IR": qir_inv,
}

print_results_table(forward_data, "FORWARD DYNAMICS")
print_results_table(inverse_data, "INVERSE DYNAMICS")
