import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import torch

plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})


def load(directory):
    f_list = glob.glob(directory + "*/*.json")
    iters = []
    viol = []
    time = []
    for file in f_list:
        with open(file, "r") as f:
            data = json.load(f)
            iters.append(data["sqp_iterations"])
            viol.append(data["constraint_violation"])
            time.append(data["solve_time"])

    vram = "N/A"
    vram_path = os.path.join(directory, "run_1", "vram_peak.txt")
    if os.path.exists(vram_path):
        with open(vram_path, "r") as f:
            vram = f.read().strip()

    return (iters, viol, time, vram)


main_path = "./results/benchmarks/"
data_1 = load(main_path + "turbo_mpc/batch_1/")
data_8 = load(main_path + "turbo_mpc/batch_8/")
data_128 = load(main_path + "turbo_mpc/batch_128/")
data_1024 = load(main_path + "turbo_mpc/batch_1024/")

print(
    np.median(data_1[0]),
    "|||",
    np.max(data_1[1]),
    "|||",
    np.median(data_1[2]),
    "|||",
    data_1[3],
)
print(
    np.median(data_8[0]),
    "|||",
    np.max(data_8[1]),
    "|||",
    np.median(data_8[2]),
    "|||",
    data_8[3],
)
print(
    np.median(data_128[0]),
    "|||",
    np.max(data_128[1]),
    "|||",
    np.median(data_128[2]),
    "|||",
    data_128[3],
)
print(
    np.median(data_1024[0]),
    "|||",
    np.max(data_1024[1]),
    "|||",
    np.median(data_1024[2]),
    "|||",
    data_1024[3],
)

main_path = "./results/benchmarks/"
data_1 = load(main_path + "mpc_pytorch/batch_1/")
data_8 = load(main_path + "mpc_pytorch/batch_8/")
data_128 = load(main_path + "mpc_pytorch/batch_128/")
data_1024 = load(main_path + "mpc_pytorch/batch_1024/")

print()
print()
print()
print()
print(
    np.median(data_1[0]),
    "|||",
    np.max(data_1[1]),
    "|||",
    np.median(data_1[2]),
    "|||",
    data_1[3],
)
print(
    np.median(data_8[0]),
    "|||",
    np.max(data_8[1]),
    "|||",
    np.median(data_8[2]),
    "|||",
    data_8[3],
)
print(
    np.median(data_128[0]),
    "|||",
    np.max(data_128[1]),
    "|||",
    np.median(data_128[2]),
    "|||",
    data_128[3],
)
print(
    np.median(data_1024[0]),
    "|||",
    np.max(data_1024[1]),
    "|||",
    np.median(data_1024[2]),
    "|||",
    data_1024[3],
)
