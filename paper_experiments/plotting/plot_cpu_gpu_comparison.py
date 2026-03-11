import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt

from plot_utils import *

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": "Computer Modern",
    }
)


def load(directory):
    child_list = next(os.walk(directory))[1]
    child_list = sorted(child_list, key=lambda x: int(x.split("_")[0]))

    envs_t_solve = []
    envs_ls_iters = []
    for child in child_list:
        runs = glob.glob(directory + child + "/*/log.json")
        t_solve_hist = []
        iter_ls_hist = []
        for run in runs:
            with open(run, "r") as r:
                data = json.load(r)
                t_solve_hist.append(data["t_solve_s"])
                iter_ls_hist.append(data["line_search_iters"])
        envs_t_solve.append(t_solve_hist)
        envs_ls_iters.append(iter_ls_hist)

    return {"t_sqp": np.array(envs_t_solve).T, "i_ls": np.array(envs_ls_iters).T}


cpu_f = load("../results/execution_time/cpu_parallel/forward_dynamics/")
cpu_i = load("../results/execution_time/cpu_parallel/inverse_dynamics/")
gpu_f = load("../results/execution_time/cuda_parallel/forward_dynamics/")
gpu_i = load("../results/execution_time/cuda_parallel/inverse_dynamics/")

line_colors = [
    "#056fa1",
    "#23c0d3",
    "#e0b266",
    "#963d4d",
    "#00969f",
    "#aa8b96",
]

text_color = "#221F21"

fig = create_fig()
main_title(fig, "Performance Across Architectures")
secondary_title(fig, "Wall-time to solution as batch size increases")

ax = fig.add_subplot(111)

plot_median_and_percentiles(ax, cpu_f["t_sqp"], "CPU For. Dyn.", line_colors[0], 5, 95)
plot_median_and_percentiles(ax, cpu_i["t_sqp"], "CPU Inv. Dyn.", line_colors[1], 5, 95)
plot_median_and_percentiles(ax, gpu_f["t_sqp"], "GPU For. Dyn.", line_colors[2], 5, 95)
plot_median_and_percentiles(ax, gpu_i["t_sqp"], "GPU Inv. Dyn.", line_colors[3], 5, 95)

add_legend(ax)

data_x = np.arange(0, cpu_f["t_sqp"].shape[-1], 1)
xticks = data_x
# xlabels = data_x
xlabels = np.array([128, 512, 2048, 8192, 32768])
yticks = np.arange(0, 351, 50)
ylabels = yticks

add_ticks(ax, xticks, yticks, xlabels, ylabels, x_title="Batch Size", y_title="Seconds")
# ax.set_xscale("log", base=2)

# add_patch(fig, ax1)

adjust_layout_for_titles(fig)
plt.savefig("figs/cpu_gpu_wall_times.pdf")
plt.show()
