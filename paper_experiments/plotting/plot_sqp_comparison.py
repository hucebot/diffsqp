import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt

from plot_utils import *

# plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": "Computer Modern",
    }
)


def load_tri(directory):
    child_list = next(os.walk(directory))[1]
    child_list = sorted(child_list, key=lambda x: int(x.split("_")[0]))
    env_t_hist = []
    env_iter_hist = []
    env_err_hist = []
    for child in child_list:
        runs = glob.glob(directory + child + "/*/log.json")
        t_hist = []
        iter_hist = []
        err_hist = []
        for run in runs:
            with open(run, "r") as r:
                data = json.load(r)
                t_hist.append(data["t_solve_s"])
                iter_hist.append(max(data["num_iter"]))
                err_hist.append(max(data["convergence_error"]))
        env_t_hist.append(t_hist)
        env_iter_hist.append(iter_hist)
        env_err_hist.append(err_hist)

    env_t = np.array(env_t_hist).T
    env_iter = np.array(env_iter_hist).T
    env_err = np.array(env_err_hist).T
    return {
        "env_t": env_t,
        "env_iter": env_iter,
        "env_err": env_err,
    }


def load_sqp(directory):
    child_list = next(os.walk(directory))[1]
    child_list = sorted(child_list, key=lambda x: int(x.split("_")[0]))

    env_t_hist = []
    env_iter_hist = []
    env_err_hist = []
    for child in child_list:
        runs = glob.glob(directory + child + "/*/log.json")
        t_hist = []
        iter_hist = []
        err_hist = []
        for run in runs:
            with open(run, "r") as r:
                data = json.load(r)
                t_hist.append(data["t_solve_s"])
                iter_hist.append(data["ssqp_iterations"])
                err_hist.append(max([data["max_dyn_viol"], data["max_uact_viol"]]))
        env_t_hist.append(t_hist)
        env_iter_hist.append(iter_hist)
        env_err_hist.append(err_hist)

    env_t = np.array(env_t_hist).T
    env_iter = np.array(env_iter_hist).T
    env_err = np.array(env_err_hist).T
    return {
        "env_t": env_t,
        "env_iter": env_iter,
        "env_err": env_err,
    }


tri_for = load_tri("../results/sqp_comparison/diffmpc_results/")
sqp_for = load_sqp("../results/sqp_comparison/ssqp_results/forward_dynamics/")
sqp_inv = load_sqp("../results/sqp_comparison/ssqp_results/inverse_dynamics/")

line_colors = [
    "#056fa1",
    "#23c0d3",
    "#e0b266",
    "#963d4d",
    "#00969f",
    "#aa8b96",
]

text_color = "#221F21"

main = "Scalability Comparison "
secondary = "Wall-time in seconds for increasing batch size"

fig = create_fig()
ax1 = fig.add_subplot(111)
# ax2 = fig.add_subplot(122)

plot_median_and_percentiles(
    ax1, tri_for["env_t"], "\\emph{diffmpc}", line_colors[0], low=5, high=95
)
# plot_median_and_percentiles(ax1, sqp_for["env_t"], "Ours(Forw)", line_colors[1])
plot_median_and_percentiles(
    ax1, sqp_inv["env_t"], "SQP-EQ", line_colors[2], low=5, high=95
)

add_legend(ax1)
# add_legend(ax2)
# add_legend(ax3)

xticks = np.arange(0, 6, 1)
xlabels = np.array([1, 8, 32, 128, 512, 2048])
yticks = np.linspace(0, 35, 6)
ylabels = np.linspace(0, 35, 6)

add_ticks(
    ax1, xticks, yticks, xlabels, ylabels, x_title="Batch Size", y_title="Seconds"
)
# add_ticks(ax2, xticks, yticks, xlabels, ylabels)

# add_patch(fig, ax1)

main_title(fig, main)
secondary_title(fig, secondary)

adjust_layout_for_titles(fig)
plt.savefig("figs/sqp_time_comp.pdf")
plt.show()
