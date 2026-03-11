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


def load(directory):
    file_list = glob.glob(directory + "*/log.json")
    t_qp_hist = []
    t_ls_hist = []
    iter_ls_hist = []
    for file in file_list:
        with open(file, "r") as f:
            data = json.load(f)
            t_qp_hist.append(data["t_qp_solve"])
            t_ls_hist.append(data["t_line_search"])
            iter_ls_hist.append(data["line_search_iters"])

    return {
        "t_qp": np.array(t_qp_hist),
        "t_ls": np.array(t_ls_hist),
        "i_ls": np.array(iter_ls_hist),
    }


lqr_f = load("../results/qp_comparison/lqr/forward_dynamics/")
kkt_f = load("../results/qp_comparison/kkt/forward_dynamics/")
qpt_f = load("../results/qp_comparison/qpth/forward_dynamics/")
qir_f = load("../results/qp_comparison/qpth_ir/forward_dynamics/")

lqr_i = load("../results/qp_comparison/lqr/inverse_dynamics/")
kkt_i = load("../results/qp_comparison/kkt/inverse_dynamics/")
qpt_i = load("../results/qp_comparison/qpth/inverse_dynamics/")
qir_i = load("../results/qp_comparison/qpth_ir/inverse_dynamics/")

line_colors = [
    "#056fa1",
    "#23c0d3",
    "#e0b266",
    "#963d4d",
    "#00969f",
    "#aa8b96",
]

text_color = "#221F21"

# Plot QP Solve times figure
fig1 = create_fig()
ax1 = fig1.add_subplot(121)
ax2 = fig1.add_subplot(122)

main_title(fig1, "Internal QP solver")
secondary_title(fig1, "Wall-times per SQP iterationForward and inverse dynamics")

xticks = np.array([0, 9, 19, 29, 39, 49])
xlabels = xticks + 1
yticks = np.arange(0, 1.7, 0.2)
ylabels = np.round(yticks, 2)

plot_median_and_percentiles(ax1, lqr_f["t_qp"][:, :50], "LQR (Ours)", line_colors[0])
plot_median_and_percentiles(ax1, kkt_f["t_qp"][:, :50], "KKT", line_colors[1])
plot_median_and_percentiles(ax1, qpt_f["t_qp"][:, :50], "QPTH", line_colors[2])
plot_median_and_percentiles(ax1, qir_f["t_qp"][:, :50], "QPTH-IR", line_colors[3])

plot_median_and_percentiles(ax2, lqr_i["t_qp"][:, :50], "LQR (Ours)", line_colors[0])
plot_median_and_percentiles(ax2, kkt_i["t_qp"][:, :50], "KKT", line_colors[1])
plot_median_and_percentiles(ax2, qpt_i["t_qp"][:, :50], "QPTH", line_colors[2])
plot_median_and_percentiles(ax2, qir_i["t_qp"][:, :50], "QPTH-IR", line_colors[3])

add_legend(ax1)
add_legend(ax2)

add_ticks(ax1, xticks, yticks, xlabels, ylabels)
add_ticks(ax2, xticks, yticks, xlabels, ylabels)

# plt.savefig("figs/qp_wall_times.pdf")

# Plot LS Solve times
fig2 = create_fig()
ax3 = fig2.add_subplot(121)
ax4 = fig2.add_subplot(122)

main_title(fig2, "Line search solve times")
secondary_title(fig2, "Forward and inverse dynamics")

xticks = np.array([0, 9, 19, 29, 39, 49])
xlabels = xticks + 1
yticks = np.arange(0, 0.9, 0.2)
ylabels = np.round(yticks, 2)

plot_median_and_percentiles(ax3, lqr_f["t_ls"][:, :50], "LQR (Ours)", line_colors[0])
plot_median_and_percentiles(ax3, kkt_f["t_ls"][:, :50], "KKT", line_colors[1])
plot_median_and_percentiles(ax3, qpt_f["t_ls"][:, :50], "QPTH", line_colors[2])
plot_median_and_percentiles(ax3, qir_f["t_ls"][:, :50], "QPTH-IR", line_colors[3])

plot_median_and_percentiles(ax4, lqr_i["t_ls"][:, :50], "LQR (Ours)", line_colors[0])
plot_median_and_percentiles(ax4, kkt_i["t_ls"][:, :50], "KKT", line_colors[1])
plot_median_and_percentiles(ax4, qpt_i["t_ls"][:, :50], "QPTH", line_colors[2])
plot_median_and_percentiles(ax4, qir_i["t_ls"][:, :50], "QPTH-IR", line_colors[3])

add_legend(ax3)
add_legend(ax4)

add_ticks(ax3, xticks, yticks, xlabels, ylabels)
add_ticks(ax4, xticks, yticks, xlabels, ylabels)

# plt.savefig("qp_times.pdf")

# Plot LS Solve times
fig3 = create_fig()
ax5 = fig3.add_subplot(211)
ax6 = fig3.add_subplot(212)

main_title(fig3, "Line Search Iterations Per SQP step ")
secondary_title(fig3, "Forward (top) and inverse (bottom) dynamics")

xticks = np.array([0, 9, 19, 29, 39, 49])
xlabels = xticks + 1
yticks = np.arange(0, 6, 1)
ylabels = np.round(yticks, 2)

plot_median_and_percentiles(ax5, lqr_f["i_ls"][:, :50], "LQR", line_colors[0])
plot_median_and_percentiles(ax5, kkt_f["i_ls"][:, :50], "LU", line_colors[1])
plot_median_and_percentiles(ax5, qpt_f["i_ls"][:, :50], "QPTH", line_colors[2])
plot_median_and_percentiles(ax5, qir_f["i_ls"][:, :50], "QPTH-IR", line_colors[3])

plot_median_and_percentiles(ax6, lqr_i["i_ls"][:, :50], "LQR-EQ", line_colors[0])
plot_median_and_percentiles(ax6, kkt_i["i_ls"][:, :50], "LU", line_colors[1])
plot_median_and_percentiles(ax6, qpt_i["i_ls"][:, :50], "QPTH", line_colors[2])
plot_median_and_percentiles(ax6, qir_i["i_ls"][:, :50], "QPTH-IR", line_colors[3])

add_legend(ax5)
add_legend(ax6)

add_ticks(ax5, xticks, yticks, xlabels, ylabels, y_title="LS Iterations")
yticks = np.arange(0, 3, 1)
ylabels = np.round(yticks, 2)
add_ticks(
    ax6,
    xticks,
    yticks,
    xlabels,
    ylabels,
    x_title="SQP Iterations",
    y_title="LS Iterations",
)

# fig3.tight_layout()
adjust_layout_for_titles(fig3)
plt.savefig("figs/line_search_iters.pdf")
# plt.show()
