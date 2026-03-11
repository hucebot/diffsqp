import numpy as np
import matplotlib.pyplot as plt


def create_fig():
    # Set up figure and axes with 5pt margin
    fig_width_pt, fig_height_pt = 332, 0.9 * 332
    fig_width_in, fig_height_in = fig_width_pt / 72, fig_height_pt / 72

    side_margin_in = 6 / 72
    l = side_margin_in / fig_width_in
    r = 1 - l

    bottom_margin_in = (10 + 10 + 10 + 6) / 72
    top_margin_in = (6 + 12.5 + 12.5 + 11.5 + 10) / 72
    b = bottom_margin_in / fig_height_in
    t = 1 - (top_margin_in / fig_height_in)

    fig = plt.figure(figsize=(fig_width_in, fig_height_in))
    fig.subplots_adjust(
        left=l,
        bottom=b,
        right=r,
        top=t,
        wspace=None,
        hspace=None,
    )

    return fig


def main_title(fig, text, text_color="#221F21"):
    margin_left_in = 6 / 72
    margin_top_in = (6 + 13) / 72
    fig_w, fig_h = fig.get_size_inches()
    x = margin_left_in / fig_w
    y = 1 - (margin_top_in / fig_h)
    fig.text(
        x=x,
        y=y,
        s="\\textbf{" + text + "}",
        ha="left",
        fontsize=11,
        weight="bold",
        color=text_color,
    )


def secondary_title(fig, text, text_color="#221F21"):
    margin_left_in = 6 / 72
    margin_top_in = (6 + 13 + 13) / 72
    fig_w, fig_h = fig.get_size_inches()
    x = margin_left_in / fig_w
    y = 1 - (margin_top_in / fig_h)
    fig.text(
        x=x,
        y=y,
        s=text,
        ha="left",
        fontsize=9.5,
        color=text_color,
    )


def adjust_layout_for_titles(fig):
    """
    Pushes the top of the plotting area down to make room for fig.text() titles.
    """
    fig.tight_layout()
    # 6 (margin) + 13 (main) + 13 (secondary) + 15 (padding below titles)
    total_top_margin_in = (6 + 13 + 13 + 17) / 72

    fig_h = fig.get_size_inches()[1]

    # Calculate the top boundary as a fraction of figure height (0 to 1)
    top_boundary = 1 - (total_top_margin_in / fig_h)

    # Push the top of the plot down
    fig.subplots_adjust(top=top_boundary)


def add_ticks(
    ax,
    xticks,
    yticks,
    xlabels,
    ylabels,
    x_title=None,
    y_title=None,
    text_color="#221F21",
):
    # Hide spines and griate y grid
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(which="major", axis="y", color="#AFC0CA", zorder=1)
    ax.xaxis.set_ticks(
        xticks,
        labels=xlabels,
    )
    # ax.set_xticklabels(xlabels)
    ax.xaxis.set_tick_params(
        labelbottom=True, bottom=True, labelsize=7, color=text_color
    )
    ax.yaxis.set_ticks(yticks, labels=ylabels, ha="right", verticalalignment="bottom")
    ax.yaxis.tick_right()
    ax.yaxis.set_tick_params(
        pad=-2,
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelsize=7,
        color=text_color,
    )

    if y_title is not None:
        ax.set_ylabel(
            y_title,
            fontsize=9,
            fontweight="bold",
            color=text_color,
        )
        ax.yaxis.set_label_coords(1.05, 0.5)
    if x_title is not None:
        ax.set_xlabel(
            x_title, fontsize=9, ha="center", fontweight="bold", color=text_color
        )
        # ax.xaxis.set_label_coords(0.0, -0.05)  # Moves title to the top right

    # TODO: Use ax.text instead of title
    # ax.text(
    #     x=1,
    #     y=1.02,
    #     s="Your Y-Axis Title",
    #     transform=ax.transAxes,
    #     ha='right',
    #     va='bottom',
    #     fontsize=9,
    #     color=text_color
    # )


def add_legend(ax):
    ax.legend(
        loc=(0.0, 1.02),
        ncol=4,
        fontsize=7.5,
        frameon=False,
        handletextpad=0.4,
        handlelength=0.8,
    )


def plot_line(ax, data, label, color="#056fa1"):
    data_x = np.arange(0, data.shape[-1], 1)
    data_y = data
    ax.plot(data_x, data_y, linewidth=3, label=label, color=color)


def plot_median_and_percentiles(ax, data, label, color="#056fa1", low=0, high=100):
    data_x = np.arange(0, data.shape[-1], 1)
    data_y = np.median(data, axis=0)
    plow = np.percentile(data, low, axis=0)
    phigh = np.percentile(data, high, axis=0)
    ax.plot(data_x, data_y, linewidth=3, label=label, color=color)
    ax.fill_between(data_x, plow, phigh, color=color, alpha=0.25, linewidth=0)


def add_patch(fig, ax):
    fig_width_in, fig_height_in = fig.get_size_inches()
    rect = plt.Rectangle(
        ((6 / 72) / fig_width_in, 1.0),
        ((15 + 6) / 72) / fig_width_in,  # Width of rectangle
        -(5 / 72) / fig_height_in,  # Height of rectangle. Negative so it goes down.
        facecolor="#E3120B",
        transform=fig.transFigure,
        clip_on=False,
        linewidth=0,
    )
    ax.add_patch(rect)
