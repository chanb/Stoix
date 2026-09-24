"""Combines the hardness-vs-compute-time and hardness-vs-episode-length plots
from both `analysis-compute_time-hardness.ipynb` (Lights Out) and
`analysis-compute_time-hardness-slidingpuzzle.ipynb` (sliding puzzle) into a
single 1x4 figure: [lightsout compute time, lightsout episode length,
slidingpuzzle compute time, slidingpuzzle episode length].

Pure numpy/pandas/matplotlib - no JAX/Flax/hydra/checkpoint restore needed.
Reads `plot_data-lightsout.pkl`/`plot_data-slidingpuzzle.pkl`, produced by the
"Export plot data for the standalone plotting scripts" cell near the end of
each notebook - run that cell (in each notebook) at least once before running
this script.
"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch

import analysis_utils as base

set_size = base.set_size
pgf_with_latex = base.pgf_with_latex
doc_width_pt = base.doc_width_pt
plt.rcParams.update(pgf_with_latex)
sns.set_palette("colorblind")

HERE = Path(__file__).resolve().parent
CI95_Z = 1.96  # normal-approximation 95% CI half-width, in units of SEM


def load_plot_data(name):
    path = HERE / f"plot_data-{name}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def violin_groups_for(hardness, values):
    """One (hardness_value, values) group per distinct hardness value, KDE-
    plottable groups only (>= 2 points, non-degenerate) - see the notebooks'
    own hardness-vs-* cells for the same convention."""
    groups = {}
    for h, v in zip(hardness, values):
        groups.setdefault(h, []).append(v)
    groups = {h: np.asarray(v) for h, v in groups.items()}
    return [(h, v) for h, v in sorted(groups.items()) if len(v) >= 2 and np.std(v) > 0]


def plot_hardness_panel(ax, hardness, values, ylabel, title, diagonal=False):
    violin_color = sns.color_palette("colorblind")[0]
    mean_color = sns.color_palette("colorblind")[3]

    pearson_r = np.corrcoef(hardness, values)[0, 1]

    groups = violin_groups_for(hardness, values)
    if groups:
        parts = ax.violinplot(
            [v for _, v in groups],
            positions=[h for h, _ in groups],
            widths=0.8,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body in parts["bodies"]:
            body.set_facecolor(violin_color)
            body.set_edgecolor(violin_color)
            body.set_alpha(0.4)

    unique_h = np.unique(hardness)
    means = np.array([values[hardness == h].mean() for h in unique_h])
    sems = np.array(
        [
            values[hardness == h].std(ddof=1) / np.sqrt((hardness == h).sum())
            if (hardness == h).sum() > 1
            else 0.0
            for h in unique_h
        ]
    )

    handles = [Patch(facecolor=violin_color, alpha=0.4, label="episode distribution")]

    if diagonal:
        axis_min = min(hardness.min(), values.min())
        (diagonal_line,) = ax.plot(
            [axis_min, hardness.max()],
            [axis_min, hardness.max()],
            linestyle="--",
            color="gray",
            linewidth=1.2,
            alpha=0.7,
            zorder=0,
            label="optimal (episode length = shortest path)",
        )
        handles.append(diagonal_line)

    mean_errorbar = ax.errorbar(
        unique_h,
        means,
        yerr=CI95_Z * sems,
        marker="o",
        ms=1,
        linewidth=1,
        color=mean_color,
        label=r"mean $\pm$ 95\% CI",
    )
    handles.append(mean_errorbar)

    # Pearson r as an in-axis annotation (not the title): a 4-up figure has no
    # room for a title long enough to spell out both the env label and the
    # correlation without adjacent panels' titles colliding. fontsize=8
    # matches analysis-unshared_iru.ipynb's analogous delta annotation.
    ax.text(
        0.95,
        0.05,
        f"$r={pearson_r:.2f}$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.0),
    )
    # ax.set_xlabel("Shortest path length")
    # ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return handles


def main():
    lightsout = load_plot_data("lightsout")
    slidingpuzzle = load_plot_data("slidingpuzzle")

    env_map = {
        "lightsout-5x4": "Lightsout",
        "slidingtile-3x3": "Sliding puzzle",
    }

    nrows = 1
    ncols = 4
    # doc_width_in = doc_width_pt / 72.27
    # fig, axes = plt.subplots(nrows, ncols, figsize=(doc_width_in * 1.3, doc_width_in * 0.42))
    figsize = set_size(doc_width_pt, fraction=0.95, subplots=(nrows, ncols), use_golden_ratio=False)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    plot_hardness_panel(
        axes[0],
        lightsout["hardness"],
        lightsout["first_step_compute_time"],
        ylabel="Compute steps",
        title="Compute steps",
        # title=env_map[lightsout["env_label"]],
    )
    handles = plot_hardness_panel(
        axes[1],
        lightsout["hardness"],
        lightsout["episode_length"],
        ylabel="Ep. length",
        title="Ep. length",
        # title=env_map[lightsout["env_label"]],
        diagonal=True,
    )
    plot_hardness_panel(
        axes[2],
        slidingpuzzle["hardness"],
        slidingpuzzle["first_step_compute_time"],
        ylabel="Compute steps",
        title="Compute steps",
        # title=env_map[slidingpuzzle["env_label"]],
    )
    plot_hardness_panel(
        axes[3],
        slidingpuzzle["hardness"],
        slidingpuzzle["episode_length"],
        ylabel="Ep. length",
        title="Ep. length",
        # title=env_map[slidingpuzzle["env_label"]],
        diagonal=True,
    )

    # Matches analysis-unshared_iru.ipynb's plot_pareto_row exactly: a single
    # tight_layout call (before adding the group titles/legend/x label below),
    # pad=0.2/w_pad=1.0 for a squarer per-panel look, then those three sit
    # *outside* the resulting figure box (y > 1 or y < 0) - bbox_inches=
    # "tight" at save time expands the saved page to include them. A second
    # tight_layout call after placing them would move the axes again and
    # invalidate the positions those placements were computed from, so unlike
    # an earlier version of this script, there is only one here.
    fig.tight_layout(pad=0.2, w_pad=1.0)

    XLABEL_OFFSET_IN = 0.19  # x label's bottom edge below the figure box
    GROUP_TITLE_GAP_IN = 0.07  # group titles' bottom edge above the figure box
    GROUP_TITLE_HEIGHT_IN = 0.2  # room the legend leaves for the group titles
    fig_h = fig.get_figheight()
    fig.supxlabel("Shortest path length", y=-XLABEL_OFFSET_IN / fig_h)
    PANEL_GROUPS = [("Lightsout", [0, 1]), ("Sliding puzzle", [2, 3])]
    for title, cols in PANEL_GROUPS:
        x0 = min(axes[i].get_position().x0 for i in cols)
        x1 = max(axes[i].get_position().x1 for i in cols)
        fig.text((x0 + x1) / 2, 1.0 + GROUP_TITLE_GAP_IN / fig_h, title, ha="center", va="bottom", fontsize=12)

    by_label = {}
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            by_label.setdefault(l, h)
    fig.legend(
        list(by_label.values()),
        list(by_label.keys()),
        bbox_to_anchor=(0.0, 1.0 + (GROUP_TITLE_GAP_IN + GROUP_TITLE_HEIGHT_IN) / fig_h, 1.0, 0.0),
        loc="lower center",
        ncols=len(by_label),
        borderaxespad=0.0,
        frameon=True,
    )
    out_path = HERE / "analysis-hardness_grid.pdf"
    fig.savefig(out_path, dpi=600, format="pdf", bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
