"""Combines the hardness-vs-compute-time and action-vs-compute-time plots
from both `analysis-compute_time-hardness.ipynb` (Lights Out) and
`analysis-compute_time-hardness-slidingpuzzle.ipynb` (sliding puzzle) into a
single 1x4 figure: [lightsout hardness, lightsout action, slidingpuzzle
hardness, slidingpuzzle action], all against compute time.
"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from scipy.stats import trim_mean

import analysis_utils as base

set_size = base.set_size
pgf_with_latex = base.pgf_with_latex
doc_width_pt = base.doc_width_pt
plt.rcParams.update(pgf_with_latex)
sns.set_palette("colorblind")

HERE = Path(__file__).resolve().parent
N_BOOTSTRAP = 2000
CI_LEVEL = 0.95
BOOTSTRAP_SEED = 0

# Rollout caches the notebooks' per-step action-vs-compute-time cells read
# (and that plot_data-*.pkl's hardness data was exported from).
ROLLOUT_CACHES = {
    "lightsout": "rollout_cache-lightsout_5x4-trained-20260923032800-n50000-seed0-v3.pkl",
    "slidingpuzzle": "rollout_cache-slidingtile_gs3-nrm200-20260923063537-n50000-seed0-v2.pkl",
}
SLIDINGPUZZLE_ACTION_NAMES = [r"$\uparrow$", r"$\rightarrow$", r"$\downarrow$", r"$\leftarrow$"]  # jumanji's MOVES order (Up, Right, Down, Left)


def iqm(x, axis=None):
    """Interquartile mean: mean of the middle 50% (25% trimmed each side)."""
    return trim_mean(x, 0.25, axis=axis)


def iqm_bootstrap_ci(x, rng, n_bootstrap=N_BOOTSTRAP, ci=CI_LEVEL, max_chunk_elems=20_000_000):
    """Percentile bootstrap CI of the IQM of `x`."""
    x = np.asarray(x)
    if len(x) < 2:
        return x.mean(), x.mean()
    chunk = max(1, min(n_bootstrap, max_chunk_elems // len(x)))
    stats = []
    for start in range(0, n_bootstrap, chunk):
        size = min(chunk, n_bootstrap - start)
        idx = rng.integers(0, len(x), size=(size, len(x)))
        stats.append(iqm(x[idx], axis=1))
    stats = np.concatenate(stats)
    alpha = (1 - ci) / 2
    return np.quantile(stats, alpha), np.quantile(stats, 1 - alpha)


def load_plot_data(name):
    path = HERE / f"plot_data-{name}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def load_step_actions(name):
    """Every real (non-padding) step's action and compute time, from solved episodes only -
    matching the notebooks' build_step_dataframe, which the action-vs-compute-time cells use."""
    df = pd.read_pickle(HERE / ROLLOUT_CACHES[name])
    df = df[df["solved"] & (df["shortest_path_length"] > 0)]
    valid = np.stack(df["valid_steps"].to_numpy())
    actions = np.stack(df["actions"].to_numpy())[valid]
    compute_times = np.stack(df["compute_times"].to_numpy())[valid]
    return actions, compute_times


def eta_squared(groups, values):
    """One-way ANOVA effect size SS_between / SS_total: the share of `values`'
    variance explained by `groups`, treating groups as unordered categories
    (unlike Pearson r, which would treat action indices as ordinal)."""
    grand_mean = values.mean()
    ss_between = sum(
        (groups == g).sum() * (values[groups == g].mean() - grand_mean) ** 2 for g in np.unique(groups)
    )
    ss_total = ((values - grand_mean) ** 2).sum()
    return ss_between / ss_total


def violin_groups_for(hardness, values):
    """One (hardness_value, values) group per distinct hardness value, KDE-
    plottable groups only (>= 2 points, non-degenerate) - see the notebooks'
    own hardness-vs-* cells for the same convention."""
    groups = {}
    for h, v in zip(hardness, values):
        groups.setdefault(h, []).append(v)
    groups = {h: np.asarray(v) for h, v in groups.items()}
    return [(h, v) for h, v in sorted(groups.items()) if len(v) >= 2 and np.std(v) > 0]


def plot_violin_panel(ax, hardness, values, xlabel, categorical=False, xticklabels=None):
    """Violins + IQM of `values` per distinct `hardness` value."""
    violin_color = sns.color_palette("colorblind")[0]
    mean_color = sns.color_palette("colorblind")[3]

    if categorical:
        stat_label = rf"$\eta^2={eta_squared(hardness, values):.2f}$"
    else:
        stat_label = f"$r={np.corrcoef(hardness, values)[0, 1]:.2f}$"

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
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    iqms = np.array([iqm(values[hardness == h]) for h in unique_h])
    cis = np.array([iqm_bootstrap_ci(values[hardness == h], rng) for h in unique_h])
    # errorbar wants non-negative (below, above) offsets; clip guards against
    # tiny negative offsets when the IQM sits on a CI endpoint.
    yerr = np.clip(np.stack([iqms - cis[:, 0], cis[:, 1] - iqms]), 0, None)

    handles = [Patch(facecolor=violin_color, alpha=0.4, label="distribution")]

    ax.set_yticks([1, 2, 3, 4, 5])
    if xticklabels is not None:
        ax.set_xticks(unique_h, xticklabels)

    mean_errorbar = ax.errorbar(
        unique_h,
        iqms,
        yerr=yerr,
        marker="o",
        # Unordered actions: no connecting line (it would imply a trend
        # between neighbouring indices), so the markers need to be visible.
        ms=1,
        linestyle="none" if categorical else "-",
        linewidth=1,
        color=mean_color,
        label=r"IQM $\pm$ 95\% bootstrap CI",
    )
    handles.append(mean_errorbar)

    # Pearson r / eta^2 as an in-axis annotation (not the title): a 4-up figure has no room for a
    # title long enough to spell out both the env label and the statistic without adjacent panels'
    # titles colliding.
    ax.text(
        0.95,
        0.05,
        stat_label,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.0),
    )
    ax.set_xlabel(xlabel)
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

    lightsout_actions, lightsout_step_compute_time = load_step_actions("lightsout")
    slidingpuzzle_actions, slidingpuzzle_step_compute_time = load_step_actions("slidingpuzzle")

    handles = plot_violin_panel(
        axes[0],
        lightsout["hardness"],
        lightsout["first_step_compute_time"],
        xlabel="Shortest path length",
    )
    plot_violin_panel(
        axes[1],
        lightsout_actions,
        lightsout_step_compute_time,
        xlabel="Action (button)",
        categorical=True,
    )
    # 20 buttons don't fit as individual tick labels in a quarter-width panel.
    axes[1].set_xticks([0, 5, 10, 15])
    plot_violin_panel(
        axes[2],
        slidingpuzzle["hardness"],
        slidingpuzzle["first_step_compute_time"],
        xlabel="Shortest path length",
    )
    plot_violin_panel(
        axes[3],
        slidingpuzzle_actions,
        slidingpuzzle_step_compute_time,
        xlabel="Action",
        categorical=True,
        xticklabels=SLIDINGPUZZLE_ACTION_NAMES,
    )
    axes[0].set_ylabel("Compute steps")

    # Matches analysis-unshared_iru.ipynb's plot_pareto_row exactly: a single
    # tight_layout call (before adding the group titles/legend below),
    # pad=0.2/w_pad=1.0 for a squarer per-panel look, then those three sit
    # *outside* the resulting figure box (y > 1 or y < 0) - bbox_inches=
    # "tight" at save time expands the saved page to include them. A second
    # tight_layout call after placing them would move the axes again and
    # invalidate the positions those placements were computed from, so unlike
    # an earlier version of this script, there is only one here.
    fig.tight_layout(pad=0.2, w_pad=1.0)

    GROUP_TITLE_GAP_IN = 0.07  # group titles' bottom edge above the figure box
    GROUP_TITLE_HEIGHT_IN = 0.2  # room the legend leaves for the group titles
    fig_h = fig.get_figheight()
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
