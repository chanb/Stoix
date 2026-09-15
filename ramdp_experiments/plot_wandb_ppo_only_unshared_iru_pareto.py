#!/usr/bin/env python
"""Compute-vs-performance pareto plot for the `lightsout_sweep-ppo_only`
wandb project (fetch_wandb_ppo_only_unshared_iru.py), restricted to
network.actor_network.pre_torso._target_ ==
UnsharedIRUAdaptiveComputationTimeTorso and system.qac_variant ==
"reinforce" - the only architecture/variant combo in that project, so unlike
plot_wandb_lightsout_hd64_extra.py's plot_pareto_row (one column per
architecture) there is just a single column here.

Same visual as plot_pareto_row: fixed-budget points (mean +/- SEM over
seeds) as plain gray markers, plus one PPO/reinforce-colored diamond for the
adaptive[1-16] budget - final performance vs. final compute steps. Produces
two files (actor, evaluator), each a 1x2 figure with episode return and
discounted return side by side (mirroring plot_wandb_lightsout_hd64_extra
.py's plot_pareto, which pairs the two return metrics in one figure, rather
than plot_pareto_row's one-file-per-metric split).

Note: this project's actor/* episode metrics never include compute_time
(see fetch_wandb_ppo_only_unshared_iru.py's docstring), so
trainer/compute_time is used as the x-axis for both the actor-return and
evaluator-return panels.

Pass --trim N to drop the N highest- and N lowest-performing seeds from each
budget/qac_variant group before averaging (a trimmed mean/SEM), to keep one
or two outlier seeds from dominating a group of only ~10.

Usage:
  python ramdp_experiments/plot_wandb_ppo_only_unshared_iru_pareto.py \\
      ramdp_experiments/wandb_cache_ppo_only_unshared_iru.csv [--trim N]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from matplotlib.ticker import FormatStrFormatter, NullLocator
from matplotlib.transforms import blended_transform_factory

import plot_wandb_lightsout_hd64 as base

plt = base.plt
sns = base.sns
pd = base.pd
np = base.np

set_size = base.set_size
pgf_with_latex = base.pgf_with_latex
compute_final_values = base.compute_final_values
doc_width_pt = base.doc_width_pt

COMPUTE_METRIC = "trainer/compute_time"

# Local label overrides - "PPO"/"Uniform budget" are the shared names used
# by variant_row_label/plot_pareto_row elsewhere, but "Adaptive compute"/
# "Uniform compute" read more clearly on their own in this single-arch plot.
UNIFORM_LABEL = "Uniform compute"
ADAPTIVE_LABEL = {"reinforce": "Adaptive compute"}

FS = dict(title=15, panel_title=13, label=13, tick=11, legend=11)


def place_legend_and_title_tight(fig, handles, labels, ncols: int, figsize, title: str) -> None:
    """Same idea as base.place_legend_and_title (legend just above the axes,
    title just above the legend) but with smaller fixed-inch offsets, so the
    legend/title don't float far off the top of the panels."""
    legend_rows = -(-len(labels) // ncols)
    legend_top_frac = 1.0 + (0.14 * legend_rows) / figsize[1]
    title_y = 0.30 / figsize[1]

    fig.legend(
        handles,
        labels,
        bbox_to_anchor=(0.0, 1.0, 1.0, 0.0),
        loc="lower center",
        ncols=ncols,
        borderaxespad=0.0,
        frameon=True,
        fontsize=FS["legend"],
    )
    # fig.suptitle(title, y=title_y, fontsize=FS["title"])


PARETO_FIGURES = [
    dict(
        prefix="actor",
        fname="actor",
        metrics=[
            ("actor/episode_return/mean", "Episode return"),
            ("actor/episode_discounted_return/mean", "Discounted return"),
        ],
    ),
    dict(
        prefix="evaluator",
        fname="evaluator",
        metrics=[
            ("evaluator/episode_return/mean", "Episode return"),
            ("evaluator/episode_discounted_return/mean", "Discounted return"),
        ],
    ),
]

SQUARE_FIGURES = [
    dict(prefix="actor", fname="actor_discounted_return_square", metric="actor/episode_discounted_return/mean"),
    dict(
        prefix="evaluator",
        fname="evaluator_discounted_return_square",
        metric="evaluator/episode_discounted_return/mean",
    ),
]


def _trim_by_performance(g: pd.DataFrame, trim: int) -> pd.DataFrame:
    """Drops the `trim` seeds with the lowest and `trim` seeds with the
    highest `value_perf` from a (budget or qac_variant) group, so the
    reported mean/SEM is a trimmed statistic robust to one or two outlier
    seeds. No-op if there aren't enough seeds left over (need > 2*trim)."""
    if trim <= 0 or len(g) <= 2 * trim:
        return g
    return g.sort_values("value_perf").iloc[trim : len(g) - trim]


def _group_stats(g: pd.DataFrame, trim: int) -> "tuple[float, float, float, float, int]":
    """(perf_mean, perf_sem, comp_mean, comp_sem, n) for one group, after
    trimming (see _trim_by_performance)."""
    g = _trim_by_performance(g, trim)
    n = len(g)
    perf_mean = g["value_perf"].mean()
    perf_sem = g["value_perf"].std() / np.sqrt(n) if n > 1 else 0.0
    comp_mean = g["value_compute"].mean()
    comp_sem = g["value_compute"].std() / np.sqrt(n) if n > 1 else 0.0
    return perf_mean, perf_sem, comp_mean, comp_sem, n


def _draw_panel(ax, df: pd.DataFrame, metric: str, panel_label: str, show_ylabel: bool, trim: int = 0) -> None:
    """Draws one metric's uniform/adaptive-compute pareto points (plus, for
    the discounted-return metric, the best-uniform-vs-adaptive delta
    annotation) onto `ax`. Shared by plot_pareto (multi-panel) and
    plot_pareto_square (single square panel). `trim` drops the `trim`
    highest- and lowest-performing seeds from each budget/variant group
    before averaging (see _trim_by_performance)."""
    qac_markers = {"reinforce": "D"}
    qac_colors = {"reinforce": sns.color_palette("colorblind", n_colors=3)[0]}
    fixed_color = "0.25"

    perf = compute_final_values(df, metric)
    comp = compute_final_values(df, COMPUTE_METRIC)
    merged = perf.merge(comp[["run_id", "value"]], on="run_id", suffixes=("_perf", "_compute"))

    fixed = merged[merged["min_steps"] == merged["max_steps"]]
    fixed_perf_means = {}
    if not fixed.empty:
        for i, (mn, g) in enumerate(sorted(fixed.groupby("min_steps"))):
            perf_mean, perf_sem, comp_mean, comp_sem, n = _group_stats(g, trim)
            fixed_perf_means[mn] = perf_mean
            ax.errorbar(
                [comp_mean],
                [perf_mean],
                xerr=[comp_sem],
                yerr=[perf_sem],
                marker="o",
                markersize=6,
                color=fixed_color,
                linestyle="none",
                capsize=3,
                label=UNIFORM_LABEL if i == 0 else None,
            )

    adaptive = merged[merged["min_steps"] != merged["max_steps"]]
    adaptive_perf_means = {}
    for qac_variant, g in adaptive.groupby("qac_variant"):
        perf_mean, perf_sem, comp_mean, comp_sem, n = _group_stats(g, trim)
        adaptive_perf_means[qac_variant] = perf_mean
        ax.errorbar(
            [comp_mean],
            [perf_mean],
            xerr=[comp_sem],
            yerr=[perf_sem],
            marker=qac_markers.get(qac_variant, "*"),
            markersize=9,
            color=qac_colors.get(qac_variant, "0.5"),
            linestyle="none",
            capsize=3,
            label=ADAPTIVE_LABEL.get(qac_variant, qac_variant),
        )

    if panel_label == "Discounted return" and fixed_perf_means and adaptive_perf_means:
        adaptive_perf_mean = adaptive_perf_means.get("reinforce", next(iter(adaptive_perf_means.values())))
        best_uniform_perf = max(fixed_perf_means.values())
        adaptive_color = qac_colors.get("reinforce", "0.5")
        ax.axhline(adaptive_perf_mean, color=adaptive_color, linestyle="--", linewidth=1.2, alpha=0.8)
        ax.axhline(best_uniform_perf, color=fixed_color, linestyle="--", linewidth=1.2, alpha=0.8)
        diff = adaptive_perf_mean - best_uniform_perf
        trans = blended_transform_factory(ax.transAxes, ax.transData)
        ax.text(
            0.94,
            (adaptive_perf_mean + best_uniform_perf) / 2,
            f"$\\Delta$ = {diff:+.3f}",
            transform=trans,
            ha="right",
            va="center",
            fontsize=FS["tick"],
            clip_on=False,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.6", alpha=0.9),
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.tick_params(labelsize=FS["tick"])
    ax.set_title(panel_label, fontsize=FS["panel_title"])
    if show_ylabel:
        ax.set_ylabel("Final performance", fontsize=FS["label"])


def plot_pareto(
    df: pd.DataFrame, metrics: "list[tuple[str, str]]", output_path: Path, out_dir: Path, title: str, trim: int = 0
) -> None:
    """1x2 figure: one panel per metric in `metrics` (episode return,
    discounted return), sharing one legend/title. `trim` - see _draw_panel."""
    if os.path.abspath(out_dir).startswith("/Users"):
        plt.rcParams.update(pgf_with_latex)

    figsize = set_size(doc_width_pt, fraction=1.8, subplots=(1, len(metrics)), use_golden_ratio=True)
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize, squeeze=False)
    axes = axes[0]

    for col, (metric, panel_label) in enumerate(metrics):
        _draw_panel(axes[col], df, metric, panel_label, show_ylabel=(col == 0), trim=trim)

    fig.supxlabel("Compute steps $c$", fontsize=FS["label"])

    by_label: dict = {}
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            by_label.setdefault(ll, hh)
    place_legend_and_title_tight(
        fig, list(by_label.values()), list(by_label.keys()), min(len(by_label), 2), figsize, title
    )
    fig.tight_layout()
    fig.savefig(output_path, format="pdf", bbox_inches="tight", dpi=600)
    print(f"Saved {output_path}")
    plt.close(fig)


def plot_pareto_square(
    df: pd.DataFrame, metric: str, panel_label: str, output_path: Path, out_dir: Path, title: str, trim: int = 0
) -> None:
    """Single-panel, square-aspect figure for one metric (used for the
    discounted-return-only plots). `trim` - see _draw_panel."""
    if os.path.abspath(out_dir).startswith("/Users"):
        plt.rcParams.update(pgf_with_latex)

    side = set_size(doc_width_pt, fraction=0.5, subplots=(1, 1), use_golden_ratio=False)[0]
    fig, ax = plt.subplots(figsize=(side, side))

    _draw_panel(ax, df, metric, panel_label, show_ylabel=True, trim=trim)
    ax.set_xlabel("Compute steps $c$", fontsize=FS["label"])

    handles, labels = ax.get_legend_handles_labels()
    place_legend_and_title_tight(fig, handles, labels, min(len(labels), 2), (side, side), title)
    fig.tight_layout()
    fig.savefig(output_path, format="pdf", bbox_inches="tight", dpi=600)
    print(f"Saved {output_path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", type=Path, help="CSV produced by fetch_wandb_ppo_only_unshared_iru.py")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("ramdp_experiments/wandb_plots_ppo_only_unshared_iru")
    )
    parser.add_argument(
        "--trim",
        type=int,
        default=0,
        help="Drop this many highest- and lowest-performing seeds from each budget/variant group before "
        "averaging (trimmed mean/SEM), e.g. --trim 1 drops 1 top + 1 bottom seed out of 10. 0 (default) "
        "keeps the plain mean/SEM over all seeds.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df = df[df["state"] == "finished"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_trim{args.trim}" if args.trim > 0 else ""

    for spec in PARETO_FIGURES:
        plot_pareto(
            df,
            spec["metrics"],
            args.output_dir / f"lightsout_ppo_only_unshared_iru_pareto_{spec['fname']}{suffix}.pdf",
            args.output_dir,
            title=f"Unshared IRU-ACT (PPO), {spec['prefix']}: performance vs. compute",
            trim=args.trim,
        )

    for spec in SQUARE_FIGURES:
        plot_pareto_square(
            df,
            spec["metric"],
            "Discounted return",
            args.output_dir / f"lightsout_ppo_only_unshared_iru_pareto_{spec['fname']}{suffix}.pdf",
            args.output_dir,
            title=f"Unshared IRU-ACT (PPO), {spec['prefix']}: discounted return vs. compute",
            trim=args.trim,
        )


if __name__ == "__main__":
    main()
