#!/usr/bin/env python
"""Additional/alternative visualizations for the lightsout-sep7 wandb data
(hidden_dim=64 only) - complements plot_wandb_lightsout_hd64.py rather than
replacing it. Reads the same CSV cache (see fetch_wandb_lightsout_hd64.py)
and reuses that module's filtering, style, and helper functions.

Produces, per architecture:
  1. lightsout_hd64_<arch>_pareto.pdf - final compute (ponder steps) vs.
     final performance scatter: fixed-budget points plus adaptive-budget
     points, so the compute/performance trade-off is readable in one panel
     instead of cross-referencing separate figures. For
     Transformer-ExplicitCoT, which has a vocab_size axis, this is one file
     per vocab_size instead (lightsout_hd64_transformer-explicitcot_vocab
     <N>_pareto.pdf), including vocab_size=1 - each vocab_size now has its
     own budget=1 run, so there's no shared/borrowed point to fold in.
  2. lightsout_hd64_<arch>_seed_variance.pdf - per-seed strip plot + mean/CI
     point, budget on x, instead of collapsing straight to mean with a 95% CI;
     shows whether spread is genuine or one outlier seed.
  3. lightsout_hd64_<arch>_heatmap.pdf - (variant x budget) heatmap of final
     performance; a compact grid instead of many line/row panels.
  4. lightsout_hd64_<arch>_steps_to_threshold.pdf - timesteps needed to
     reach 80% of each run's own final performance, as a bar chart; isolates
     learning *speed* from final performance.
  5. lightsout_hd64_<arch>_compute_spaghetti.pdf - per-seed compute-time
     trajectories (adaptive-budget configs only) as individual thin lines
     instead of a mean with a 95% CI band, to see whether halting behavior is
     consistent across seeds or not.
  6. lightsout_hd64_pareto_{actor,evaluator}_{episode,discounted}_return.pdf
     - four combined 1x7 figures (paper-sized fonts), one per actor/evaluator
     x episode/discounted-return metric, each with one column per the same
     (architecture, vocab_size) grouping as the per-file pareto plots above
     (IRU-ACT, Transformer-CoT, Transformer-ExplicitCoT x vocab={1,2,4,8,16}).
     The actor figures share a fixed y-axis of [0.5, 1.0] so performance is
     directly comparable across architectures; all four share fixed x-ticks
     at [1, 2, 3, 4, 5].
  7. lightsout_hd64_learning_curve_{actor,evaluator}_{episode,discounted}
     _return.pdf - four more combined 1x7 figures, same 7-column (arch,
     vocab_size) grouping, but plotting the full training curve (metric vs.
     timesteps) instead of a single final-performance point: every fixed
     budget (grayscale) and every adaptive-budget qac_variant (colored) is
     overlaid in the same panel, so one figure shows the whole compute/
     algorithm sweep's training dynamics per architecture.

Pass --trim N to the pareto plots (plot_pareto, plot_pareto_row) to drop the
N highest- and N lowest-performing seeds from each budget/qac_variant group
before averaging (a trimmed mean/CI), so one or two outlier seeds don't
dominate a group of only ~5.

Usage:
  python ramdp_experiments/plot_wandb_lightsout_hd64_extra.py \\
      ramdp_experiments/wandb_cache_hd64.csv --output-dir ramdp_experiments/wandb_plots_extra [--trim N]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from matplotlib.ticker import FormatStrFormatter

import plot_wandb_lightsout_hd64 as base

last_k_eval = 3
plt = base.plt
sns = base.sns
pd = base.pd
np = base.np

set_size = base.set_size
pgf_with_latex = base.pgf_with_latex
place_legend_and_title = base.place_legend_and_title
budget_label = base.budget_label
variant_row_label = base.variant_row_label
compute_final_values = base.compute_final_values
resolve_ent_coef = base.resolve_ent_coef
mean_ci_curve = base.mean_ci_curve
step_axis = base.step_axis
CI95_Z = base.CI95_Z
METRICS = base.METRICS
METRIC_LABELS = base.METRIC_LABELS
ARCH_ORDER = base.ARCH_ORDER
arch_label = base.arch_label
VOCAB_SIZE_NA = base.VOCAB_SIZE_NA
doc_width_pt = base.doc_width_pt


def variant_rows_for_arch(sub_arch: pd.DataFrame):
    """(qac_variant, vocab_size) combos present, ordered like the main
    script's rows (reinforce first, then by vocab_size, then variant name)."""
    return sorted(
        sub_arch[["qac_variant", "vocab_size"]].drop_duplicates().itertuples(index=False, name=None),
        key=lambda k: (k[0] != "reinforce", k[1], k[0]),
    )


def all_variant_vocab_palette(df: pd.DataFrame) -> dict:
    """Consistent color per (qac_variant, vocab_size) row label, shared
    across every plot in this file (keyed by the label string, for direct
    use as a seaborn `palette=` dict)."""
    variants = sorted(
        df[["qac_variant", "vocab_size"]].drop_duplicates().itertuples(index=False, name=None),
        key=lambda k: (k[0] != "reinforce", k[1], k[0]),
    )
    colors = sns.color_palette("colorblind", n_colors=len(variants))
    return {variant_row_label(qv, False, vs): c for (qv, vs), c in zip(variants, colors)}


def build_labeled_final_df(sub_arch: pd.DataFrame, metric: str, row_keys) -> pd.DataFrame:
    """compute_final_values, called separately per (qac_variant, vocab_size)
    row and concatenated, so each row gets its own variant_label/budget_label
    columns attached directly rather than re-derived after the fact."""
    frames = []
    for qac_variant, vocab_size in row_keys:
        row_sub = sub_arch[(sub_arch["qac_variant"] == qac_variant) & (sub_arch["vocab_size"] == vocab_size)]
        if row_sub.empty:
            continue
        final = compute_final_values(row_sub, metric)
        final["variant_label"] = variant_row_label(qac_variant, False, vocab_size)
        final["budget_label"] = [budget_label(mn, mx) for mn, mx in zip(final["min_steps"], final["max_steps"])]
        frames.append(final)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def determine_primary_vocab(sub_arch: pd.DataFrame) -> int:
    """The vocab_size with the widest fixed-budget coverage for this arch
    (the "main" sweep) - sentinel VOCAB_SIZE_NA for archs without a vocab
    axis, where this is a no-op."""
    fixed = sub_arch[sub_arch["min_steps"] == sub_arch["max_steps"]]
    if fixed.empty:
        return VOCAB_SIZE_NA
    counts = fixed.groupby("vocab_size")["min_steps"].nunique()
    return counts.idxmax()


def _trim_by_performance(g: pd.DataFrame, trim: int) -> pd.DataFrame:
    """Drops the `trim` lowest- and `trim` highest-value_perf seeds from a
    (budget or qac_variant) group, for a trimmed mean/CI robust to one or
    two outlier seeds. No-op if there aren't enough seeds left over (need
    more than 2*trim to begin with)."""
    if trim <= 0 or len(g) <= 2 * trim:
        return g
    return g.sort_values("value_perf").iloc[trim : len(g) - trim]


def _group_stats(g: pd.DataFrame, trim: int) -> "tuple[float, float, float, float, int]":
    """(perf_mean, perf_ci, comp_mean, comp_ci, n) for one group, after
    trimming (see _trim_by_performance). perf_ci/comp_ci are normal-
    approximation 95% CI half-widths (CI95_Z * SEM)."""
    g = _trim_by_performance(g, trim)
    n = len(g)
    perf_mean = g["value_perf"].mean()
    perf_ci = CI95_Z * g["value_perf"].std() / np.sqrt(n) if n > 1 else 0.0
    comp_mean = g["value_compute"].mean()
    comp_ci = CI95_Z * g["value_compute"].std() / np.sqrt(n) if n > 1 else 0.0
    return perf_mean, perf_ci, comp_mean, comp_ci, n


def plot_pareto(
    df: pd.DataFrame, arch: str, output_path: Path, out_dir: Path, vocab_size: int | None = None, trim: int = 0
) -> None:
    """Final compute (ponder steps) vs. final performance for one
    architecture (and, for Transformer-ExplicitCoT, one vocab_size). Fixed-
    budget points are plain markers (no connecting line - budget isn't an
    ordered path through compute/performance space, so a line between them
    implies a trend that isn't really there); adaptive-budget points (one
    per qac_variant) are separate markers.

    `vocab_size`, when given, restricts to that vocab_size (the caller loops
    over every vocab_size present, including vocab_size=1, and makes one
    file each - every vocab_size now has its own real budget=1 run). When
    omitted (non-eCoT architectures, which have no vocab_size axis), falls
    back to the "primary" vocab_size (the sentinel, a no-op for those archs).

    `trim` drops the `trim` highest- and lowest-performing seeds from each
    budget/qac_variant group before averaging (see _trim_by_performance).

    One panel per PARETO_ROWS entry (actor episode/discounted return, then
    evaluator episode/discounted return), each against its own prefix's
    compute metric (actor panels vs. actor/compute_time/mean, evaluator
    panels vs. evaluator/compute_time/mean)."""
    panels = PARETO_ROWS
    qac_markers = {"reinforce": "D", "cond_fac": "s", "cond_naive": "^"}
    qac_colors = dict(zip(["reinforce", "cond_fac", "cond_naive"], sns.color_palette("colorblind", n_colors=3)))
    fixed_color = "0.25"

    sub_arch = df[df["arch"] == arch]
    if vocab_size is None:
        vocab_size = determine_primary_vocab(sub_arch)
    arch_sub = sub_arch[sub_arch["vocab_size"] == vocab_size]

    if os.path.abspath(out_dir).startswith("/Users"):
        plt.rcParams.update(pgf_with_latex)

    figsize = set_size(doc_width_pt, fraction=0.9 * len(panels), subplots=(1, len(panels)), use_golden_ratio=True)
    fig, axes = plt.subplots(1, len(panels), figsize=figsize, squeeze=False)
    axes = axes[0]

    for col, spec in enumerate(panels):
        ax = axes[col]
        metric = spec["metric"]
        compute_metric = f"{spec['prefix']}/compute_time/mean"
        perf = compute_final_values(arch_sub, metric)
        comp = compute_final_values(arch_sub, compute_metric)
        merged = perf.merge(comp[["run_id", "value"]], on="run_id", suffixes=("_perf", "_compute"))

        fixed = merged[merged["min_steps"] == merged["max_steps"]]
        if not fixed.empty:
            for i, (mn, g) in enumerate(sorted(fixed.groupby("min_steps"))):
                perf_mean, perf_ci, comp_mean, comp_ci, n = _group_stats(g, trim)
                ax.errorbar(
                    [comp_mean],
                    [perf_mean],
                    xerr=[comp_ci],
                    yerr=[perf_ci],
                    marker="o",
                    color=fixed_color,
                    linestyle="none",
                    capsize=2,
                    label="Uniform budget" if i == 0 else None,
                )

        adaptive = merged[merged["min_steps"] != merged["max_steps"]]
        for qac_variant, g in adaptive.groupby("qac_variant"):
            perf_mean, perf_ci, comp_mean, comp_ci, n = _group_stats(g, trim)
            ax.errorbar(
                [comp_mean],
                [perf_mean],
                xerr=[comp_ci],
                yerr=[perf_ci],
                marker=qac_markers.get(qac_variant, "*"),
                markersize=7,
                color=qac_colors.get(qac_variant, "0.5"),
                linestyle="none",
                capsize=2,
                label=variant_row_label(qac_variant, False),
            )

        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.set_title(f"{spec['prefix'].capitalize()}: {spec['label']}", fontsize=9)
        if col == 0:
            ax.set_ylabel("Final performance\n(mean of last {} evals, 95% CI)".format(last_k_eval), fontsize=8)

    fig.supxlabel("Final mean compute steps $c$")
    title = f"{arch_label(arch)}: Compute vs. Performance"
    if vocab_size != VOCAB_SIZE_NA:
        title += f", vocab={vocab_size}"

    by_label: dict = {}
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            by_label.setdefault(ll, hh)
    place_legend_and_title(
        fig,
        list(by_label.values()),
        list(by_label.keys()),
        min(len(by_label), 4),
        figsize,
        title,
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=600)
    print(f"Saved {output_path}")
    plt.close(fig)


PARETO_ROWS = [
    dict(prefix="actor", metric="actor/episode_return/mean", label="Episode return",
         ylim=(0.5, 1.0), fname="actor_episode_return"),
    dict(prefix="actor", metric="actor/episode_discounted_return/mean", label="Discounted return",
         ylim=(0.5, 1.0), fname="actor_discounted_return"),
    dict(prefix="evaluator", metric="evaluator/episode_return/mean", label="Episode return",
         ylim=None, fname="evaluator_episode_return"),
    dict(prefix="evaluator", metric="evaluator/episode_discounted_return/mean", label="Discounted return",
         ylim=None, fname="evaluator_discounted_return"),
]

PARETO_GRID_XTICKS = [1, 2, 3, 4, 5]
PARETO_GRID_FONTSIZES = dict(title=20, col_title=18, row_label=16, tick=12, xlabel=16, legend=14)


def pareto_columns(df: pd.DataFrame):
    """Ordered (arch, vocab_size, column_title) triples: one column per
    architecture, expanded into one column per vocab_size for architectures
    with a vocab_size axis (Transformer-ExplicitCoT) - the same grouping
    plot_pareto uses to produce one file per (arch, vocab_size)."""
    columns = []
    for arch in ARCH_ORDER:
        if arch not in df["arch"].unique():
            continue
        sub_arch = df[df["arch"] == arch]
        vocab_values = sorted(v for v in sub_arch["vocab_size"].unique() if v != VOCAB_SIZE_NA)
        if vocab_values:
            for vocab_size in vocab_values:
                columns.append(
                    (arch, vocab_size, f"{arch_label(arch)}\n$\\vert \\mathcal{{V}} \\vert = {vocab_size}$")
                )
        else:
            columns.append((arch, VOCAB_SIZE_NA, arch_label(arch)))
    return columns


def plot_pareto_row(
    df: pd.DataFrame,
    metric: str,
    compute_metric: str,
    row_label: str,
    output_path: Path,
    out_dir: Path,
    title: str,
    ylim: "tuple[float, float] | None" = None,
    trim: int = 0,
) -> None:
    """Single combined figure, one row: one column per (architecture,
    vocab_size) - the same grouping as the per-file plot_pareto figures -
    for one actor/evaluator return metric, so that metric's compute/
    performance trade-off is visible across every architecture at a glance.
    `ylim`, when given, is applied to every panel so performance is directly
    comparable architecture-to-architecture; x-ticks are fixed at
    PARETO_GRID_XTICKS regardless of each panel's data range, for a
    consistent budget axis across panels. `trim` drops the `trim` highest-
    and lowest-performing seeds from each budget/qac_variant group before
    averaging (see _trim_by_performance)."""
    columns = pareto_columns(df)
    n_cols = len(columns)
    qac_markers = {"reinforce": "D", "cond_fac": "s", "cond_naive": "^"}
    qac_colors = dict(zip(["reinforce", "cond_fac", "cond_naive"], sns.color_palette("colorblind", n_colors=3)))
    fixed_color = "0.25"
    fs = PARETO_GRID_FONTSIZES

    if os.path.abspath(out_dir).startswith("/Users"):
        plt.rcParams.update(pgf_with_latex)

    figsize = set_size(doc_width_pt, fraction=2.6, subplots=(1, n_cols), use_golden_ratio=False)
    fig, axes = plt.subplots(1, n_cols, figsize=figsize, squeeze=False)
    axes = axes[0]
    all_y: list = []

    for col, (arch, vocab_size, col_title) in enumerate(columns):
        ax = axes[col]
        arch_sub = df[(df["arch"] == arch) & (df["vocab_size"] == vocab_size)]
        perf = compute_final_values(arch_sub, metric)
        comp = compute_final_values(arch_sub, compute_metric)
        merged = perf.merge(comp[["run_id", "value"]], on="run_id", suffixes=("_perf", "_compute"))

        fixed = merged[merged["min_steps"] == merged["max_steps"]]
        if not fixed.empty:
            for i, (mn, g) in enumerate(sorted(fixed.groupby("min_steps"))):
                perf_mean, perf_ci, comp_mean, comp_ci, n = _group_stats(g, trim)
                all_y += [perf_mean - perf_ci, perf_mean + perf_ci]
                ax.errorbar(
                    [comp_mean],
                    [perf_mean],
                    xerr=[comp_ci],
                    yerr=[perf_ci],
                    marker="o",
                    markersize=6,
                    color=fixed_color,
                    linestyle="none",
                    capsize=3,
                    label="Uniform budget" if i == 0 else None,
                )

        adaptive = merged[merged["min_steps"] != merged["max_steps"]]
        for qac_variant, g in adaptive.groupby("qac_variant"):
            perf_mean, perf_ci, comp_mean, comp_ci, n = _group_stats(g, trim)
            all_y += [perf_mean - perf_ci, perf_mean + perf_ci]
            ax.errorbar(
                [comp_mean],
                [perf_mean],
                xerr=[comp_ci],
                yerr=[perf_ci],
                marker=qac_markers.get(qac_variant, "*"),
                markersize=9,
                color=qac_colors.get(qac_variant, "0.5"),
                linestyle="none",
                capsize=3,
                label=variant_row_label(qac_variant, False),
            )

        ax.grid(True, alpha=0.3)
        ax.set_xticks(PARETO_GRID_XTICKS)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.tick_params(labelsize=fs["tick"])
        ax.set_title(col_title, fontsize=fs["col_title"])
        if col == 0:
            ax.set_ylabel(row_label, fontsize=fs["row_label"])

    # Same y-range on every column: `ylim` if the caller fixed one (e.g. the
    # actor rows' [0.5, 1.0]), otherwise the min/max of every point plotted
    # above (with 5% padding) so columns stay comparable even when no fixed
    # range was given (e.g. the evaluator rows).
    if ylim is not None:
        y_lo, y_hi = ylim
    elif all_y:
        y_lo, y_hi = min(all_y), max(all_y)
        pad = 0.05 * (y_hi - y_lo) if y_hi > y_lo else 0.05
        y_lo, y_hi = y_lo - pad, y_hi + pad
    else:
        y_lo = y_hi = None
    if y_lo is not None:
        for ax in axes:
            ax.set_ylim(y_lo, y_hi)

    fig.subplots_adjust(bottom=0.55 / figsize[1])
    max_title_lines = max(col_title.count("\n") + 1 for _, _, col_title in columns)
    fig.subplots_adjust(top=1 - (0.05 + 0.25 * max_title_lines) / figsize[1])
    fig.supxlabel("Compute steps $c$", fontsize=fs["xlabel"])
    by_label: dict = {}
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            by_label.setdefault(ll, hh)

    ncols = min(len(by_label), 4)
    legend_rows = -(-len(by_label) // ncols)
    legend_top_frac = 1.0 + (0.05 + 0.30 * legend_rows) / figsize[1]
    title_y = legend_top_frac + 0.30 / figsize[1]
    fig.suptitle(title, y=title_y, fontsize=fs["title"])
    fig.legend(
        list(by_label.values()),
        list(by_label.keys()),
        bbox_to_anchor=(0.0, 1.0, 1.0, 0.0),
        loc="lower center",
        ncols=ncols,
        borderaxespad=0.0,
        frameon=True,
        fontsize=fs["legend"],
    )
    fig.savefig(output_path, format="pdf", bbox_inches="tight", dpi=600)
    print(f"Saved {output_path}")
    plt.close(fig)


QAC_VARIANT_ORDER = ["reinforce", "cond_fac", "cond_naive"]

LEARNING_CURVE_ROWS = [
    dict(prefix="actor", metric="actor/episode_return/mean", label="Episode return", fname="actor_episode_return"),
    dict(prefix="actor", metric="actor/episode_discounted_return/mean", label="Discounted return",
         fname="actor_discounted_return"),
    dict(prefix="evaluator", metric="evaluator/episode_return/mean", label="Episode return",
         fname="evaluator_episode_return"),
    dict(prefix="evaluator", metric="evaluator/episode_discounted_return/mean", label="Discounted return",
         fname="evaluator_discounted_return"),
]


def learning_curve_palette(df: pd.DataFrame):
    """Grayscale (light->dark, more compute = darker) per fixed budget, plus
    the same 3 colorblind colors used for qac_variant everywhere else in
    this file - so fixed- and adaptive-budget curves read as two distinct
    families instead of competing for hues in one panel."""
    fixed_budgets = sorted(df.loc[df["min_steps"] == df["max_steps"], "min_steps"].unique())
    grays = plt.cm.Greys(np.linspace(0.35, 0.85, len(fixed_budgets))) if fixed_budgets else []
    fixed_colors = dict(zip(fixed_budgets, grays))
    qac_colors = dict(zip(QAC_VARIANT_ORDER, sns.color_palette("colorblind", n_colors=len(QAC_VARIANT_ORDER))))
    return fixed_colors, qac_colors


def plot_learning_curves(
    df: pd.DataFrame,
    metric: str,
    row_label: str,
    output_path: Path,
    out_dir: Path,
    title: str,
) -> None:
    """Single combined figure, one row: one subplot per (architecture,
    vocab_size) - the same 7-column grouping as plot_pareto_row - with every
    fixed-budget curve (grayscale) and every adaptive-budget qac_variant
    curve (PPO/Factorized/Separated, each its own colorblind color)
    overlaid in the same panel, mean with a 95% CI across seeds. Unlike the main
    script's plot_arch (one row per qac_variant), this puts the full
    compute/algorithm sweep for one architecture in a single panel. Every
    column shares the same y-range (min/max mean+-CI seen across all
    columns, with 5% padding) so performance is comparable panel-to-panel."""
    columns = pareto_columns(df)
    n_cols = len(columns)
    fixed_colors, qac_colors = learning_curve_palette(df)
    fs = PARETO_GRID_FONTSIZES

    if os.path.abspath(out_dir).startswith("/Users"):
        plt.rcParams.update(pgf_with_latex)

    figsize = set_size(doc_width_pt, fraction=2.6, subplots=(1, n_cols), use_golden_ratio=False)
    fig, axes = plt.subplots(1, n_cols, figsize=figsize, squeeze=False)
    axes = axes[0]
    all_y: list = []

    for col, (arch, vocab_size, col_title) in enumerate(columns):
        ax = axes[col]
        col_df = df[(df["arch"] == arch) & (df["vocab_size"] == vocab_size)]

        fixed_budgets = sorted(col_df.loc[col_df["min_steps"] == col_df["max_steps"], "min_steps"].unique())
        for mn in fixed_budgets:
            b_df = col_df[(col_df["min_steps"] == mn) & (col_df["max_steps"] == mn)]
            result = mean_ci_curve(b_df, metric)
            if result is None:
                continue
            eval_idx, mean, ci = result
            steps = step_axis(b_df).reindex(eval_idx).to_numpy()
            color = fixed_colors[mn]
            ax.plot(steps, mean, color=color, linewidth=1.3, label=f"Uniform budget={mn}")
            ax.fill_between(steps, mean - ci, mean + ci, color=color, alpha=0.15)
            all_y += [float(np.min(mean - ci)), float(np.max(mean + ci))]

        adaptive_df = col_df[col_df["min_steps"] != col_df["max_steps"]]
        for qac_variant in QAC_VARIANT_ORDER:
            v_df = adaptive_df[adaptive_df["qac_variant"] == qac_variant]
            result = mean_ci_curve(v_df, metric)
            if result is None:
                continue
            eval_idx, mean, ci = result
            steps = step_axis(v_df).reindex(eval_idx).to_numpy()
            color = qac_colors[qac_variant]
            ax.plot(
                steps, mean, color=color, linewidth=1.8,
                label=f"Adaptive {variant_row_label(qac_variant, False)}",
            )
            ax.fill_between(steps, mean - ci, mean + ci, color=color, alpha=0.15)
            all_y += [float(np.min(mean - ci)), float(np.max(mean + ci))]

        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=fs["tick"])
        ax.set_title(col_title, fontsize=fs["col_title"])
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        if col == 0:
            ax.set_ylabel(row_label, fontsize=fs["row_label"])

    if all_y:
        y_lo, y_hi = min(all_y), max(all_y)
        pad = 0.05 * (y_hi - y_lo) if y_hi > y_lo else 0.05
        for ax in axes:
            ax.set_ylim(y_lo - pad, y_hi + pad)

    fig.subplots_adjust(bottom=0.55 / figsize[1])
    max_title_lines = max(col_title.count("\n") + 1 for _, _, col_title in columns)
    fig.subplots_adjust(top=1 - (0.05 + 0.25 * max_title_lines) / figsize[1])
    fig.supxlabel("Timesteps", fontsize=fs["xlabel"])

    by_label: dict = {}
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            by_label.setdefault(ll, hh)

    ncols = min(len(by_label), 4)
    legend_rows = -(-len(by_label) // ncols)
    legend_top_frac = 1.0 + (0.05 + 0.30 * legend_rows) / figsize[1]
    title_y = legend_top_frac + 0.30 / figsize[1]
    fig.suptitle(title, y=title_y, fontsize=fs["title"])
    fig.legend(
        list(by_label.values()),
        list(by_label.keys()),
        bbox_to_anchor=(0.0, 1.0, 1.0, 0.0),
        loc="lower center",
        ncols=ncols,
        borderaxespad=0.0,
        frameon=True,
        fontsize=fs["legend"],
    )
    fig.savefig(output_path, format="pdf", bbox_inches="tight", dpi=600)
    print(f"Saved {output_path}")
    plt.close(fig)


def plot_seed_variance(
    df: pd.DataFrame, arch: str, variant_palette: dict, output_path: Path, out_dir: Path
) -> None:
    """Per-seed final performance as a strip plot (budget on x, one column
    per seed's dot) plus a mean +/- SE point marker, instead of the main
    script's mean with a 95% CI line - makes it possible to spot a single outlier
    seed driving a wide error bar."""
    sub_arch = df[df["arch"] == arch]
    return_metrics = METRICS[:2]
    row_keys = variant_rows_for_arch(sub_arch)
    hue_order = [variant_row_label(qv, False, vs) for qv, vs in row_keys]
    budgets_all = sorted(
        sub_arch[["min_steps", "max_steps"]].drop_duplicates().itertuples(index=False, name=None),
        key=lambda mm: (mm[0] != mm[1], mm[0], mm[1]),
    )
    cat_order = [budget_label(mn, mx) for mn, mx in budgets_all]

    if os.path.abspath(out_dir).startswith("/Users"):
        plt.rcParams.update(pgf_with_latex)

    figsize = set_size(doc_width_pt, fraction=1.8, subplots=(1, len(return_metrics)), use_golden_ratio=True)
    fig, axes = plt.subplots(1, len(return_metrics), figsize=figsize, squeeze=False)
    axes = axes[0]

    for col, metric in enumerate(return_metrics):
        ax = axes[col]
        labeled = build_labeled_final_df(sub_arch, metric, row_keys)
        if labeled.empty:
            continue
        dodge = len(hue_order) > 1
        sns.stripplot(
            data=labeled,
            x="budget_label",
            y="value",
            hue="variant_label",
            order=cat_order,
            hue_order=hue_order,
            palette=variant_palette,
            ax=ax,
            dodge=dodge,
            size=4,
            alpha=0.75,
            linewidth=0.3,
            edgecolor="white",
        )
        sns.pointplot(
            data=labeled,
            x="budget_label",
            y="value",
            hue="variant_label",
            order=cat_order,
            hue_order=hue_order,
            palette=variant_palette,
            ax=ax,
            dodge=0.4 if dodge else False,
            errorbar=("se", CI95_Z),
            markers="D",
            markersize=4,
            linestyle="none",
            capsize=0.1,
            err_kws={"linewidth": 1.2},
        )
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        ax.set_xlabel("Budget")
        ax.set_title(METRIC_LABELS[metric], fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        ax.tick_params(axis="x", rotation=45)
        if col == 0:
            ax.set_ylabel("Final performance\n(per-seed dots + mean, 95% CI)", fontsize=8)
        else:
            ax.set_ylabel("")

    handles = [plt.Line2D([0], [0], marker="o", linestyle="none", color=variant_palette[lbl]) for lbl in hue_order]
    place_legend_and_title(
        fig, handles, hue_order, min(len(hue_order), 3), figsize, f"{arch_label(arch)}: per-seed final performance"
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=600)
    print(f"Saved {output_path}")
    plt.close(fig)


def plot_heatmap(df: pd.DataFrame, arch: str, output_path: Path, out_dir: Path) -> None:
    """(qac_variant x budget) -> final performance heatmap, one subplot per
    return metric (columns - actor and evaluator episode/discounted return,
    per PARETO_ROWS) and, for architectures with a vocab_size axis
    (Transformer-ExplicitCoT), one subplot row per vocab_size too - a
    compact grid alternative to the many row/line panels in the main
    script's figures, without folding vocab_size into the row labels of a
    single heatmap."""
    sub_arch = df[df["arch"] == arch]
    panels = PARETO_ROWS
    row_keys = variant_rows_for_arch(sub_arch)
    vocab_values = sorted({vs for _, vs in row_keys}, key=lambda v: (v != VOCAB_SIZE_NA, v))
    qac_by_vocab = {
        vs: sorted({qv for qv, v2 in row_keys if v2 == vs}, key=lambda q: (q != "reinforce", q))
        for vs in vocab_values
    }
    budgets_all = sorted(
        sub_arch[["min_steps", "max_steps"]].drop_duplicates().itertuples(index=False, name=None),
        key=lambda mm: (mm[0] != mm[1], mm[0], mm[1]),
    )
    col_labels = [budget_label(mn, mx) for mn, mx in budgets_all]

    if os.path.abspath(out_dir).startswith("/Users"):
        plt.rcParams.update(pgf_with_latex)

    # Sized directly from the grid shape (cells ~square) rather than via
    # set_size's subplot-grid formula, which assumes stacked subplot rows
    # and badly stretches a wide-and-short annotated heatmap.
    n_vocab_rows = max(len(vocab_values), 1)
    n_cols = len(panels)
    max_qac_rows = max((len(v) for v in qac_by_vocab.values()), default=1)
    n_col_labels = max(len(col_labels), 1)
    cell_size = 0.5
    figsize = (
        n_col_labels * cell_size * n_cols + 1.5 * n_cols,
        max_qac_rows * cell_size * n_vocab_rows + 1.5 * n_vocab_rows,
    )
    fig, axes = plt.subplots(n_vocab_rows, n_cols, figsize=figsize, squeeze=False)

    for row, vocab_size in enumerate(vocab_values):
        row_labels = [variant_row_label(qv, False) for qv in qac_by_vocab[vocab_size]]
        row_row_keys = [(qv, vocab_size) for qv in qac_by_vocab[vocab_size]]
        for col, spec in enumerate(panels):
            metric = spec["metric"]
            col_title = f"{spec['prefix'].capitalize()}: {spec['label']}"
            ax = axes[row, col]
            labeled = build_labeled_final_df(sub_arch, metric, row_row_keys)
            # Row labels here are per-qac_variant only (vocab_size is fixed
            # for this subplot row), so re-derive them without the vocab
            # suffix that build_labeled_final_df's variant_label carries.
            pivot = pd.DataFrame(index=row_labels, columns=col_labels, dtype=float)
            if not labeled.empty:
                labeled = labeled.assign(
                    qac_row_label=[variant_row_label(qv, False) for qv in labeled["qac_variant"]]
                )
                grouped = labeled.groupby(["qac_row_label", "budget_label"])["value"].mean()
                for (rlabel, clabel), val in grouped.items():
                    if rlabel in pivot.index and clabel in pivot.columns:
                        pivot.loc[rlabel, clabel] = val
            sns.heatmap(
                pivot.astype(float),
                annot=True,
                fmt=".2f",
                cmap="viridis",
                ax=ax,
                cbar_kws={"label": col_title},
                linewidths=0.5,
                linecolor="white",
                annot_kws={"fontsize": 7},
            )
            if row == 0:
                ax.set_title(col_title, fontsize=9)
            if row == n_vocab_rows - 1:
                ax.set_xlabel("Budget")
            else:
                ax.set_xlabel("")
            ax.set_ylabel(f"vocab={vocab_size}" if vocab_size != VOCAB_SIZE_NA and col == 0 else "", fontsize=8)
            ax.tick_params(axis="x", rotation=45, labelsize=7)
            ax.tick_params(axis="y", rotation=0, labelsize=7)

    fig.suptitle(f"{arch_label(arch)}: final performance heatmap", y=1.0 + 0.03 * n_vocab_rows, fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=600)
    print(f"Saved {output_path}")
    plt.close(fig)


def compute_steps_to_threshold(sub_arch: pd.DataFrame, metric: str, row_keys, frac: float = 0.8) -> pd.DataFrame:
    """Per run, the first real timestep at which `metric` reaches `frac` of
    that run's own final value (mean of its last 3 evals). NaN if the run's
    final value is <= 0 (threshold undefined) or never reached (shouldn't
    happen given the definition, but eval noise can make the last-3-mean
    dip below points earlier in a noisy tail)."""
    records = []
    for qac_variant, vocab_size in row_keys:
        row_sub = sub_arch[(sub_arch["qac_variant"] == qac_variant) & (sub_arch["vocab_size"] == vocab_size)]
        if row_sub.empty:
            continue
        for run_id, g in row_sub.groupby("run_id"):
            g = g.sort_values("eval_idx")
            final = g[metric].tail(last_k_eval).mean()
            if pd.isna(final) or final <= 0:
                step = np.nan
            else:
                reached = g[g[metric] >= frac * final]
                step = reached["_step"].iloc[0] if not reached.empty else np.nan
            mn, mx = g["min_steps"].iloc[0], g["max_steps"].iloc[0]
            records.append(
                {
                    "run_id": run_id,
                    "seed": g["seed"].iloc[0],
                    "variant_label": variant_row_label(qac_variant, False, vocab_size),
                    "budget_label": budget_label(mn, mx),
                    "step": step,
                }
            )
    return pd.DataFrame(records)


def plot_steps_to_threshold(
    df: pd.DataFrame, arch: str, variant_palette: dict, output_path: Path, out_dir: Path, frac: float = 0.8
) -> None:
    """Timesteps needed to reach `frac` of each run's own final performance,
    as a bar chart - isolates learning *speed* from final performance,
    which the raw training curves make hard to compare when lines cross."""
    sub_arch = df[df["arch"] == arch]
    return_metrics = METRICS[:2]
    row_keys = variant_rows_for_arch(sub_arch)
    hue_order = [variant_row_label(qv, False, vs) for qv, vs in row_keys]
    budgets_all = sorted(
        sub_arch[["min_steps", "max_steps"]].drop_duplicates().itertuples(index=False, name=None),
        key=lambda mm: (mm[0] != mm[1], mm[0], mm[1]),
    )
    cat_order = [budget_label(mn, mx) for mn, mx in budgets_all]

    if os.path.abspath(out_dir).startswith("/Users"):
        plt.rcParams.update(pgf_with_latex)

    figsize = set_size(doc_width_pt, fraction=1.8, subplots=(1, len(return_metrics)), use_golden_ratio=True)
    fig, axes = plt.subplots(1, len(return_metrics), figsize=figsize, squeeze=False)
    axes = axes[0]

    for col, metric in enumerate(return_metrics):
        ax = axes[col]
        records = compute_steps_to_threshold(sub_arch, metric, row_keys, frac=frac)
        if records.empty:
            continue
        sns.barplot(
            data=records,
            x="budget_label",
            y="step",
            hue="variant_label",
            order=cat_order,
            hue_order=hue_order,
            palette=variant_palette,
            ax=ax,
            errorbar=("se", CI95_Z),
            capsize=0.08,
            err_kws={"linewidth": 1.0},
        )
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        ax.set_xlabel("Budget")
        ax.set_title(METRIC_LABELS[metric], fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        ax.tick_params(axis="x", rotation=45)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        if col == 0:
            ax.set_ylabel(f"Timesteps to {int(frac * 100)}% of final\n(mean, 95% CI across seeds)", fontsize=8)
        else:
            ax.set_ylabel("")

    handles = [plt.Line2D([0], [0], marker="s", linestyle="none", color=variant_palette[lbl]) for lbl in hue_order]
    place_legend_and_title(
        fig,
        handles,
        hue_order,
        min(len(hue_order), 3),
        figsize,
        f"{arch_label(arch)}: learning speed (steps to {int(frac * 100)}% of final performance)",
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=600)
    print(f"Saved {output_path}")
    plt.close(fig)


def plot_compute_spaghetti(df: pd.DataFrame, arch: str, output_path: Path, out_dir: Path) -> None:
    """Per-seed compute-time trajectories for adaptive-budget configs only,
    as individual thin lines rather than a mean with a 95% CI band - the main
    script's compute-time panels only show the aggregate, which can hide a
    seed that never learns to halt early."""
    sub_arch = df[df["arch"] == arch]
    adaptive = sub_arch[sub_arch["min_steps"] != sub_arch["max_steps"]]
    if adaptive.empty:
        print(f"  (skipping compute spaghetti for {arch}: no adaptive-budget runs)")
        return
    row_keys = sorted(
        adaptive[["qac_variant", "vocab_size"]].drop_duplicates().itertuples(index=False, name=None),
        key=lambda k: (k[0] != "reinforce", k[1], k[0]),
    )
    n_rows = len(row_keys)
    compute_metric = METRICS[2]

    if os.path.abspath(out_dir).startswith("/Users"):
        plt.rcParams.update(pgf_with_latex)

    figsize = set_size(doc_width_pt, fraction=1.1, subplots=(n_rows, 1), use_golden_ratio=True)
    fig, axes = plt.subplots(n_rows, 1, figsize=figsize, squeeze=False)

    seed_colors = sns.color_palette("colorblind", n_colors=5)

    for row, (qac_variant, vocab_size) in enumerate(row_keys):
        ax = axes[row, 0]
        row_df = adaptive[(adaptive["qac_variant"] == qac_variant) & (adaptive["vocab_size"] == vocab_size)]
        for seed, g in row_df.groupby("seed"):
            g = g.sort_values("eval_idx")
            ax.plot(
                g["_step"],
                g[compute_metric],
                color=seed_colors[int(seed) % 5],
                alpha=0.85,
                linewidth=1.0,
                label=f"seed {int(seed)}",
            )
        mn, mx = row_df[["min_steps", "max_steps"]].iloc[0]
        ax.set_ylabel(f"{variant_row_label(qac_variant, False, vocab_size)}\n\nCompute steps", fontsize=8)
        ax.grid(True, alpha=0.3)
        if row == 0:
            ax.set_title(f"Per-seed compute-time trajectories (adaptive[{mn}-{mx}])", fontsize=9)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

    fig.supxlabel("Timesteps")
    handles = [plt.Line2D([0], [0], color=seed_colors[s], label=f"seed {s}") for s in range(5)]
    place_legend_and_title(fig, handles, [f"seed {s}" for s in range(5)], 5, figsize, arch_label(arch))
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=600)
    print(f"Saved {output_path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", type=Path, help="CSV produced by fetch_wandb_lightsout_hd64.py")
    parser.add_argument("--output-dir", type=Path, default=Path("ramdp_experiments/wandb_plots_extra"))
    parser.add_argument(
        "--trim",
        type=int,
        default=0,
        help="For the pareto plots (plot_pareto, plot_pareto_row), drop this many highest- and "
        "lowest-performing seeds from each budget/qac_variant group before averaging (trimmed "
        "mean/CI). 0 (default) keeps the plain mean/CI over all seeds.",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=None,
        help="Keep only rows at this system.ent_coef, for CSVs fetched from a project that sweeps "
        "it (e.g. the eCoT sweep). Default: no fixed value - the best-performing ent_coef is picked "
        "automatically per (arch, qac_variant, vocab_size, budget) group (see select_best_ent_coef "
        "in the main script).",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    # Same filters as plot_wandb_lightsout_hd64.py's main(): finished runs
    # only (config identity dedup keeps a still-running run only when no
    # finished run with that exact config exists yet), and IRU-ACT has a
    # secondary 1e8-timestep sweep alongside the main 3e8 one.
    df = df[df["state"] == "finished"]
    df = df[~df["sgh"]]
    df = df[(df["arch"] != "IRU-ACT") | (df["total_timesteps"] == 300_000_000)]
    df = resolve_ent_coef(df, args.ent_coef)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_trim{args.trim}" if args.trim > 0 else ""

    variant_palette = all_variant_vocab_palette(df)

    for arch in ARCH_ORDER:
        if arch not in df["arch"].unique():
            continue
        safe_name = arch.lower().replace(" ", "_")
        sub_arch = df[df["arch"] == arch]
        vocab_values = sorted(v for v in sub_arch["vocab_size"].unique() if v != VOCAB_SIZE_NA)
        if vocab_values:
            for vocab_size in vocab_values:
                plot_pareto(
                    df,
                    arch,
                    args.output_dir / f"lightsout_hd64_{safe_name}_vocab{vocab_size}_pareto{suffix}.pdf",
                    args.output_dir,
                    vocab_size=vocab_size,
                    trim=args.trim,
                )
        else:
            plot_pareto(
                df,
                arch,
                args.output_dir / f"lightsout_hd64_{safe_name}_pareto{suffix}.pdf",
                args.output_dir,
                trim=args.trim,
            )
        plot_seed_variance(
            df, arch, variant_palette, args.output_dir / f"lightsout_hd64_{safe_name}_seed_variance.pdf", args.output_dir
        )
        plot_heatmap(df, arch, args.output_dir / f"lightsout_hd64_{safe_name}_heatmap.pdf", args.output_dir)
        plot_steps_to_threshold(
            df,
            arch,
            variant_palette,
            args.output_dir / f"lightsout_hd64_{safe_name}_steps_to_threshold.pdf",
            args.output_dir,
        )
        plot_compute_spaghetti(
            df, arch, args.output_dir / f"lightsout_hd64_{safe_name}_compute_spaghetti.pdf", args.output_dir
        )

    for spec in PARETO_ROWS:
        compute_metric = f"{spec['prefix']}/compute_time/mean"
        plot_pareto_row(
            df,
            spec["metric"],
            compute_metric,
            spec["label"],
            args.output_dir / f"lightsout_hd64_pareto_{spec['fname']}{suffix}.pdf",
            args.output_dir,
            title=f"{spec['label']} vs. compute (all architectures)",
            ylim=spec["ylim"],
            trim=args.trim,
        )

    for spec in LEARNING_CURVE_ROWS:
        plot_learning_curves(
            df,
            spec["metric"],
            spec["label"],
            args.output_dir / f"lightsout_hd64_learning_curve_{spec['fname']}.pdf",
            args.output_dir,
            title=f"{spec['label']} vs. timesteps (all budgets and variants, all architectures)",
        )


if __name__ == "__main__":
    main()
