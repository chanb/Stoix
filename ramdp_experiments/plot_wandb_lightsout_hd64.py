#!/usr/bin/env python
"""Plot actor/{episode_return,episode_discounted_return,compute_time} training
curves for the lightsout-sep7 wandb project, hidden_dim=64 only (see
fetch_wandb_lightsout_hd64.py, which produces the input CSV).

For each architecture, produces one figure: rows are (qac_variant, stop-
gradient-halting) settings actually swept for that architecture, columns are
the three metrics, and each line is one compute budget (mean +/- standard
error across 5 seeds). qac_variant only varies at the adaptive budget (min_steps=1,
max_steps=5) - fixed budgets (min_steps == max_steps) have a single
"reinforce" row since there's no halting decision to train.

Also produces one "headline" comparison figure overlaying all three
architectures at their default adaptive-budget setting (reinforce,
no stop-gradient) so they can be compared head-to-head.

Usage:
  python ramdp_experiments/plot_wandb_lightsout_hd64.py \\
      ramdp_experiments/wandb_cache_hd64.csv --output-dir ramdp_experiments/wandb_plots
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_palette("colorblind")
doc_width_pt = 452.9679

METRICS = [
    "actor/episode_return/mean",
    "actor/episode_discounted_return/mean",
    "actor/compute_time/mean",
]
METRIC_LABELS = {
    "actor/episode_return/mean": "Episode return",
    "actor/episode_discounted_return/mean": "Discounted return",
    "actor/compute_time/mean": "Mean compute (ponder) steps",
}
ARCH_ORDER = ["IRU-ACT", "Transformer-CoT", "Transformer-ExplicitCoT"]


def set_size(width_pt, fraction=1, subplots=(1, 1), use_golden_ratio=True):
    """
    Reference: https://jwalton.info/Matplotlib-latex-PGF/
    Set figure dimensions to sit nicely in our document.

    Parameters
    ----------
    width_pt: float
            Document width in points
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    fig_width_pt = width_pt * fraction
    inches_per_pt = 1 / 72.27

    fig_width_in = fig_width_pt * inches_per_pt
    if use_golden_ratio:
        golden_ratio = (5**0.5 - 1) / 2
        fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])
    else:
        fig_height_in = fig_width_in * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


pgf_with_latex = {
    "pgf.texsystem": "pdflatex",
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": [],
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 10,
    "font.size": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "pgf.rcfonts": False,
}


def place_legend_and_title(fig, handles, labels, ncols: int, figsize, title: str) -> None:
    """Places a top legend just above the axes, and a suptitle just above
    the legend - both offset by a fixed inch amount (converted to figure
    fraction via `figsize`) so the gap stays small regardless of how tall
    the overall figure is."""
    legend_rows = -(-len(labels) // ncols)  # ceil division
    legend_top_frac = 1.0 + (0.22 * legend_rows) / figsize[1]
    title_y = legend_top_frac + 0.28 / figsize[1]

    fig.legend(
        handles,
        labels,
        bbox_to_anchor=(0.0, 1.0, 1.0, 0.0),
        loc="lower center",
        ncols=ncols,
        borderaxespad=0.0,
        frameon=True,
        fontsize="8",
    )
    fig.suptitle(title, y=title_y, fontsize=12)


def budget_label(min_steps: int, max_steps: int) -> str:
    return f"fixed={min_steps}" if min_steps == max_steps else f"adaptive[{min_steps}-{max_steps}]"


VOCAB_SIZE_NA = -1  # sentinel: architecture has no vocab_size (only Transformer-ExplicitCoT does)


def variant_row_label(qac_variant: str, sgh: bool, vocab_size: int = VOCAB_SIZE_NA) -> str:
    label = {"reinforce": "REINFORCE", "cond_fac": "Cond. factorized", "cond_naive": "Cond. naive"}.get(
        qac_variant, qac_variant
    )
    if sgh:
        label = f"{label} + stop-grad halt"
    if vocab_size != VOCAB_SIZE_NA:
        label = f"{label}, vocab={vocab_size}"
    return label


def mean_sem_curve(sub: pd.DataFrame, metric: str):
    """sub has one row per (seed, eval_idx). Returns (eval_idx, mean, sem)
    aggregated across seeds, restricted to eval_idx present for every seed.
    sem = std / sqrt(n_seeds)."""
    pivot = sub.pivot_table(index="eval_idx", columns="seed", values=metric)
    pivot = pivot.dropna(how="any")
    if pivot.empty:
        return None
    x = pivot.index.to_numpy()
    mean = pivot.mean(axis=1).to_numpy()
    sem = pivot.std(axis=1).to_numpy() / np.sqrt(pivot.shape[1])
    return x, mean, sem


def step_axis(sub: pd.DataFrame):
    """Representative real-timestep x-axis (mean `_step` per eval_idx)."""
    return sub.groupby("eval_idx")["_step"].mean()


def expand_vocab_agnostic_budget1(df: pd.DataFrame) -> pd.DataFrame:
    """For Transformer-ExplicitCoT, budget=1 (min_steps == max_steps == 1)
    never emits any CoT tokens - only one step is ever taken - so
    vocab_size doesn't affect it. The vocab_size=1 budget=1 runs are
    therefore valid data for every other vocab_size too; duplicate them
    (relabeled to each other vocab_size present) and drop the now-redundant
    standalone vocab_size=1 grouping, since it would otherwise just repeat
    what's now folded into every other vocab_size row."""
    is_ecot = df["arch"] == "Transformer-ExplicitCoT"
    vocab1_budget1 = df[is_ecot & (df["vocab_size"] == 1) & (df["min_steps"] == 1) & (df["max_steps"] == 1)]
    if vocab1_budget1.empty:
        return df
    other_vocabs = sorted(v for v in df.loc[is_ecot, "vocab_size"].unique() if v not in (1, VOCAB_SIZE_NA))
    dup_frames = []
    for vocab_size in other_vocabs:
        dup = vocab1_budget1.copy()
        dup["vocab_size"] = vocab_size
        dup_frames.append(dup)
    df = df[~(is_ecot & (df["vocab_size"] == 1))]
    return pd.concat([df] + dup_frames, ignore_index=True) if dup_frames else df


def all_budgets_palette(df: pd.DataFrame):
    budgets = sorted(
        {budget_label(mn, mx) for mn, mx in df[["min_steps", "max_steps"]].drop_duplicates().itertuples(index=False)},
        key=lambda b: (b.startswith("adaptive"), b),
    )
    colors = sns.color_palette("colorblind", n_colors=len(budgets))
    return dict(zip(budgets, colors))


def all_variants_palette(df: pd.DataFrame):
    """Consistent color per (qac_variant, sgh) combo, shared across archs."""
    variants = sorted(
        df[["qac_variant", "sgh"]].drop_duplicates().itertuples(index=False, name=None),
        key=lambda k: (k[0] != "reinforce", k[1], k[0]),
    )
    colors = sns.color_palette("colorblind", n_colors=len(variants))
    return dict(zip(variants, colors))


def compute_final_values(df: pd.DataFrame, metric: str, n_tail: int = 3) -> pd.DataFrame:
    """Per-run final performance: mean of the last `n_tail` eval points."""
    tail = df.sort_values("eval_idx").groupby("run_id").tail(n_tail)
    return (
        tail.groupby("run_id")
        .agg(
            value=(metric, "mean"),
            arch=("arch", "first"),
            qac_variant=("qac_variant", "first"),
            min_steps=("min_steps", "first"),
            max_steps=("max_steps", "first"),
            sgh=("sgh", "first"),
            vocab_size=("vocab_size", "first"),
            seed=("seed", "first"),
        )
        .reset_index()
    )


def plot_budget_vs_performance(
    df: pd.DataFrame, arch: str, variant_colors: dict, output_path: Path, out_dir: Path
) -> None:
    """For one architecture: x-axis is the fixed compute budget, y-axis is
    final performance (mean of the last 3 evals, mean +/- standard error
    across 5 seeds); one panel per return metric. Adaptive-budget models (which don't
    have a single x position) are drawn as horizontal reference lines
    spanning the fixed-budget range instead. Architectures with a
    `vocab_size` axis (Transformer-ExplicitCoT) get one row per vocab_size,
    since a given fixed budget can otherwise map to more than one vocab_size
    (an explicit-CoT-only ablation) and mixing them together would be
    misleading."""
    sub = df[df["arch"] == arch]
    return_metrics = METRICS[:2]

    has_vocab = (sub["vocab_size"] != VOCAB_SIZE_NA).any()
    row_values = sorted(v for v in sub["vocab_size"].unique() if v != VOCAB_SIZE_NA) if has_vocab else [VOCAB_SIZE_NA]
    n_rows = len(row_values)
    n_cols = len(return_metrics)

    all_fixed_budgets = sorted(sub.loc[sub["min_steps"] == sub["max_steps"], "min_steps"].unique())

    if os.path.abspath(out_dir).startswith("/Users"):
        plt.rcParams.update(pgf_with_latex)

    figsize = set_size(doc_width_pt, fraction=1.6, subplots=(n_rows, n_cols), use_golden_ratio=False)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    for row, vocab_size in enumerate(row_values):
        row_sub = sub if not has_vocab else sub[sub["vocab_size"] == vocab_size]
        for col, metric in enumerate(return_metrics):
            ax = axes[row, col]
            final_df = compute_final_values(row_sub, metric)

            fixed = final_df[final_df["min_steps"] == final_df["max_steps"]]
            fixed_stats = fixed.groupby("min_steps")["value"].agg(["mean", "std", "count"]).sort_index()
            fixed_sem = fixed_stats["std"] / np.sqrt(fixed_stats["count"])
            ax.errorbar(
                fixed_stats.index,
                fixed_stats["mean"],
                yerr=fixed_sem,
                marker="o",
                capsize=3,
                color="0.25",
                linewidth=1.2,
                label="Fixed budget",
            )

            adaptive = final_df[final_df["min_steps"] != final_df["max_steps"]]
            for qac_variant, g in adaptive.groupby("qac_variant"):
                mean = g["value"].mean()
                sem = g["value"].std() / np.sqrt(len(g))
                color = variant_colors[(qac_variant, False)]
                mn, mx = g["min_steps"].iloc[0], g["max_steps"].iloc[0]
                label = f"{variant_row_label(qac_variant, False)} (adaptive[{mn}-{mx}])"
                ax.axhline(mean, color=color, linestyle="--", linewidth=1.3, label=label)
                ax.axhspan(mean - sem, mean + sem, color=color, alpha=0.12)

            ax.set_xticks(all_fixed_budgets)
            ax.set_xlim(min(all_fixed_budgets) - 0.5, max(all_fixed_budgets) + 0.5)
            ax.grid(True, alpha=0.3)
            if row == 0:
                ax.set_title(METRIC_LABELS[metric], fontsize=9)
            if row == n_rows - 1:
                ax.set_xlabel("Fixed compute budget")
            if col == 0:
                row_label = f"vocab={vocab_size}\n\n" if has_vocab else ""
                ax.set_ylabel(f"{row_label}Final performance\n(mean of last 3 evals, ± SEM)", fontsize=8)

    by_label: Dict[str, object] = {}
    for r in range(n_rows):
        for c in range(n_cols):
            h, l = axes[r, c].get_legend_handles_labels()
            for hh, ll in zip(h, l):
                by_label.setdefault(ll, hh)

    place_legend_and_title(
        fig,
        list(by_label.values()),
        list(by_label.keys()),
        min(len(by_label), 3),
        figsize,
        f"{arch}: fixed budget vs. final performance",
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    print(f"Saved {output_path}")
    plt.close(fig)


def plot_arch(df: pd.DataFrame, arch: str, budget_colors: dict, output_path: Path, out_dir: Path) -> None:
    sub_arch = df[df["arch"] == arch]
    row_keys = sorted(
        sub_arch[["qac_variant", "sgh", "vocab_size"]].drop_duplicates().itertuples(index=False, name=None),
        key=lambda k: (k[0] != "reinforce", k[1], k[2], k[0]),
    )
    n_rows = len(row_keys)
    n_cols = len(METRICS)

    if os.path.abspath(out_dir).startswith("/Users"):
        plt.rcParams.update(pgf_with_latex)

    figsize = set_size(doc_width_pt, fraction=2.2, subplots=(n_rows, n_cols), use_golden_ratio=False)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    for row, (qac_variant, sgh, vocab_size) in enumerate(row_keys):
        row_df = sub_arch[
            (sub_arch["qac_variant"] == qac_variant)
            & (sub_arch["sgh"] == sgh)
            & (sub_arch["vocab_size"] == vocab_size)
        ]
        budgets = sorted(
            row_df[["min_steps", "max_steps"]].drop_duplicates().itertuples(index=False, name=None),
            key=lambda mm: (mm[0] != mm[1], mm[0], mm[1]),
        )
        for col, metric in enumerate(METRICS):
            ax = axes[row, col]
            for mn, mx in budgets:
                b_label = budget_label(mn, mx)
                b_df = row_df[(row_df["min_steps"] == mn) & (row_df["max_steps"] == mx)]
                result = mean_sem_curve(b_df, metric)
                if result is None:
                    continue
                eval_idx, mean, sem = result
                steps = step_axis(b_df).reindex(eval_idx).to_numpy()
                color = budget_colors[b_label]
                ax.plot(steps, mean, label=b_label, color=color, linewidth=1.2)
                ax.fill_between(steps, mean - sem, mean + sem, color=color, alpha=0.15)
            ax.grid(True, alpha=0.3)
            if row == 0:
                ax.set_title(METRIC_LABELS[metric], fontsize=9)
            if row == n_rows - 1:
                ax.set_xlabel("Timesteps")
            if col == 0:
                ax.set_ylabel(variant_row_label(qac_variant, sgh, vocab_size), fontsize=8)
            ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

    by_label: Dict[str, object] = {}
    for r in range(n_rows):
        for c in range(n_cols):
            h, l = axes[r, c].get_legend_handles_labels()
            for hh, ll in zip(h, l):
                by_label.setdefault(ll, hh)
    place_legend_and_title(
        fig, list(by_label.values()), list(by_label.keys()), min(len(by_label), 4), figsize, arch
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    print(f"Saved {output_path}")
    plt.close(fig)


def plot_headline_comparison(df: pd.DataFrame, output_path: Path, out_dir: Path) -> None:
    """One row, one figure: all architectures overlaid at their default
    adaptive-budget setting (reinforce, no stop-gradient-halting)."""
    sub = df[(df["qac_variant"] == "reinforce") & (~df["sgh"]) & (df["min_steps"] == 1) & (df["max_steps"] == 5)]

    if os.path.abspath(out_dir).startswith("/Users"):
        plt.rcParams.update(pgf_with_latex)

    n_cols = len(METRICS)
    figsize = set_size(doc_width_pt, fraction=2.2, subplots=(1, n_cols), use_golden_ratio=False)
    fig, axes = plt.subplots(1, n_cols, figsize=figsize, squeeze=False)
    axes = axes[0]

    arch_colors = dict(zip(ARCH_ORDER, sns.color_palette("colorblind", n_colors=len(ARCH_ORDER))))

    for col, metric in enumerate(METRICS):
        ax = axes[col]
        for arch in ARCH_ORDER:
            arch_df = sub[sub["arch"] == arch]
            result = mean_sem_curve(arch_df, metric)
            if result is None:
                continue
            eval_idx, mean, sem = result
            steps = step_axis(arch_df).reindex(eval_idx).to_numpy()
            color = arch_colors[arch]
            ax.plot(steps, mean, label=arch, color=color, linewidth=1.4)
            ax.fill_between(steps, mean - sem, mean + sem, color=color, alpha=0.15)
        ax.set_title(METRIC_LABELS[metric], fontsize=9)
        ax.set_xlabel("Timesteps")
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

    handles, labels = axes[0].get_legend_handles_labels()
    place_legend_and_title(
        fig,
        handles,
        labels,
        len(labels),
        figsize,
        "Architecture comparison (adaptive budget, REINFORCE, no stop-grad halt)",
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    print(f"Saved {output_path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", type=Path, help="CSV produced by fetch_wandb_lightsout_hd64.py")
    parser.add_argument("--output-dir", type=Path, default=Path("ramdp_experiments/wandb_plots"))
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df = df[~df["sgh"]]
    # IRU-ACT has some runs launched with total_timesteps=1e8 alongside the
    # main 3e8 sweep for the same (qac_variant, budget) config; keep only
    # the 3e8 ones so curves aren't mixing runs of different lengths.
    df = df[(df["arch"] != "IRU-ACT") | (df["total_timesteps"] == 300_000_000)]
    df = expand_vocab_agnostic_budget1(df)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    budget_colors = all_budgets_palette(df)
    variant_colors = all_variants_palette(df)

    for arch in ARCH_ORDER:
        if arch not in df["arch"].unique():
            continue
        safe_name = arch.lower().replace(" ", "_")
        plot_arch(df, arch, budget_colors, args.output_dir / f"lightsout_hd64_{safe_name}.png", args.output_dir)
        plot_budget_vs_performance(
            df,
            arch,
            variant_colors,
            args.output_dir / f"lightsout_hd64_{safe_name}_budget_vs_performance.png",
            args.output_dir,
        )

    plot_headline_comparison(df, args.output_dir / "lightsout_hd64_arch_comparison.png", args.output_dir)


if __name__ == "__main__":
    main()
