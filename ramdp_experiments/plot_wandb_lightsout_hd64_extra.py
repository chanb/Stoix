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
     <N>_pareto.pdf) - the vocab_size=1/budget=1 point is shared across all
     of them (see expand_vocab_agnostic_budget1 in the main script).
  2. lightsout_hd64_<arch>_seed_variance.pdf - per-seed strip plot + mean/SE
     point, budget on x, instead of collapsing straight to mean +/- SEM;
     shows whether spread is genuine or one outlier seed.
  3. lightsout_hd64_<arch>_heatmap.pdf - (variant x budget) heatmap of final
     performance; a compact grid instead of many line/row panels.
  4. lightsout_hd64_<arch>_steps_to_threshold.pdf - timesteps needed to
     reach 80% of each run's own final performance, as a bar chart; isolates
     learning *speed* from final performance.
  5. lightsout_hd64_<arch>_compute_spaghetti.pdf - per-seed compute-time
     trajectories (adaptive-budget configs only) as individual thin lines
     instead of a mean +/- SEM band, to see whether halting behavior is
     consistent across seeds or not.

Usage:
  python ramdp_experiments/plot_wandb_lightsout_hd64_extra.py \\
      ramdp_experiments/wandb_cache_hd64.csv --output-dir ramdp_experiments/wandb_plots_extra
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import plot_wandb_lightsout_hd64 as base

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
expand_vocab_agnostic_budget1 = base.expand_vocab_agnostic_budget1
METRICS = base.METRICS
METRIC_LABELS = base.METRIC_LABELS
ARCH_ORDER = base.ARCH_ORDER
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
    row and concatenated. Calling it on the whole architecture at once would
    be wrong for Transformer-ExplicitCoT: expand_vocab_agnostic_budget1
    duplicates the same run_id under multiple vocab_size labels, and
    compute_final_values groups by run_id alone, so it would silently
    collapse those duplicates back down to one arbitrary vocab_size."""
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


def plot_pareto(
    df: pd.DataFrame, arch: str, output_path: Path, out_dir: Path, vocab_size: int | None = None
) -> None:
    """Final compute (ponder steps) vs. final performance for one
    architecture (and, for Transformer-ExplicitCoT, one vocab_size). Fixed-
    budget points are plain markers (no connecting line - budget isn't an
    ordered path through compute/performance space, so a line between them
    implies a trend that isn't really there); adaptive-budget points (one
    per qac_variant) are separate markers.

    `vocab_size`, when given, restricts to that vocab_size (the caller loops
    over every vocab_size present and makes one file each, since
    expand_vocab_agnostic_budget1 already folded the shared vocab_size=1
    budget=1 point into every vocab_size group upstream). When omitted
    (non-eCoT architectures, which have no vocab_size axis), falls back to
    the "primary" vocab_size (the sentinel, a no-op for those archs)."""
    return_metrics = METRICS[:2]
    compute_metric = METRICS[2]
    qac_markers = {"reinforce": "D", "cond_fac": "s", "cond_naive": "^"}
    qac_colors = dict(zip(["reinforce", "cond_fac", "cond_naive"], sns.color_palette("colorblind", n_colors=3)))
    fixed_color = "0.25"

    sub_arch = df[df["arch"] == arch]
    if vocab_size is None:
        vocab_size = determine_primary_vocab(sub_arch)
    arch_sub = sub_arch[sub_arch["vocab_size"] == vocab_size]

    if os.path.abspath(out_dir).startswith("/Users"):
        plt.rcParams.update(pgf_with_latex)

    figsize = set_size(doc_width_pt, fraction=1.8, subplots=(1, len(return_metrics)), use_golden_ratio=False)
    fig, axes = plt.subplots(1, len(return_metrics), figsize=figsize, squeeze=False)
    axes = axes[0]

    for col, metric in enumerate(return_metrics):
        ax = axes[col]
        perf = compute_final_values(arch_sub, metric)
        comp = compute_final_values(arch_sub, compute_metric)
        merged = perf.merge(comp[["run_id", "value"]], on="run_id", suffixes=("_perf", "_compute"))

        fixed = merged[merged["min_steps"] == merged["max_steps"]]
        if not fixed.empty:
            stats = fixed.groupby("min_steps").agg(
                perf_mean=("value_perf", "mean"),
                perf_sem=("value_perf", lambda s: s.std() / np.sqrt(len(s))),
                comp_mean=("value_compute", "mean"),
                comp_sem=("value_compute", lambda s: s.std() / np.sqrt(len(s))),
            ).sort_index()
            ax.errorbar(
                stats["comp_mean"],
                stats["perf_mean"],
                xerr=stats["comp_sem"],
                yerr=stats["perf_sem"],
                marker="o",
                color=fixed_color,
                linestyle="none",
                capsize=2,
                label="Fixed budget",
            )

        adaptive = merged[merged["min_steps"] != merged["max_steps"]]
        for qac_variant, g in adaptive.groupby("qac_variant"):
            perf_mean = g["value_perf"].mean()
            perf_sem = g["value_perf"].std() / np.sqrt(len(g))
            comp_mean = g["value_compute"].mean()
            comp_sem = g["value_compute"].std() / np.sqrt(len(g))
            ax.errorbar(
                [comp_mean],
                [perf_mean],
                xerr=[comp_sem],
                yerr=[perf_sem],
                marker=qac_markers.get(qac_variant, "*"),
                markersize=7,
                color=qac_colors.get(qac_variant, "0.5"),
                linestyle="none",
                capsize=2,
                label=variant_row_label(qac_variant, False),
            )

        ax.grid(True, alpha=0.3)
        ax.set_title(METRIC_LABELS[metric], fontsize=9)
        ax.set_xlabel("Final mean compute (ponder) steps")
        if col == 0:
            ax.set_ylabel("Final performance\n(mean of last 3 evals, ± SEM)", fontsize=8)

    title = f"{arch}: compute vs. performance (Pareto view)"
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


def plot_seed_variance(
    df: pd.DataFrame, arch: str, variant_palette: dict, output_path: Path, out_dir: Path
) -> None:
    """Per-seed final performance as a strip plot (budget on x, one column
    per seed's dot) plus a mean +/- SE point marker, instead of the main
    script's mean +/- SEM line - makes it possible to spot a single outlier
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

    figsize = set_size(doc_width_pt, fraction=1.8, subplots=(1, len(return_metrics)), use_golden_ratio=False)
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
            errorbar="se",
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
            ax.set_ylabel("Final performance\n(per-seed dots + mean ± SE)", fontsize=8)
        else:
            ax.set_ylabel("")

    handles = [plt.Line2D([0], [0], marker="o", linestyle="none", color=variant_palette[lbl]) for lbl in hue_order]
    place_legend_and_title(fig, handles, hue_order, min(len(hue_order), 3), figsize, f"{arch}: per-seed final performance")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=600)
    print(f"Saved {output_path}")
    plt.close(fig)


def plot_heatmap(df: pd.DataFrame, arch: str, output_path: Path, out_dir: Path) -> None:
    """(variant x budget) -> final performance heatmap, one per return
    metric - a compact grid alternative to the many row/line panels in the
    main script's figures."""
    sub_arch = df[df["arch"] == arch]
    return_metrics = METRICS[:2]
    row_keys = variant_rows_for_arch(sub_arch)
    row_labels = [variant_row_label(qv, False, vs) for qv, vs in row_keys]
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
    n_row_labels = max(len(row_labels), 1)
    n_col_labels = max(len(col_labels), 1)
    cell_size = 0.5
    figsize = (
        n_col_labels * cell_size * len(return_metrics) + 1.5 * len(return_metrics),
        n_row_labels * cell_size + 1.5,
    )
    fig, axes = plt.subplots(1, len(return_metrics), figsize=figsize, squeeze=False)
    axes = axes[0]

    for col, metric in enumerate(return_metrics):
        ax = axes[col]
        labeled = build_labeled_final_df(sub_arch, metric, row_keys)
        pivot = pd.DataFrame(index=row_labels, columns=col_labels, dtype=float)
        if not labeled.empty:
            grouped = labeled.groupby(["variant_label", "budget_label"])["value"].mean()
            for (rlabel, clabel), val in grouped.items():
                if rlabel in pivot.index and clabel in pivot.columns:
                    pivot.loc[rlabel, clabel] = val
        sns.heatmap(
            pivot.astype(float),
            annot=True,
            fmt=".2f",
            cmap="viridis",
            ax=ax,
            cbar_kws={"label": METRIC_LABELS[metric]},
            linewidths=0.5,
            linecolor="white",
            annot_kws={"fontsize": 7},
        )
        ax.set_title(METRIC_LABELS[metric], fontsize=9)
        ax.set_xlabel("Budget")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.tick_params(axis="y", rotation=0, labelsize=7)

    fig.suptitle(f"{arch}: final performance heatmap", y=1.04, fontsize=12)
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
            final = g[metric].tail(3).mean()
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

    figsize = set_size(doc_width_pt, fraction=1.8, subplots=(1, len(return_metrics)), use_golden_ratio=False)
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
            errorbar="se",
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
            ax.set_ylabel(f"Timesteps to {int(frac * 100)}% of final\n(mean ± SE across seeds)", fontsize=8)
        else:
            ax.set_ylabel("")

    handles = [plt.Line2D([0], [0], marker="s", linestyle="none", color=variant_palette[lbl]) for lbl in hue_order]
    place_legend_and_title(
        fig,
        handles,
        hue_order,
        min(len(hue_order), 3),
        figsize,
        f"{arch}: learning speed (steps to {int(frac * 100)}% of final performance)",
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=600)
    print(f"Saved {output_path}")
    plt.close(fig)


def plot_compute_spaghetti(df: pd.DataFrame, arch: str, output_path: Path, out_dir: Path) -> None:
    """Per-seed compute-time trajectories for adaptive-budget configs only,
    as individual thin lines rather than a mean +/- SEM band - the main
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

    figsize = set_size(doc_width_pt, fraction=1.1, subplots=(n_rows, 1), use_golden_ratio=False)
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
        if row == n_rows - 1:
            ax.set_xlabel("Timesteps")
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

    handles = [plt.Line2D([0], [0], color=seed_colors[s], label=f"seed {s}") for s in range(5)]
    place_legend_and_title(fig, handles, [f"seed {s}" for s in range(5)], 5, figsize, arch)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=600)
    print(f"Saved {output_path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", type=Path, help="CSV produced by fetch_wandb_lightsout_hd64.py")
    parser.add_argument("--output-dir", type=Path, default=Path("ramdp_experiments/wandb_plots_extra"))
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    # Same filters as plot_wandb_lightsout_hd64.py's main(): finished runs
    # only (config identity dedup keeps a still-running run only when no
    # finished run with that exact config exists yet), IRU-ACT has a
    # secondary 1e8-timestep sweep alongside the main 3e8 one, and eCoT's
    # vocab_size=1 budget=1 runs are valid for every other vocab_size too
    # (no CoT tokens are ever emitted at budget=1).
    df = df[df["state"] == "finished"]
    df = df[~df["sgh"]]
    df = df[(df["arch"] != "IRU-ACT") | (df["total_timesteps"] == 300_000_000)]
    df = expand_vocab_agnostic_budget1(df)
    args.output_dir.mkdir(parents=True, exist_ok=True)

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
                    args.output_dir / f"lightsout_hd64_{safe_name}_vocab{vocab_size}_pareto.pdf",
                    args.output_dir,
                    vocab_size=vocab_size,
                )
        else:
            plot_pareto(df, arch, args.output_dir / f"lightsout_hd64_{safe_name}_pareto.pdf", args.output_dir)
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


if __name__ == "__main__":
    main()
