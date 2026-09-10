#!/usr/bin/env python
"""IQM and probability-of-improvement comparison across qac_variants (and,
for Transformer-ExplicitCoT, vocab_size) using rliable
(https://github.com/google-research/rliable), for one or more lightsout
wandb-cache CSVs (see fetch_wandb_lightsout_hd64.py) - e.g.
wandb_cache_hd64-sep7.csv and wandb_cache_hd64-sep8.csv.

Comparisons are scoped to ONE ARCHITECTURE at a time (IRU-ACT,
Transformer-CoT, Transformer-ExplicitCoT never appear in the same figure) -
one IQM and one probability-of-improvement figure per (dataset, architecture,
metric).

Run separately for four scores (`--metric` choices, or all by default):
  - discounted_return: actor/episode_discounted_return/mean, final value
    (mean of the run's last 3 evals - see compute_final_values). Actor
    metrics come from the training-time rollout (the batch of episodes the
    policy update itself is computed from).
  - budget: actor/compute_time/mean, final value - how many ponder steps an
    adaptive-budget run actually used, not the nominal budget cap.
  - eval_discounted_return / eval_budget: same two quantities, but from
    evaluator/episode_discounted_return/mean and evaluator/compute_time/mean
    - a held-out evaluator rollout logged at the same eval_step as the actor
    metrics above (see fetch_wandb_lightsout_hd64.py), not the training
    batch. Use these to check whether an IQM/POI conclusion drawn from
    actor-side numbers also holds on held-out evaluation, or is a training-
    batch artifact.

"Algorithms" (rliable's unit of comparison) are every (qac_variant,
vocab_size) combo present AT THE ADAPTIVE BUDGET (min_steps=1, max_steps=5)
for that architecture - fixed-budget configs are excluded because they
aren't different *algorithms*, they're different resource allocations for
the same REINFORCE algorithm, and mixing them in would blur the comparison
rliable is meant for. For Transformer-ExplicitCoT, which has a vocab_size
axis, every vocab_size is its own variant (rather than picking one "primary"
vocab_size as plot_wandb_lightsout_hd64_extra.py's pareto/heatmap plots do),
so e.g. "REINFORCE, vocab=2" and "REINFORCE, vocab=8" are compared against
each other just like different qac_variants are.

This is a single task (this one lightsout env/scenario) with 5 seeds per
algorithm, so IQM and the bootstrap CIs reduce to a straightforward
run-level (not task-level) comparison. Probability of improvement is
computed for the FULL pairwise matrix within each architecture (every
variant against every other variant of that same architecture) - for
IRU-ACT/Transformer-CoT (3 qac_variants, no vocab axis) that's 3 pairs; for
Transformer-ExplicitCoT (qac_variants x vocab_sizes) it can be a couple
dozen, which is why it's one figure per architecture rather than one giant
combined figure.

Usage:
  python ramdp_experiments/rliable_analysis.py \\
      ramdp_experiments/wandb_cache_hd64-sep7.csv ramdp_experiments/wandb_cache_hd64-sep8.csv \\
      --output-dir ramdp_experiments/rliable_plots
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from rliable import library as rly
from rliable import metrics
from rliable import plot_utils

import plot_wandb_lightsout_hd64 as base

pd = base.pd
plt = base.plt
compute_final_values = base.compute_final_values
expand_vocab_agnostic_budget1 = base.expand_vocab_agnostic_budget1
variant_row_label = base.variant_row_label
ARCH_ORDER = base.ARCH_ORDER

METRIC_SPECS = {
    "discounted_return": ("actor/episode_discounted_return/mean", "IQM: discounted return"),
    "budget": ("actor/compute_time/mean", "IQM: compute (ponder) steps used"),
    "eval_discounted_return": (
        "evaluator/episode_discounted_return/mean",
        "IQM: discounted return (evaluator)",
    ),
    "eval_budget": (
        "evaluator/compute_time/mean",
        "IQM: compute (ponder) steps used (evaluator)",
    ),
}

AlgoSpec = Tuple[str, str, int]  # (arch, qac_variant, vocab_size)

DPI = 600


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Same filters as plot_wandb_lightsout_hd64.py's main(): finished runs
    only, no stop-gradient-halting variants, IRU-ACT's main 3e8-timestep
    sweep only, and eCoT's vocab-agnostic budget=1 shared across vocab
    sizes."""
    df = pd.read_csv(csv_path)
    df = df[df["state"] == "finished"]
    df = df[~df["sgh"]]
    df = df[(df["arch"] != "IRU-ACT") | (df["total_timesteps"] == 300_000_000)]
    df = expand_vocab_agnostic_budget1(df)
    return df


def build_algorithms_for_arch(sub_arch: pd.DataFrame, arch: str) -> Dict[str, AlgoSpec]:
    """{algorithm_label: (arch, qac_variant, vocab_size)} for one
    architecture's adaptive[1-5] budget. Every (qac_variant, vocab_size)
    combo present is its own variant - for Transformer-ExplicitCoT this
    means every vocab_size is separately comparable, not just a single
    "primary" one."""
    adaptive = sub_arch[sub_arch["min_steps"] != sub_arch["max_steps"]]
    combos = sorted(
        adaptive[["qac_variant", "vocab_size"]].drop_duplicates().itertuples(index=False, name=None),
        key=lambda k: (k[0] != "reinforce", k[0], k[1]),
    )
    algorithms: Dict[str, AlgoSpec] = {}
    for qac_variant, vocab_size in combos:
        label = variant_row_label(qac_variant, False, vocab_size)
        algorithms[label] = (arch, qac_variant, vocab_size)
    return algorithms


def algorithm_scores(df: pd.DataFrame, arch: str, qac_variant: str, vocab_size: int, metric: str) -> np.ndarray:
    """(n_seeds, 1) array of final values for one algorithm/metric - the
    shape rliable expects: (num_runs x num_tasks), one task here."""
    sub = df[
        (df["arch"] == arch)
        & (df["qac_variant"] == qac_variant)
        & (df["vocab_size"] == vocab_size)
        & (df["min_steps"] != df["max_steps"])
    ]
    final = compute_final_values(sub, metric)
    return final["value"].to_numpy().reshape(-1, 1)


def build_score_dict(df: pd.DataFrame, algorithms: Dict[str, AlgoSpec], metric: str) -> Dict[str, np.ndarray]:
    return {label: algorithm_scores(df, *spec, metric) for label, spec in algorithms.items()}


def print_compute_time_spread(df: pd.DataFrame, algorithms: Dict[str, AlgoSpec], arch: str, prefix: str) -> None:
    """Print, per algorithm, the final (mean of last-3-eval) compute_time
    mean/min/max averaged across seeds, plus max-min spread - a quick check
    of whether an adaptive-budget agent's per-episode compute usage
    actually varies or is effectively constant. `prefix` is "actor" or
    "evaluator"."""
    min_col, max_col = f"{prefix}/compute_time/min", f"{prefix}/compute_time/max"
    if min_col not in df.columns or max_col not in df.columns:
        print(
            f"  [{arch}] {prefix}/compute_time spread: skipped - {min_col}/{max_col} not in this CSV "
            "(re-run fetch_wandb_lightsout_hd64.py, which now scrapes them from the console log - "
            "see fetch_compute_time_extrema)"
        )
        return
    print(f"  [{arch}] {prefix}/compute_time spread (final value, mean across seeds):")
    for label, spec in algorithms.items():
        mean_scores = algorithm_scores(df, *spec, f"{prefix}/compute_time/mean").ravel()
        if mean_scores.size == 0:
            continue
        min_scores = algorithm_scores(df, *spec, min_col).ravel()
        max_scores = algorithm_scores(df, *spec, max_col).ravel()
        mean_of_min = min_scores.mean() if min_scores.size else float("nan")
        mean_of_max = max_scores.mean() if max_scores.size else float("nan")
        print(
            f"    {label:35s} mean={mean_scores.mean():6.2f}  min={mean_of_min:6.2f}  "
            f"max={mean_of_max:6.2f}  spread={mean_of_max - mean_of_min:6.2f}  (n={mean_scores.size} seeds)"
        )


def build_poi_pairs(labels: List[str]) -> List[Tuple[str, str]]:
    """Full pairwise matrix among the given algorithm labels (all from the
    same architecture - see module docstring)."""
    pairs = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            pairs.append((labels[i], labels[j]))
    return pairs


def run_iqm(df: pd.DataFrame, algorithms: Dict[str, AlgoSpec], metric: str, xlabel: str, output_path: Path) -> None:
    score_dict = build_score_dict(df, algorithms, metric)
    algo_order = [label for label in algorithms if score_dict[label].size > 0]
    if not algo_order:
        print(f"  (no data for {output_path}, skipping)")
        return
    score_dict = {label: score_dict[label] for label in algo_order}

    iqm_func = lambda scores: np.array([metrics.aggregate_iqm(scores)])  # noqa: E731
    point_estimates, interval_estimates = rly.get_interval_estimates(score_dict, iqm_func, reps=50000)

    fig, axes = plot_utils.plot_interval_estimates(
        point_estimates,
        interval_estimates,
        metric_names=["IQM"],
        algorithms=algo_order,
        xlabel=xlabel,
    )
    fig.savefig(output_path, bbox_inches="tight", dpi=DPI)
    print(f"Saved {output_path}")
    plt.close(fig)


def run_poi(df: pd.DataFrame, algorithms: Dict[str, AlgoSpec], metric: str, xlabel: str, output_path: Path) -> None:
    pairs = build_poi_pairs(list(algorithms.keys()))
    # Algorithm labels can themselves contain a comma (e.g. eCoT's
    # "Cond. factorized, vocab=4"), so a plain "," pair-key separator would
    # be ambiguous when rliable splits it back apart. Use a separator that
    # can't collide with label text instead.
    sep = " :: "
    pair_score_dict = {}
    for x_label, y_label in pairs:
        scores_x = algorithm_scores(df, *algorithms[x_label], metric)
        scores_y = algorithm_scores(df, *algorithms[y_label], metric)
        if scores_x.size == 0 or scores_y.size == 0:
            continue
        pair_score_dict[f"{x_label}{sep}{y_label}"] = [scores_x, scores_y]

    if not pair_score_dict:
        print(f"  (no POI pairs with data for {output_path}, skipping)")
        return

    point_estimates, interval_estimates = rly.get_interval_estimates(
        pair_score_dict, metrics.probability_of_improvement, reps=2000
    )

    fig, ax = plt.subplots(figsize=(7, 0.5 * len(pair_score_dict) + 1))
    plot_utils.plot_probability_of_improvement(
        point_estimates,
        interval_estimates,
        pair_separator=sep,
        ax=ax,
        xticks=[0.0, 0.25, 0.5, 0.75, 1.0],
        xlabel=xlabel,
    )
    ax.set_xlim(0, 1)
    ax.axvline(0.5, color="0.5", linestyle="--", linewidth=1, zorder=0)
    fig.savefig(output_path, bbox_inches="tight", dpi=DPI)
    print(f"Saved {output_path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csvs", type=Path, nargs="+", help="wandb_cache_hd64-*.csv files, one per dataset")
    parser.add_argument(
        "--metric", choices=list(METRIC_SPECS.keys()), action="append", help="Restrict to one metric (repeatable). Default: all."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("ramdp_experiments/rliable_plots"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metric_keys = args.metric or list(METRIC_SPECS.keys())

    for csv_path in args.csvs:
        m = re.search(r"-([^-.]+)$", csv_path.stem)
        dataset_label = m.group(1) if m else csv_path.stem
        print(f"=== {dataset_label} ({csv_path}) ===")
        df = load_dataset(csv_path)

        for arch in ARCH_ORDER:
            sub_arch = df[df["arch"] == arch]
            if sub_arch.empty:
                continue
            algorithms = build_algorithms_for_arch(sub_arch, arch)
            if not algorithms:
                continue
            safe_name = arch.lower().replace(" ", "-").replace(".", "")
            print(f"  [{arch}] variants: {list(algorithms.keys())}")

            for prefix in ("actor", "evaluator"):
                print_compute_time_spread(df, algorithms, arch, prefix)

            for metric_key in metric_keys:
                metric_col, iqm_label = METRIC_SPECS[metric_key]
                run_iqm(
                    df,
                    algorithms,
                    metric_col,
                    iqm_label,
                    args.output_dir / f"rliable_{dataset_label}_{safe_name}_{metric_key}_iqm.pdf",
                )
                run_poi(
                    df,
                    algorithms,
                    metric_col,
                    "P(X > Y)",
                    args.output_dir / f"rliable_{dataset_label}_{safe_name}_{metric_key}_poi.pdf",
                )


if __name__ == "__main__":
    main()
