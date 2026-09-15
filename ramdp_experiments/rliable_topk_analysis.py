#!/usr/bin/env python
"""Symmetric-trim mean comparison across qac_variants, budgets (and, for
Transformer-ExplicitCoT, vocab_size) using rliable
(https://github.com/google-research/rliable), for one or more lightsout
wandb-cache CSVs (see fetch_wandb_lightsout_hd64.py).

rliable_analysis.py's IQM is itself a symmetric trimmed mean - drop the
worst alpha/2 and best alpha/2 of scores, average what's left - fixed at
alpha=0.5 (25% off each end, i.e. the middle 50%). This script is the same
statistic with alpha (equivalently, how many seeds to drop off each end)
exposed as a CLI flag, since with only 5-10 seeds per algorithm, dropping a
quarter off each end can be too aggressive or too lenient depending on the
dataset. `--drop 1` (the default) drops the single worst and single best
seed and averages the rest - e.g. 1 of 10 seeds is close to trimming the
bottom/top 10th percentile each side. If dropping `drop` from each end
would leave no seeds, `drop` is clipped down (a printed warning notes this)
so the statistic degrades toward the plain mean rather than dividing by
zero.

Same algorithm/architecture scoping and CLI shape as rliable_analysis.py:
comparisons are scoped to one architecture at a time, and "algorithms" are
every (qac_variant, vocab_size, budget) combo present, at both the adaptive
budget (min_steps=1, max_steps=5) and every fixed ("uniform") budget - see
rliable_analysis.py's module docstring for the full rationale, and its
compute_time-spread printout for a sanity check on adaptive vs. uniform
budget usage.

Usage:
  python ramdp_experiments/rliable_topk_analysis.py \\
      ramdp_experiments/wandb_cache_hd64-sep7.csv --drop 1 \\
      --output-dir ramdp_experiments/rliable_plots
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict

import numpy as np
from rliable import library as rly
from rliable import plot_utils

import rliable_analysis as ra

pd = ra.pd
plt = ra.plt
load_dataset = ra.load_dataset
build_algorithms_for_arch = ra.build_algorithms_for_arch
build_score_dict = ra.build_score_dict
print_compute_time_spread = ra.print_compute_time_spread
AlgoSpec = ra.AlgoSpec
METRIC_SPECS = ra.METRIC_SPECS
ARCH_ORDER = ra.ARCH_ORDER
DPI = ra.DPI

# "of the return" - budget/eval_budget are still available via --metric, but
# a trimmed mean of a compute-usage metric isn't the headline use case here.
DEFAULT_METRICS = ["discounted_return", "eval_discounted_return"]


def aggregate_trimmed_mean(scores: np.ndarray, drop: int) -> float:
    """Mean after dropping the `drop` lowest and `drop` highest entries of
    `scores` (flattened across runs and tasks, matching rliable's
    aggregate_iqm/aggregate_mean convention) - e.g. drop=1 means "drop 1
    worst and 1 best". Clipped so at least one entry always survives."""
    flat = np.sort(np.asarray(scores).ravel())
    n = flat.size
    d = min(drop, (n - 1) // 2) if n > 0 else 0
    trimmed = flat[d : n - d]
    return float(np.mean(trimmed))


def run_trimmed_mean(
    df: pd.DataFrame,
    algorithms: Dict[str, AlgoSpec],
    metric: str,
    xlabel: str,
    output_path: Path,
    drop: int,
) -> None:
    score_dict = build_score_dict(df, algorithms, metric)
    algo_order = [label for label in algorithms if score_dict[label].size > 0]
    if not algo_order:
        print(f"  (no data for {output_path}, skipping)")
        return
    score_dict = {label: score_dict[label] for label in algo_order}

    for label in algo_order:
        n = score_dict[label].size
        d = min(drop, (n - 1) // 2) if n > 0 else 0
        if d < drop:
            print(
                f"    [{label}] only {n} seed(s) - dropping {d} (not {drop}) from each end "
                f"so {n - 2 * d} remain(s)"
            )

    trim_func = lambda scores: np.array([aggregate_trimmed_mean(scores, drop)])  # noqa: E731
    point_estimates, interval_estimates = rly.get_interval_estimates(score_dict, trim_func, reps=50000)

    fig, axes = plot_utils.plot_interval_estimates(
        point_estimates,
        interval_estimates,
        metric_names=[f"Trimmed mean (drop {drop}/end)"],
        algorithms=algo_order,
        xlabel=xlabel,
    )
    fig.savefig(output_path, bbox_inches="tight", dpi=DPI)
    print(f"Saved {output_path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csvs", type=Path, nargs="+", help="wandb_cache_hd64-*.csv files, one per dataset")
    parser.add_argument(
        "--drop", type=int, default=1, help="Number of worst and best seeds to drop from each end (default: 1)"
    )
    parser.add_argument(
        "--metric",
        choices=list(METRIC_SPECS.keys()),
        action="append",
        help=f"Restrict to one metric (repeatable). Default: {DEFAULT_METRICS}.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("ramdp_experiments/rliable_plots"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metric_keys = args.metric or DEFAULT_METRICS

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

            for metric_key in metric_keys:
                metric_col, iqm_label = METRIC_SPECS[metric_key]
                xlabel = iqm_label.replace("IQM:", f"Trimmed mean (drop {args.drop}/end):")
                run_trimmed_mean(
                    df,
                    algorithms,
                    metric_col,
                    xlabel,
                    args.output_dir / f"rliable_{dataset_label}_{safe_name}_{metric_key}_trim{args.drop}.pdf",
                    args.drop,
                )


if __name__ == "__main__":
    main()
