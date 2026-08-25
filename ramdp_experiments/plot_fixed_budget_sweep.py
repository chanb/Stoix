#!/usr/bin/env python
"""Plot discounted and undiscounted return vs. compute budget for a
`*_fixed_budget_sweep.py` result directory (e.g.
`results_minatar_fixed_budget_sweep`, `results_seaquest_fixed_budget_sweep`,
`results_sokoban_fixed_budget_sweep`).

Reads `manifest.jsonl` (one JSON record per launched job, written by the
sweep script's `run_job`) to find each successful run's output directory,
then reads that run's `metrics.json` (written by `marl_eval.json_tools.
JsonLogger`, see `stoix.utils.logger.JsonLogger`) to pull out:

  - its full training curve: for each eval checkpoint `step_<eval_idx>`,
    the real env timestep (`step_count`) paired with
    `mean_episode_return` / `mean_episode_discounted_return`
  - its final performance, from `absolute_metrics` (falling back to the
    last eval checkpoint if the absolute-metric eval is missing):
      - undiscounted return: `mean_episode_return`
      - discounted return:   `mean_episode_discounted_return`
        (discounted the same way the actor is trained: gamma per env step,
        further discounted by gamma per pondering/CoT step - see
        `stoix.systems.ramdp_vpg.evaluator`)

`metrics.json` nests as `data[env_name][task_name][algorithm_name]
[f"seed_{seed}"]`; the env/task names are read directly out of that nesting
rather than assumed from the manifest, since older sweep scripts didn't
record an `env` field.

For each (env, arch), multiple hyperparameter combinations (system,
hidden_dim, lr, critic_lr, delightful/eta, layer norm options) may have been
swept - every combination actually run gets its own row (no "pick the best"
selection), so nothing is hidden from the plot.

One figure is produced per (env, arch) pair. Within each figure, rows are
hyperparameter settings and there are two columns:
  1. timesteps (x) vs. performance (y): the training curve, one line per
     budget, averaged across seeds.
  2. budget (x) vs. final performance (y): mean +/- std across seeds at each
     budget.

Usage:
  python ramdp_experiments/plot_fixed_budget_sweep.py results_minatar_fixed_budget_sweep
  python ramdp_experiments/plot_fixed_budget_sweep.py results_seaquest_fixed_budget_sweep \\
      results_minatar_fixed_budget_sweep --output-dir sweep_plots
  python ramdp_experiments/plot_fixed_budget_sweep.py results_sokoban_fixed_budget_sweep \\
      --metric undiscounted --title "Sokoban fixed-budget sweep"
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class RunResult:
    env: str
    task: str
    system: str
    arch: str
    budget: int
    hidden_dim: int
    lr: float
    critic_lr: float
    delightful: bool
    delightful_eta: float
    use_layer_norm: bool
    use_input_layer_norm: bool
    seed: int
    undiscounted_return: float
    discounted_return: float
    # Training curve: (timestep, undiscounted_return, discounted_return) per
    # eval checkpoint, sorted by eval order.
    curve: List[Tuple[int, float, float]]


def load_manifest(results_dir: Path) -> List[dict]:
    """Read manifest.jsonl, deduped by (output_dir, run_name) keeping the
    last (most recent) entry - a run may appear more than once if it was
    relaunched with --no-skip-existing."""
    manifest_path = results_dir / "manifest.jsonl"
    if not manifest_path.is_file():
        print(f"warning: no manifest.jsonl in {results_dir}, skipping", file=sys.stderr)
        return []

    records: Dict[Tuple[str, str], dict] = {}
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            key = (record["output_dir"], record["run_name"])
            records[key] = record
    return list(records.values())


def find_latest_metrics_json(output_dir: Path, run_name: str) -> Optional[Path]:
    """`<output_dir>/<run_name>/json/<system_name>/<timestamp>/metrics.json`
    - system_name and timestamp aren't known ahead of time (timestamp is
    generated at run time, and reruns create a fresh one), so glob for all
    candidates and take the most recently modified."""
    candidates = list(Path(output_dir, run_name).glob("json/*/*/metrics.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _unwrap(value):
    """marl-eval's JsonLogger writes every metric as a single-element list."""
    return value[0] if isinstance(value, list) else value


def extract_training_curve(run_data: dict) -> List[Tuple[int, float, float]]:
    """Returns the full training curve as a list of
    (timestep, undiscounted_return, discounted_return) tuples, sorted by
    eval order. The `step_<eval_idx>` key suffix is only an eval-index
    counter, not the real timestep - the real timestep is the `step_count`
    field inside each eval's dict."""
    step_keys = sorted(
        (k for k in run_data if k.startswith("step_")),
        key=lambda k: int(k.split("_")[1]),
    )
    curve = []
    for k in step_keys:
        step_metrics = run_data[k]
        if (
            "step_count" not in step_metrics
            or "mean_episode_return" not in step_metrics
            or "mean_episode_discounted_return" not in step_metrics
        ):
            continue
        curve.append(
            (
                int(_unwrap(step_metrics["step_count"])),
                float(_unwrap(step_metrics["mean_episode_return"])),
                float(_unwrap(step_metrics["mean_episode_discounted_return"])),
            )
        )
    return curve


def extract_run_data(
    metrics_path: Path, seed: int
) -> Optional[Tuple[str, str, float, float, List[Tuple[int, float, float]]]]:
    """Returns (env_name, task_name, undiscounted_return, discounted_return,
    curve), or None if the run has no usable data (e.g. it crashed before
    its first evaluation)."""
    with open(metrics_path) as f:
        data = json.load(f)

    env_name, task_data = next(iter(data.items()))
    task_name, algo_data = next(iter(task_data.items()))
    _, seed_to_run_data = next(iter(algo_data.items()))
    seed_key = f"seed_{seed}"
    run_data = seed_to_run_data.get(seed_key) or next(iter(seed_to_run_data.values()), None)
    if run_data is None:
        return None

    absolute_metrics = run_data.get("absolute_metrics", {})
    if "mean_episode_return" in absolute_metrics:
        step_metrics = absolute_metrics
    else:
        # Fall back to the final evaluation point if `arch.absolute_metric`
        # was off (or the run died before the absolute-metric eval).
        step_keys = [k for k in run_data if k.startswith("step_")]
        if not step_keys:
            return None
        last_step_key = max(step_keys, key=lambda k: int(k.split("_")[1]))
        step_metrics = run_data[last_step_key]

    if "mean_episode_return" not in step_metrics or "mean_episode_discounted_return" not in step_metrics:
        return None

    undiscounted = float(_unwrap(step_metrics["mean_episode_return"]))
    discounted = float(_unwrap(step_metrics["mean_episode_discounted_return"]))
    curve = extract_training_curve(run_data)
    return env_name, task_name, undiscounted, discounted, curve


def collect_results(results_dirs: List[Path]) -> List[RunResult]:
    results = []
    n_total = 0
    n_failed = 0
    n_missing = 0
    for results_dir in results_dirs:
        for record in load_manifest(results_dir):
            n_total += 1
            if record.get("returncode", 1) != 0:
                n_failed += 1
                continue
            metrics_path = find_latest_metrics_json(
                Path(record["output_dir"]), record["run_name"]
            )
            if metrics_path is None:
                n_missing += 1
                continue
            extracted = extract_run_data(metrics_path, record["seed"])
            if extracted is None:
                n_missing += 1
                continue
            _env_name, task_name, undiscounted, discounted, curve = extracted
            results.append(
                RunResult(
                    env=record.get("env", task_name),
                    task=task_name,
                    system=record["system"],
                    arch=record["arch"],
                    budget=record["budget"],
                    hidden_dim=record["hidden_dim"],
                    lr=record["lr"],
                    critic_lr=record["critic_lr"],
                    delightful=record.get("delightful", False),
                    delightful_eta=record.get("delightful_eta", 1.0),
                    use_layer_norm=record.get("use_layer_norm", False),
                    use_input_layer_norm=record.get("use_input_layer_norm", False),
                    seed=record["seed"],
                    undiscounted_return=undiscounted,
                    discounted_return=discounted,
                    curve=curve,
                )
            )
    print(
        f"Loaded {len(results)}/{n_total} runs "
        f"({n_failed} failed, {n_missing} succeeded but had no usable metrics.json)."
    )
    return results


# env, arch, system, hidden_dim, lr, critic_lr, delightful, delightful_eta,
# use_layer_norm, use_input_layer_norm - every axis except `budget` and `seed`,
# which are what's aggregated/plotted within a row (see plot_env_arch).
RowKey = Tuple[str, str, str, int, float, float, bool, float, bool, bool]


def format_hyperparam_row_label(row_key: RowKey, show_system: bool) -> str:
    (
        _env,
        _arch,
        system,
        hidden_dim,
        lr,
        critic_lr,
        delightful,
        delightful_eta,
        use_layer_norm,
        use_input_layer_norm,
    ) = row_key
    parts = [f"hidden_dim={hidden_dim}", f"lr={lr:g}", f"critic_lr={critic_lr:g}"]
    if delightful:
        parts.append(f"delightful_eta={delightful_eta:g}")
    if use_layer_norm:
        parts.append("ln")
    if use_input_layer_norm:
        parts.append("input_ln")
    label = ", ".join(parts)
    return f"{system} | {label}" if show_system else label


def group_all_variants(results: List[RunResult]) -> Dict[RowKey, Dict[int, List[RunResult]]]:
    """Groups runs by every hyperparameter axis (env, arch, system,
    hidden_dim, lr, critic_lr, delightful, delightful_eta, use_layer_norm,
    use_input_layer_norm) -> {budget: [RunResult per seed]}. Unlike a
    "pick-the-best" selection, every combination actually swept becomes its
    own row when plotted - nothing is hidden."""
    groups: Dict[RowKey, Dict[int, List[RunResult]]] = {}
    for r in results:
        row_key = (
            r.env,
            r.arch,
            r.system,
            r.hidden_dim,
            r.lr,
            r.critic_lr,
            r.delightful,
            r.delightful_eta,
            r.use_layer_norm,
            r.use_input_layer_norm,
        )
        groups.setdefault(row_key, {}).setdefault(r.budget, []).append(r)
    return groups


def plot_env_arch(
    env: str,
    arch: str,
    row_keys: List[RowKey],
    grouped: Dict[RowKey, Dict[int, List[RunResult]]],
    metric: str,
    output_path: Path,
    title: Optional[str],
) -> None:
    """One figure for (env, arch): rows are hyperparameter settings, columns
    are (timesteps vs. performance) and (budget vs. final performance)."""
    metric_attr = "discounted_return" if metric == "discounted" else "undiscounted_return"
    curve_idx = 2 if metric == "discounted" else 1
    metric_label = "Discounted return" if metric == "discounted" else "Undiscounted return"

    all_budgets = sorted({budget for row_key in row_keys for budget in grouped[row_key]})
    color_map = plt.get_cmap("viridis")
    budget_colors = {
        budget: color_map(i / max(len(all_budgets) - 1, 1)) for i, budget in enumerate(all_budgets)
    }
    show_system = len({row_key[2] for row_key in row_keys}) > 1

    fig, axes = plt.subplots(
        len(row_keys),
        2,
        figsize=(11, 4 * len(row_keys)),
        squeeze=False,
    )

    for row, row_key in enumerate(row_keys):
        ax_curve, ax_budget = axes[row]
        budgets_data = grouped[row_key]
        row_budgets = sorted(budgets_data)

        # Column 1: timesteps (x) vs. performance (y), one line per budget,
        # averaged across seeds.
        for budget in row_budgets:
            curves = [r.curve for r in budgets_data[budget] if r.curve]
            if not curves:
                continue
            min_len = min(len(c) for c in curves)
            if min_len == 0:
                continue
            steps = [curves[0][i][0] for i in range(min_len)]
            values = np.array([[c[i][curve_idx] for i in range(min_len)] for c in curves])
            mean_curve = values.mean(axis=0)
            std_curve = values.std(axis=0)
            ax_curve.plot(
                steps, mean_curve, label=f"budget={budget}", color=budget_colors[budget]
            )
            ax_curve.fill_between(
                steps,
                mean_curve - std_curve,
                mean_curve + std_curve,
                color=budget_colors[budget],
                alpha=0.15,
            )
        ax_curve.set_xlabel("Timesteps")
        row_label = format_hyperparam_row_label(row_key, show_system)
        ax_curve.set_ylabel(f"{metric_label}\n\n{row_label}", fontsize=9)
        if row == 0:
            ax_curve.set_title("Training curve")
        ax_curve.grid(True, alpha=0.3)
        ax_curve.legend(fontsize=8)

        # Column 2: budget (x) vs. final performance (y), mean +/- std
        # across seeds.
        means = []
        stds = []
        for budget in row_budgets:
            values = [getattr(r, metric_attr) for r in budgets_data[budget]]
            means.append(np.mean(values))
            stds.append(np.std(values))
        ax_budget.errorbar(row_budgets, means, yerr=stds, marker="o", capsize=3, color="tab:blue")
        ax_budget.set_xscale("log", base=2)
        ax_budget.set_xticks(all_budgets)
        ax_budget.set_xticklabels(all_budgets)
        ax_budget.set_xlabel("Compute budget (min_steps == max_steps)")
        if row == 0:
            ax_budget.set_title("Final performance vs. budget")
        ax_budget.grid(True, alpha=0.3)

    fig.suptitle(title or f"{env} - {arch}", y=1.0, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    print(f"Saved plot to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "results_dirs", type=Path, nargs="+", help="One or more *_fixed_budget_sweep.py output directories."
    )
    parser.add_argument(
        "--metric",
        choices=["discounted", "undiscounted"],
        default="discounted",
        help="Which return to plot as 'performance' (default: discounted, matching the actual "
        "training objective).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory to write one '<env>_<arch>_fixed_budget_sweep.png' plot into per "
        "(env, arch) pair.",
    )
    parser.add_argument(
        "--title", default=None, help="Optional title prefix (env and arch are appended)."
    )
    args = parser.parse_args()

    results = collect_results(args.results_dirs)
    if not results:
        print("No usable runs found.", file=sys.stderr)
        sys.exit(1)

    grouped = group_all_variants(results)
    env_archs = sorted({(row_key[0], row_key[1]) for row_key in grouped})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for env, arch in env_archs:
        row_keys = sorted(rk for rk in grouped if rk[0] == env and rk[1] == arch)
        title = f"{args.title} - {env} - {arch}" if args.title else f"{env} - {arch}"
        output_path = args.output_dir / f"{env}_{arch}_fixed_budget_sweep.png"
        plot_env_arch(env, arch, row_keys, grouped, args.metric, output_path, title)


if __name__ == "__main__":
    main()
