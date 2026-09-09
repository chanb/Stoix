#!/usr/bin/env python
"""LaTeX table of actor compute-time statistics (min/max/mean/std), read
back from each run's console log (see fetch_wandb_lightsout_hd64.py's module
docstring for why min/max/std aren't in wandb's structured history - they're
printed by ConsoleLogger but dropped by WandBLogger unless
`logger.loggers.wandb.detailed_logging: true`, which none of these runs set).

Restricted to the adaptive[1-5] budget, no stop-gradient-halting runs (fixed-
budget compute time is trivial by construction: min=max=mean=budget, std=0),
broken out per (architecture, vocab_size, qac_variant) rather than merged
across qac_variant - 5 (arch, vocab_size) groups x 3 qac_variants = 15 rows,
each averaged (mean +/- SEM) over that group's 5 seeds. Columns are the four
console-log statistics: Min, Max, Mean, Std.

Per-run console-log stats are cached to a CSV next to the output .tex file
(<output>.stats_cache.csv) so re-running (e.g. after tweaking table
formatting) doesn't re-hit the wandb API.

Usage:
  python ramdp_experiments/make_compute_time_table.py \\
      ramdp_experiments/wandb_cache_hd64-sep7.csv \\
      --project bpychan-university-of-alberta/lightsout-sep7 \\
      --output ramdp_experiments/sep7/compute_time_table.tex
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

import fetch_wandb_lightsout_hd64 as fetch
import plot_wandb_lightsout_hd64 as base

ARCH_ORDER = base.ARCH_ORDER
VOCAB_SIZE_NA = base.VOCAB_SIZE_NA

STATS = ["min", "max", "mean", "std"]  # fetched/cached from the console log
TABLE_STATS = ["min", "mean", "std"]  # shown in the rendered table (max omitted)
STAT_ROW_LABELS = {"min": "Min", "max": "Max", "mean": "Mean", "std": "Std"}
VARIANT_ORDER = ["reinforce", "cond_fac", "cond_naive"]
VARIANT_LABELS = {"reinforce": "PPO", "cond_fac": "Cond. fac.", "cond_naive": "Cond. naive"}


def select_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Unique (run_id, arch, vocab_size, qac_variant, seed) rows for the
    adaptive[1-5], no-stop-grad-halting runs - one row per run. IRU-ACT has
    some runs launched with total_timesteps=1e8 alongside the main 3e8 sweep
    for the same (qac_variant, seed) config (see plot_wandb_lightsout_hd64's
    main()); keep only the 3e8 ones so seeds aren't double-counted."""
    sub = df[
        (df["state"] == "finished")
        & (df["min_steps"] == 1)
        & (df["max_steps"] == 5)
        & (df["sgh"].astype(str) == "False")
        & ((df["arch"] != "IRU-ACT") | (df["total_timesteps"] == 300_000_000))
    ]
    return sub[["run_id", "arch", "vocab_size", "qac_variant", "seed"]].drop_duplicates().reset_index(drop=True)


def fetch_actor_compute_stats(project: str, run_id: str) -> "dict[str, float]":
    """{"min": ..., "max": ..., "mean": ..., "std": ...} for ACTOR compute
    time, averaged over the last N_TAIL_EVALS eval checkpoints found in the
    run's console log (same tail-averaging convention as
    fetch_compute_time_extrema). Missing keys mean that stat never appeared
    in the fetched tail."""
    api = fetch.wandb.Api()
    run = api.run(f"{project}/{run_id}")
    try:
        lines = list(run.console_logs(last=fetch.CONSOLE_TAIL_LINES))
    except Exception as e:
        print(f"    (console log fetch failed for {run_id}: {e})")
        lines = []
    series: "dict[str, list[float]]" = {stat: [] for stat in STATS}
    for line in lines:
        parsed = fetch.parse_console_log_line(line.content)
        if parsed is None:
            continue
        label, stats = parsed
        if fetch._CONSOLE_EVENT_PREFIX.get(label) != "actor":
            continue
        for stat in STATS:
            key = f"compute_time_{stat}"
            if key in stats:
                series[stat].append(stats[key])
    return {stat: float(np.mean(vals[-fetch.N_TAIL_EVALS :])) for stat, vals in series.items() if vals}


def collect_stats_cache(runs: pd.DataFrame, project: str, cache_path: Path) -> pd.DataFrame:
    """Per-run actor compute-time stats, reusing `cache_path` if present
    (keyed by run_id) and fetching only the runs missing from it."""
    cached = pd.read_csv(cache_path) if cache_path.exists() else pd.DataFrame(columns=["run_id"] + STATS)
    have = set(cached["run_id"])
    rows = [cached]
    todo = runs[~runs["run_id"].isin(have)]
    for i, r in enumerate(todo.itertuples(index=False)):
        print(f"  [{i + 1}/{len(todo)}] fetching console log for {r.run_id} ({r.arch}, {r.qac_variant}, seed={r.seed})")
        stats = fetch_actor_compute_stats(project, r.run_id)
        rows.append(pd.DataFrame([{"run_id": r.run_id, **stats}]))
    merged = pd.concat(rows, ignore_index=True).drop_duplicates(subset="run_id", keep="last")
    merged.to_csv(cache_path, index=False)
    return merged


def table_columns(runs: pd.DataFrame):
    """Ordered (arch, vocab_size, qac_variant) triples: the same (arch,
    vocab_size) grouping as pareto_columns in plot_wandb_lightsout_hd64_extra
    (IRU-ACT, Transformer-CoT, Transformer-ExplicitCoT x vocab={2,4,8}),
    each split into its 3 qac_variant sub-columns."""
    columns = []
    for arch in ARCH_ORDER:
        arch_runs = runs[runs["arch"] == arch]
        if arch_runs.empty:
            continue
        vocab_values = sorted(v for v in arch_runs["vocab_size"].unique() if v != VOCAB_SIZE_NA)
        vocab_groups = vocab_values if vocab_values else [VOCAB_SIZE_NA]
        for vocab_size in vocab_groups:
            for variant in VARIANT_ORDER:
                if not arch_runs[
                    (arch_runs["vocab_size"] == vocab_size) & (arch_runs["qac_variant"] == variant)
                ].empty:
                    columns.append((arch, vocab_size, variant))
    return columns


def aggregate(runs: pd.DataFrame, per_run_stats: pd.DataFrame, columns) -> "dict[tuple, dict[str, tuple[float, float]]]":
    """{(arch, vocab_size, qac_variant): {stat: (mean_over_seeds, sem_over_seeds)}}."""
    merged = runs.merge(per_run_stats, on="run_id", how="left")
    out = {}
    for arch, vocab_size, variant in columns:
        g = merged[(merged["arch"] == arch) & (merged["vocab_size"] == vocab_size) & (merged["qac_variant"] == variant)]
        out[(arch, vocab_size, variant)] = {
            stat: (g[stat].mean(), g[stat].std() / np.sqrt(len(g))) for stat in STATS
        }
    return out


def escape_tex(s: str) -> str:
    return s.replace("_", r"\_")


def render_latex(columns, agg, caption: str, label: str) -> str:
    """One row per (arch, vocab_size, qac_variant) combo (15), one column
    per stat (4) - transpose of the earlier 4x15 layout. Architecture/vocab
    labels are only printed on the first row of their block (blank
    otherwise), with a \\midrule between architecture blocks, so repeated
    labels don't have to be read down every row. Wrapped in \\resizebox so it
    scales to \\textwidth regardless of the surrounding document's column
    width (requires \\usepackage{graphicx})."""
    has_vocab = any(vocab_size != VOCAB_SIZE_NA for _arch, vocab_size, _variant in columns)

    header = ["Architecture"]
    if has_vocab:
        header.append("Vocab")
    header += ["Variant"] + [STAT_ROW_LABELS[stat] for stat in TABLE_STATS]
    col_spec = "l" + ("l" if has_vocab else "") + "l" + "c" * len(TABLE_STATS)

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")

    prev_arch = None
    prev_vocab_key = None
    for i, (arch, vocab_size, variant) in enumerate(columns):
        if arch != prev_arch and i > 0:
            lines.append(r"\midrule")
        row = [escape_tex(arch) if arch != prev_arch else ""]
        if has_vocab:
            vocab_key = (arch, vocab_size)
            if vocab_key == prev_vocab_key:
                vocab_label = ""
            else:
                vocab_label = str(vocab_size) if vocab_size != VOCAB_SIZE_NA else "--"
            row.append(vocab_label)
            prev_vocab_key = vocab_key
        row.append(VARIANT_LABELS[variant])
        for stat in TABLE_STATS:
            mean, sem = agg[(arch, vocab_size, variant)][stat]
            row.append("--" if np.isnan(mean) else f"{mean:.3f} $\\pm$ {sem:.3f}")
        lines.append(" & ".join(row) + r" \\")
        prev_arch = arch

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", type=Path, help="CSV produced by fetch_wandb_lightsout_hd64.py")
    parser.add_argument("--project", required=True, help="wandb project, e.g. bpychan-university-of-alberta/lightsout-sep7")
    parser.add_argument("--output", type=Path, default=Path("ramdp_experiments/compute_time_table.tex"))
    parser.add_argument("--label", default=None, help="defaults to tab:compute_time_<output stem>")
    args = parser.parse_args()
    if args.label is None:
        # output paths are typically <run-tag-dir>/compute_time_table.tex
        # (e.g. sep7/, sep8/), so the parent directory name is what
        # actually distinguishes tables sharing that same filename.
        args.label = f"tab:compute_time_{args.output.parent.name}_{args.output.stem}"

    df = pd.read_csv(args.csv)
    runs = select_runs(df)
    print(f"{len(runs)} runs match adaptive[1-5], no-sgh across {runs['run_id'].nunique()} unique run ids")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cache_path = args.output.with_suffix(".stats_cache.csv")
    per_run_stats = collect_stats_cache(runs, args.project, cache_path)

    columns = table_columns(runs)
    agg = aggregate(runs, per_run_stats, columns)

    caption = (
        "Actor compute time (ponder steps) statistics -- mean $\\pm$ SEM across 5 seeds -- read from each "
        "run's console log, for the adaptive[1--5] budget (no stop-gradient halting)."
    )
    tex = render_latex(columns, agg, caption, args.label)
    args.output.write_text(tex)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
