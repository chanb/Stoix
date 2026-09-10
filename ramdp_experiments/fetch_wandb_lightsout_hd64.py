#!/usr/bin/env python
"""Fetch actor/{episode_return,episode_discounted_return,compute_time} training
curves, and the same three quantities from held-out evaluator rollouts
(evaluator/{episode_return,episode_discounted_return,compute_time}), from the
wandb project `bpychan-university-of-alberta/lightsout-sep7`, restricted to
`network.actor_network.pre_torso.hidden_dim == 64`, and cache them to a local
parquet file for plotting (see plot_wandb_lightsout_hd64.py) and rliable
analysis (see rliable_analysis.py).

Also fetches compute_time/{min,max} for both actor and evaluator, so
downstream analysis can check how much an adaptive-budget agent's
per-episode compute usage actually varies rather than only seeing the mean.
These two are NOT in `wandb.log`'s structured history - `WandBLogger.log_stat`
(stoix/utils/logger.py) drops every non-mean stat unless a run's config sets
`logger.loggers.wandb.detailed_logging: true` (default False, see
stoix/configs/logger/logger.yaml), and none of these runs did. But
`ConsoleLogger.log_stat` has no such gate, so mean/std/min/max for every
metric were all printed to stdout regardless - e.g. "EVALUATOR - ... |
Compute time mean: 3.480 | Compute time std: 0.596 | Compute time min: 2.200
| Compute time max: 4.500 | ..." - and wandb captures stdout as the run's
console log (the "Logs" tab) independently of what got `wandb.log`'d. So
`fetch_compute_time_extrema` reads back the tail of that console log via
`run.console_logs(last=...)` and regex-parses out the last few ACTOR/
EVALUATOR blocks' min/max, averaged the same way `compute_final_values`
(plot_wandb_lightsout_hd64.py) averages the last 3 eval points - it's a
single scalar per run, broadcast onto every row of that run's history so
the existing per-eval-point machinery doesn't need to change.

The project mixes runs logged under a couple of config schema versions (e.g.
`stop_gradient_halting_input` / `halting_ent_coef` only exist on newer runs,
and some configs were relaunched and so have duplicate rows). This script:
  - normalizes `stop_gradient_halting_input` (absent == False)
  - dedupes by the run's full config (everything except the `logger`
    subtree, which is pure logging plumbing - exp path, wandb run id, tags -
    not experiment identity), keeping the most-recently-created run per
    identical config, preferring `finished` over other states.

    Deliberately NOT deduped by a hand-picked tuple of fields (arch,
    qac_variant, budget, ...): earlier versions of this script did that and
    it kept silently mixing distinct runs together every time a new
    hyperparameter axis turned out to vary (stop_gradient_halting_input,
    vocab_size, total_timesteps each caused this in turn) but wasn't yet in
    the tuple. The run's naming convention (`logger.base_exp_path` /
    `group_tag`) has the exact same blind spot - it's built from a fixed set
    of tag components that new hyperparameters aren't automatically added
    to. Keying off the actual config sidesteps the whole class of bug: any
    config difference, known or not-yet-discovered, makes two runs distinct.

Usage:
  python ramdp_experiments/fetch_wandb_lightsout_hd64.py --out wandb_cache_hd64.parquet
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import wandb

PROJECT = "bpychan-university-of-alberta/lightsout-sep7"

METRICS = [
    "actor/episode_return/mean",
    "actor/episode_discounted_return/mean",
    "actor/compute_time/mean",
    # Evaluator metrics: same quantities but from held-out eval rollouts
    # (deterministic-ish, not the training batch) rather than the actor's
    # own training-time episodes - logged at the same eval_step/wandb step
    # as the actor/* metrics above (see ff_reinforce.py's eval loop), so
    # they line up 1:1 with the actor rows already fetched here.
    "evaluator/episode_return/mean",
    "evaluator/episode_discounted_return/mean",
    "evaluator/compute_time/mean",
    # compute_time/{min,max} are deliberately NOT here - Run.history(keys=...)
    # requires every requested key to be present on a row, and these were
    # never sent via wandb.log (see module docstring), so including them
    # would make every run's history query return empty. Fetched separately
    # from the console log instead - see fetch_compute_time_extrema.
]

# How many of the most recent console log lines to pull per run when looking
# for the tail few eval checkpoints' compute_time min/max (see
# fetch_compute_time_extrema). Each eval_step contributes only a handful of
# lines (MISC/TRAINER/ACTOR/EVALUATOR, occasionally ABSOLUTE), so this is
# generous headroom for N_TAIL_EVALS worth of them in one request.
CONSOLE_TAIL_LINES = 500
N_TAIL_EVALS = 3  # matches compute_final_values' n_tail default

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_CONSOLE_EVENT_PREFIX = {"ACTOR": "actor", "EVALUATOR": "evaluator"}

ARCH_SHORT = {
    "stoix.networks.torso_compute.IRUAdaptiveComputationTimeTorso": "IRU-ACT",
    "stoix.networks.torso_compute_explicit_cot.TransformerExplicitCoTTorso": "Transformer-ExplicitCoT",
    "stoix.networks.torso_compute_transformer.TransformerChainOfThoughtTorso": "Transformer-CoT",
}


def cfg_get(c: dict, path: str, default=None):
    cur = c
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


VOCAB_SIZE_NA = -1  # sentinel: architecture has no vocab_size (only Transformer-ExplicitCoT does)

# Config subtrees that are logging plumbing, not experiment identity - two
# runs with identical hyperparameters but different exp paths / wandb run
# ids / tags are the same experiment (a relaunch), not different ones.
IDENTITY_EXCLUDE_KEYS = {"logger"}


def config_identity_key(c: dict) -> str:
    """Canonical string identity for a run's config: every field except
    `logger`. Two runs get the same key iff every hyperparameter matches -
    this is what dedup should key on, not a hand-picked subset of fields."""
    identity = {k: v for k, v in c.items() if k not in IDENTITY_EXCLUDE_KEYS}
    return json.dumps(identity, sort_keys=True)


@dataclass
class RunMeta:
    run_id: str
    arch: str
    qac_variant: str
    min_steps: int
    max_steps: int
    sgh: bool
    vocab_size: int
    total_timesteps: int
    seed: int
    config_key: str
    created_at: str
    state: str


def fetch_run_metas(project: str) -> List[Tuple["wandb.apis.public.Run", RunMeta]]:
    api = wandb.Api()
    runs = api.runs(project, filters={
        "config.network.actor_network.pre_torso.hidden_dim": "64",
        "config.system.gamma": "0.999",
    })
    out = []
    for r in runs:
        c = r.config
        arch_full = cfg_get(c, "network.actor_network.pre_torso._target_")
        if arch_full not in ARCH_SHORT:
            continue
        qac = cfg_get(c, "system.qac_variant", "reinforce")
        mn = cfg_get(c, "network.actor_network.pre_torso.min_steps")
        mx = cfg_get(c, "network.actor_network.pre_torso.max_steps")
        if mn is None or mx is None:
            continue
        sgh_raw = cfg_get(c, "network.actor_network.pre_torso.stop_gradient_halting_input", False)
        sgh = str(sgh_raw) == "True"
        vocab_size_raw = cfg_get(c, "network.actor_network.pre_torso.vocab_size")
        vocab_size = int(vocab_size_raw) if vocab_size_raw is not None else VOCAB_SIZE_NA
        total_timesteps_raw = cfg_get(c, "arch.total_timesteps")
        if total_timesteps_raw is None:
            continue
        seed = cfg_get(c, "arch.seed")
        if seed is None:
            continue
        out.append(
            (
                r,
                RunMeta(
                    run_id=r.id,
                    arch=ARCH_SHORT[arch_full],
                    qac_variant=qac,
                    min_steps=int(mn),
                    max_steps=int(mx),
                    sgh=sgh,
                    vocab_size=vocab_size,
                    total_timesteps=int(float(total_timesteps_raw)),
                    seed=int(seed),
                    config_key=config_identity_key(c),
                    created_at=str(r.created_at),
                    state=r.state,
                ),
            )
        )
    return out


def dedupe_latest(
    rows: List[Tuple["wandb.apis.public.Run", RunMeta]]
) -> List[Tuple["wandb.apis.public.Run", RunMeta]]:
    """Keep, per identical config (meta.config_key - see config_identity_key),
    the best run: finished beats non-finished, then most-recently-created
    wins."""
    best: Dict[str, Tuple["wandb.apis.public.Run", RunMeta]] = {}
    for run, meta in rows:
        cur = best.get(meta.config_key)
        if cur is None:
            best[meta.config_key] = (run, meta)
            continue
        _, cur_meta = cur
        rank = (meta.state == "finished", meta.created_at)
        cur_rank = (cur_meta.state == "finished", cur_meta.created_at)
        if rank > cur_rank:
            best[meta.config_key] = (run, meta)
    return list(best.values())


def parse_console_log_line(content: str) -> Optional[Tuple[str, Dict[str, float]]]:
    """Parse one ConsoleLogger-formatted line (see stoix/utils/logger.py's
    `ConsoleLogger.log_dict`), e.g. "EVALUATOR - Compute time mean: 3.480 |
    Compute time min: 2.200 | ..." into (event_label, {stat_key: value}),
    e.g. ("EVALUATOR", {"compute_time_mean": 3.48, "compute_time_min": 2.2,
    ...}). Returns None for lines that don't match this format (config
    dumps, non-Stoix log lines, etc.)."""
    content = _ANSI_RE.sub("", content).strip()
    if " - " not in content:
        return None
    label, _, rest = content.partition(" - ")
    stats: Dict[str, float] = {}
    for frag in rest.split(" | "):
        key, sep, value = frag.partition(": ")
        if not sep:
            continue
        try:
            stats[key.strip().lower().replace(" ", "_")] = float(value)
        except ValueError:
            continue
    return label.strip(), stats


def fetch_compute_time_extrema(
    run: "wandb.apis.public.Run", n_tail: int = N_TAIL_EVALS, tail_lines: int = CONSOLE_TAIL_LINES
) -> Dict[str, float]:
    """{"actor/compute_time/min": ..., "evaluator/compute_time/max": ...,
    ...} - the mean of the last `n_tail` ACTOR/EVALUATOR eval checkpoints'
    compute_time min and max, read back from the run's console log (see
    module docstring for why min/max aren't in wandb's structured history).
    Missing keys mean nothing was found (e.g. the run has no console log,
    or ACTOR never logged compute_time in the fetched tail)."""
    mins: Dict[str, List[float]] = {"actor": [], "evaluator": []}
    maxs: Dict[str, List[float]] = {"actor": [], "evaluator": []}
    try:
        lines = list(run.console_logs(last=tail_lines))
    except Exception as e:
        print(f"    (console log fetch failed for {run.id}: {e})")
        lines = []
    for line in lines:
        parsed = parse_console_log_line(line.content)
        if parsed is None:
            continue
        label, stats = parsed
        prefix = _CONSOLE_EVENT_PREFIX.get(label)
        if prefix is None:
            continue
        if "compute_time_min" in stats:
            mins[prefix].append(stats["compute_time_min"])
        if "compute_time_max" in stats:
            maxs[prefix].append(stats["compute_time_max"])
    result: Dict[str, float] = {}
    for prefix in ("actor", "evaluator"):
        if mins[prefix]:
            result[f"{prefix}/compute_time/min"] = float(np.mean(mins[prefix][-n_tail:]))
        if maxs[prefix]:
            result[f"{prefix}/compute_time/max"] = float(np.mean(maxs[prefix][-n_tail:]))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--out", default="ramdp_experiments/wandb_cache_hd64.csv")
    args = parser.parse_args()

    print(f"Fetching run list from {args.project} (hidden_dim=64) ...")
    all_rows = fetch_run_metas(args.project)
    print(f"  {len(all_rows)} runs match hidden_dim=64")
    deduped = dedupe_latest(all_rows)
    print(f"  {len(deduped)} runs after dedup by full config identity")

    frames = []
    for i, (run, meta) in enumerate(deduped):
        hist = run.history(keys=METRICS, pandas=True)
        if hist.empty:
            print(f"  [{i+1}/{len(deduped)}] {run.id} ({meta.arch}, {meta.state}): no history, skipping")
            continue
        hist = hist.reset_index(drop=True)
        hist["eval_idx"] = hist.index
        for col, value in fetch_compute_time_extrema(run).items():
            hist[col] = value
        hist["arch"] = meta.arch
        hist["qac_variant"] = meta.qac_variant
        hist["min_steps"] = meta.min_steps
        hist["max_steps"] = meta.max_steps
        hist["sgh"] = meta.sgh
        hist["vocab_size"] = meta.vocab_size
        hist["total_timesteps"] = meta.total_timesteps
        hist["seed"] = meta.seed
        hist["run_id"] = meta.run_id
        hist["state"] = meta.state
        frames.append(hist)
        print(f"  [{i+1}/{len(deduped)}] {run.id} ({meta.arch}, seed={meta.seed}): {len(hist)} rows")

    df = pd.concat(frames, ignore_index=True)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} rows from {len(frames)} runs to {args.out}")


if __name__ == "__main__":
    main()
