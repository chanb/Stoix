#!/usr/bin/env python
"""Fetch wandb data based on filters
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

METRICS = [
    "actor/episode_return/mean",
    "actor/episode_discounted_return/mean",
    "actor/compute_time/mean",
    "actor/episode_length/mean",
    "evaluator/episode_return/mean",
    "evaluator/episode_discounted_return/mean",
    "evaluator/compute_time/mean",
    "evaluator/episode_length/mean",
]

# The "absolute metric" (see stoix/evaluator.py's get_ff_evaluator_fn docstring): logged exactly
# once per run, at the very end of training, by re-evaluating the best-performing checkpoint for 10x
# as many episodes as a single actor/evaluator eval step - a much less noisy final-performance
# estimate than evaluator/* (which is itself still a fine per-checkpoint curve, just noisier at any
# single eval_idx).
ABSOLUTE_METRICS = [
    "absolute/episode_return/mean",
    "absolute/episode_discounted_return/mean",
    "absolute/compute_time/mean",
    "absolute/episode_length/mean",
]

# Metrics only some runs log - solved_episode exists only for envs that set
# `solved_final_reward_threshold` (Sokoban, sliding tile; not Lights Out), see
# stoix.systems.ramdp_vpg.ramdp_vpg_types.solved_episode_info.
OPTIONAL_METRICS = [
    "actor/solved_episode/mean",
    "evaluator/solved_episode/mean",
]
OPTIONAL_ABSOLUTE_METRICS = [
    "absolute/solved_episode/mean",
]

# How many of the most recent console log lines to pull per run when looking for the tail few eval
# checkpoints' compute_time min/max (see fetch_compute_time_extrema).
CONSOLE_TAIL_LINES = 500
N_TAIL_EVALS = 3  # matches compute_final_values' n_tail default

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_CONSOLE_EVENT_PREFIX = {"ACTOR": "actor", "EVALUATOR": "evaluator"}

ARCH_SHORT = {
    "stoix.networks.torso_compute.IRUAdaptiveComputationTimeTorso": "IRU-ACT",
    "stoix.networks.torso_compute.UnsharedIRUAdaptiveComputationTimeTorso": "IRU-ACT",
    "stoix.networks.torso_compute_explicit_cot.TransformerExplicitCoTTorso": "Transformer-ExplicitCoT",
    "stoix.networks.torso_compute_transformer.TransformerChainOfThoughtTorso": "Transformer-CoT",
    "stoix.networks.torso_compute_explicit_cot_merged.TransformerMergedActionCoTTorso": "Transformer-ExplicitCoT",
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
    """Canonical string identity for a run's config: every field except `logger`."""
    identity = {k: v for k, v in c.items() if k not in IDENTITY_EXCLUDE_KEYS}
    return json.dumps(identity, sort_keys=True)


@dataclass
class RunMeta:
    run_id: str
    arch: str
    min_steps: int
    max_steps: int
    sgh: bool
    vocab_size: int
    total_timesteps: int
    seed: int
    ent_coef: float
    gamma: float
    halting_temperature: float
    halting_ent_coef: float
    actor_lr: float
    clip_halting_head: bool
    halting_lr: float
    config_key: str
    created_at: str
    state: str


def lightsout_filter(c):
    mn = cfg_get(c, "network.actor_network.pre_torso.min_steps")
    mx = cfg_get(c, "network.actor_network.pre_torso.max_steps")
    if int(mn) == int(mx):
        return False
    
    halt_lr = cfg_get(c, "system.halting_lr")
    halt_temp = cfg_get(c, "network.actor_network.pre_torso.halting_temperature")
    stop_grad = cfg_get(c, "network.actor_network.pre_torso.stop_gradient_halting_input")
    clip_halt = cfg_get(c, "system.clip_halting_head")
    if (
        (
            halt_lr is None
            and (halt_temp is not None and np.isclose(float(halt_temp), 5.0))
            and (stop_grad is not None and stop_grad == "False")
            and clip_halt is None
        )
        or (halt_lr is not None and np.isclose(float(halt_lr), 0.001))
        or (halt_lr is not None and np.isclose(float(halt_lr), 0.0001))
    ):
        return False
    return True

def slidingpuzzle_filter(c):
    mn = cfg_get(c, "network.actor_network.pre_torso.min_steps")
    mx = cfg_get(c, "network.actor_network.pre_torso.max_steps")
    if int(mn) == int(mx):
        return False

    halt_lr = cfg_get(c, "system.halting_lr")
    halt_ent_coef = cfg_get(c, "system.halting_ent_coef")
    clip_halt = cfg_get(c, "system.clip_halting_head")
    halt_temp = cfg_get(c, "network.actor_network.pre_torso.halting_temperature")

    if (
        halt_lr is not None
        and halt_ent_coef is not None
        and clip_halt is not None
        and halt_temp is not None
        and np.isclose(float(halt_lr), 0.0001)
        and np.isclose(float(halt_ent_coef), 0.0)
        and np.isclose(float(halt_temp), 1.0)
        and clip_halt == "True"
    ):
        return False
    return True

def sokoban_filter(c):
    mn = cfg_get(c, "network.actor_network.pre_torso.min_steps")
    mx = cfg_get(c, "network.actor_network.pre_torso.max_steps")
    if int(mn) == int(mx):
        return False

    halt_lr = cfg_get(c, "system.halting_lr")
    halt_ent_coef = cfg_get(c, "system.halting_ent_coef")
    clip_halt = cfg_get(c, "system.clip_halting_head")

    if (
        halt_lr is not None
        and halt_ent_coef is not None
        and clip_halt is not None
        and np.isclose(float(halt_lr), 0.0001)
        and np.isclose(float(halt_ent_coef), 0.0)
        and clip_halt == "True"
    ):
        return False
    return True


def fetch_run_metas(project: str) -> List[Tuple["wandb.apis.public.Run", RunMeta]]:
    api = wandb.Api()
    runs = api.runs(project, filters={
        
    })
    out = []

    if project == "lightsout-icot-icot_sweep-qkv":
        run_filter = lightsout_filter
    elif project == "slidingpuzzle-icot_sweep_2":
        run_filter = slidingpuzzle_filter
    elif project == "sokoban-shallow_cnn":
        run_filter = sokoban_filter
    else:
        run_filter = lambda c: True
    for r in runs:
        c = r.config
        arch_full = cfg_get(c, "network.actor_network.pre_torso._target_")
        if arch_full not in ARCH_SHORT:
            continue
        mn = cfg_get(c, "network.actor_network.pre_torso.min_steps")
        mx = cfg_get(c, "network.actor_network.pre_torso.max_steps")
        if mn is None or mx is None:
            continue
        if run_filter(c):
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
        ent_coef_raw = cfg_get(c, "system.ent_coef")
        gamma_raw = cfg_get(c, "system.gamma")
        halting_temperature_raw = cfg_get(c, "network.actor_network.pre_torso.halting_temperature")
        halting_ent_coef_raw = cfg_get(c, "system.halting_ent_coef")
        actor_lr_raw = cfg_get(c, "system.actor_lr")
        halting_lr_raw = cfg_get(c, "system.halting_lr")
        # Default True matches stoix/configs/system/ramdp_vpg/ff_ppo.yaml.
        clip_halting_head = str(cfg_get(c, "system.clip_halting_head", True)) == "True"
        out.append(
            (
                r,
                RunMeta(
                    run_id=r.id,
                    arch=ARCH_SHORT[arch_full],
                    min_steps=int(mn),
                    max_steps=int(mx),
                    sgh=sgh,
                    vocab_size=vocab_size,
                    total_timesteps=int(float(total_timesteps_raw)),
                    seed=int(seed),
                    ent_coef=float(ent_coef_raw) if ent_coef_raw is not None else float("nan"),
                    gamma=float(gamma_raw) if gamma_raw is not None else float("nan"),
                    halting_temperature=(
                        float(halting_temperature_raw) if halting_temperature_raw is not None else 1.0
                    ),
                    halting_ent_coef=(
                        float(halting_ent_coef_raw) if halting_ent_coef_raw is not None else 0.0
                    ),
                    actor_lr=float(actor_lr_raw) if actor_lr_raw is not None else float("nan"),
                    clip_halting_head=clip_halting_head,
                    halting_lr=float(halting_lr_raw) if halting_lr_raw is not None else float("nan"),
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
    `ConsoleLogger.log_dict`), e.g. "EVALUATOR - Compute time mean: 3.480 | Compute time min:
    2.200 | ..." into (event_label, {stat_key: value}), e.g. ("EVALUATOR",
    {"compute_time_mean": 3.48, "compute_time_min": 2.2, ...})."""
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
    """{"actor/compute_time/min": ..., "evaluator/compute_time/max": ..., ...} - the mean of the
    last `n_tail` ACTOR/EVALUATOR eval checkpoints' compute_time min and max, read back from
    the run's console log (see module docstring for why min/max aren't in wandb's structured
    history)."""
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


def fetch_absolute_metrics(run: "wandb.apis.public.Run") -> Dict[str, float]:
    """{"absolute/episode_return/mean": ..., ...} - the single post-training absolute-metric row
    (see ABSOLUTE_METRICS), fetched with its own `run.history` call so it doesn't collapse the
    main per-eval METRICS query (see ABSOLUTE_METRICS docstring)."""
    hist = run.history(keys=ABSOLUTE_METRICS, pandas=True)
    if hist.empty:
        return {}
    row = hist.iloc[-1]
    result = {col: float(row[col]) for col in ABSOLUTE_METRICS if col in row and pd.notna(row[col])}
    for col in OPTIONAL_ABSOLUTE_METRICS:
        opt = run.history(keys=[col], pandas=True)
        if not opt.empty and col in opt and pd.notna(opt[col].iloc[-1]):
            result[col] = float(opt[col].iloc[-1])
    return result


def merge_optional_metrics(run: "wandb.apis.public.Run", hist: pd.DataFrame) -> pd.DataFrame:
    """Left-join each OPTIONAL_METRICS series the run logged onto the per-eval
    `hist` rows by `_step` (they're logged on the same row as METRICS)."""
    for col in OPTIONAL_METRICS:
        opt = run.history(keys=[col], pandas=True)
        if opt.empty or col not in opt:
            continue
        hist = hist.merge(opt[["_step", col]], on="_step", how="left")
    return hist


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, type=str)
    parser.add_argument("--out", required=True, type=str)
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
        hist = merge_optional_metrics(run, hist.reset_index(drop=True))
        hist["eval_idx"] = hist.index
        for col, value in fetch_compute_time_extrema(run).items():
            hist[col] = value
        for col, value in fetch_absolute_metrics(run).items():
            hist[col] = value
        hist["arch"] = meta.arch
        hist["min_steps"] = meta.min_steps
        hist["max_steps"] = meta.max_steps
        hist["sgh"] = meta.sgh
        hist["vocab_size"] = meta.vocab_size
        hist["total_timesteps"] = meta.total_timesteps
        hist["seed"] = meta.seed
        hist["ent_coef"] = meta.ent_coef
        hist["gamma"] = meta.gamma
        hist["halting_temperature"] = meta.halting_temperature
        hist["halting_ent_coef"] = meta.halting_ent_coef
        hist["actor_lr"] = meta.actor_lr
        hist["clip_halting_head"] = meta.clip_halting_head
        hist["halting_lr"] = meta.halting_lr
        hist["run_id"] = meta.run_id
        hist["state"] = meta.state
        frames.append(hist)
        print(f"  [{i+1}/{len(deduped)}] {run.id} ({meta.arch}, seed={meta.seed}): {len(hist)} rows")

    df = pd.concat(frames, ignore_index=True)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} rows from {len(frames)} runs to {args.out}")


if __name__ == "__main__":
    main()
