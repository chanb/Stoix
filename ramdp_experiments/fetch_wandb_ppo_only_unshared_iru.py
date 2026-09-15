#!/usr/bin/env python
"""Fetch actor/evaluator episode-return training curves for the
`lightsout_sweep-ppo_only` wandb project, restricted to
network.actor_network.pre_torso._target_ ==
stoix.networks.torso_compute.UnsharedIRUAdaptiveComputationTimeTorso and
system.qac_variant == "reinforce" (the project's only qac_variant - see
experiments.md's "IRU without parameter sharing" section: grid 3x3,
hidden_dim=16, num_layers=1, 1e8 timesteps, budgets {1,2,4,8,16} fixed plus
adaptive[1-16], 10 seeds each - 60 finished runs total as of writing).

Unlike lightsout-sep7 (fetch_wandb_lightsout_hd64.py), this project's
actor/* episode metrics never include compute_time - only
episode_return/episode_discounted_return/episode_length. In ff_ppo.py's
hydra_entry_point, the actor's per-rollout episode_metrics come from
`get_final_step_metrics(learner_output.episode_metrics)`, which has no
compute_time. So this script instead fetches trainer/compute_time - the
mean realised compute_time over the training rollout/minibatches
(train_metrics, logged every eval_step under LogEvent.TRAIN) - and
downstream plotting uses it as the x-axis for both the actor-return and
evaluator-return panels.

Runs are deduped by full config identity (config_identity_key - everything
except the `logger` subtree), same as fetch_wandb_lightsout_hd64.py, in case
a config was relaunched.

Usage:
  python ramdp_experiments/fetch_wandb_ppo_only_unshared_iru.py \\
      --out ramdp_experiments/wandb_cache_ppo_only_unshared_iru.csv
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd
import wandb

PROJECT = "bpychan-university-of-alberta/lightsout_sweep-ppo_only"
TARGET = "stoix.networks.torso_compute.UnsharedIRUAdaptiveComputationTimeTorso"
ARCH = "Unshared-IRU-ACT"
VOCAB_SIZE_NA = -1  # sentinel: this architecture has no vocab_size axis

METRICS = [
    "actor/episode_return/mean",
    "actor/episode_discounted_return/mean",
    "evaluator/episode_return/mean",
    "evaluator/episode_discounted_return/mean",
    "trainer/compute_time",
]

# Config subtrees that are logging plumbing, not experiment identity.
IDENTITY_EXCLUDE_KEYS = {"logger"}


def cfg_get(c: dict, path: str, default=None):
    cur = c
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def config_identity_key(c: dict) -> str:
    """Canonical string identity for a run's config: every field except
    `logger`. Two runs get the same key iff every hyperparameter matches."""
    identity = {k: v for k, v in c.items() if k not in IDENTITY_EXCLUDE_KEYS}
    return json.dumps(identity, sort_keys=True)


@dataclass
class RunMeta:
    run_id: str
    min_steps: int
    max_steps: int
    seed: int
    total_timesteps: int
    config_key: str
    created_at: str
    state: str


def fetch_run_metas(project: str) -> List[Tuple["wandb.apis.public.Run", RunMeta]]:
    api = wandb.Api()
    runs = api.runs(
        project,
        filters={
            "config.network.actor_network.pre_torso._target_": TARGET,
            "config.system.qac_variant": "reinforce",
        },
    )
    out = []
    for r in runs:
        c = r.config
        mn = cfg_get(c, "network.actor_network.pre_torso.min_steps")
        mx = cfg_get(c, "network.actor_network.pre_torso.max_steps")
        if mn is None or mx is None:
            continue
        total_timesteps_raw = cfg_get(c, "arch.total_timesteps")
        seed = cfg_get(c, "arch.seed")
        if total_timesteps_raw is None or seed is None:
            continue
        out.append(
            (
                r,
                RunMeta(
                    run_id=r.id,
                    min_steps=int(mn),
                    max_steps=int(mx),
                    seed=int(seed),
                    total_timesteps=int(float(total_timesteps_raw)),
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
    """Keep, per identical config, the best run: finished beats non-finished,
    then most-recently-created wins."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--out", default="ramdp_experiments/wandb_cache_ppo_only_unshared_iru.csv")
    args = parser.parse_args()

    print(f"Fetching run list from {args.project} ({TARGET}, qac_variant=reinforce) ...")
    all_rows = fetch_run_metas(args.project)
    print(f"  {len(all_rows)} runs match")
    deduped = dedupe_latest(all_rows)
    print(f"  {len(deduped)} runs after dedup by full config identity")

    frames = []
    for i, (run, meta) in enumerate(deduped):
        hist = run.history(keys=METRICS, pandas=True)
        if hist.empty:
            print(f"  [{i+1}/{len(deduped)}] {run.id} ({meta.state}): no history, skipping")
            continue
        hist = hist.reset_index(drop=True)
        hist["eval_idx"] = hist.index
        hist["arch"] = ARCH
        hist["qac_variant"] = "reinforce"
        hist["min_steps"] = meta.min_steps
        hist["max_steps"] = meta.max_steps
        hist["sgh"] = False
        hist["vocab_size"] = VOCAB_SIZE_NA
        hist["total_timesteps"] = meta.total_timesteps
        hist["seed"] = meta.seed
        hist["run_id"] = meta.run_id
        hist["state"] = meta.state
        frames.append(hist)
        print(
            f"  [{i+1}/{len(deduped)}] {run.id} (seed={meta.seed}, "
            f"budget={meta.min_steps}-{meta.max_steps}): {len(hist)} rows"
        )

    df = pd.concat(frames, ignore_index=True)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} rows from {len(frames)} runs to {args.out}")


if __name__ == "__main__":
    main()
