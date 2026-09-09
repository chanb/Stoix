#!/usr/bin/env python
"""Fetch actor/{episode_return,episode_discounted_return,compute_time} training
curves from the wandb project `bpychan-university-of-alberta/lightsout-sep7`,
restricted to `network.actor_network.pre_torso.hidden_dim == 64`, and cache them
to a local parquet file for plotting (see plot_wandb_lightsout_hd64.py).

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
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd
import wandb

PROJECT = "bpychan-university-of-alberta/lightsout-sep7"

METRICS = [
    "actor/episode_return/mean",
    "actor/episode_discounted_return/mean",
    "actor/compute_time/mean",
]

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
