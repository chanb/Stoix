#!/usr/bin/env python
"""Fetch actor/{episode_return,episode_discounted_return,compute_time} training
curves from the wandb project `bpychan-university-of-alberta/lightsout-sep7`,
restricted to `network.actor_network.pre_torso.hidden_dim == 64`, and cache them
to a local parquet file for plotting (see plot_wandb_lightsout_hd64.py).

The project mixes runs logged under a couple of config schema versions (e.g.
`stop_gradient_halting_input` / `halting_ent_coef` only exist on newer runs,
and some (arch, qac_variant, budget, seed) combos were relaunched and so have
duplicate rows). This script:
  - normalizes `stop_gradient_halting_input` (absent == False)
  - dedupes by (arch, qac_variant, min_steps, max_steps, sgh, vocab_size,
    total_timesteps, seed), keeping the most-recently-created run,
    preferring `finished` over other states (IRU-ACT has both 1e8- and
    3e8-timestep runs for the same otherwise-matching config; without this
    axis they'd collide in the dedup key and get silently mixed)

Usage:
  python ramdp_experiments/fetch_wandb_lightsout_hd64.py --out wandb_cache_hd64.parquet
"""

from __future__ import annotations

import argparse
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
    created_at: str
    state: str


def fetch_run_metas(project: str) -> List[Tuple["wandb.apis.public.Run", RunMeta]]:
    api = wandb.Api()
    runs = api.runs(project, filters={"config.network.actor_network.pre_torso.hidden_dim": "64"})
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
                    created_at=str(r.created_at),
                    state=r.state,
                ),
            )
        )
    return out


def dedupe_latest(
    rows: List[Tuple["wandb.apis.public.Run", RunMeta]]
) -> List[Tuple["wandb.apis.public.Run", RunMeta]]:
    """Keep, per (arch, qac_variant, min_steps, max_steps, sgh, seed), the
    best run: finished beats non-finished, then most-recently-created wins."""
    best: Dict[tuple, Tuple["wandb.apis.public.Run", RunMeta]] = {}
    for run, meta in rows:
        key = (
            meta.arch,
            meta.qac_variant,
            meta.min_steps,
            meta.max_steps,
            meta.sgh,
            meta.vocab_size,
            meta.total_timesteps,
            meta.seed,
        )
        cur = best.get(key)
        if cur is None:
            best[key] = (run, meta)
            continue
        _, cur_meta = cur
        rank = (meta.state == "finished", meta.created_at)
        cur_rank = (cur_meta.state == "finished", cur_meta.created_at)
        if rank > cur_rank:
            best[key] = (run, meta)
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
    print(f"  {len(deduped)} runs after dedup by (arch, qac_variant, budget, sgh, seed)")

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
        frames.append(hist)
        print(f"  [{i+1}/{len(deduped)}] {run.id} ({meta.arch}, seed={meta.seed}): {len(hist)} rows")

    df = pd.concat(frames, ignore_index=True)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} rows from {len(frames)} runs to {args.out}")


if __name__ == "__main__":
    main()
