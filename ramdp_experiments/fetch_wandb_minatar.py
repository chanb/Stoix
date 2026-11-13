#!/usr/bin/env python
"""Fetch MinAtar runs (one gamma) to a CSV, reusing fetch_wandb's run parsing plus an `env` column.

python fetch_wandb_minatar.py --project minatar-icot-nov06-g0.999 --gamma 0.99999 --out minatar-icot-g0.99999.csv
"""

import argparse

import numpy as np
import pandas as pd
import wandb

import fetch_wandb as fw


def fetch_run_metas(project: str, gamma: float):
    """(run, RunMeta) for every run in `project` at `gamma`. fw.fetch_run_metas can't be reused
    as-is: its fallback run_filter drops every run of a project it doesn't know."""
    out = []
    for r in wandb.Api().runs(project):
        c = r.config
        g = lambda path, default=None: fw.cfg_get(c, path, default)
        arch_full = g("network.actor_network.pre_torso._target_")
        if arch_full not in fw.ARCH_SHORT or g("system.gamma") is None:
            continue
        if not np.isclose(float(g("system.gamma")), gamma, rtol=0, atol=1e-9):
            continue
        num = lambda path, default=float("nan"): float(g(path)) if g(path) is not None else default
        out.append((r, fw.RunMeta(
            run_id=r.id,
            arch=fw.ARCH_SHORT[arch_full],
            min_steps=int(g("network.actor_network.pre_torso.min_steps")),
            max_steps=int(g("network.actor_network.pre_torso.max_steps")),
            sgh=str(g("network.actor_network.pre_torso.stop_gradient_halting_input", False)) == "True",
            vocab_size=int(num("network.actor_network.pre_torso.vocab_size", fw.VOCAB_SIZE_NA)),
            total_timesteps=int(num("arch.total_timesteps")),
            seed=int(g("arch.seed")),
            ent_coef=num("system.ent_coef"),
            gamma=num("system.gamma"),
            halting_temperature=num("network.actor_network.pre_torso.halting_temperature", 1.0),
            halting_ent_coef=num("system.halting_ent_coef", 0.0),
            actor_lr=num("system.actor_lr"),
            clip_halting_head=str(g("system.clip_halting_head", True)) == "True",
            halting_lr=num("system.halting_lr"),
            config_key=fw.config_identity_key(c),
            created_at=str(r.created_at),
            state=r.state,
        )))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, type=str)
    parser.add_argument("--gamma", required=True, type=float)
    parser.add_argument("--out", required=True, type=str)
    args = parser.parse_args()

    rows = fw.dedupe_latest(fetch_run_metas(args.project, args.gamma))
    print(f"{len(rows)} runs with gamma={args.gamma}")

    frames = []
    for i, (run, meta) in enumerate(rows):
        hist = run.history(keys=fw.METRICS, pandas=True, samples=10000)
        if hist.empty:
            print(f"  [{i+1}/{len(rows)}] {run.id}: no history, skipping")
            continue
        hist = hist.sort_values("_step").reset_index(drop=True)
        hist["eval_idx"] = hist.index
        for col, value in fw.fetch_absolute_metrics(run).items():
            hist[col] = value
        hist["env"] = fw.cfg_get(run.config, "env.scenario.name")
        for k, v in vars(meta).items():
            if k != "config_key":
                hist[k] = v
        frames.append(hist)
        print(f"  [{i+1}/{len(rows)}] {run.id} {hist['env'].iat[0]} [{meta.min_steps}-{meta.max_steps}] seed={meta.seed}: {len(hist)} rows")

    df = pd.concat(frames, ignore_index=True)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} rows from {len(frames)} runs to {args.out}")


if __name__ == "__main__":
    main()
