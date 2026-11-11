"""Success rate vs hardness of the fixed-budget N=8 models, forced to c = 1..8 CoT steps at test time.

Usage (from the repo root): python ramdp_experiments/forced_budget.py {lightsout,slidingpuzzle} {train,test} [N=8|var]  (var = adaptive mn1-mx8 run; c=0 rows = its own greedy halting)
"""
import sys

import numpy as np
import pandas as pd

from hardness_rollout import (FIXED8_CKPT, FIXED_CKPT, NUM_EPISODES, REPO_ROOT, RUNS, TEST_MAX_STEPS, build, forced_act_fn,
                              gf2_min_weight, rollout, sliding_bfs)

BEST_SEED = {("lightsout", 2): 3, ("slidingpuzzle", 2): 4, ("lightsout", 1): 2, ("lightsout", 4): 4, ("slidingpuzzle", 1): 3, ("slidingpuzzle", 4): 2}

name, split = sys.argv[1], sys.argv[2]
N = sys.argv[3] if len(sys.argv) > 3 else "8"
if N == "var":
    ckpt, train_steps = RUNS[name]["ckpt"], 8
else:
    N = int(N)
    ckpt = FIXED8_CKPT[name] if N == 8 else FIXED_CKPT[(name, N)].format(BEST_SEED[(name, N)])
    train_steps = N
env, halting_act_fn, params, actor = build(name, ckpt, split, train_steps=train_steps)
max_steps = RUNS[name]["max_steps"] if split == "train" else TEST_MAX_STEPS[name]
if name == "lightsout":
    from stoix.envs.lightsout.lightsout_env import LightsOutEnv, default_config

    cfg = default_config()
    cfg.m, cfg.n = 5, 4
    A = np.array(LightsOutEnv(cfg, eval=False).action_matrix)
    board, goal = (lambda s: s.data.grid), (lambda s: s.data.goal)
else:
    dist = sliding_bfs(3)
    board, goal = (lambda s: s.puzzle), (lambda s: s.puzzle * 0)

rows = []
for c in ([0] if N == "var" else []) + list(range(1, 9)):
    b = rollout(env, forced_act_fn(actor, c) if c else halting_act_fn, params, max_steps, board, goal)
    if not rows:  # same keys for every c -> same instances
        if name == "lightsout":
            hard = [gf2_min_weight(A, np.logical_xor(i, g).reshape(-1)) for i, g in zip(b["initial"], b["goal"])]
        else:
            hard = [dist.get(tuple(p.reshape(-1).astype(int))) for p in b["initial"]]
    rows.append(pd.DataFrame({"c": c, "hardness": hard, "solved": b["solved"], "episode_length": b["episode_length"]}))
    print(f"c={c} solved={b['solved'].mean():.3f}", flush=True)
out = REPO_ROOT / "ramdp_experiments" / f"forced_budget-{name}-{split}-cnn16x4-N{N}-n{NUM_EPISODES}.pkl"
pd.concat(rows).to_pickle(out)
print("Saved", out)
