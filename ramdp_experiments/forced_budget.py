"""Success rate vs hardness of the fixed-budget N=8 models, forced to c = 1..8 CoT steps at test time.

Usage (from the repo root): python ramdp_experiments/forced_budget.py {lightsout,slidingpuzzle} {train,test}
"""
import sys

import numpy as np
import pandas as pd

from hardness_rollout import (FIXED8_CKPT, NUM_EPISODES, REPO_ROOT, RUNS, TEST_MAX_STEPS, build, forced_act_fn,
                              gf2_min_weight, rollout, sliding_bfs)

name, split = sys.argv[1], sys.argv[2]
env, _, params, actor = build(name, FIXED8_CKPT[name], split)
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
for c in range(1, 9):
    b = rollout(env, forced_act_fn(actor, c), params, max_steps, board, goal)
    if c == 1:  # same keys for every c -> same instances
        if name == "lightsout":
            hard = [gf2_min_weight(A, np.logical_xor(i, g).reshape(-1)) for i, g in zip(b["initial"], b["goal"])]
        else:
            hard = [dist.get(tuple(p.reshape(-1).astype(int))) for p in b["initial"]]
    rows.append(pd.DataFrame({"c": c, "hardness": hard, "solved": b["solved"], "episode_length": b["episode_length"]}))
    print(f"c={c} solved={b['solved'].mean():.3f}", flush=True)
out = REPO_ROOT / "ramdp_experiments" / f"forced_budget-{name}-{split}-cnn16x4-N8-n{NUM_EPISODES}.pkl"
pd.concat(rows).to_pickle(out)
print("Saved", out)
