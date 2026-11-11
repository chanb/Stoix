"""plot_data-<name>.pkl for make_tsne_step_grid_plot.py, from a hardness_rollout.py train rollout cache.

Same recipe as the t-SNE export cell of analysis-compute_time-hardness-slidingpuzzle.ipynb: 3000 episodes
sampled with random_state=0, the latent after every CoT step 1..max_steps of each episode's initial puzzle
(states_history, replaying max_steps steps), coloured by first-step compute time and shortest-path hardness.

Usage (from the repo root): python ramdp_experiments/tsne_plot_data.py slidingpuzzle-bankfinal
"""
import pickle
import sys

import jax.numpy as jnp
import numpy as np
import pandas as pd

from hardness_rollout import NUM_EPISODES, REPO_ROOT, RUNS, SEED, build

name = sys.argv[1]
env, _, params, actor = build(name)
df = pd.read_pickle(REPO_ROOT / "ramdp_experiments" / f"rollout_cache-{name}-{RUNS[name]['tag']}-n{NUM_EPISODES}-seed{SEED}.pkl")
sample = df.sample(n=3000, random_state=0)
w = env
while not hasattr(w, "_prepare"):  # GridObservationWrapper: puzzle -> one-hot obs
    w = w._env
obs = w._prepare(jnp.asarray(np.stack([s[0] for s in sample["states"]])))
max_steps = actor.torso.max_steps
states_history = actor.apply(params, obs, torso_kwargs={"target_compute_time": jnp.full((len(obs),), float(max_steps))})[2]
plot_data = {
    "env_label": "slidingtile-3x3" if name.startswith("sliding") else "lightsout-5x4",
    "max_steps": max_steps,
    "tsne_sample_hardness": sample["shortest_path_length"].to_numpy(),
    "tsne_sample_compute_time": np.array([ct[0] for ct in sample["compute_times"]]),
    "step_latents": np.asarray(states_history),  # (n, max_steps, hidden_dim)
}
out = REPO_ROOT / "ramdp_experiments" / f"plot_data-{name}.pkl"
pickle.dump(plot_data, open(out, "wb"))
print("Saved", out, plot_data["step_latents"].shape, "compute time counts:",
      dict(zip(*np.unique(plot_data["tsne_sample_compute_time"], return_counts=True))))
