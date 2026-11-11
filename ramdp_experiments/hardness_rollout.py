"""Rolls out a trained checkpoint on its training distribution and saves a rollout cache
(per-episode hardness + per-step actions/compute times) for make_hardness_grid_plot.py.

Script form of the rollout/solver cells of analysis-compute_time-hardness{,-slidingpuzzle}.ipynb,
for the cnn16x4 concat runs in sem-round2 (best seed by final
evaluator/train/episode_discounted_return).

Usage (from the repo root): python ramdp_experiments/hardness_rollout.py {lightsout,slidingpuzzle}
"""

import collections
import sys
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from hydra import compose, initialize_config_dir
from omegaconf import open_dict

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from stoix.base_types import ActorCriticParams  # noqa: E402
from stoix.networks.base import FeedForwardCritic  # noqa: E402
from stoix.networks.base_compute import FeedForwardActorWithComputeTime as Actor  # noqa: E402
from stoix.systems.ramdp_vpg.ff_ppo import get_distribution_act_fn_with_compute_time  # noqa: E402
from stoix.utils.checkpointing import Checkpointer  # noqa: E402
from stoix.utils.make_env import make as make_env, make_train_eval_env  # noqa: E402

SEM = "/home/bryanpu1/scratch/logs/ramdp/sem-round2"
NUM_EPISODES = 50000
SEED = 0

# Overrides shared by both runs (copied from their launch commands).
COMMON = """network=cnn_transformer_compute system.gamma=0.999
network.actor_network.pre_torso.hidden_dim=128 ++network.actor_network.pre_torso.num_layers=2
network.actor_network.pre_torso.max_steps=8 network.actor_network.pre_torso.min_steps=1
system.halting_loss_type=pg ++network.actor_network.pre_torso.num_heads=4 ++network.actor_network.pre_torso.mlp_dim=512
++network.actor_network.pre_torso.use_sandwich_norm=False ++network.actor_network.pre_torso.use_rmsnorm=True
++network.actor_network.input_layer.channel_tokens=True ++network.actor_network.pre_torso.input_is_tokens=True
++network.actor_network.pre_torso.sem_group_size=0 ++network.actor_network.pre_torso.latent_rms_norm=True
++network.actor_network.pre_torso.sem_readout_only=False ++network.actor_network.pre_torso.input_injection=True
++network.actor_network.pre_torso.residual_scaling=False ++network.actor_network.pre_torso.block_init=trunc_normal
network.critic_network.pre_torso.layer_sizes=[256,256,256] network.critic_network.pre_torso.activation=gelu
++network.actor_network.pre_torso.use_input_layer_norm=True ++network.actor_network.pre_torso.stop_gradient_halting_input=False
network.actor_network.pre_torso.halting_temperature=1 ++network.actor_network.pre_torso.halting_hidden_dims=[]
network.actor_network.input_layer.channel_sizes=[16,4] network.actor_network.input_layer.kernel_sizes=[3,3]
network.actor_network.input_layer.strides=[1,1] network.actor_network.input_layer.activation=gelu
network.critic_network.input_layer.channel_sizes=[32,64] network.critic_network.input_layer.kernel_sizes=[3,3]
network.critic_network.input_layer.strides=[1,1] network.critic_network.input_layer.activation=gelu
network.critic_network.input_layer.hidden_sizes=[] network.actor_network.pre_torso.activation=gelu
++network.actor_network.pre_torso.concat_input_tokens=True ++network.actor_network.pre_torso.action_logit_accum=last
++network.actor_network.pre_torso.action_logit_discount=0.5 arch.evaluation_greedy=True"""

RUNS = {
    "lightsout": dict(
        ckpt=f"{SEM}/lightsout_cnn16x4concat_clip0.2_noresscale_nosem_acclast_var8_hent0_do0.0/last_Nvar_seed_4/checkpoints",
        overrides="""env=lightsout/lightsout_3x3 env.scenario.name=lightsout-5x4 env.scenario.task_name=lightsout_5x4
env.kwargs.episode_length=10 ++env.kwargs.eval_episode_length=20 env.kwargs.difficulty_threshold=0.5
network.actor_network.input_dropout=0.0 ++network.actor_network.pre_torso.context_dropout=0.0""",
        max_steps=10,
    ),
    "slidingpuzzle": dict(
        ckpt=f"{SEM}/slidingpuzzle3x3_cnn16x4concat_clip0.2_acclast_var8_hent0_do0.0/slidingtile_bank-bank-gs3-ppo_reinf-cnn+TF-iCoT-mn1-mx8-g0.999-hd128-lr0.0003-clr0.0003-ec0.01-mgn1-nl2-nh4-md512-cnn16x4-hent0-seed_2/checkpoints",
        overrides="""env=jumanji/slidingtile_bank_grid env.kwargs.generator.grid_size=3 env.eval_kwargs.generator.grid_size=3
network.actor_network.input_layer.hidden_sizes=[128]""",
        max_steps=40,
    ),
}


FIXED8_CKPT = {  # fixed budget N=8 twins of RUNS (same overrides except min_steps=8); best seed by final
    # evaluator/train/episode_discounted_return. Weights load into the min_steps=1 module unchanged.
    "lightsout": f"{SEM}/lightsout_cnn16x4concat_clip0.2_noresscale_nosem_acclast-sum_fixed_do0.0/last_N8_seed_4/checkpoints",
    "slidingpuzzle": f"{SEM}/slidingpuzzle3x3_cnn16x4concat_clip0.2_acclast_fixed_do0.0/slidingtile_bank-bank-gs3-ppo_reinf-cnn+TF-iCoT-b8-g0.999-hd128-lr0.0003-clr0.0003-ec0.01-mgn1-nl2-nh4-md512-4ba1ea16-seed_0/checkpoints",
}


def forced_act_fn(actor, c):
    """Greedy action after exactly `c` CoT steps (replay mode with target_compute_time=c)."""

    def act(params, obs, key):
        ct = jnp.full((jax.tree_util.tree_leaves(obs)[0].shape[0],), float(c))
        pi = actor.apply(params, obs, torso_kwargs={"target_compute_time": ct})[0]
        return pi.mode(), ct, None, None

    return act


TEST_MAX_STEPS = {"lightsout": 20, "slidingpuzzle": 40}  # eval_episode_length / eval time_limit


FIXED_CKPT = {  # fixed budget N twins of RUNS; best seed by final evaluator/train/episode_discounted_return
    ("lightsout", 1): f"{SEM}/lightsout_cnn16x4concat_clip0.2_noresscale_nosem_acclast-sum_fixed_do0.0/last_N1_seed_{{}}/checkpoints",
    ("lightsout", 4): f"{SEM}/lightsout_cnn16x4concat_clip0.2_noresscale_nosem_acclast-sum_fixed_do0.0/last_N4_seed_{{}}/checkpoints",
    ("lightsout", 2): f"{SEM}/lightsout_cnn16x4concat_clip0.2_noresscale_nosem_acclast-sum_fixed_do0.0/last_N2_seed_{{}}/checkpoints",
    ("slidingpuzzle", 2): f"{SEM}/slidingpuzzle3x3_cnn16x4concat_clip0.2_acclast_fixed_do0.0/slidingtile_bank-bank-gs3-ppo_reinf-cnn+TF-iCoT-b2-g0.999-hd128-lr0.0003-clr0.0003-ec0.01-mgn1-nl2-nh4-md512-fixedN2-seed_{{}}/checkpoints",
    ("slidingpuzzle", 1): f"{SEM}/slidingpuzzle3x3_cnn16x4concat_clip0.2_acclast_fixed_do0.0/slidingtile_bank-bank-gs3-ppo_reinf-cnn+TF-iCoT-b1-g0.999-hd128-lr0.0003-clr0.0003-ec0.01-mgn1-nl2-nh4-md512-39e20ab8-seed_{{}}/checkpoints",
    ("slidingpuzzle", 4): f"{SEM}/slidingpuzzle3x3_cnn16x4concat_clip0.2_acclast_fixed_do0.0/slidingtile_bank-bank-gs3-ppo_reinf-cnn+TF-iCoT-b4-g0.999-hd128-lr0.0003-clr0.0003-ec0.01-mgn1-nl2-nh4-md512-da3ac6f7-seed_{{}}/checkpoints",
}


def pad_pos_embedding(params, max_steps):
    """Grow a N-step model's pos_embedding ((N+1, d), rows 0..N-1 trained) to max_steps+1 rows, reusing
    the last trained row (step N-1) for every step past N, so the 8-step module can run it for c > N."""

    def pad(path, x):
        if "pos_embedding" not in jax.tree_util.keystr(path) or x.shape[0] >= max_steps + 1:
            return x
        n = x.shape[0] - 1
        return jnp.concatenate([x[:n], jnp.repeat(x[n - 1 : n], max_steps + 1 - n, axis=0)])

    return jax.tree_util.tree_map_with_path(pad, params)


def build(name, ckpt=None, split="train", train_steps=8):
    run = RUNS[name]
    with initialize_config_dir(config_dir=str(REPO_ROOT / "stoix/configs/default/anakin"), version_base="1.2"):
        config = compose("default_ramdp_ff_ppo.yaml", overrides=(COMMON + " " + run["overrides"]).split())
    # Both un-batched: training distribution, or the held-out test instances (LightsOut hard goals / test bank).
    env = make_train_eval_env(config) if split == "train" else make_env(config)[1]
    num_actions = int(env.action_space().num_values)
    with open_dict(config):
        config.system.action_dim = num_actions

    a, c = config.network.actor_network, config.network.critic_network
    make_actor = lambda steps: Actor(
        input_layer=hydra.utils.instantiate(a.input_layer),
        torso=hydra.utils.instantiate(a.pre_torso, max_steps=steps, min_steps=1),
        action_head=hydra.utils.instantiate(a.action_head, action_dim=num_actions),
        input_dropout=a.get("input_dropout", 0.0),
    )
    actor, ckpt_actor = make_actor(a.pre_torso.max_steps), make_actor(train_steps)
    critic = FeedForwardCritic(
        input_layer=hydra.utils.instantiate(c.input_layer),
        torso=hydra.utils.instantiate(c.pre_torso),
        critic_head=hydra.utils.instantiate(c.critic_head),
    )
    x = jax.tree_util.tree_map(lambda v: v[None], env.observation_space().generate_value())
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    params = ActorCriticParams(ckpt_actor.init(k1, x, torso_kwargs={"rng": k1}), critic.init(k2, x))
    params, _ = Checkpointer(model_name="ramdp_ff_ppo", rel_dir=ckpt or run["ckpt"], checkpoint_uid="final").restore_params(
        input_params=params
    )
    actor_params = pad_pos_embedding(params.actor_params, a.pre_torso.max_steps)
    return env, get_distribution_act_fn_with_compute_time(config, actor.apply), actor_params, actor


def rollout(env, act_fn, params, max_steps, get_board, get_goal):
    """vmapped fixed-length scan rollouts; env frozen once done (see the notebooks)."""

    def one(key):
        reset_key, key = jax.random.split(key)
        state, ts = env.reset(reset_key)
        board0, goal = get_board(state), get_goal(state)

        def step(carry, _):
            key, state, ts, done = carry
            key, act_key = jax.random.split(key)
            obs = jax.tree_util.tree_map(lambda v: v[None], ts.observation)
            action, ct, _, _ = act_fn(params, obs, act_key)
            new_state, new_ts = env.step(state, action.squeeze(0))
            keep = lambda new, old: jnp.where(done, old, new)
            rec = dict(board=get_board(state), action=action.squeeze(), ct=ct.squeeze(), valid=~done)
            return (key, jax.tree_util.tree_map(keep, new_state, state),
                    jax.tree_util.tree_map(keep, new_ts, ts), done | new_ts.last()), rec

        (_, _, final_ts, _), r = jax.lax.scan(step, (key, state, ts, jnp.array(False)), None, max_steps)
        return dict(initial=board0, goal=goal, states=r["board"], actions=r["action"], compute_times=r["ct"],
                    valid_steps=r["valid"], episode_length=r["valid"].sum(),
                    solved=final_ts.reward.squeeze() >= 1.0)

    keys = jax.random.split(jax.random.PRNGKey(SEED), NUM_EPISODES)
    return jax.tree_util.tree_map(np.asarray, jax.jit(jax.vmap(one))(keys))


# --- Exact hardness solvers (from the notebooks) ---
def gf2_min_weight(A, diff):
    """Min Hamming weight x with A x = diff (mod 2), brute-forcing the kernel."""
    n = A.shape[0]
    M = np.concatenate([A.astype(np.uint8) % 2, diff.astype(np.uint8)[:, None]], axis=1)
    pivots, row = [], 0
    for col in range(n):
        p = next((r for r in range(row, n) if M[r, col]), None)
        if p is None:
            continue
        M[[row, p]] = M[[p, row]]
        for r in range(n):
            if r != row and M[r, col]:
                M[r] ^= M[row]
        pivots.append(col)
        row += 1
    if M[row:, n].any():
        return None
    free = [c for c in range(n) if c not in pivots]
    x0 = np.zeros(n, np.uint8)
    x0[pivots] = M[: len(pivots), n]
    basis = []
    for fc in free:
        v = np.zeros(n, np.uint8)
        v[fc] = 1
        v[pivots] = M[: len(pivots), fc]
        basis.append(v)
    best = int(x0.sum())
    for mask in range(1, 1 << len(basis)):
        v = x0.copy()
        for i, b in enumerate(basis):
            if mask >> i & 1:
                v ^= b
        best = min(best, int(v.sum()))
    return best


def sliding_bfs(n=3):
    solved = tuple(list(range(1, n * n)) + [0])
    dist, q = {solved: 0}, collections.deque([solved])
    while q:
        s = q.popleft()
        b = s.index(0)
        r, c = divmod(b, n)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            if 0 <= r + dr < n and 0 <= c + dc < n:
                t = list(s)
                j = (r + dr) * n + c + dc
                t[b], t[j] = t[j], t[b]
                t = tuple(t)
                if t not in dist:
                    dist[t] = dist[s] + 1
                    q.append(t)
    return dist


def main(name):
    env, act_fn, params, _ = build(name)
    if name == "lightsout":
        from stoix.envs.lightsout.lightsout_env import LightsOutEnv, default_config

        cfg = default_config()
        cfg.m, cfg.n = 5, 4
        A = np.array(LightsOutEnv(cfg, eval=False).action_matrix)
        b = rollout(env, act_fn, params, RUNS[name]["max_steps"], lambda s: s.data.grid, lambda s: s.data.goal)
        hard = [gf2_min_weight(A, np.logical_xor(i, g).reshape(-1)) for i, g in zip(b["initial"], b["goal"])]
    else:
        b = rollout(env, act_fn, params, RUNS[name]["max_steps"], lambda s: s.puzzle, lambda s: s.puzzle * 0)
        dist = sliding_bfs(3)
        hard = [dist.get(tuple(p.reshape(-1).astype(int))) for p in b["initial"]]

    df = pd.DataFrame({k: list(b[k]) for k in ("states", "actions", "compute_times", "valid_steps")})
    df["episode_length"] = b["episode_length"]
    df["solved"] = b["solved"]
    df["shortest_path_length"] = hard
    out = REPO_ROOT / "ramdp_experiments" / f"rollout_cache-{name}-cnn16x4-hent0-n{NUM_EPISODES}-seed{SEED}.pkl"
    df.to_pickle(out)
    print(f"Saved {out}: solved={df.solved.mean():.3f} mean_ct0={np.mean([c[0] for c in df.compute_times]):.3f} "
          f"hardness range={df.shortest_path_length.min()}-{df.shortest_path_length.max()}")


if __name__ == "__main__":
    main(sys.argv[1])
