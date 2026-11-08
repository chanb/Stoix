import os

import jax
import numpy as np
import pytest
from hydra import compose, initialize_config_dir

from stoix.utils.make_env import make, make_train_eval_env

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "stoix", "configs", "default", "anakin")
BANK = os.path.join(os.path.dirname(__file__), "..", "data", "slidingpuzzle", "3x3")


def _cfg(*overrides):
    with initialize_config_dir(config_dir=os.path.abspath(CONFIG_DIR), version_base=None):
        return compose("default_ramdp_ff_ppo", overrides=list(overrides))


def _base(env):
    while not hasattr(env, "_eval"):
        env = env._env
    return env


@pytest.mark.parametrize(
    "env_cfg", ["lightsout/lightsout_3x3", "jumanji/slidingtile", "jumanji/slidingtile_grid",
                "jumanji/slidingtile_bank", "jumanji/slidingtile_bank_grid", "jumanji/sokoban",
                "jumanji/sokoban_grid"],
)
def test_train_eval_flag_on(env_cfg):
    assert _cfg(f"env={env_cfg}").env.get("train_eval", False)


def test_train_eval_flag_off_elsewhere():
    assert not _cfg("env=jumanji/knapsack").env.get("train_eval", False)


def test_lightsout_train_eval_env_uses_easy_goals():
    cfg = _cfg("env=lightsout/lightsout_3x3", "+env.kwargs.eval_episode_length=9")
    _, eval_env = make(cfg)
    train_eval_env = make_train_eval_env(cfg)
    assert _base(eval_env)._eval and not _base(train_eval_env)._eval
    assert _base(train_eval_env)._config.episode_length == cfg.env.kwargs.episode_length


@pytest.mark.skipif(not os.path.exists(os.path.join(BANK, "test.npz")), reason="3x3 bank not built")
def test_slidingtile_train_eval_env_uses_train_bank():
    cfg = _cfg("env=jumanji/slidingtile_bank_grid")
    train_eval_env = make_train_eval_env(cfg)
    _, ts = jax.vmap(train_eval_env.reset)(jax.random.split(jax.random.PRNGKey(0), 256))
    train = {p.astype(np.int8).tobytes() for p in np.load(os.path.join(BANK, "train.npz"))["puzzle"]}
    # Observations are one-hot tile ids (GridObservationWrapper one_hot=True).
    boards = np.asarray(getattr(ts.observation, "agent_view", ts.observation)).argmax(-1).astype(np.int8)
    assert all(b.tobytes() in train for b in boards)
