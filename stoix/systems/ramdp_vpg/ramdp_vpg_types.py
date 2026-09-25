from typing import Dict, Tuple

import chex
import jax.numpy as jnp
from omegaconf import DictConfig
from stoa import TimeStep, WrapperState
from typing_extensions import NamedTuple

from stoix.base_types import OptStates, Parameters


class RamdpOnPolicyLearnerState(NamedTuple):
    """Like `stoix.base_types.OnPolicyLearnerState`, plus per-env running state needed to report
    `episode_discounted_return` as an actor metric: the actual RAMDP compute-discounted return
    realised over a whole episode - `gamma^(compute_time - 1)` applied per step, see
    `ff_ppo.py`'s module docstring for the `G_h` derivation this mirrors."""

    params: Parameters
    opt_states: OptStates
    key: chex.PRNGKey
    env_state: WrapperState
    timestep: TimeStep
    running_cum_compute_time: chex.Array
    running_discounted_return: chex.Array
    episode_discounted_return: chex.Array


def update_discounted_return(
    running_cum_compute_time: chex.Array,
    running_discounted_return: chex.Array,
    episode_discounted_return: chex.Array,
    compute_time: chex.Array,
    reward: chex.Array,
    done: chex.Array,
    gamma: float,
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """Forward-accumulate the RAMDP compute-discounted return by one step, resetting on episode
    completion - see `RamdpOnPolicyLearnerState`'s docstring."""
    not_done = 1.0 - done.astype(jnp.float32)
    new_cum_compute_time = running_cum_compute_time + compute_time.astype(jnp.float32)
    step_weight = gamma ** (new_cum_compute_time - 1.0)
    new_running_discounted_return = running_discounted_return + step_weight * reward
    episode_discounted_return_info = (
        episode_discounted_return * not_done + new_running_discounted_return * done
    )
    return (
        new_cum_compute_time * not_done,
        new_running_discounted_return * not_done,
        episode_discounted_return_info,
    )


def solved_episode_info(
    config: DictConfig, reward: chex.Array, done: chex.Array
) -> Dict[str, chex.Array]:
    """Per-step `solved_episode` entry for the actor's `episode_metrics` info dict, or `{}` if
    `config.env.solved_final_reward_threshold` isn't set."""
    threshold = config.env.get("solved_final_reward_threshold", None)
    if threshold is None:
        return {}
    solved = (reward.reshape(done.shape) >= threshold) & done.astype(bool)
    return {"solved_episode": solved.astype(jnp.float32)}
