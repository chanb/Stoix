from typing import Dict, Tuple

import chex
import jax.numpy as jnp
from stoa import TimeStep, WrapperState
from typing_extensions import NamedTuple

from stoix.base_types import Action, Done, OptStates, Parameters, Value


class Transition(NamedTuple):
    done: Done
    action: Action
    value: Value
    reward: chex.Array
    obs: chex.Array
    info: Dict
    compute_time: chex.Array
    first_convergence_step: chex.Array
    num_close_steps: chex.Array


class RamdpOnPolicyLearnerState(NamedTuple):
    """Like `stoix.base_types.OnPolicyLearnerState`, plus per-env running
    state needed to report `episode_discounted_return` as an actor metric:
    the actual RAMDP compute-discounted return realised over a whole
    episode - `gamma^(compute_time - 1)` applied per step, see
    `ff_reinforce.py`'s module docstring for the `G_h` derivation this
    mirrors. This can't live in `stoa`'s `RecordEpisodeMetrics` env wrapper
    (which only ever sees reward/done, not `compute_time` - an actor-network
    output) so it's carried in the learner state instead, using the same
    running-count/reset-on-done pattern `RecordEpisodeMetrics` itself uses
    for `episode_return` (see `stoa.core_wrappers.episode_metrics`).

    Forward-accumulable, unlike the *training* target `G_h`
    (`batch_discounted_returns` in `ff_reinforce.py`, which bootstraps
    backward from a value estimate): the weight applied to reward `r_h`,
    `gamma^(cum_compute_time_h - 1)`, depends only on compute times up to
    and including step h, not on the (unknown until the episode ends)
    remaining trajectory - see `update_discounted_return`.

    `running_cum_compute_time`/`running_discounted_return` reset to 0 the
    step after an episode ends; `episode_discounted_return` instead holds
    the last *completed* episode's total in between - mirroring
    `RecordEpisodeMetricsState`'s `running_count_episode_return` vs.
    `episode_return`.
    """

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
    """Forward-accumulate the RAMDP compute-discounted return by one step,
    resetting on episode completion - see `RamdpOnPolicyLearnerState`'s
    docstring. Returns the updated `(running_cum_compute_time,
    running_discounted_return, episode_discounted_return)`; the last of
    these is what should be merged into that step's `episode_metrics` info
    dict (it's only ever read at the terminal step, like
    `RecordEpisodeMetrics`'s own fields - see `get_final_step_metrics`)."""
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
