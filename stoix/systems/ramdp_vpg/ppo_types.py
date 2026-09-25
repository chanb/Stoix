from typing import Dict

import chex
from typing_extensions import NamedTuple

from stoix.base_types import Action, Done, Value


class PPOTransition(NamedTuple):
    """Transition tuple for RAMDP-PPO (compute-time-aware PPO, `ff_ppo.py`)."""

    done: Done
    action: Action
    value: Value
    reward: chex.Array
    obs: chex.Array
    info: Dict
    compute_time: chex.Array
    first_convergence_step: chex.Array
    num_close_steps: chex.Array
    env_log_prob: chex.Array
    halting_log_prob: chex.Array
    old_latent_states: chex.Array