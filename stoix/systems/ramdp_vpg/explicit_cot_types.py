from typing import Dict

import chex
from typing_extensions import NamedTuple

from stoix.base_types import Action, Done, Value


class PPOMergedActionCoTTransition(NamedTuple):
    """Like `stoix.systems.ramdp_vpg.ppo_types.PPOTransition`, but for
    `TransformerMergedActionCoTTorso`: stores the explicit thought tokens emitted by the
    actor's torso (in place of `first_convergence_step`/`num_close_steps`) so the exact token
    trajectory can be replayed when computing the actor loss at each PPO epoch's parameters."""

    done: Done
    action: Action
    value: Value
    reward: chex.Array
    obs: chex.Array
    info: Dict
    compute_time: chex.Array
    thought_tokens: chex.Array
    cot_log_prob: chex.Array
