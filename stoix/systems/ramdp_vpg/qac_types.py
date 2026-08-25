from typing import Dict

import chex
from typing_extensions import NamedTuple

from stoix.base_types import Action, Done, Value


class Transition(NamedTuple):
    """Like `stoix.systems.ramdp_vpg.ramdp_vpg_types.Transition`, but for a
    Q actor-critic: `value` is V(s_t) (an independent state-value estimate,
    used to bootstrap the n-step return target) and `q_value` is
    Q(s_t, a_t, c_t) (the taken action's, and realised compute time's,
    Q-value), so the advantage can be computed directly as
    `q_value - value` without any Monte-Carlo return."""

    done: Done
    action: Action
    value: Value
    q_value: Value
    reward: chex.Array
    obs: chex.Array
    info: Dict
    compute_time: chex.Array
    first_convergence_step: chex.Array
    num_close_steps: chex.Array
