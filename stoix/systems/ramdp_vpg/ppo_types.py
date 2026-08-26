from typing import Dict

import chex
from typing_extensions import NamedTuple

from stoix.base_types import Action, Done, Value


class PPOTransition(NamedTuple):
    """Transition tuple for RAMDP-PPO (compute-time-aware PPO, `ff_ppo.py`).

    Like `stoix.systems.ramdp_vpg.ramdp_vpg_types.Transition`/
    `stoix.systems.ramdp_vpg.qac_types.Transition`, plus `log_prob`: the
    *joint* log-probability (environment action + halting trajectory) of the
    sampled transition under the rollout-time policy. PPO needs this because,
    unlike REINFORCE/QAC (one gradient step per rollout, always evaluated at
    the current parameters), it takes several epochs of minibatch gradient
    steps over the same rollout - so the ratio in `ppo_clip_loss` needs a
    fixed "old" log-probability to compare each epoch's updated parameters
    against. See `ff_ppo.py`'s module docstring for how this is captured at
    rollout time.

    `q_value` is Q(s_t, a_t, c_t) under the rollout-time critic - only
    meaningful when `config.system.qac_variant` is "naive"/"fac" (the Q-V
    advantage variant, mirroring `ff_qac.py`'s `Transition`); it is zeros for
    the "reinforce" (G-V) variant, which only ever reads `value`.
    """

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
    log_prob: chex.Array