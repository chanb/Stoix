from typing import Dict

import chex
from typing_extensions import NamedTuple

from stoix.base_types import Action, Done, Value


class PPOTransition(NamedTuple):
    """Transition tuple for RAMDP-PPO (compute-time-aware PPO, `ff_ppo.py`).

    Like `stoix.systems.ramdp_vpg.ramdp_vpg_types.Transition`/
    `stoix.systems.ramdp_vpg.qac_types.Transition`, plus `env_log_prob` and
    `halting_log_prob`: the environment action's and the halting
    trajectory's log-probabilities under the rollout-time policy, kept
    *separate* (rather than summed into one joint `log_prob`, as earlier
    versions of this transition did) so `ff_ppo.py` can PPO-clip the action's
    ratio and each pondering step's ratio individually instead of one ratio
    over their sum - see that file's module docstring. PPO needs these
    because, unlike REINFORCE/QAC (one gradient step per rollout, always
    evaluated at the current parameters), it takes several epochs of
    minibatch gradient steps over the same rollout - so the ratio needs a
    fixed "old" log-probability to compare each epoch's updated parameters
    against. `halting_log_prob` has shape `(*batch, max_steps)`, one entry
    per pondering step (zeroed at forced steps or past the step the
    trajectory actually halted at - see
    `stoix.networks.torso_compute.AdaptiveComputationTimeTorso`'s
    `per_step_halting_log_prob`).

    `q_value` is Q(s_t, a_t, c_t) under the rollout-time critic - only
    meaningful when `config.system.qac_variant` is "naive"/"fac" (the Q-V
    advantage variant, mirroring `ff_qac.py`'s `Transition`); it is zeros for
    the "reinforce" (G-V) variant, which only ever reads `value`.

    `old_latent_states` (shape `(*batch, max_steps, hidden_dim)`) is the
    rollout-time torso's per-step `states_history` (see
    `stoix.networks.torso_compute.AdaptiveComputationTimeTorso` and
    `stoix.networks.torso_compute_transformer.TransformerChainOfThoughtTorso`),
    recorded from the same replay-mode pass that produces `halting_log_prob`.
    Used by `ff_ppo.py`'s optional latent trust-region penalty to compare
    against each epoch's freshly-computed `new_latent_states` at the same
    `compute_time` trajectory - not needed (and zero-cost, since it's read
    off a pass already being made) when that penalty is disabled
    (`config.system.latent_kl_coef == 0`).
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
    env_log_prob: chex.Array
    halting_log_prob: chex.Array
    old_latent_states: chex.Array