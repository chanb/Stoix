from typing import Dict

import chex
from typing_extensions import NamedTuple

from stoix.base_types import Action, Done, Value


class ExplicitCoTTransition(NamedTuple):
    """Like `stoix.systems.ramdp_vpg.ramdp_vpg_types.Transition`, but also
    stores the explicit thought tokens emitted by the actor's torso (see
    `stoix.networks.torso_compute_explicit_cot.TransformerExplicitCoTTorso`),
    so the exact token trajectory - thoughts and the halting "act now" token
    together - can be replayed when computing the actor loss."""

    done: Done
    action: Action
    value: Value
    reward: chex.Array
    obs: chex.Array
    info: Dict
    compute_time: chex.Array
    thought_tokens: chex.Array


class PPOExplicitCoTTransition(NamedTuple):
    """Like `stoix.systems.ramdp_vpg.ppo_types.PPOTransition`, but also
    stores the explicit thought tokens emitted by the actor's torso (see
    `stoix.networks.torso_compute_explicit_cot.TransformerExplicitCoTTorso`)
    in place of `first_convergence_step`/`num_close_steps` (which
    `TransformerExplicitCoTTorso` has no equivalent of), so the exact token
    trajectory can be replayed when computing the actor loss at each PPO
    epoch's parameters.

    `env_log_prob` and `cot_log_prob` are kept separate (rather than summed
    into one joint `log_prob`, as earlier versions of this transition did)
    so `ff_ppo_explicit_cot.py` can clip the environment action's ratio and
    each CoT step's ratio individually instead of one ratio over their sum -
    see that file's module docstring. `cot_log_prob` has shape
    `(*batch, max_steps)`, one entry per CoT step (zeroed past the step the
    trajectory actually halted at - see `TransformerExplicitCoTTorso`)."""

    done: Done
    action: Action
    value: Value
    q_value: Value
    reward: chex.Array
    obs: chex.Array
    info: Dict
    compute_time: chex.Array
    thought_tokens: chex.Array
    env_log_prob: chex.Array
    cot_log_prob: chex.Array
