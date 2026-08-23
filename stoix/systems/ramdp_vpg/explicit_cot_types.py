from typing import Dict

import chex
from typing_extensions import NamedTuple

from stoix.base_types import Action, Done, Value


class ExplicitCoTTransition(NamedTuple):
    """Like `stoix.systems.ramdp_vpg.ramdp_vpg_types.Transition`, but also
    stores the explicit thought tokens emitted by the actor's torso (see
    `stoix.networks.torso_compute_explicit_cot.TransformerExplicitCoTTorso`),
    so the exact halting-and-token trajectory can be replayed when computing
    the actor loss."""

    done: Done
    action: Action
    value: Value
    reward: chex.Array
    obs: chex.Array
    info: Dict
    compute_time: chex.Array
    thought_tokens: chex.Array
