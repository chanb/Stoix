"""A critic network for Q actor-critic: one shared torso, two heads.

`ValueAndQCritic` exposes `.value(obs)` (a state-value head V(s)) and
`.q_value(obs)` (a state-action-value head Q(s, ·)) as separate `apply(...,
method=...)` entry points sharing the same embedding, so V and Q are always
trained from - and read off - the same representation rather than two
independently-initialised networks. See `stoix.systems.ramdp_vpg.ff_qac` for
how the two heads are used together (advantage = Q - V) and for what shape
`q_value`'s output takes under the "naive" vs "fac" (runtime-factorized)
variants. This is a separate module so `stoix/networks/base.py` is left
untouched.
"""

import chex
from flax import linen as nn

from stoix.base_types import Observation
from stoix.networks.inputs import ArrayInput


class ValueAndQCritic(nn.Module):
    """Shared-torso critic with a state-value head and a state-action-value head."""

    torso: nn.Module
    value_head: nn.Module
    q_head: nn.Module
    input_layer: nn.Module = ArrayInput()

    def _embed(self, observation: Observation) -> chex.Array:
        return self.torso(self.input_layer(observation))

    def value(self, observation: Observation) -> chex.Array:
        """V(s)."""
        return self.value_head(self._embed(observation))

    def q_value(self, observation: Observation) -> chex.Array:
        """Q(s, ·): shape depends on how `q_head` was configured (a full
        `(num_actions, max_steps)` table, or just `(num_actions,)` for the
        runtime-factorized Q(s, ·, 1))."""
        return self.q_head(self._embed(observation))

    def __call__(self, observation: Observation) -> chex.Array:
        embedding = self._embed(observation)
        return self.value_head(embedding), self.q_head(embedding)
