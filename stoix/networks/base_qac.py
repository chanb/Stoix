"""Critic networks for Q actor-critic.

`ValueAndQCritic` is one shared torso, two heads: it exposes `.value(obs)`
(a state-value head V(s)) and `.q_value(obs, compute_time=None)` (a
state-action-value head Q(s, ·) or Q(s, ·, c)) as separate
`apply(..., method=...)` entry points sharing the same embedding, so V and Q
are always trained from - and read off - the same representation.

`SeparateValueAndQCritic` is the independently-initialised counterpart: V
and Q each get their own torso (and, optionally, their own input_layer), so
the two heads share no parameters at all. It exposes the same
`value`/`q_value`/`__call__` entry points, so callers only need to change
how the critic is constructed (two torsos instead of one), not how it's
invoked.

See `stoix.systems.ramdp_vpg.ff_ppo` for how the two heads are used together
(advantage = Q - V) and for what shape `q_value`'s output takes, and how
it's obtained, under the "naive"/"fac" (output-side, table/analytic-scaling)
vs "cond_naive"/"cond_fac" (input-side, `compute_time` fed into the network)
variants. This is a separate module so `stoix/networks/base.py` is left
untouched.
"""

from typing import Optional

import chex
from flax import linen as nn
from flax.linen.initializers import Initializer, orthogonal
from jax import numpy as jnp

from stoix.base_types import Observation
from stoix.networks.inputs import ArrayInput


def _q_input(
    compute_time_dense: nn.Dense, embedding: chex.Array, compute_time: Optional[chex.Array]
) -> chex.Array:
    """The input `q_head` reads: just the embedding, or - for the
    "cond_naive"/"cond_fac" `qac_variant`s (see `ff_ppo.py`) - the embedding
    concatenated with a single learned linear feature of `compute_time` (a
    genuine `nn.Dense` mapping `c -> R`, not a table/one-hot lookup), so
    `q_head` can learn a dependence on `c` directly instead of only ever
    seeing Q(s,·,1) ("fac") or needing a `(num_actions, max_steps)`-shaped
    output ("naive"). Shared by `ValueAndQCritic` and
    `SeparateValueAndQCritic`, which differ only in how `embedding` itself
    is produced."""
    if compute_time is None:
        return embedding
    c = compute_time.astype(jnp.float32)[..., jnp.newaxis]
    c_feature = compute_time_dense(c)
    return jnp.concatenate([embedding, c_feature], axis=-1)


class ValueAndQCritic(nn.Module):
    """Shared-torso critic with a state-value head and a state-action-value head."""

    torso: nn.Module
    value_head: nn.Module
    q_head: nn.Module
    input_layer: nn.Module = ArrayInput()
    kernel_init: Initializer = orthogonal(1.0)

    def setup(self) -> None:
        # A persistent submodule (rather than one created inline in
        # `_q_input`) since this class exposes three separate `apply(...,
        # method=...)` entry points (`value`/`q_value`/`__call__`) - Flax
        # only allows a single `@nn.compact` method per Module, which
        # inline submodule creation would require, so submodules not passed
        # in as constructor fields are created here in `setup()` instead.
        self.compute_time_dense = nn.Dense(1, kernel_init=self.kernel_init)

    def _embed(self, observation: Observation) -> chex.Array:
        return self.torso(self.input_layer(observation))

    def value(self, observation: Observation) -> chex.Array:
        """V(s)."""
        return self.value_head(self._embed(observation))

    def q_value(
        self, observation: Observation, compute_time: Optional[chex.Array] = None
    ) -> chex.Array:
        """Q(s, ·) or Q(s, ·, c) if `compute_time` is given. Shape/meaning
        depends on `qac_variant` (see `ff_ppo.py`'s module docstring):

          - "naive": `compute_time=None`, output is a full
            `(num_actions, max_steps)` table.
          - "fac": `compute_time=None`, output is `(num_actions,)`,
            representing Q(s,·,1) (the runtime-factorized Q(s,·,c) is
            recovered by the caller scaling this by `gamma ** (c - 1)`).
          - "cond_naive"/"cond_fac": `compute_time` given, output is
            `(num_actions,)`, already conditioned on the realised `c` via
            `_q_input`'s linear feature - "cond_naive" uses it directly as
            Q(s,·,c); "cond_fac" is still scaled by the caller (see
            `ff_ppo.py`), to test whether that analytic prior helps *given*
            the same c-conditioning capacity (and, since both variants share
            this exact architecture, the same parameter count) as
            "cond_naive", rather than conflating the prior with a
            parameter-count difference the way plain "naive" vs "fac" do.
        """
        return self.q_head(
            _q_input(self.compute_time_dense, self._embed(observation), compute_time)
        )

    def __call__(
        self, observation: Observation, compute_time: Optional[chex.Array] = None
    ) -> chex.Array:
        embedding = self._embed(observation)
        return self.value_head(embedding), self.q_head(
            _q_input(self.compute_time_dense, embedding, compute_time)
        )


class SeparateValueAndQCritic(nn.Module):
    """Independently-initialised critic: V and Q each get their own torso
    (and, optionally, their own `input_layer`), so the two heads share no
    parameters at all - the opposite design point from `ValueAndQCritic`.
    Exposes the same `value`/`q_value`/`__call__` entry points."""

    value_torso: nn.Module
    q_torso: nn.Module
    value_head: nn.Module
    q_head: nn.Module
    value_input_layer: nn.Module = ArrayInput()
    q_input_layer: nn.Module = ArrayInput()
    kernel_init: Initializer = orthogonal(1.0)

    def setup(self) -> None:
        self.compute_time_dense = nn.Dense(1, kernel_init=self.kernel_init)

    def value(self, observation: Observation) -> chex.Array:
        """V(s)."""
        return self.value_head(self.value_torso(self.value_input_layer(observation)))

    def _q_embed(self, observation: Observation) -> chex.Array:
        return self.q_torso(self.q_input_layer(observation))

    def q_value(
        self, observation: Observation, compute_time: Optional[chex.Array] = None
    ) -> chex.Array:
        """Q(s, ·) or Q(s, ·, c) if `compute_time` is given - see
        `ValueAndQCritic.q_value` for the `qac_variant` shape/meaning
        breakdown, which applies unchanged here."""
        return self.q_head(
            _q_input(self.compute_time_dense, self._q_embed(observation), compute_time)
        )

    def __call__(
        self, observation: Observation, compute_time: Optional[chex.Array] = None
    ) -> chex.Array:
        return self.value(observation), self.q_value(observation, compute_time)
