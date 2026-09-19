"""Wrapper that flattens Jumanji Knapsack's observation into a vector - like
`ConcatObservationWrapper` - and additionally appends the bag's remaining
capacity, which jumanji's own `Observation` (weights/values/packed_items/
action_mask) never exposes even though `state.remaining_budget` is tracked
internally throughout (see jumanji.environments.packing.knapsack.types.State)
and both the action mask and the episode-termination condition depend on it.

Without this, an agent has no direct signal for how much room is left in the
bag - only the indirect, entangled signal of which items still fit
(`action_mask`, which conflates "too heavy" with "already packed"). This
matters most now that env=jumanji/knapsack's capacity is resampled every
episode (see stoix.envs.knapsack.generator.IntegerRandomGenerator) rather
than being one fixed constant: without observing it, the agent can no longer
even calibrate a fixed decision threshold across episodes.

`KnapsackConcatWithBudgetWrapper` does the exact flatten-and-concatenate
`ConcatObservationWrapper` does, plus one extra scalar -
`state.remaining_budget` - appended to the end of the flattened vector. It
reads `state` (not `observation`) for that scalar, since only `state` (not
jumanji's `Observation`) carries it - see `JumanjiToStoa.step`/`reset` in
stoa/env_adapters/jumanji.py, which passes the raw Jumanji `State` through
unmodified.
"""

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from chex import PRNGKey
from stoa.core_wrappers.wrapper import Wrapper
from stoa.env_types import Action, EnvParams, State, TimeStep
from stoa.environment import Environment
from stoa.spaces import ArraySpace, DictSpace, Space, TupleSpace


def _flat_size(space: Space) -> int:
    """Total flattened element count of a (possibly nested Dict/Tuple) Space
    - see `stoix.wrappers.concat_observation._flat_size` for why this is
    computed from declared shapes rather than via `Space.sample()`."""
    if isinstance(space, DictSpace):
        return sum(_flat_size(s) for s in space.spaces.values())
    if isinstance(space, TupleSpace):
        return sum(_flat_size(s) for s in space.spaces)
    return int(np.prod(space.shape, dtype=int)) if space.shape else 1


class KnapsackConcatWithBudgetWrapper(Wrapper[State]):
    """Flattens Knapsack's observation into a vector (same as
    `ConcatObservationWrapper`) and appends `state.remaining_budget` as one
    extra scalar."""

    def __init__(self, env: Environment):
        super().__init__(env)
        self._flat_dim = _flat_size(env.observation_space()) + 1

    def _concat(self, state: State, observation: object) -> jnp.ndarray:
        leaves = jax.tree_util.tree_leaves(observation)
        flat_leaves = [jnp.reshape(jnp.asarray(leaf, dtype=jnp.float32), (-1,)) for leaf in leaves]
        budget = jnp.reshape(jnp.asarray(state.remaining_budget, dtype=jnp.float32), (1,))
        return jnp.concatenate([*flat_leaves, budget], axis=0)

    def reset(
        self, rng_key: PRNGKey, env_params: Optional[EnvParams] = None
    ) -> Tuple[State, TimeStep]:
        state, timestep = self._env.reset(rng_key, env_params)
        return state, timestep.replace(observation=self._concat(state, timestep.observation))

    def step(
        self, state: State, action: Action, env_params: Optional[EnvParams] = None
    ) -> Tuple[State, TimeStep]:
        state, timestep = self._env.step(state, action, env_params)
        return state, timestep.replace(observation=self._concat(state, timestep.observation))

    def observation_space(self, env_params: Optional[EnvParams] = None) -> Space:
        return ArraySpace(shape=(self._flat_dim,), dtype=jnp.float32, name="observation")
