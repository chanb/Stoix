"""Wrapper that flattens and concatenates every array leaf of a structured
(dict/NamedTuple) observation into a single float32 vector.

Some Jumanji environments' `Observation` bundles several fields that are
*all* necessary to act well - e.g. Knapsack's weights/values/packed_items/
action_mask, or Maze's agent_position/target_position/walls/action_mask -
unlike environments whose native observation already bundles everything
needed into one array (Sokoban's `grid` stacks fixed+variable elements as two
channels; SlidingTilePuzzle's `puzzle` alone already determines the empty
tile's position). For those single-array cases, `stoa.ObservationExtractWrapper`
picking one named attribute is enough. For Knapsack/Maze, extracting only one
field would silently drop information the network needs (e.g. Knapsack's
`values` without `weights` makes the packing decision unsolvable) - so
`make_jumanji_env` (see stoix/utils/make_env.py) skips
`ObservationExtractWrapper` entirely when `env.observation_attribute` isn't
set, and this wrapper is used instead (via `env.wrapper`) to keep every
field, flattened into one vector - the same flattening
`stoa.FlattenObservationWrapper` does for an already-single-array
observation, generalized (via `jax.tree_util.tree_leaves`) to a whole pytree
of separate fields, recursing through nested NamedTuples too (e.g. Maze's
`agent_position: Position(row, col)`).
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
    """Total flattened element count of a (possibly nested Dict/Tuple) Space,
    computed from declared shapes alone - deliberately not via
    `Space.generate_value()`/`sample()`, which crashes for some real spaces
    here (`stoa.spaces.ArraySpace.sample()` unconditionally calls
    `jax.random.normal`, which rejects unbounded integer-dtype leaves such as
    Jumanji Maze's `step_count: Array(int32, shape=())` - see
    `stoix.wrappers.grid_observation.GridObservationWrapper`'s docstring for
    the same underlying issue)."""
    if isinstance(space, DictSpace):
        return sum(_flat_size(s) for s in space.spaces.values())
    if isinstance(space, TupleSpace):
        return sum(_flat_size(s) for s in space.spaces)
    return int(np.prod(space.shape, dtype=int)) if space.shape else 1


class ConcatObservationWrapper(Wrapper[State]):
    """Flattens and concatenates every leaf of a structured observation into
    a single float32 vector, preserving field order as returned by
    `jax.tree_util.tree_leaves` (stable/deterministic for a given pytree
    structure)."""

    def __init__(self, env: Environment):
        super().__init__(env)
        self._flat_dim = _flat_size(env.observation_space())

    def _concat(self, observation: object) -> jnp.ndarray:
        leaves = jax.tree_util.tree_leaves(observation)
        flat_leaves = [jnp.reshape(jnp.asarray(leaf, dtype=jnp.float32), (-1,)) for leaf in leaves]
        return jnp.concatenate(flat_leaves, axis=0)

    def reset(
        self, rng_key: PRNGKey, env_params: Optional[EnvParams] = None
    ) -> Tuple[State, TimeStep]:
        state, timestep = self._env.reset(rng_key, env_params)
        return state, timestep.replace(observation=self._concat(timestep.observation))

    def step(
        self, state: State, action: Action, env_params: Optional[EnvParams] = None
    ) -> Tuple[State, TimeStep]:
        state, timestep = self._env.step(state, action, env_params)
        return state, timestep.replace(observation=self._concat(timestep.observation))

    def observation_space(self, env_params: Optional[EnvParams] = None) -> Space:
        return ArraySpace(shape=(self._flat_dim,), dtype=jnp.float32, name="observation")
