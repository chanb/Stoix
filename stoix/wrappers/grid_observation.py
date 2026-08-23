"""Wrapper that prepares a channel-less integer grid observation for a CNN torso.

Some environments expose a spatial observation with no channel axis and no
declared bounds - e.g. jumanji's 2048 `board`: shape (height, width), dtype
int32, unbounded (tile values can in principle grow without limit). Two
things break for that combination:

  1. `CNNTorso` (stoix.networks.torso.CNNTorso) expects a trailing channel
     axis - (height, width, channels) - which a bare (height, width) board
     doesn't have.
  2. `stoa.spaces.ArraySpace.sample()` (used by every Stoix system via
     `env.observation_space().generate_value()` to build a dummy array for
     network initialisation) unconditionally samples via `jax.random.normal`,
     which requires a float/complex dtype - it crashes on an unbounded
     integer-dtype space such as this one. (Bounded spaces, e.g. Sokoban's
     grid, don't hit this: `BoundedArraySpace.sample()` always samples in
     float32 internally and casts to the target dtype only at the very end.)
     `stoa.FlattenObservationWrapper` avoids the crash only as a side effect
     of also casting to float32 while flattening - which destroys the
     spatial structure a CNN needs.

`GridObservationWrapper` fixes both without flattening: it casts the
observation to `dtype` (default float32) and, if `add_channel_dim=True`
(default), appends a trailing singleton channel axis.
"""

from typing import Optional, Tuple

import jax.numpy as jnp
from chex import PRNGKey
from stoa.core_wrappers.wrapper import Wrapper
from stoa.env_types import Action, EnvParams, State, TimeStep
from stoa.environment import Environment
from stoa.spaces import ArraySpace, BoundedArraySpace, Space


class GridObservationWrapper(Wrapper[State]):
    """Casts a grid observation's dtype and optionally adds a channel axis."""

    def __init__(self, env: Environment, dtype: str = "float32", add_channel_dim: bool = True):
        super().__init__(env)
        self._dtype = jnp.dtype(dtype)
        self._add_channel_dim = add_channel_dim

    def _prepare(self, observation: jnp.ndarray) -> jnp.ndarray:
        observation = jnp.asarray(observation, dtype=self._dtype)
        if self._add_channel_dim:
            observation = observation[..., None]
        return observation

    def reset(
        self, rng_key: PRNGKey, env_params: Optional[EnvParams] = None
    ) -> Tuple[State, TimeStep]:
        state, timestep = self._env.reset(rng_key, env_params)
        new_timestep = timestep.replace(observation=self._prepare(timestep.observation))  # type: ignore
        return state, new_timestep

    def step(
        self, state: State, action: Action, env_params: Optional[EnvParams] = None
    ) -> Tuple[State, TimeStep]:
        new_state, timestep = self._env.step(state, action, env_params)
        new_timestep = timestep.replace(observation=self._prepare(timestep.observation))  # type: ignore
        return new_state, new_timestep

    def observation_space(self, env_params: Optional[EnvParams] = None) -> Space:
        orig_space = self._env.observation_space(env_params)
        new_shape = (*orig_space.shape, 1) if self._add_channel_dim else orig_space.shape

        if isinstance(orig_space, BoundedArraySpace):
            return BoundedArraySpace(
                shape=new_shape,
                dtype=self._dtype,
                minimum=orig_space.minimum,
                maximum=orig_space.maximum,
                name=orig_space.name,
            )
        elif isinstance(orig_space, ArraySpace):
            return ArraySpace(shape=new_shape, dtype=self._dtype, name=orig_space.name)
        else:
            raise ValueError(f"Unsupported space type for GridObservationWrapper: {type(orig_space)}")
