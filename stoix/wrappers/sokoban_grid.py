"""Wrapper that one-hot encodes Jumanji Sokoban's grid observation for a CNN torso.

Sokoban's native `grid` observation (extracted by `ObservationExtractWrapper`
via `observation_attribute: grid`, see sokoban_grid.yaml) is a
(num_rows, num_cols, 2) uint8 array whose two channels bundle several object
types together as small integer codes rather than one grid per object:
channel 0 is jumanji's `variable_grid` (0 = empty, `AGENT` = 3, `BOX` = 4) and
channel 1 is its `fixed_grid` (0 = empty, `WALL` = 1, `TARGET` = 2) - see
`jumanji.environments.routing.sokoban.env.Sokoban.observe` and
`generator.convert_level_to_array`. Feeding those raw codes straight into a
CNN would make it treat unrelated object types (e.g. agent=3 vs. wall=1) as
ordinally related, which they aren't.

`SokobanGridObservationWrapper` mirrors the one-hot preprocessing jumanji's
own `make_sokoban_cnn` network applies (`jumanji.training.networks.sokoban.
actor_critic.preprocess_input`) so Stoix's CNN networks (network=
cnn_mlp_compute / cnn_mlp_compute_qac / cnn_transformer_compute /
cnn_transformer_compute_qac) get the same four semantic channels - agent, box,
wall, target - as one-hot floats, stacked into one
(num_rows, num_cols, 4) array. "Empty" is left as the implicit all-zero
background, same as jumanji's version.
"""

from typing import Optional, Tuple

import jax.numpy as jnp
from chex import PRNGKey
from jumanji.environments.routing.sokoban.constants import AGENT, BOX, TARGET, WALL
from stoa.core_wrappers.wrapper import Wrapper
from stoa.env_types import Action, EnvParams, State, TimeStep
from stoa.environment import Environment
from stoa.spaces import BoundedArraySpace, Space


class SokobanGridObservationWrapper(Wrapper[State]):
    """One-hot encodes Sokoban's (rows, cols, 2) coded grid into a
    (rows, cols, 4) [agent, box, wall, target] float32 grid observation."""

    def _prepare(self, observation: jnp.ndarray) -> jnp.ndarray:
        variable_grid = observation[..., 0:1]
        fixed_grid = observation[..., 1:2]

        agent_and_box = jnp.equal(variable_grid, jnp.array([AGENT, BOX])).astype(jnp.float32)
        wall_and_target = jnp.equal(fixed_grid, jnp.array([WALL, TARGET])).astype(jnp.float32)

        return jnp.concatenate([agent_and_box, wall_and_target], axis=-1)

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
        new_shape = (*orig_space.shape[:-1], 4)
        return BoundedArraySpace(
            shape=new_shape,
            dtype=jnp.float32,
            minimum=0.0,
            maximum=1.0,
            name=orig_space.name,
        )
