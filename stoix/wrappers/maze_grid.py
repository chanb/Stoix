"""Wrapper that turns Jumanji Maze's structured observation into a spatial
grid observation for a CNN torso.

Maze's native observation has no single field a CNN can operate on alone:
`walls` is a genuine (num_rows, num_cols) grid, but on its own it doesn't say
where the agent or the target are (see stoix/configs/env/jumanji/maze.yaml,
which uses ConcatObservationWrapper instead, for the same reason). Dropping
`agent_position`/`target_position` the way `ObservationExtractWrapper`/
`GridObservationWrapper` drop everything but one field (as sokoban_grid/
2048_grid do) would make the task unsolvable from the observation alone.

`MazeGridObservationWrapper` fixes this without flattening: it keeps the
(num_rows, num_cols) spatial layout and stacks three channels - walls, a
one-hot of the agent's position, and a one-hot of the target's position -
into one (num_rows, num_cols, 3) float32 array. `action_mask`/`step_count`
are dropped, matching how the other CNN/grid scenarios (sokoban_grid,
2048_grid) already drop them.
"""

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from chex import PRNGKey
from stoa.core_wrappers.wrapper import Wrapper
from stoa.env_types import Action, EnvParams, State, TimeStep
from stoa.environment import Environment
from stoa.spaces import BoundedArraySpace, Space


class MazeGridObservationWrapper(Wrapper[State]):
    """Stacks Maze's walls/agent_position/target_position into one
    (num_rows, num_cols, 3) float32 grid observation."""

    def _prepare(self, observation: object) -> jnp.ndarray:
        walls = jnp.asarray(observation.walls, dtype=jnp.float32)
        num_rows, num_cols = walls.shape
        num_cells = num_rows * num_cols

        agent_index = observation.agent_position.row * num_cols + observation.agent_position.col
        agent = jax.nn.one_hot(agent_index, num_cells, dtype=jnp.float32).reshape(
            num_rows, num_cols
        )

        target_index = (
            observation.target_position.row * num_cols + observation.target_position.col
        )
        target = jax.nn.one_hot(target_index, num_cells, dtype=jnp.float32).reshape(
            num_rows, num_cols
        )

        return jnp.stack([walls, agent, target], axis=-1)

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
        num_rows, num_cols = orig_space.spaces["walls"].shape
        return BoundedArraySpace(
            shape=(num_rows, num_cols, 3),
            dtype=jnp.float32,
            minimum=0.0,
            maximum=1.0,
            name="maze_grid",
        )
