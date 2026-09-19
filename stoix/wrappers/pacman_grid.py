"""Wrapper that turns Jumanji PacMan's structured observation into a spatial
grid observation for a CNN torso.

PacMan's native `grid` field is only the maze's walls - it doesn't say where
the player, ghosts, pellets, or power-ups are (see
jumanji.environments.routing.pac_man.types.Observation), so (like Maze's
`walls`, see stoix/wrappers/maze_grid.py) it can't be used alone. Jumanji's
own reference network (jumanji.training.networks.pac_man.actor_critic.
process_image) bakes all of these into one 3-channel RGB image via a Python
loop over pellet/power-up/ghost arrays with a Python-level `if` on a traced
value - not jit-safe, and it packs unrelated entities into shared RGB
channels rather than one channel per entity type.

`PacManGridObservationWrapper` instead keeps the (x_size, y_size) spatial
layout and stacks one one-hot/binary channel per semantic category - walls,
remaining pellets, remaining power-ups, the player, the ghosts - plus a
constant "scared timer" channel broadcasting `frightened_state_time` (clipped
to [0, 30], the actual max set by the env's `powerup_collected`, and
normalized to [0, 1]) so a CNN can read the ghosts' edible/dangerous state
without extra plumbing. `action_mask`/`score` are dropped, matching how the
other CNN/grid scenarios (sokoban_grid, maze_grid) already drop
action_mask/step_count.

Coordinate convention: `grid` is indexed [x, y] (see
jumanji...pac_man.env.PacMan.check_wall_collisions), while
`player_locations`/`ghost_locations`/`pellet_locations`/`power_up_locations`
are all [y, x] pairs (see jumanji...pac_man.utils.check_ghost_collisions'
`Position(y=ghost_pos[0], x=ghost_pos[1])`) - every placement below indexes
the grid-shaped channels as `[loc[..., 1], loc[..., 0]]` to match. Eaten
pellets/power-ups are zeroed to `[0, 0]` by the base env rather than removed
(see PacMan.check_rewards/check_power_up); scattering with an
eaten/not-eaten mask via `.add()` (rather than `.set()`) means a zeroed entry
contributes 0 at cell (0, 0) instead of falsely marking it, without needing a
non-jittable per-element Python `if`.
"""

from typing import Optional, Tuple

import jax.numpy as jnp
from chex import PRNGKey
from stoa.core_wrappers.wrapper import Wrapper
from stoa.env_types import Action, EnvParams, State, TimeStep
from stoa.environment import Environment
from stoa.spaces import BoundedArraySpace, Space

# The env resets frightened_state_time to this value on a power-up pickup
# (jumanji...pac_man.env.PacMan._update_state.powerup_collected) and never
# clips it below 0 while ticking down - used to normalize the scared-timer
# channel to [0, 1].
_MAX_FRIGHTENED_STATE_TIME = 30.0


def _scatter_presence_channel(
    x_size: int, y_size: int, locations: jnp.ndarray, present: jnp.ndarray
) -> jnp.ndarray:
    """One binary (x_size, y_size) channel with a 1 at each `[y, x]` entry of
    `locations` whose `present` flag is true (scatter-`add`, not `set`, so a
    false-flagged entry - e.g. an eaten pellet zeroed to `[0, 0]` - safely
    contributes 0 instead of marking that cell)."""
    channel = jnp.zeros((x_size, y_size), dtype=jnp.float32)
    channel = channel.at[locations[..., 1], locations[..., 0]].add(present.astype(jnp.float32))
    return jnp.clip(channel, 0.0, 1.0)


class PacManGridObservationWrapper(Wrapper[State]):
    """Stacks PacMan's walls/pellets/power-ups/player/ghosts/scared-timer into
    one (x_size, y_size, 6) float32 grid observation."""

    def _prepare(self, observation: object) -> jnp.ndarray:
        grid = jnp.asarray(observation.grid, dtype=jnp.float32)
        x_size, y_size = grid.shape
        walls = 1.0 - grid  # grid: 1 = walkable, 0 = wall.

        player = jnp.zeros((x_size, y_size), dtype=jnp.float32)
        player = player.at[observation.player_locations.x, observation.player_locations.y].set(1.0)

        ghost_locations = jnp.asarray(observation.ghost_locations)
        ghosts = _scatter_presence_channel(
            x_size, y_size, ghost_locations, jnp.ones(ghost_locations.shape[:-1], dtype=bool)
        )

        pellet_locations = jnp.asarray(observation.pellet_locations)
        pellet_present = jnp.any(pellet_locations != 0, axis=-1)
        pellets = _scatter_presence_channel(x_size, y_size, pellet_locations, pellet_present)

        power_up_locations = jnp.asarray(observation.power_up_locations)
        power_up_present = jnp.any(power_up_locations != 0, axis=-1)
        power_ups = _scatter_presence_channel(x_size, y_size, power_up_locations, power_up_present)

        frightened_norm = (
            jnp.clip(observation.frightened_state_time, 0, _MAX_FRIGHTENED_STATE_TIME)
            / _MAX_FRIGHTENED_STATE_TIME
        )
        frightened = jnp.full((x_size, y_size), frightened_norm, dtype=jnp.float32)

        return jnp.stack([walls, pellets, power_ups, player, ghosts, frightened], axis=-1)

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
        x_size, y_size = orig_space.spaces["grid"].shape
        return BoundedArraySpace(
            shape=(x_size, y_size, 6),
            dtype=jnp.float32,
            minimum=0.0,
            maximum=1.0,
            name="pacman_grid",
        )
