"""Stoa adapter for `BoxMovingEnv`.

`BoxMovingEnv` (`block_moving_env.py`) has its own established API -
`reset(key) -> (state, info)`, `step(state, action) -> (state, reward, done,
info)` - already used directly by `play.py`, `tests.py`, and
`wrappers.py`'s `SymmetryFilter`/`QuarterFilter` (domain-specific curriculum
truncation). Rather than rewriting `BoxMovingEnv` in place (which would break
those), this module wraps an already-constructed `BoxMovingEnv` (optionally
already wrapped in `SymmetryFilter`/`QuarterFilter`) in the Stoa
`Environment` interface - the same pattern `stoa.env_adapters.gymnax.
GymnaxToStoa` uses for an external gymnax environment, just for this in-repo
one instead.

The observation is the grid's "factored channels" encoding - `(has_box,
has_target, has_agent, agent_carrying)`, shape `(grid_size, grid_size, 4)`,
see `stoix.envs.block_moving.input_features._to_factored`. Unlike
`stoix.envs.lightsout.lightsout_env.LightsOutEnv`, no separate goal channel
is needed: the goal is already implicit in the grid itself (`TARGET`/
`BOX_ON_TARGET`/... states directly encode where boxes must end up).

Known gap, not addressed here: `BoxMovingEnv`'s `dense_rewards`/
`negative_sparse` constructor args are accepted but never read by `step` -
reward is always the sparse "fully solved" indicator (1.0/0.0) regardless of
those flags.
"""

from typing import Optional, Tuple

import jax.numpy as jnp
from stoa.env_types import Action, EnvParams, StepType, TimeStep
from stoa.environment import Environment
from stoa.spaces import BoundedArraySpace, DiscreteSpace, Space

from .block_moving_env import BoxMovingEnv, BoxMovingState
from .input_features import _to_factored


class BoxMovingToStoa(Environment):
    """Wraps a `BoxMovingEnv` (or a `wrappers.py` filter around one) in the
    Stoa `Environment` interface."""

    def __init__(self, env: BoxMovingEnv):
        super().__init__()
        self._env = env

    def reset(
        self, rng_key, env_params: Optional[EnvParams] = None
    ) -> Tuple[BoxMovingState, TimeStep]:
        state, info = self._env.reset(rng_key)
        timestep = TimeStep(
            step_type=StepType.FIRST,
            reward=jnp.array(0.0, dtype=jnp.float32),
            discount=jnp.array(1.0, dtype=jnp.float32),
            observation=_to_factored(state.grid),
            extras={"boxes_on_target": info["boxes_on_target"]},
        )
        return state, timestep

    def step(
        self, state: BoxMovingState, action: Action, env_params: Optional[EnvParams] = None
    ) -> Tuple[BoxMovingState, TimeStep]:
        new_state, reward, done, info = self._env.step(state, action)

        # `done` only ever comes from `success` when `terminate_when_success`
        # is set (a Python-level bool at BoxMovingEnv construction time) -
        # otherwise it's the Python constant `False`, which broadcasts fine
        # below. Truncation and success are mutually exclusive by
        # construction (`get_reward` can't fire past `episode_length`), but
        # guard against double-counting anyway.
        truncated = info["truncated"] & (~done)
        step_type = jnp.where(
            done,
            StepType.TERMINATED,
            jnp.where(truncated, StepType.TRUNCATED, StepType.MID),
        )
        discount = jnp.where(done | truncated, 0.0, 1.0).astype(jnp.float32)

        timestep = TimeStep(
            step_type=step_type,
            reward=jnp.asarray(reward, dtype=jnp.float32),
            discount=discount,
            observation=_to_factored(new_state.grid),
            extras={"boxes_on_target": info["boxes_on_target"]},
        )
        return new_state, timestep

    def observation_space(self, env_params: Optional[EnvParams] = None) -> Space:
        grid_size = self._env.grid_size
        return BoundedArraySpace(
            shape=(grid_size, grid_size, 4),
            dtype=jnp.float32,
            minimum=0.0,
            maximum=1.0,
            name="observation",
        )

    def action_space(self, env_params: Optional[EnvParams] = None) -> Space:
        return DiscreteSpace(num_values=self._env.action_space, dtype=jnp.int32, name="action")

    def state_space(self, env_params: Optional[EnvParams] = None) -> Space:
        grid_size = self._env.grid_size
        return BoundedArraySpace(
            shape=(grid_size, grid_size, 4),
            dtype=jnp.float32,
            minimum=0.0,
            maximum=1.0,
            name="state",
        )
