import jax
import jax.numpy as jnp
import jumanji
import numpy as np
import pytest
from stoa import ObservationExtractWrapper
from stoa.env_adapters.jumanji import JumanjiToStoa

from stoix.wrappers.grid_observation import GridObservationWrapper


def _env(grid_size, **kwargs):
    from jumanji.environments.logic.sliding_tile_puzzle.generator import RandomWalkGenerator

    env = jumanji.make("SlidingTilePuzzle-v0", generator=RandomWalkGenerator(grid_size, 20))
    return GridObservationWrapper(ObservationExtractWrapper(JumanjiToStoa(env), "puzzle"), **kwargs)


@pytest.mark.parametrize("grid_size", [3, 4])
def test_one_hot_puzzle(grid_size):
    env = _env(grid_size, one_hot=True)
    k = grid_size * grid_size
    space = env.observation_space()
    assert space.shape == (grid_size, grid_size, k)
    state, ts = env.reset(jax.random.PRNGKey(0))
    obs = ts.observation
    assert obs.shape == (grid_size, grid_size, k) and obs.dtype == jnp.float32
    np.testing.assert_array_equal(obs.sum(-1), 1.0)
    np.testing.assert_array_equal(jnp.argmax(obs, -1), state.puzzle)
    for action in range(4):  # whichever moves are legal, the encoding tracks the board
        next_state, ts = env.step(state, jnp.int32(action))
        np.testing.assert_array_equal(jnp.argmax(ts.observation, -1), next_state.puzzle)
    assert space.generate_value().shape == (grid_size, grid_size, k)


def test_default_is_unchanged():
    env = _env(4)
    assert env.observation_space().shape == (4, 4, 1)
