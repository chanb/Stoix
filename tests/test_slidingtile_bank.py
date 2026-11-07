import jax
import jax.numpy as jnp
import jumanji
import numpy as np
from jumanji.environments.logic.sliding_tile_puzzle.reward import SparseRewardFn

from stoix.envs.slidingpuzzle.generator import BankGenerator

SOLVED = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]], np.int8)
ONE_MOVE = np.array([[1, 2, 3], [4, 5, 6], [7, 0, 8]], np.int8)  # blank moves right (action 1) to solve
TWO_MOVES = np.array([[1, 2, 3], [4, 5, 6], [0, 7, 8]], np.int8)


def _bank(tmp_path):
    # 1 board of length 1, 9 boards of length 2; unsorted on purpose.
    puzzle = np.stack([TWO_MOVES] * 5 + [ONE_MOVE] + [TWO_MOVES] * 4)
    moves = np.array([2] * 5 + [1] + [2] * 4, np.int16)
    (tmp_path / "3x3").mkdir()
    np.savez(tmp_path / "3x3" / "train.npz", puzzle=puzzle, moves=moves)
    return str(tmp_path)


def _lengths(gen, n=4000):
    states = jax.vmap(gen)(jax.random.split(jax.random.PRNGKey(0), n))
    flat = states.puzzle.reshape(n, -1)
    blank = states.empty_tile_position
    assert jnp.all(flat[jnp.arange(n), blank[:, 0] * 3 + blank[:, 1]] == 0)
    return np.where(np.all(np.asarray(flat) == ONE_MOVE.reshape(-1), -1), 1, 2)


def test_train_draws_length_uniformly_eval_draws_board_uniformly(tmp_path):
    d = _bank(tmp_path)
    train = _lengths(BankGenerator(3, "train", by_length=True, data_dir=d))
    evals = _lengths(BankGenerator(3, "train", by_length=False, data_dir=d))
    assert abs((train == 1).mean() - 0.5) < 0.05  # uniform over lengths {1, 2}
    assert abs((evals == 1).mean() - 0.1) < 0.03  # uniform over the 10 boards


def test_jumanji_env_solves_bank_board(tmp_path):
    d = _bank(tmp_path)
    gen = BankGenerator(3, "train", by_length=True, data_dir=d)
    env = jumanji.make("SlidingTilePuzzle-v0", generator=gen, reward_fn=SparseRewardFn(), time_limit=40)
    for seed in range(20):
        state, ts = env.reset(jax.random.PRNGKey(seed))
        if np.array_equal(state.puzzle, ONE_MOVE):
            state, ts = env.step(state, jnp.array(1))
            assert ts.reward == 1.0 and ts.last()
            return
    raise AssertionError("no length-1 board drawn in 20 resets")
