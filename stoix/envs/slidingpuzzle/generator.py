"""Sliding tile puzzle level banks split by optimal solution length, for jumanji's SlidingTilePuzzle.

Port of LoopedActor's envs/slidingpuzzle_env.py bank resets. A bank is <data_dir>/<N>x<N>/<split>.npz with
puzzle [L, N, N] int8 (0 = blank) and moves [L] int16 (optimal length), built by
data_scripts/build_slidingpuzzle_banks.py (3x3: train = optimal length <= 28, test = > 28; 4x4: train <= 60,
test > 60). No scrambling is done.
"""

import os

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jumanji.environments.logic.sliding_tile_puzzle.generator import Generator
from jumanji.environments.logic.sliding_tile_puzzle.types import State

DATA_DIR = os.environ.get(
    "SLIDINGPUZZLE_DATA_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data", "slidingpuzzle"),
)


class BankGenerator(Generator):
    """`by_length=True` (training): draw an optimal length uniformly from the lengths in the bank, then a
    board of that length uniformly. `by_length=False` (evaluation): draw a board uniformly from the bank."""

    def __init__(self, grid_size: int, split: str, by_length: bool = False, data_dir: str = ""):
        super().__init__(grid_size)
        z = np.load(os.path.join(data_dir or DATA_DIR, f"{grid_size}x{grid_size}", f"{split}.npz"))
        order = np.argsort(z["moves"], kind="stable")
        moves = z["moves"][order]
        _, start, count = np.unique(moves, return_index=True, return_counts=True)  # per-length slices
        self.by_length = by_length
        self.puzzles = jnp.asarray(z["puzzle"][order].astype(np.int32))
        self.moves = jnp.asarray(moves.astype(np.int32))
        self.start = jnp.asarray(start.astype(np.int32))
        self.count = jnp.asarray(count.astype(np.int32))

    def __call__(self, key: chex.PRNGKey) -> State:
        key, key_idx, key_len = jax.random.split(key, 3)
        if self.by_length:
            j = jax.random.randint(key_len, (), 0, self.start.shape[0])
            idx = self.start[j] + jax.random.randint(key_idx, (), 0, self.count[j])
        else:
            idx = jax.random.randint(key_idx, (), 0, self.puzzles.shape[0])
        puzzle = self.puzzles[idx]
        b = jnp.argmin(puzzle.reshape(-1))  # the blank (0) is the only zero
        return State(
            puzzle=puzzle,
            empty_tile_position=jnp.stack([b // self.grid_size, b % self.grid_size]),
            key=key,
            step_count=jnp.zeros((), jnp.int32),
        )
