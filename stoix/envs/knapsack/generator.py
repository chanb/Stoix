"""Integer-valued (classic textbook) 0-1 knapsack instance generator with a
fresh, per-episode-random capacity.

Jumanji's own `RandomGenerator` (jumanji.environments.packing.knapsack.
generator) samples both item weights and values as continuous floats on
[0, 1], and takes a single fixed `total_budget` at construction time - every
episode drawn from a given generator instance sees the identical capacity.
Item selection is already 0-1 (each item is packed at most once, see
`Knapsack.step`'s `packed_items` masking) regardless of how weights/values
are drawn - `IntegerRandomGenerator` only changes *what* gets drawn: integer
weights/values from `{1, ..., max_weight}`/`{1, ..., max_value}` (the classic
0-1 knapsack formulation), and a fresh integer capacity from
`Uniform{1, ..., max_budget}` on every `__call__` (i.e. every reset) instead
of reusing one fixed value, so problem difficulty varies per episode rather
than being pinned to one sweep-level constant.

`total_budget` (the `Generator` base class attribute) is kept at
`max_budget` purely for `KnapsackViewer`'s rendering scale -
`Knapsack.step`/`_update_state` only ever read `state.remaining_budget` (the
per-instance sampled value), never `self.total_budget`/`env.total_budget`,
so this has no effect on the actual dynamics.
"""

import chex
import jax
import jax.numpy as jnp
from jumanji.environments.packing.knapsack.generator import Generator
from jumanji.environments.packing.knapsack.types import State


class IntegerRandomGenerator(Generator):
    """0-1 knapsack instance generator with integer item weights/values and a
    fresh integer capacity - Uniform{1, ..., max_budget} - drawn every reset."""

    def __init__(self, num_items: int, max_weight: int, max_value: int, max_budget: int):
        super().__init__(num_items, total_budget=float(max_budget))
        self.max_weight = max_weight
        self.max_value = max_value
        self.max_budget = max_budget

    def __call__(self, key: chex.PRNGKey) -> State:
        key, weight_key, value_key, budget_key = jax.random.split(key, 4)

        weights = jax.random.randint(
            weight_key, (self.num_items,), minval=1, maxval=self.max_weight + 1
        ).astype(float)
        values = jax.random.randint(
            value_key, (self.num_items,), minval=1, maxval=self.max_value + 1
        ).astype(float)

        packed_items = jnp.zeros(self.num_items, dtype=bool)

        remaining_budget = jax.random.randint(
            budget_key, (), minval=1, maxval=self.max_budget + 1
        ).astype(float)

        return State(
            weights=weights,
            values=values,
            packed_items=packed_items,
            remaining_budget=remaining_budget,
            key=key,
        )
