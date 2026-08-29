"""Sokoban reward variant with no per-step penalty.

Jumanji's `DenseReward` (jumanji.environments.routing.sokoban.reward) adds a
constant `STEP_BONUS = -0.1` (jumanji.environments.routing.sokoban.constants)
on every transition, on top of the +1-per-box-newly-on-target and
+10-on-full-solve bonuses. That per-step penalty rewards *speed*, not just
solving - which conflates "did more computation help" with "did the agent
dawdle", the opposite of what the RAMDP fixed/adaptive compute-budget sweeps
(see ramdp_experiments/jumanji_fixed_budget_sweep.py) want to isolate.

`DenseRewardNoStepPenalty` keeps the same box-on-target/level-complete
shaping as `DenseReward` but drops the `STEP_BONUS` term.
"""

import chex
from jumanji.environments.routing.sokoban.constants import LEVEL_COMPLETE_BONUS, N_BOXES, SINGLE_BOX_BONUS
from jumanji.environments.routing.sokoban.reward import RewardFn
from jumanji.environments.routing.sokoban.types import State


class DenseRewardNoStepPenalty(RewardFn):
    def __call__(
        self,
        state: State,
        action: chex.Array,
        next_state: State,
    ) -> chex.Array:
        num_box_target = self.count_targets(state)
        next_num_box_target = self.count_targets(next_state)

        level_completed = next_num_box_target == N_BOXES

        return (
            SINGLE_BOX_BONUS * (next_num_box_target - num_box_target)
            + LEVEL_COMPLETE_BONUS * level_completed
        )
