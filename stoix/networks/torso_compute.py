"""Adaptive-computation-time torsos.

Unlike a normal torso that runs a fixed number of layers, an
`AdaptiveComputationTimeTorso` repeatedly applies a *shared* computation step
to its own output, and at every step actually *samples* a Bernoulli "halt
now?" decision from a learned per-step halting probability - rather than
Graves' (2016) original ACT, which never samples and instead takes a
deterministic weighted sum over all steps' outputs. `compute_time` here is
the literal number of steps sampled before halting: a genuine, measured
count (like the number of hidden layers/CoT steps used), not a value
predicted by a separate head, and not a soft expectation over steps.

Because halting is now a discrete sampled decision, it cannot be trained by
ordinary backprop. Instead - exactly like the environment action in
REINFORCE - it is trained via the score-function (REINFORCE) estimator using
the log-probability of the sampled halting trajectory. To support this, this
module has two modes:

  - Rollout mode (`target_compute_time=None`): samples a halting trajectory
    using `rng`, and returns `(embedding, compute_time, halting_log_prob)`.
  - Replay mode (`target_compute_time=<array from a stored transition>`):
    deterministically replays exactly that many steps (no sampling/rng
    needed) and returns the log-probability, under the current parameters,
    of having produced that exact halting trajectory. This is what the loss
    function should use, mirroring how `distribution.log_prob(stored_action)`
    is used for the environment action rather than re-sampling it.

A transformer-with-chain-of-thought torso would plug into the same
interface: sample whether to keep "thinking" after each CoT step, with
`compute_time` becoming the number of CoT steps actually taken.
"""

from typing import Optional, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.linen.initializers import Initializer, orthogonal

from stoix.networks.utils import parse_activation_fn

_PROB_EPS = 1e-6


class ACTStep(nn.Module):
    """A single shared computation step: updates the running state and
    predicts a halting probability for that step."""

    hidden_dim: int
    activation: str = "relu"
    kernel_init: Initializer = orthogonal(np.sqrt(2.0))
    use_layer_norm: bool = False

    @nn.compact
    def __call__(self, state: chex.Array) -> Tuple[chex.Array, chex.Array]:
        if self.use_layer_norm:
            state = nn.LayerNorm()(state)
        state = nn.Dense(self.hidden_dim, kernel_init=self.kernel_init)(state)
        state = parse_activation_fn(self.activation)(state)
        halting_prob = nn.sigmoid(nn.Dense(1, kernel_init=self.kernel_init)(state))
        halting_prob = jnp.clip(halting_prob.squeeze(axis=-1), _PROB_EPS, 1.0 - _PROB_EPS)
        return state, halting_prob


class AdaptiveComputationTimeTorso(nn.Module):
    """Adaptive Computation Time torso with genuinely sampled halting.

    Applies a shared computation step up to `max_steps` times. At each step a
    halting unit predicts a probability of stopping, and a "halt now?"
    decision is sampled (or, in replay mode, is a known fixed outcome). The
    output embedding is the state at the step the trajectory actually
    halted - not a weighted combination of every step's state.

    Because `max_steps` must be static for JAX, this unrolls a fixed number
    of steps and masks out steps after halting, rather than using a
    dynamically-shaped `while_loop`.

    `min_steps` forbids halting (voluntarily, in replay, or greedily) before
    that many steps have been taken - `is_final_step` still forces a halt at
    `max_steps` regardless. Setting `min_steps == max_steps` therefore removes
    adaptivity entirely: every example always takes exactly `max_steps` -
    useful as a fixed-budget baseline against the adaptive policy.

    `use_input_layer_norm` normalizes the raw `observation` before it's
    projected into the shared recurrent width, once, up front.
    `use_layer_norm` normalizes the running `state` inside every shared
    `ACTStep` before that step's Dense layer (pre-LN), reused across all
    `max_steps` applications since the step's parameters are shared.
    """

    hidden_dim: int
    max_steps: int = 10
    min_steps: int = 1
    activation: str = "relu"
    kernel_init: Initializer = orthogonal(np.sqrt(2.0))
    use_input_layer_norm: bool = False
    use_layer_norm: bool = False

    @nn.compact
    def __call__(
        self,
        observation: chex.Array,
        rng: Optional[chex.PRNGKey] = None,
        target_compute_time: Optional[chex.Array] = None,
        deterministic: bool = False,
    ) -> Tuple[chex.Array, chex.Array]:
        """
        Args:
            observation: the input embedding to ponder over.
            rng: PRNG key used to sample halting decisions. Required unless
                `target_compute_time` is given or `deterministic=True`.
            target_compute_time: if given (e.g. `Transition.compute_time`
                from a prior rollout), halting is *not* sampled - instead the
                trajectory is replayed deterministically to halt at exactly
                this many steps per example, and the returned second output
                is `halting_log_prob`: the log-probability of that exact
                halting trajectory under the current parameters. Use this in
                loss functions.
            deterministic: if True (and `target_compute_time` is None), halt
                as soon as the halting probability crosses 0.5 instead of
                sampling. Useful for greedy evaluation.

        Returns:
            `(embedding, compute_time)` when `target_compute_time` is None,
            or `(embedding, halting_log_prob)` when replaying a known
            trajectory.
        """
        batch_shape = observation.shape[:-1]
        replaying = target_compute_time is not None
        if not replaying and not deterministic:
            if rng is None:
                raise ValueError(
                    "rng must be provided to AdaptiveComputationTimeTorso when sampling "
                    "(i.e. target_compute_time is None and deterministic=False)."
                )
        if not (1 <= self.min_steps <= self.max_steps):
            raise ValueError(
                f"min_steps must be between 1 and max_steps ({self.max_steps}), "
                f"got min_steps={self.min_steps}."
            )

        # Project into the shared recurrent width once up front, so the shared
        # step below always sees `hidden_dim`-sized states (its parameters are
        # reused across every pondering step).
        if self.use_input_layer_norm:
            observation = nn.LayerNorm()(observation)
        state = parse_activation_fn(self.activation)(
            nn.Dense(self.hidden_dim, kernel_init=self.kernel_init)(observation)
        )
        step_fn = ACTStep(
            self.hidden_dim, self.activation, self.kernel_init, use_layer_norm=self.use_layer_norm
        )

        still_running = jnp.ones(batch_shape, dtype=bool)
        num_steps_taken = jnp.zeros(batch_shape)
        halting_log_prob = jnp.zeros(batch_shape)
        final_state = state

        for step in range(self.max_steps):
            state, halting_prob = step_fn(state)
            step_count = step + 1
            is_final_step = step_count == self.max_steps
            can_halt = step_count >= self.min_steps

            if replaying:
                halts_this_step = still_running & (
                    ((step_count >= target_compute_time) & can_halt) | is_final_step
                )
            elif deterministic:
                halts_this_step = still_running & (
                    ((halting_prob >= 0.5) & can_halt) | is_final_step
                )
            else:
                rng, step_rng = jax.random.split(rng)
                sampled_halt = jax.random.bernoulli(step_rng, halting_prob)
                halts_this_step = still_running & ((sampled_halt & can_halt) | is_final_step)

            # Halting is forced (not a free policy choice) before min_steps or at
            # max_steps - stop-gradient halting_prob there so REINFORCE doesn't
            # credit/blame the halting head for an outcome it didn't control.
            is_forced_step = (step_count < self.min_steps) or is_final_step
            halting_prob_for_log = jnp.where(
                is_forced_step, jax.lax.stop_gradient(halting_prob), halting_prob
            )
            step_log_prob = jnp.where(
                halts_this_step,
                jnp.log(halting_prob_for_log),
                jnp.log(1.0 - halting_prob_for_log),
            )
            halting_log_prob = halting_log_prob + jnp.where(still_running, step_log_prob, 0.0)
            num_steps_taken = num_steps_taken + still_running.astype(jnp.float32)
            final_state = jnp.where(halts_this_step[..., None], state, final_state)

            still_running = still_running & (~halts_this_step)

        if replaying:
            return final_state, halting_log_prob
        return final_state, num_steps_taken
