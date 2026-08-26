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
    using `rng`, and returns `(embedding, compute_time,
    first_convergence_step, num_close_steps)` - the last two are latent-
    convergence diagnostics, see `AdaptiveComputationTimeTorso`'s docstring.
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
    predicts a halting probability for that step.

    `num_layers` stacks that many independently-parameterized
    `Dense -> activation` transformations one after another *within* this one
    step (each with its own weights - not shared across the stack, unlike the
    step itself, which is shared/reused across every pondering step). The
    optional pre-LN (`use_layer_norm`) is applied once, before the stack, not
    between its layers."""

    hidden_dim: int
    num_layers: int = 1
    activation: str = "relu"
    kernel_init: Initializer = orthogonal(np.sqrt(2.0))
    use_layer_norm: bool = False

    @nn.compact
    def __call__(self, state: chex.Array) -> Tuple[chex.Array, chex.Array]:
        if self.use_layer_norm:
            state = nn.LayerNorm()(state)
        for i in range(self.num_layers):
            state = nn.Dense(self.hidden_dim, kernel_init=self.kernel_init, name=f"dense_{i}")(
                state
            )
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

    `num_layers` stacks that many `Dense -> activation` transformations
    inside each shared `ACTStep` (see that class) - i.e. how "deep" a single
    pondering step is, independent of `max_steps`/`min_steps` (how many
    pondering steps are taken).

    Also tracks, per example, how quickly the running `state` settles: the
    L2 distance between consecutive steps' states (step `t` vs `t - 1`,
    starting at `t = 2` since there's no step 0 to compare step 1 against) is
    compared against `convergence_threshold`. This gives two diagnostics,
    only while the example hasn't halted yet:

      - `first_convergence_step`: the (1-indexed) step count `t` at which
        that distance first drops below `convergence_threshold`, or `-1` if
        it never does within the steps actually taken.
      - `num_close_steps`: how many steps (not necessarily consecutive) had
        a distance below `convergence_threshold`.
    """

    hidden_dim: int
    max_steps: int = 10
    min_steps: int = 1
    num_layers: int = 1
    activation: str = "relu"
    kernel_init: Initializer = orthogonal(np.sqrt(2.0))
    use_input_layer_norm: bool = False
    use_layer_norm: bool = False
    convergence_threshold: float = 0.1

    @nn.compact
    def __call__(
        self,
        observation: chex.Array,
        rng: Optional[chex.PRNGKey] = None,
        target_compute_time: Optional[chex.Array] = None,
        deterministic: bool = False,
    ) -> Tuple[chex.Array, ...]:
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
            `(embedding, compute_time, first_convergence_step,
            num_close_steps)` when `target_compute_time` is None, or
            `(embedding, halting_log_prob)` when replaying a known
            trajectory (see class docstring for the convergence diagnostics).
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
            self.hidden_dim,
            self.num_layers,
            self.activation,
            self.kernel_init,
            use_layer_norm=self.use_layer_norm,
        )

        still_running = jnp.ones(batch_shape, dtype=bool)
        num_steps_taken = jnp.zeros(batch_shape)
        halting_log_prob = jnp.zeros(batch_shape)
        final_state = state
        first_convergence_step = jnp.full(batch_shape, -1.0)
        num_close_steps = jnp.zeros(batch_shape)

        for step in range(self.max_steps):
            prev_state = state
            state, halting_prob = step_fn(state)
            step_count = step + 1
            is_final_step = step_count == self.max_steps
            can_halt = step_count >= self.min_steps

            if not replaying and step > 0:
                # Only meaningful from the second step on (step 0 has no
                # previous ACT-step state to compare against), and only while
                # still running - once an example has halted, `state` keeps
                # being recomputed (the loop is unrolled over every example)
                # but that continuation is discarded, so it says nothing
                # about the example's real trajectory.
                l2_dist = jnp.linalg.norm(state - prev_state, axis=-1)
                is_close = still_running & (l2_dist < self.convergence_threshold)
                num_close_steps = num_close_steps + is_close.astype(jnp.float32)
                first_convergence_step = jnp.where(
                    is_close & (first_convergence_step < 0),
                    jnp.float32(step_count),
                    first_convergence_step,
                )

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
            # Forced steps (before min_steps, or the forced halt at max_steps)
            # contribute no log prob: the true probability of a forced outcome
            # is 1, so log(1) = 0 - not `step_log_prob`, which would otherwise
            # reflect a "choice" the policy never actually got to make.
            halting_log_prob = halting_log_prob + jnp.where(
                jnp.logical_and(still_running, not is_forced_step), step_log_prob, 0.0
            )
            num_steps_taken = num_steps_taken + still_running.astype(jnp.float32)
            final_state = jnp.where(halts_this_step[..., None], state, final_state)

            still_running = still_running & (~halts_this_step)

        if replaying:
            return final_state, halting_log_prob
        return final_state, num_steps_taken, first_convergence_step, num_close_steps


class UnsharedAdaptiveComputationTimeTorso(nn.Module):
    """Like `AdaptiveComputationTimeTorso`, but with no weight sharing across
    pondering steps: each of the `max_steps` steps gets its own
    independently-parameterized `ACTStep` (a fresh `Dense[+...] ->
    activation -> halting head`), rather than one shared step reused at
    every iteration. So this is a genuinely deep, unshared stack of up to
    `max_steps` distinct layers - not a weight-tied recurrent block -
    "adaptive computation time" in the sense that halting can still stop
    early anywhere in that stack, but each depth reached is its own
    dedicated layer rather than another application of the same one.
    `compute_time` accordingly means "how many of these distinct layers were
    used", not "how many times was the shared step applied."

    Deliberately a separate class from `AdaptiveComputationTimeTorso` (not a
    shared-vs-unshared flag on it), since the two have a genuinely different
    parameter count/growth behaviour as `max_steps` changes - increasing
    `max_steps` here adds a whole new layer's worth of parameters, whereas
    for the shared version it's free.

    Same sampled/replayed/deterministic halting modes, `min_steps`/
    `max_steps` semantics (see that class's docstring), `num_layers`
    stacking *within* each step (see `ACTStep`), and latent-convergence
    diagnostics as `AdaptiveComputationTimeTorso` - only how `step_fn` is
    instantiated differs (see `__call__`).
    """

    hidden_dim: int
    max_steps: int = 10
    min_steps: int = 1
    num_layers: int = 1
    activation: str = "relu"
    kernel_init: Initializer = orthogonal(np.sqrt(2.0))
    use_input_layer_norm: bool = False
    use_layer_norm: bool = False
    convergence_threshold: float = 0.1

    @nn.compact
    def __call__(
        self,
        observation: chex.Array,
        rng: Optional[chex.PRNGKey] = None,
        target_compute_time: Optional[chex.Array] = None,
        deterministic: bool = False,
    ) -> Tuple[chex.Array, ...]:
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
            `(embedding, compute_time, first_convergence_step,
            num_close_steps)` when `target_compute_time` is None, or
            `(embedding, halting_log_prob)` when replaying a known
            trajectory (see class docstring for the convergence diagnostics).
        """
        batch_shape = observation.shape[:-1]
        replaying = target_compute_time is not None
        if not replaying and not deterministic:
            if rng is None:
                raise ValueError(
                    "rng must be provided to UnsharedAdaptiveComputationTimeTorso when sampling "
                    "(i.e. target_compute_time is None and deterministic=False)."
                )
        if not (1 <= self.min_steps <= self.max_steps):
            raise ValueError(
                f"min_steps must be between 1 and max_steps ({self.max_steps}), "
                f"got min_steps={self.min_steps}."
            )

        # Project into the pondering width once up front, so every step's
        # (separately-parameterized) ACTStep sees a `hidden_dim`-sized state.
        if self.use_input_layer_norm:
            observation = nn.LayerNorm()(observation)
        state = parse_activation_fn(self.activation)(
            nn.Dense(self.hidden_dim, kernel_init=self.kernel_init)(observation)
        )

        still_running = jnp.ones(batch_shape, dtype=bool)
        num_steps_taken = jnp.zeros(batch_shape)
        halting_log_prob = jnp.zeros(batch_shape)
        final_state = state
        first_convergence_step = jnp.full(batch_shape, -1.0)
        num_close_steps = jnp.zeros(batch_shape)

        for step in range(self.max_steps):
            prev_state = state
            # A fresh ACTStep instance per step (unique `name`), unlike
            # AdaptiveComputationTimeTorso's single `step_fn` created once
            # outside this loop and reused - this is what makes every step's
            # parameters independent rather than shared/tied.
            step_fn = ACTStep(
                self.hidden_dim,
                self.num_layers,
                self.activation,
                self.kernel_init,
                use_layer_norm=self.use_layer_norm,
                name=f"act_step_{step}",
            )
            state, halting_prob = step_fn(state)
            step_count = step + 1
            is_final_step = step_count == self.max_steps
            can_halt = step_count >= self.min_steps

            if not replaying and step > 0:
                # Only meaningful from the second step on (step 0 has no
                # previous ACT-step state to compare against), and only while
                # still running - once an example has halted, `state` keeps
                # being recomputed (the loop is unrolled over every example)
                # but that continuation is discarded, so it says nothing
                # about the example's real trajectory.
                l2_dist = jnp.linalg.norm(state - prev_state, axis=-1)
                is_close = still_running & (l2_dist < self.convergence_threshold)
                num_close_steps = num_close_steps + is_close.astype(jnp.float32)
                first_convergence_step = jnp.where(
                    is_close & (first_convergence_step < 0),
                    jnp.float32(step_count),
                    first_convergence_step,
                )

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
            # Forced steps (before min_steps, or the forced halt at max_steps)
            # contribute no log prob: the true probability of a forced outcome
            # is 1, so log(1) = 0 - not `step_log_prob`, which would otherwise
            # reflect a "choice" the policy never actually got to make.
            halting_log_prob = halting_log_prob + jnp.where(
                jnp.logical_and(still_running, not is_forced_step), step_log_prob, 0.0
            )
            num_steps_taken = num_steps_taken + still_running.astype(jnp.float32)
            final_state = jnp.where(halts_this_step[..., None], state, final_state)

            still_running = still_running & (~halts_this_step)

        if replaying:
            return final_state, halting_log_prob
        return final_state, num_steps_taken, first_convergence_step, num_close_steps


class RecurrentACTStep(nn.Module):
    """A single shared computation step for `GRUAdaptiveComputationTimeTorso`:
    unlike `ACTStep` (which only ever sees its own running state, having
    discarded the original observation after the first projection), this
    re-feeds the fixed, once-encoded input embedding into a GRU cell
    alongside the running recurrent state `carry` at every step - i.e. the
    step function is `carry_{t+1} = GRUCell(carry_t, input_embedding)` -
    then predicts a halting probability for that step from the new carry.

    `num_layers` stacks that many independently-parameterized GRU cells
    within this one step - like a multi-layer GRU: layer 0 takes the fixed
    `input_embedding`, layer `i > 0` takes layer `i - 1`'s output as its
    input, and each layer keeps its *own* carry across pondering steps (so
    `carry` here is `(num_layers, ..., hidden_dim)`, not `(..., hidden_dim)`).
    The halting probability is predicted from the top (last) layer's new
    carry, which is also what gets used as this step's "public" state (e.g.
    for the latent-convergence diagnostics and the final returned
    embedding) - see `GRUAdaptiveComputationTimeTorso`."""

    hidden_dim: int
    num_layers: int = 1
    kernel_init: Initializer = orthogonal(np.sqrt(2.0))

    @nn.compact
    def __call__(
        self, carry: chex.Array, input_embedding: chex.Array
    ) -> Tuple[chex.Array, chex.Array]:
        x = input_embedding
        new_carries = []
        for i in range(self.num_layers):
            layer_carry, x = nn.GRUCell(
                features=self.hidden_dim, kernel_init=self.kernel_init, name=f"gru_{i}"
            )(carry[i], x)
            new_carries.append(layer_carry)
        new_carry = jnp.stack(new_carries, axis=0)
        halting_prob = nn.sigmoid(nn.Dense(1, kernel_init=self.kernel_init)(new_carries[-1]))
        halting_prob = jnp.clip(halting_prob.squeeze(axis=-1), _PROB_EPS, 1.0 - _PROB_EPS)
        return new_carry, halting_prob


class GRUAdaptiveComputationTimeTorso(nn.Module):
    """Adaptive Computation Time torso whose shared step is a genuine
    recurrent cell that re-feeds the (fixed) encoded input at every
    pondering step - i.e. `S -> Linear -> LayerNorm -> Recurrent Block`,
    where the Recurrent Block is a GRU cell taking both that encoded `S` and
    its own running carry `C_i` at every step, producing `C_{i+1}`. This is
    the key structural difference from `AdaptiveComputationTimeTorso`, whose
    shared `ACTStep` only ever operates on its own running state and never
    sees the raw observation again after the very first projection into it -
    here, the block can always "look back" at the original input, not just
    whatever survived being carried forward through its own state.

    Otherwise identical to `AdaptiveComputationTimeTorso`: same sampled/
    replayed/deterministic halting modes, same `min_steps`/`max_steps`
    semantics, and the same latent-convergence diagnostics
    (`first_convergence_step`, `num_close_steps` - see that class's
    docstring), computed here on the GRU's carry instead of the MLP step's
    state.

    The recurrent carry `C_0` is initialized to zero, the standard RNN
    convention: the block only learns anything about the observation once it
    first consumes the encoded input as the GRU's `inputs` argument. The
    final carry is squashed through `tanh` before being returned (matching
    the "Tanh" applied after the Recurrent Block, ahead of the
    action/value heads' own `Linear` projection, which live outside this
    torso as separate head modules - see `stoix.networks.base_compute.
    FeedForwardActorWithComputeTime`).

    `use_input_layer_norm` normalizes the encoded input embedding (after its
    `Linear` projection, before it's fed into the recurrent block every
    step) - unlike the other torsos in this module, where `use_input_layer_norm`
    normalizes the *raw* observation before its projection.

    `num_layers` stacks that many GRU cells inside each shared step (see
    `RecurrentACTStep`) - a multi-layer GRU applied at every pondering step,
    independent of `max_steps`/`min_steps`. Each stacked layer keeps its own
    carry across pondering steps; the diagnostics/final embedding below use
    only the top layer's carry, matching what `RecurrentACTStep` returns as
    its halting-probability input.
    """

    hidden_dim: int
    max_steps: int = 10
    min_steps: int = 1
    num_layers: int = 1
    kernel_init: Initializer = orthogonal(np.sqrt(2.0))
    use_input_layer_norm: bool = False
    convergence_threshold: float = 0.1

    @nn.compact
    def __call__(
        self,
        observation: chex.Array,
        rng: Optional[chex.PRNGKey] = None,
        target_compute_time: Optional[chex.Array] = None,
        deterministic: bool = False,
    ) -> Tuple[chex.Array, ...]:
        """
        Args:
            observation: the input embedding to ponder over; re-fed as the
                GRU's `inputs` at every pondering step (see class docstring).
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
            `(embedding, compute_time, first_convergence_step,
            num_close_steps)` when `target_compute_time` is None, or
            `(embedding, halting_log_prob)` when replaying a known
            trajectory (see class docstring for the convergence diagnostics).
        """
        batch_shape = observation.shape[:-1]
        replaying = target_compute_time is not None
        if not replaying and not deterministic:
            if rng is None:
                raise ValueError(
                    "rng must be provided to GRUAdaptiveComputationTimeTorso when sampling "
                    "(i.e. target_compute_time is None and deterministic=False)."
                )
        if not (1 <= self.min_steps <= self.max_steps):
            raise ValueError(
                f"min_steps must be between 1 and max_steps ({self.max_steps}), "
                f"got min_steps={self.min_steps}."
            )

        # Encoded once up front, then re-fed into the recurrent block at
        # every pondering step (unlike `AdaptiveComputationTimeTorso`, which
        # only uses the observation to seed the very first state).
        input_embedding = nn.Dense(self.hidden_dim, kernel_init=self.kernel_init)(observation)
        if self.use_input_layer_norm:
            input_embedding = nn.LayerNorm()(input_embedding)
        step_fn = RecurrentACTStep(self.hidden_dim, self.num_layers, self.kernel_init)

        # `carry` holds every stacked layer's own recurrent state
        # (`(num_layers, *batch_shape, hidden_dim)`), threaded between
        # pondering steps by `step_fn`. `state` (below) is just the top
        # layer's carry - what everything else here (diagnostics, halting,
        # the final returned embedding) actually uses, same as when
        # `num_layers == 1`.
        carry = jnp.zeros((self.num_layers,) + batch_shape + (self.hidden_dim,))
        state = carry[-1]

        still_running = jnp.ones(batch_shape, dtype=bool)
        num_steps_taken = jnp.zeros(batch_shape)
        halting_log_prob = jnp.zeros(batch_shape)
        final_state = state
        first_convergence_step = jnp.full(batch_shape, -1.0)
        num_close_steps = jnp.zeros(batch_shape)

        for step in range(self.max_steps):
            prev_state = state
            carry, halting_prob = step_fn(carry, input_embedding)
            state = carry[-1]
            step_count = step + 1
            is_final_step = step_count == self.max_steps
            can_halt = step_count >= self.min_steps

            if not replaying and step > 0:
                # Only meaningful from the second step on (step 0 has no
                # previous carry to compare against - `C_0` is zeros, not a
                # computed step), and only while still running - once an
                # example has halted, the carry keeps being recomputed (the
                # loop is unrolled over every example) but that continuation
                # is discarded, so it says nothing about the example's real
                # trajectory.
                l2_dist = jnp.linalg.norm(state - prev_state, axis=-1)
                is_close = still_running & (l2_dist < self.convergence_threshold)
                num_close_steps = num_close_steps + is_close.astype(jnp.float32)
                first_convergence_step = jnp.where(
                    is_close & (first_convergence_step < 0),
                    jnp.float32(step_count),
                    first_convergence_step,
                )

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
            # Forced steps (before min_steps, or the forced halt at max_steps)
            # contribute no log prob: the true probability of a forced outcome
            # is 1, so log(1) = 0 - not `step_log_prob`, which would otherwise
            # reflect a "choice" the policy never actually got to make.
            halting_log_prob = halting_log_prob + jnp.where(
                jnp.logical_and(still_running, not is_forced_step), step_log_prob, 0.0
            )
            num_steps_taken = num_steps_taken + still_running.astype(jnp.float32)
            final_state = jnp.where(halts_this_step[..., None], state, final_state)

            still_running = still_running & (~halts_this_step)

        final_state = jnp.tanh(final_state)

        if replaying:
            return final_state, halting_log_prob
        return final_state, num_steps_taken, first_convergence_step, num_close_steps


class IRUCell(nn.Module):
    """The "interpolation recurrent unit" (IRU) cell math, factored out of
    `IRUStep` so it can be stacked: the running cell state `c_i` is updated
    as an interpolation between its previous value and a fresh candidate
    value, gated by a learned per-step "forget" probability - both computed
    from the same concatenation of the fixed input `x` and `tanh(c_i)`:

        f_i = sigmoid(FORGET_theta([x, tanh(c_i)]))
        I_i = tanh(INPUT_theta([x, tanh(c_i)]))
        c_{i+1} = f_i * c_i + (1 - f_i) * I_i

    where `FORGET_theta`/`INPUT_theta` are each a single learned `Dense`
    layer. No halting head here - see `IRUStep`, which wraps one or more of
    these into a shared pondering step."""

    hidden_dim: int
    kernel_init: Initializer = orthogonal(np.sqrt(2.0))

    @nn.compact
    def __call__(self, carry: chex.Array, x: chex.Array) -> chex.Array:
        gate_input = jnp.concatenate([x, jnp.tanh(carry)], axis=-1)
        forget_gate = nn.sigmoid(
            nn.Dense(self.hidden_dim, kernel_init=self.kernel_init, name="forget")(gate_input)
        )
        candidate = jnp.tanh(
            nn.Dense(self.hidden_dim, kernel_init=self.kernel_init, name="input")(gate_input)
        )
        return forget_gate * carry + (1.0 - forget_gate) * candidate


class IRUStep(nn.Module):
    """A single shared computation step for `IRUAdaptiveComputationTimeTorso`:
    updates the running cell state via `IRUCell` and predicts a halting
    probability for that step from the new cell state.

    `num_layers` stacks that many independently-parameterized `IRUCell`s
    within this one step - layer 0 takes the fixed `input_embedding`, layer
    `i > 0` takes `tanh` of layer `i - 1`'s new cell state as its own `x`
    (mirroring how `RecurrentACTStep`'s stacked GRU layers feed each other,
    and matching what `IRUCell` itself already applies `tanh` to for its own
    gate inputs). Each stacked layer keeps its own carry across pondering
    steps (so `carry` here is `(num_layers, ..., hidden_dim)`, not
    `(..., hidden_dim)`). The halting probability is predicted from the top
    (last) layer's new carry, which is also what gets used as this step's
    "public" state - see `IRUAdaptiveComputationTimeTorso`."""

    hidden_dim: int
    num_layers: int = 1
    kernel_init: Initializer = orthogonal(np.sqrt(2.0))

    @nn.compact
    def __call__(
        self, carry: chex.Array, input_embedding: chex.Array
    ) -> Tuple[chex.Array, chex.Array]:
        x = input_embedding
        new_carries = []
        for i in range(self.num_layers):
            layer_carry = IRUCell(self.hidden_dim, self.kernel_init, name=f"cell_{i}")(
                carry[i], x
            )
            new_carries.append(layer_carry)
            x = jnp.tanh(layer_carry)
        new_carry = jnp.stack(new_carries, axis=0)

        halting_prob = nn.sigmoid(nn.Dense(1, kernel_init=self.kernel_init)(new_carries[-1]))
        halting_prob = jnp.clip(halting_prob.squeeze(axis=-1), _PROB_EPS, 1.0 - _PROB_EPS)
        return new_carry, halting_prob


class IRUAdaptiveComputationTimeTorso(nn.Module):
    """Adaptive Computation Time torso whose shared step is the
    "interpolation recurrent unit" (IRU, see `IRUStep`) rather than a GRU
    cell (`GRUAdaptiveComputationTimeTorso`) or a plain residual MLP
    (`AdaptiveComputationTimeTorso`): `S -> Linear -> LayerNorm -> Recurrent
    Block`, where the Recurrent Block interpolates its own running cell
    state `C_i` towards a candidate computed from `[S, tanh(C_i)]`, gated by
    a learned forget probability computed the same way - see `IRUStep`'s
    docstring for the exact equations. Like `GRUAdaptiveComputationTimeTorso`,
    the fixed encoded input `S` is re-fed into the block at every pondering
    step, so the block can always "look back" at the original input, not
    just whatever survived being carried forward through its own state.

    Otherwise identical to `AdaptiveComputationTimeTorso`/
    `GRUAdaptiveComputationTimeTorso`: same sampled/replayed/deterministic
    halting modes, same `min_steps`/`max_steps` semantics, the same
    latent-convergence diagnostics (`first_convergence_step`,
    `num_close_steps`, computed here on the IRU's cell state), a zero-
    initialized `C_0`, and a final `tanh` applied before the embedding is
    returned (ahead of the action/value heads' own `Linear` projection,
    which live outside this torso - see `stoix.networks.base_compute.
    FeedForwardActorWithComputeTime`).

    `use_input_layer_norm` normalizes the encoded input embedding (after its
    `Linear` projection, before it's fed into the recurrent block every
    step).

    `num_layers` stacks that many `IRUCell`s inside each shared step (see
    `IRUStep`) - independent of `max_steps`/`min_steps`. Each stacked layer
    keeps its own carry across pondering steps; the diagnostics/final
    embedding below use only the top layer's carry, matching what `IRUStep`
    returns as its halting-probability input.
    """

    hidden_dim: int
    max_steps: int = 10
    min_steps: int = 1
    num_layers: int = 1
    kernel_init: Initializer = orthogonal(np.sqrt(2.0))
    use_input_layer_norm: bool = False
    convergence_threshold: float = 0.1

    @nn.compact
    def __call__(
        self,
        observation: chex.Array,
        rng: Optional[chex.PRNGKey] = None,
        target_compute_time: Optional[chex.Array] = None,
        deterministic: bool = False,
    ) -> Tuple[chex.Array, ...]:
        """
        Args:
            observation: the input embedding to ponder over; re-fed as the
                IRU's `x` at every pondering step (see class docstring).
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
            `(embedding, compute_time, first_convergence_step,
            num_close_steps)` when `target_compute_time` is None, or
            `(embedding, halting_log_prob)` when replaying a known
            trajectory (see class docstring for the convergence diagnostics).
        """
        batch_shape = observation.shape[:-1]
        replaying = target_compute_time is not None
        if not replaying and not deterministic:
            if rng is None:
                raise ValueError(
                    "rng must be provided to IRUAdaptiveComputationTimeTorso when sampling "
                    "(i.e. target_compute_time is None and deterministic=False)."
                )
        if not (1 <= self.min_steps <= self.max_steps):
            raise ValueError(
                f"min_steps must be between 1 and max_steps ({self.max_steps}), "
                f"got min_steps={self.min_steps}."
            )

        # Encoded once up front, then re-fed into the recurrent block at
        # every pondering step (unlike `AdaptiveComputationTimeTorso`, which
        # only uses the observation to seed the very first state).
        input_embedding = nn.Dense(self.hidden_dim, kernel_init=self.kernel_init)(observation)
        if self.use_input_layer_norm:
            input_embedding = nn.LayerNorm()(input_embedding)
        step_fn = IRUStep(self.hidden_dim, self.num_layers, self.kernel_init)

        # `carry` holds every stacked layer's own cell state
        # (`(num_layers, *batch_shape, hidden_dim)`), threaded between
        # pondering steps by `step_fn`. `state` (below) is just the top
        # layer's carry - what everything else here (diagnostics, halting,
        # the final returned embedding) actually uses, same as when
        # `num_layers == 1`.
        carry = jnp.zeros((self.num_layers,) + batch_shape + (self.hidden_dim,))
        state = carry[-1]

        still_running = jnp.ones(batch_shape, dtype=bool)
        num_steps_taken = jnp.zeros(batch_shape)
        halting_log_prob = jnp.zeros(batch_shape)
        final_state = state
        first_convergence_step = jnp.full(batch_shape, -1.0)
        num_close_steps = jnp.zeros(batch_shape)

        for step in range(self.max_steps):
            prev_state = state
            carry, halting_prob = step_fn(carry, input_embedding)
            state = carry[-1]
            step_count = step + 1
            is_final_step = step_count == self.max_steps
            can_halt = step_count >= self.min_steps

            if not replaying and step > 0:
                # Only meaningful from the second step on (step 0 has no
                # previous cell state to compare against - `C_0` is zeros,
                # not a computed step), and only while still running - once
                # an example has halted, the cell state keeps being
                # recomputed (the loop is unrolled over every example) but
                # that continuation is discarded, so it says nothing about
                # the example's real trajectory.
                l2_dist = jnp.linalg.norm(state - prev_state, axis=-1)
                is_close = still_running & (l2_dist < self.convergence_threshold)
                num_close_steps = num_close_steps + is_close.astype(jnp.float32)
                first_convergence_step = jnp.where(
                    is_close & (first_convergence_step < 0),
                    jnp.float32(step_count),
                    first_convergence_step,
                )

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
            # Forced steps (before min_steps, or the forced halt at max_steps)
            # contribute no log prob: the true probability of a forced outcome
            # is 1, so log(1) = 0 - not `step_log_prob`, which would otherwise
            # reflect a "choice" the policy never actually got to make.
            halting_log_prob = halting_log_prob + jnp.where(
                jnp.logical_and(still_running, not is_forced_step), step_log_prob, 0.0
            )
            num_steps_taken = num_steps_taken + still_running.astype(jnp.float32)
            final_state = jnp.where(halts_this_step[..., None], state, final_state)

            still_running = still_running & (~halts_this_step)

        final_state = jnp.tanh(final_state)

        if replaying:
            return final_state, halting_log_prob
        return final_state, num_steps_taken, first_convergence_step, num_close_steps
