"""A transformer torso whose adaptive computation is a chain of thought.

`TransformerChainOfThoughtTorso` is a transformer analogue of
`stoix.networks.torso_compute.AdaptiveComputationTimeTorso`: instead of a
shared MLP step repeatedly refining a single hidden state, it maintains a
growing sequence of "thought" tokens - a scratchpad - and, at every step, runs
a (weight-shared, recurrent-in-depth) transformer over the whole scratchpad
so far to produce the next thought. A halting unit reads that thought and
samples (or, in replay mode, replays) a "keep thinking?" decision, exactly
like the MLP ACT torso. `compute_time` here is the actual number of CoT steps
taken before the model chose to stop and act - the interface is identical to
`AdaptiveComputationTimeTorso`, so this is a drop-in replacement for it as an
actor's `pre_torso` (see `stoix.networks.base_compute.FeedForwardActorWithComputeTime`).

Concretely, per CoT step `t` (0-indexed):
  1. Run `num_layers` shared transformer blocks, *causally masked*, over the
     scratchpad tokens produced so far (`t + 1` of them, including the
     initial token derived from the observation), with learned positional
     embeddings.
  2. Take the representation at the last (most recent) position as this
     step's "thought" - this is what the halting unit and, if halting, the
     action head see.
  3. Decide whether to halt (sampled / replayed / deterministic, same as
     `AdaptiveComputationTimeTorso`). If not halting, append the thought to
     the scratchpad and continue.

The causal mask means each position's representation is a function only of
its own past, at every layer. Because the whole (causal) scratchpad is
visible via self-attention, a step's thought can depend on every earlier
thought, not just the immediately preceding one - the natural inductive bias
for a chain of thought, as opposed to the Markovian single-hidden-state
recurrence of the MLP ACT torso.

Since `max_steps` must be static for JAX, the CoT loop runs as an `nn.scan`
(a weight-tied `jax.lax.scan` over the shared transformer blocks/halting
head), not a Python loop. Each layer maintains a KV-cache across steps (see
`TransformerBlock.step`), so a step only ever runs the *single new token*
through the stack - attention reads the growing cache instead of
recomputing it, and the MLP only ever touches one token per step - giving
O(max_steps) total compute rather than O(max_steps^2) for a design that
recomputed the whole scratchpad from scratch every step.
"""

from typing import Optional, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.linen.initializers import Initializer, normal, orthogonal

from stoix.networks.utils import parse_activation_fn

_PROB_EPS = 1e-6
_NEG_INF = jnp.finfo(jnp.float32).min


class TransformerBlock(nn.Module):
    """A single pre-norm transformer block: self-attention + MLP.

    Exposes two forward modes, built on the *same* projection weights
    (`setup()`, not `@nn.compact`, specifically so both are available
    regardless of which is called first - see below):

      - `__call__(tokens, mask)`: full-sequence self-attention over
        `(*batch, seq_len, hidden_dim)` in one call. `mask` should be a
        causal mask (e.g. from `nn.make_causal_mask`) so each position's
        output is a function only of its own past.
      - `step(token, cached_keys, cached_values, step_idx, max_steps)`:
        incremental decoding - processes a *single* new token
        (`(*batch, hidden_dim)`), reading/extending a per-layer KV-cache
        (`(*batch, max_steps + 1, num_heads, head_dim)`) instead of
        reprocessing a whole sequence - O(1) work per call instead of
        O(seq_len).

    Sharing weights between the two matters beyond just avoiding waste:
    `stoix.networks.torso_compute_explicit_cot.TransformerExplicitCoTTorso`
    uses `__call__` for one code path (its parallel one-shot replay pass) and
    `step` for another (rollout/latent-feedback-replay), for the *same*
    logical model - if those weights could drift apart, replay would be
    scoring a rollout against a different policy than the one that actually
    produced it, corrupting the policy gradient.
    """

    hidden_dim: int
    num_heads: int
    mlp_dim: int
    activation: str = "relu"
    kernel_init: Initializer = orthogonal(np.sqrt(2.0))

    def setup(self) -> None:
        assert self.hidden_dim % self.num_heads == 0, (
            f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads})."
        )
        head_dim = self.hidden_dim // self.num_heads
        dense = lambda name: nn.DenseGeneral(  # noqa: E731
            axis=-1, features=(self.num_heads, head_dim), kernel_init=self.kernel_init, name=name
        )
        self.query_proj = dense("query")
        self.key_proj = dense("key")
        self.value_proj = dense("value")
        self.out_proj = nn.DenseGeneral(
            features=self.hidden_dim, axis=(-2, -1), kernel_init=self.kernel_init, name="out"
        )
        self.attn_norm = nn.LayerNorm()
        self.mlp_norm = nn.LayerNorm()
        self.mlp_dense_0 = nn.Dense(self.mlp_dim, kernel_init=self.kernel_init)
        self.mlp_dense_1 = nn.Dense(self.hidden_dim, kernel_init=self.kernel_init)

    def _mlp(self, tokens: chex.Array) -> chex.Array:
        y = self.mlp_norm(tokens)
        y = self.mlp_dense_0(y)
        y = parse_activation_fn(self.activation)(y)
        y = self.mlp_dense_1(y)
        return tokens + y

    def __call__(self, tokens: chex.Array, mask: Optional[chex.Array] = None) -> chex.Array:
        y = self.attn_norm(tokens)
        q, k, v = self.query_proj(y), self.key_proj(y), self.value_proj(y)
        head_dim = q.shape[-1]
        scale = 1.0 / jnp.sqrt(jnp.array(head_dim, dtype=q.dtype))
        scores = jnp.einsum("...qhd,...khd->...hqk", q, k) * scale
        if mask is not None:
            scores = jnp.where(mask, scores, _NEG_INF)
        weights = jax.nn.softmax(scores, axis=-1)
        attn_out = jnp.einsum("...hqk,...khd->...qhd", weights, v)
        tokens = tokens + self.out_proj(attn_out)
        return self._mlp(tokens)

    def step(
        self,
        token: chex.Array,
        cached_keys: chex.Array,
        cached_values: chex.Array,
        step_idx: chex.Array,
        max_steps: int,
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """
        token: `(*batch, hidden_dim)` - this layer's input at this step.
        cached_keys, cached_values: `(*batch, max_steps + 1, num_heads,
            head_dim)` - this layer's cache (positions > step_idx are
            not-yet-written).
        step_idx: traced scalar - the position to write/attend through.

        Returns `(new_token, updated_keys, updated_values)`.
        """
        y = self.attn_norm(token)
        q, k, v = self.query_proj(y), self.key_proj(y), self.value_proj(y)

        cached_keys = cached_keys.at[..., step_idx, :, :].set(k)
        cached_values = cached_values.at[..., step_idx, :, :].set(v)

        # True for cache positions written by step `step_idx` or earlier;
        # `key`/`value` for later positions are still zero (not yet written).
        key_mask = jnp.arange(max_steps + 1) <= step_idx
        head_dim = q.shape[-1]
        scale = 1.0 / jnp.sqrt(jnp.array(head_dim, dtype=q.dtype))
        scores = jnp.einsum("...hd,...khd->...hk", q, cached_keys) * scale
        scores = jnp.where(key_mask, scores, _NEG_INF)
        weights = jax.nn.softmax(scores, axis=-1)
        attn_out = jnp.einsum("...hk,...khd->...hd", weights, cached_values)

        token = token + self.out_proj(attn_out)
        return self._mlp(token), cached_keys, cached_values


class _CoTStep(nn.Module):
    """One CoT step, meant to be lifted into a weight-tied loop via `nn.scan`.

    Holds the transformer blocks, halting head and positional embedding as
    submodules/params created on first trace; `nn.scan(..., variable_broadcast
    ="params")` then shares those same params across every step instead of
    creating a fresh set per step, exactly reproducing the weight-tying of an
    unrolled Python loop, without the O(max_steps) compile-time cost of
    actually unrolling one.

    Since scan traces this body once for *all* steps, it needs a fixed-shape
    carry - unlike a Python loop, it can't grow the KV-cache array itself.
    Instead each layer's cache is pre-allocated at its final `max_steps + 1`
    size, `TransformerBlock.step` writes into position `step_idx` each call,
    and a `step_idx`-dependent mask keeps not-yet-written positions from
    being attended to (see `TransformerBlock.step`).
    """

    hidden_dim: int
    num_heads: int
    num_layers: int
    mlp_dim: int
    max_steps: int
    min_steps: int
    activation: str
    kernel_init: Initializer
    convergence_threshold: float
    replaying: bool
    deterministic: bool

    @nn.compact
    def __call__(
        self, carry: Tuple[chex.Array, ...], step_idx: chex.Array
    ) -> Tuple[Tuple[chex.Array, ...], None]:
        (
            current_token,
            cached_keys,
            cached_values,
            still_running,
            num_steps_taken,
            halting_log_prob,
            final_state,
            first_convergence_step,
            num_close_steps,
            prev_state,
            rng,
            target_compute_time,
        ) = carry

        pos_embedding = self.param(
            "pos_embedding",
            normal(stddev=0.02),
            (self.max_steps + 1, self.hidden_dim),
        )
        blocks = [
            TransformerBlock(
                self.hidden_dim, self.num_heads, self.mlp_dim, self.activation, self.kernel_init
            )
            for _ in range(self.num_layers)
        ]
        halting_head = nn.Dense(1, kernel_init=self.kernel_init)

        x = current_token + pos_embedding[step_idx]
        new_cached_keys = []
        new_cached_values = []
        for layer_idx, block in enumerate(blocks):
            x, k, v = block.step(
                x, cached_keys[layer_idx], cached_values[layer_idx], step_idx, self.max_steps
            )
            new_cached_keys.append(k)
            new_cached_values.append(v)
        state = x
        cached_keys = new_cached_keys
        cached_values = new_cached_values

        halting_prob = nn.sigmoid(halting_head(nn.LayerNorm()(state)))
        halting_prob = jnp.clip(halting_prob.squeeze(axis=-1), _PROB_EPS, 1.0 - _PROB_EPS)

        step_count = step_idx + 1
        is_final_step = step_count == self.max_steps
        can_halt = step_count >= self.min_steps

        if not self.replaying:
            # Only meaningful from the second step on (step 0 has no
            # previous thought to compare against - gated via `step_idx > 0`
            # rather than skipped, since `step_idx` is traced under scan),
            # and only while still running - once an example has halted,
            # later thoughts keep being computed (every example runs the
            # same fixed number of scan iterations) but are discarded, so
            # they say nothing about the example's real trajectory.
            l2_dist = jnp.linalg.norm(state - prev_state, axis=-1)
            is_close = still_running & (l2_dist < self.convergence_threshold) & (step_idx > 0)
            num_close_steps = num_close_steps + is_close.astype(jnp.float32)
            first_convergence_step = jnp.where(
                is_close & (first_convergence_step < 0),
                step_count.astype(jnp.float32),
                first_convergence_step,
            )
        prev_state = state

        if self.replaying:
            halts_this_step = still_running & (
                ((step_count >= target_compute_time) & can_halt) | is_final_step
            )
        elif self.deterministic:
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
        is_forced_step = (step_count < self.min_steps) | is_final_step
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
            still_running & (~is_forced_step), step_log_prob, 0.0
        )
        num_steps_taken = num_steps_taken + still_running.astype(jnp.float32)
        final_state = jnp.where(halts_this_step[..., None], state, final_state)

        still_running = still_running & (~halts_this_step)
        current_token = state

        new_carry = (
            current_token,
            cached_keys,
            cached_values,
            still_running,
            num_steps_taken,
            halting_log_prob,
            final_state,
            first_convergence_step,
            num_close_steps,
            prev_state,
            rng,
            target_compute_time,
        )
        return new_carry, None


class TransformerChainOfThoughtTorso(nn.Module):
    """Chain-of-Thought transformer torso with adaptively-sampled halting.

    See module docstring for the full mechanism. Has the same three-mode
    interface as `AdaptiveComputationTimeTorso`:

      - Rollout mode (`target_compute_time=None`, `rng` given): samples a CoT
        halting trajectory and returns `(embedding, compute_time)`.
      - Replay mode (`target_compute_time=<array>`): deterministically
        replays exactly that many CoT steps (no rng) and returns
        `(embedding, halting_log_prob)` - use this in loss functions.
      - Deterministic mode (`deterministic=True`): halts as soon as the
        halting probability crosses 0.5, for greedy evaluation.

    `min_steps` forbids halting (voluntarily, in replay, or greedily) before
    that many CoT steps have been taken - `is_final_step` still forces a halt
    at `max_steps` regardless. Setting `min_steps == max_steps` therefore
    removes adaptivity entirely: every example always takes exactly
    `max_steps` - useful as a fixed-budget baseline against the adaptive
    policy.

    `use_input_layer_norm` normalizes the raw `observation` before it becomes
    the first scratchpad token (i.e. before its projection into `hidden_dim`).

    Also tracks, per example, how quickly the "thought" (the representation
    read off the last scratchpad position) settles: the L2 distance between
    consecutive steps' thoughts (step `t` vs `t - 1`, starting at `t = 2`
    since there's no step 0 to compare step 1 against) is compared against
    `convergence_threshold`. This gives two diagnostics, only while the
    example hasn't halted yet:

      - `first_convergence_step`: the (1-indexed) step count `t` at which
        that distance first drops below `convergence_threshold`, or `-1` if
        it never does within the steps actually taken.
      - `num_close_steps`: how many steps (not necessarily consecutive) had
        a distance below `convergence_threshold`.
    """

    hidden_dim: int
    num_heads: int = 4
    num_layers: int = 2
    mlp_dim: int = 512
    max_steps: int = 8
    min_steps: int = 1
    activation: str = "relu"
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
            observation: the input embedding to think about; becomes the
                first scratchpad token.
            rng: PRNG key used to sample halting decisions. Required unless
                `target_compute_time` is given or `deterministic=True`.
            target_compute_time: if given (e.g. `Transition.compute_time`
                from a prior rollout), halting is *not* sampled - instead the
                CoT is replayed deterministically to halt at exactly this
                many steps per example, and the second output is
                `halting_log_prob`: the log-probability of that exact
                halting trajectory under the current parameters.
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
        if not replaying and not deterministic and rng is None:
            raise ValueError(
                "rng must be provided to TransformerChainOfThoughtTorso when sampling "
                "(i.e. target_compute_time is None and deterministic=False)."
            )
        if not (1 <= self.min_steps <= self.max_steps):
            raise ValueError(
                f"min_steps must be between 1 and max_steps ({self.max_steps}), "
                f"got min_steps={self.min_steps}."
            )
        assert self.hidden_dim % self.num_heads == 0, (
            f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads})."
        )

        # The first scratchpad token is the observation projected into the
        # model width.
        initial_token = nn.Dense(self.hidden_dim, kernel_init=self.kernel_init)(observation)
        if self.use_input_layer_norm:
            initial_token = nn.LayerNorm()(initial_token)

        # Pre-allocate each layer's KV-cache; positions written by
        # `_CoTStep`/`TransformerBlock.step` are set incrementally, and
        # not-yet-written positions are masked out of attention (see
        # `TransformerBlock.step`), never read.
        head_dim = self.hidden_dim // self.num_heads
        cache_shape = batch_shape + (self.max_steps + 1, self.num_heads, head_dim)
        cached_keys = [jnp.zeros(cache_shape) for _ in range(self.num_layers)]
        cached_values = [jnp.zeros(cache_shape) for _ in range(self.num_layers)]

        still_running = jnp.ones(batch_shape, dtype=bool)
        num_steps_taken = jnp.zeros(batch_shape)
        halting_log_prob = jnp.zeros(batch_shape)
        final_state = initial_token
        first_convergence_step = jnp.full(batch_shape, -1.0)
        num_close_steps = jnp.zeros(batch_shape)
        # Unused (never read) unless sampling, but must still be a concrete
        # array: it's carried through every scan step regardless of mode.
        step_rng = rng if rng is not None else jax.random.PRNGKey(0)

        cot_step = nn.scan(
            _CoTStep,
            variable_broadcast="params",
            split_rngs={"params": False},
        )(
            self.hidden_dim,
            self.num_heads,
            self.num_layers,
            self.mlp_dim,
            self.max_steps,
            self.min_steps,
            self.activation,
            self.kernel_init,
            self.convergence_threshold,
            replaying,
            deterministic,
        )

        initial_carry = (
            initial_token,  # current_token
            cached_keys,
            cached_values,
            still_running,
            num_steps_taken,
            halting_log_prob,
            final_state,
            first_convergence_step,
            num_close_steps,
            initial_token,  # prev_state
            step_rng,
            target_compute_time,
        )
        (
            _,
            _,
            _,
            _,
            num_steps_taken,
            halting_log_prob,
            final_state,
            first_convergence_step,
            num_close_steps,
            _,
            _,
            _,
        ), _ = cot_step(initial_carry, jnp.arange(self.max_steps))

        if replaying:
            return final_state, halting_log_prob
        return final_state, num_steps_taken, first_convergence_step, num_close_steps
