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


def _norm_cls(use_rmsnorm: bool):
    """`nn.RMSNorm` if `use_rmsnorm` else `nn.LayerNorm` - every norm site in
    `TransformerBlock` (and, via it,
    `stoix.networks.torso_compute_explicit_cot`, which reuses
    `TransformerBlock`) switches to RMSNorm when `use_rmsnorm=True`. RMSNorm
    rescales by the root-mean-square activation only (no mean-centering, no
    learned bias), so it's cheaper per call and, unlike LayerNorm, leaves the
    residual stream's mean untouched - the standard swap in recurrent-depth/
    weight-tied transformer designs that also use sandwich norm (see
    `TransformerBlock`'s `use_sandwich_norm`)."""
    return nn.RMSNorm if use_rmsnorm else nn.LayerNorm


def _resolve_qkv_dim(hidden_dim: int, qkv_dim: Optional[int]) -> int:
    """`hidden_dim` if `qkv_dim` is unset (`None`, the default everywhere it
    appears), else `qkv_dim` itself. Shared between `TransformerBlock` (whose
    `head_dim = qkv_dim // num_heads`) and every caller that pre-allocates
    `TransformerBlock.step`'s per-layer KV-cache, whose shape depends on that
    same `head_dim` - both need the identical default-resolution so the
    cache is sized to match what the block actually writes into it."""
    return hidden_dim if qkv_dim is None else qkv_dim


class TransformerBlock(nn.Module):
    """A single transformer block: self-attention + MLP, pre-norm by default.

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

    `use_sandwich_norm` (default `False`) additionally normalizes each
    sub-layer's residual sum, not just its input - i.e. `n2(x + Attn(n1(x)))`
    then `n4(x' + MLP(n3(x')))` instead of the plain pre-norm `x +
    Attn(n1(x))`/`x' + MLP(n3(x'))`, matching the "sandwich" layer-norm
    placement used in some recurrent-depth transformers (e.g. the block
    design described in arXiv:2502.05171's section 3.2) to keep a residual
    stream that gets reused across many weight-tied iterations - as it is
    here, `num_layers` blocks re-applied every CoT step, up to `max_steps`
    times, all sharing one set of weights - from drifting to ever-larger
    magnitude the more times it's iterated. Without it, only each sub-layer's
    *input* is normalized (standard pre-norm); the residual itself is never
    rescaled, so the same weights have to work correctly regardless of how
    much the accumulated residual has grown by the time they're applied
    again.

    `use_rmsnorm` (default `False`) switches every norm in this block from
    `nn.LayerNorm` to `nn.RMSNorm` - see `_norm_cls`. Independent of
    `use_sandwich_norm`: it only changes which norm module is used
    everywhere it's already placed, not where norms are placed.

    `qkv_dim` (default `None`, meaning "same as `hidden_dim`") sets the total
    Q/K/V projection width (`num_heads * head_dim`), decoupled from
    `hidden_dim` - the residual-stream width that the block's input/output,
    `out_proj`, and the MLP all still use. `out_proj` (a `DenseGeneral`
    contracting over `(num_heads, head_dim)`) maps attention's output back to
    `hidden_dim` regardless of how `qkv_dim` compares to it, so attention can
    run narrower or wider than the residual stream without changing anything
    else in the block.
    """

    hidden_dim: int
    num_heads: int
    mlp_dim: int
    activation: str = "relu"
    kernel_init: Initializer = orthogonal(np.sqrt(2.0))
    use_sandwich_norm: bool = False
    use_rmsnorm: bool = False
    qkv_dim: Optional[int] = None

    def setup(self) -> None:
        qkv_dim = _resolve_qkv_dim(self.hidden_dim, self.qkv_dim)
        assert qkv_dim % self.num_heads == 0, (
            f"qkv_dim ({qkv_dim}) must be divisible by num_heads ({self.num_heads})."
        )
        head_dim = qkv_dim // self.num_heads
        dense = lambda name: nn.DenseGeneral(  # noqa: E731
            axis=-1, features=(self.num_heads, head_dim), kernel_init=self.kernel_init, name=name
        )
        self.query_proj = dense("query")
        self.key_proj = dense("key")
        self.value_proj = dense("value")
        self.out_proj = nn.DenseGeneral(
            features=self.hidden_dim, axis=(-2, -1), kernel_init=self.kernel_init, name="out"
        )
        norm_cls = _norm_cls(self.use_rmsnorm)
        self.attn_norm = norm_cls()
        self.mlp_norm = norm_cls()
        self.mlp_dense_0 = nn.Dense(self.mlp_dim, kernel_init=self.kernel_init)
        self.mlp_dense_1 = nn.Dense(self.hidden_dim, kernel_init=self.kernel_init)
        if self.use_sandwich_norm:
            self.attn_post_norm = norm_cls()
            self.mlp_post_norm = norm_cls()

    def _mlp(self, tokens: chex.Array) -> chex.Array:
        y = self.mlp_norm(tokens)
        y = self.mlp_dense_0(y)
        y = parse_activation_fn(self.activation)(y)
        y = self.mlp_dense_1(y)
        y = tokens + y
        return self.mlp_post_norm(y) if self.use_sandwich_norm else y

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
        if self.use_sandwich_norm:
            tokens = self.attn_post_norm(tokens)
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
        if self.use_sandwich_norm:
            token = self.attn_post_norm(token)
        return self._mlp(token), cached_keys, cached_values


class HaltingHead(nn.Module):
    """Maps a "thought" to a single halting logit.

    `hidden_dims=()` (the default) is a bare linear readout - the original
    design, and still what every existing config gets. A non-empty
    `hidden_dims` instead runs the input through that many `Dense +
    activation` hidden layers (each of the corresponding width) before the
    final linear readout to logit space, giving the halting decision a
    nonlinear MLP instead of a single linear projection of the shared
    transformer state.
    """

    hidden_dims: Tuple[int, ...] = ()
    activation: str = "relu"
    kernel_init: Initializer = orthogonal(np.sqrt(2.0))

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        for dim in self.hidden_dims:
            x = nn.Dense(dim, kernel_init=self.kernel_init)(x)
            x = parse_activation_fn(self.activation)(x)
        return nn.Dense(1, kernel_init=self.kernel_init)(x)


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
    stop_gradient_halting_input: bool = False
    halting_temperature: float = 1.0
    halting_hidden_dims: Tuple[int, ...] = ()
    use_sandwich_norm: bool = False
    use_rmsnorm: bool = False
    qkv_dim: Optional[int] = None

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
            states_history,
            per_step_halting_log_prob,
            per_step_halting_entropy,
        ) = carry

        pos_embedding = self.param(
            "pos_embedding",
            normal(stddev=0.02),
            (self.max_steps + 1, self.hidden_dim),
        )
        blocks = [
            TransformerBlock(
                self.hidden_dim,
                self.num_heads,
                self.mlp_dim,
                self.activation,
                self.kernel_init,
                self.use_sandwich_norm,
                self.use_rmsnorm,
                self.qkv_dim,
            )
            for _ in range(self.num_layers)
        ]
        halting_head = HaltingHead(
            self.halting_hidden_dims, self.activation, self.kernel_init
        )

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
        if self.replaying:
            # Only ever populated when replaying - see
            # `TransformerChainOfThoughtTorso`'s docstring for `states_history`.
            states_history = states_history.at[..., step_idx, :].set(state)

        halting_input = state
        if self.stop_gradient_halting_input:
            halting_input = jax.lax.stop_gradient(halting_input)
        halting_logit = halting_head(halting_input)
        halting_prob = nn.sigmoid(halting_logit / self.halting_temperature)
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
        step_contribution = jnp.where(still_running & (~is_forced_step), step_log_prob, 0.0)
        halting_log_prob = halting_log_prob + step_contribution
        if self.replaying:
            per_step_halting_log_prob = per_step_halting_log_prob.at[..., step_idx].set(
                step_contribution
            )
            # Bernoulli entropy of this step's halting decision - masked
            # identically to `step_contribution` above (zero before
            # min_steps, at the forced max_steps halt, and once an example
            # has already halted), since a forced decision isn't a "choice"
            # to encourage exploration in. Uses `halting_prob` directly (not
            # `halting_prob_for_log`): a caller after an entropy *bonus*
            # wants gradient into the halting head here, exactly as
            # `actor_policy.entropy()` does for the environment action - see
            # ff_ppo.py.
            halting_entropy_this_step = -(
                halting_prob * jnp.log(halting_prob)
                + (1.0 - halting_prob) * jnp.log(1.0 - halting_prob)
            )
            per_step_halting_entropy = per_step_halting_entropy.at[..., step_idx].set(
                jnp.where(still_running & (~is_forced_step), halting_entropy_this_step, 0.0)
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
            states_history,
            per_step_halting_log_prob,
            per_step_halting_entropy,
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
        `(embedding, halting_log_prob, states_history,
        per_step_halting_log_prob, per_step_halting_entropy)` - use
        `halting_log_prob` in a REINFORCE-style loss; `states_history`
        (shape `(*batch, max_steps, hidden_dim)`) is the "thought" at every
        step, including steps past the one the trajectory actually halted
        at (mask by `step_idx < compute_time` before using it, same as the
        convergence diagnostics below already implicitly do). Meant for
        comparing the state trajectory produced by two parameter sets while
        replaying the same halting trajectory - e.g. a PPO trust-region
        penalty on the "thoughts" themselves, not just the halting decision
        - see `ff_ppo.py`. `per_step_halting_log_prob` (shape `(*batch,
        max_steps)`) is `halting_log_prob` left unsummed, one entry per CoT
        step, zeroed at forced steps or past the actual halt - for PPO to
        clip each step's ratio individually instead of one joint ratio over
        the trajectory's summed log-prob (also see `ff_ppo.py`).
        `per_step_halting_entropy` (same shape/masking) is the Bernoulli
        entropy of each step's halting probability, for a caller that wants
        an entropy *bonus* on the halting decision itself (see `ff_ppo.py`).
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

    `stop_gradient_halting_input` (default `False`) detaches this step's
    "thought" (`state`) before it's read by the halting head's `Dense(1)`,
    mirroring `stoix.networks.torso_compute.IRUStep`'s knob of the same
    name: the halting head's own weights still get trained via REINFORCE,
    but that gradient can no longer backprop into the shared transformer
    blocks that also produce the action head's representation - severing
    that gradient-sharing channel between the two objectives. A no-op when
    `min_steps == max_steps` (every step is already forced, so no
    halting-loss gradient reaches the torso regardless).

    `halting_temperature` (default `1.0`) divides the halting head's logit
    before the sigmoid - below `1.0` sharpens the halting probability towards
    0/1 (lower-entropy, more decisive halting decisions), above `1.0` softens
    it towards 0.5 (higher-entropy, more exploratory halting decisions);
    `1.0` is the standard sigmoid with no rescaling. Forwarded to the shared
    `_CoTStep`'s halting head - see that class.

    `halting_hidden_dims` (default `()`, a bare linear readout) sets the
    hidden layer widths of the halting head's MLP - e.g. `(64,)` for one
    ReLU hidden layer before the final linear logit. Uses `activation` for
    the hidden layers. See `HaltingHead`.

    `use_sandwich_norm` (default `False`) is forwarded to every shared
    `TransformerBlock` - see that class's docstring for what it changes and
    why it matters specifically for a weight-tied, recurrent-in-depth
    architecture like this one.

    `use_rmsnorm` (default `False`) switches every norm in this torso -
    `use_input_layer_norm`'s norm on the initial token, and every norm inside
    the shared `TransformerBlock`s - from `nn.LayerNorm` to `nn.RMSNorm`.

    `qkv_dim` (default `None`, meaning "same as `hidden_dim`") is forwarded
    to every shared `TransformerBlock` - see that class's docstring. Setting
    it lets the Q/K/V projections (and thus attention) run at a different
    width than `hidden_dim`, which stays the width of the initial
    observation-projection Dense layer, the residual stream, and everything
    else (positional embedding, `out_proj`, MLP, halting head).
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
    stop_gradient_halting_input: bool = False
    halting_temperature: float = 1.0
    halting_hidden_dims: Tuple[int, ...] = ()
    use_sandwich_norm: bool = False
    use_rmsnorm: bool = False
    qkv_dim: Optional[int] = None

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
            `(embedding, halting_log_prob, states_history,
            per_step_halting_log_prob, per_step_halting_entropy)` when
            replaying a known trajectory (see class docstring for the
            convergence diagnostics, `states_history`,
            `per_step_halting_log_prob` and `per_step_halting_entropy`).
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
        qkv_dim = _resolve_qkv_dim(self.hidden_dim, self.qkv_dim)
        assert qkv_dim % self.num_heads == 0, (
            f"qkv_dim ({qkv_dim}) must be divisible by num_heads ({self.num_heads})."
        )

        # The first scratchpad token is the observation projected into the
        # model width.
        initial_token = nn.Dense(self.hidden_dim, kernel_init=self.kernel_init)(observation)
        if self.use_input_layer_norm:
            initial_token = _norm_cls(self.use_rmsnorm)()(initial_token)

        # Pre-allocate each layer's KV-cache; positions written by
        # `_CoTStep`/`TransformerBlock.step` are set incrementally, and
        # not-yet-written positions are masked out of attention (see
        # `TransformerBlock.step`), never read.
        head_dim = qkv_dim // self.num_heads
        cache_shape = batch_shape + (self.max_steps + 1, self.num_heads, head_dim)
        cached_keys = [jnp.zeros(cache_shape) for _ in range(self.num_layers)]
        cached_values = [jnp.zeros(cache_shape) for _ in range(self.num_layers)]

        still_running = jnp.ones(batch_shape, dtype=bool)
        num_steps_taken = jnp.zeros(batch_shape)
        halting_log_prob = jnp.zeros(batch_shape)
        final_state = initial_token
        first_convergence_step = jnp.full(batch_shape, -1.0)
        num_close_steps = jnp.zeros(batch_shape)
        # Only ever written when replaying (see `_CoTStep`); carried
        # regardless of mode since `nn.scan` needs a fixed carry structure.
        states_history = jnp.zeros(batch_shape + (self.max_steps, self.hidden_dim))
        per_step_halting_log_prob = jnp.zeros(batch_shape + (self.max_steps,))
        per_step_halting_entropy = jnp.zeros(batch_shape + (self.max_steps,))
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
            self.stop_gradient_halting_input,
            self.halting_temperature,
            self.halting_hidden_dims,
            self.use_sandwich_norm,
            self.use_rmsnorm,
            self.qkv_dim,
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
            states_history,
            per_step_halting_log_prob,
            per_step_halting_entropy,
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
            states_history,
            per_step_halting_log_prob,
            per_step_halting_entropy,
        ), _ = cot_step(initial_carry, jnp.arange(self.max_steps))

        if replaying:
            return (
                final_state,
                halting_log_prob,
                states_history,
                per_step_halting_log_prob,
                per_step_halting_entropy,
            )
        return final_state, num_steps_taken, first_convergence_step, num_close_steps
