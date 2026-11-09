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


def rms_norm(x: chex.Array, eps: float = 1e-5) -> chex.Array:
    """Parameter-free RMSNorm."""
    return x * jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps)


def _logit(p: float) -> float:
    return float(np.log(p / (1.0 - p)))


# `truncated_normal(stddev=1/sqrt(fan_in))`, truncated at +-2 stddev without
# rescaling, so the realized std is ~0.88 / sqrt(fan_in) (flax
# `truncated_normal` semantics). `variance_scaling(truncated_normal)` divides
# its stddev by 0.87962566, so a scale of 0.87962566^2 cancels that. Fan-in for
# DenseGeneral's `out` is the product of its contracted (heads, head_dim) axes.
_TRUNC_NORMAL_CORRECTION = 0.87962566103423978
trunc_normal_kernel_init = nn.initializers.variance_scaling(
    _TRUNC_NORMAL_CORRECTION**2, "fan_in", "truncated_normal"
)


def _block_kernel_init(block_init: str, default: Initializer) -> Initializer:
    """Kernel init of every `TransformerBlock` projection in the CoT loop:
    `"orthogonal"` keeps the torso's `kernel_init` (orthogonal(sqrt(2)) by
    default), `"trunc_normal"` uses `trunc_normal_kernel_init`."""
    if block_init == "orthogonal":
        return default
    if block_init == "trunc_normal":
        return trunc_normal_kernel_init
    raise ValueError(f"block_init must be 'orthogonal' or 'trunc_normal', got {block_init!r}")


def simplicial_embedding(x: chex.Array, group_size: int, temperature: float) -> chex.Array:
    """Simplicial embedding (Lavoie et al., 2022). Splits the last axis into
    groups of `group_size` and applies a softmax to each group. Each group
    then is a point on a simplex. `group_size <= 0` disables it."""
    if group_size <= 0:
        return x
    groups = x.reshape(x.shape[:-1] + (-1, group_size))
    return jax.nn.softmax(groups / temperature, axis=-1).reshape(x.shape)


def _readout_and_feedback(
    x: chex.Array,
    latent_rms_norm: bool,
    group_size: int,
    temperature: float,
    readout_only: bool = False,
) -> Tuple[chex.Array, chex.Array]:
    """Returns `(state, fed_back)`: `state` is what the halting/action heads
    read, `fed_back` is what re-enters the residual stream as a scratchpad token.

    `latent_rms_norm=False` (original): both are `SEM(x)`. `True`: `state =
    SEM(rms_norm(x))`, so the softmax groups see unit-RMS
    logits instead of the raw residual, and `fed_back = sqrt(V) * state` (V =
    `group_size`) brings the token back to roughly unit scale. With SEM off
    (`group_size <= 0`) both are `rms_norm(x)`.

    `readout_only=True`: SEM is used only on `state`; `fed_back` is
    `rms_norm(x)` (or `x` without `latent_rms_norm`).
    """
    if readout_only:
        x = rms_norm(x) if latent_rms_norm else x
        return simplicial_embedding(x, group_size, temperature), x
    if not latent_rms_norm:
        state = simplicial_embedding(x, group_size, temperature)
        return state, state
    state = simplicial_embedding(rms_norm(x), group_size, temperature)
    scale = float(np.sqrt(group_size)) if group_size > 0 else 1.0
    return state, scale * state


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
    # Attention-weight dropout in `step`, applied only when `dropout_key` is given.
    attention_dropout: float = 0.0

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

    @staticmethod
    def _residual(
        h: chex.Array, out: chex.Array, res_scale: Optional[Tuple[chex.Array, chex.Array]]
    ) -> chex.Array:
        """`h + out`, or the coupled `a1 * h + b1 * out` when
        `res_scale = (a1, b1)` (see `_CoTStep`'s `residual_scaling`)."""
        if res_scale is None:
            return h + out
        a1, b1 = res_scale
        return a1 * h + b1 * out

    def _mlp(
        self, tokens: chex.Array, res_scale: Optional[Tuple[chex.Array, chex.Array]] = None
    ) -> chex.Array:
        y = self.mlp_norm(tokens)
        y = self.mlp_dense_0(y)
        y = parse_activation_fn(self.activation)(y)
        y = self.mlp_dense_1(y)
        y = self._residual(tokens, y, res_scale)
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
        res_scale: Optional[Tuple[chex.Array, chex.Array]] = None,
        dropout_key: Optional[chex.Array] = None,
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """
        dropout_key: optional per-example keys `(*batch, 2)`. With
            `attention_dropout > 0`, the mask is a function of (key, step_idx)
            only, so the same keys drop the same weights in rollout and replay.
        res_scale: optional `(a1, b1)` - both residuals become `a1 * h + b1 *
            sublayer(norm(h))` instead of `h + sublayer(norm(h))`.
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
        if self.attention_dropout > 0.0 and dropout_key is not None:
            keep_prob = 1.0 - self.attention_dropout
            mask_fn = lambda k: jax.random.bernoulli(  # noqa: E731
                jax.random.fold_in(k, step_idx), keep_prob, weights.shape[-2:]
            )
            keep = jax.vmap(mask_fn)(dropout_key.reshape(-1, 2)).reshape(weights.shape)
            weights = jnp.where(keep, weights / keep_prob, 0.0)
        attn_out = jnp.einsum("...hk,...khd->...hd", weights, cached_values)

        token = self._residual(token, self.out_proj(attn_out), res_scale)
        if self.use_sandwich_norm:
            token = self.attn_post_norm(token)
        return self._mlp(token, res_scale), cached_keys, cached_values


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
    bias_init: float = 0.0  # initial halting logit offset; < 0 starts with low halting probs

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        for dim in self.hidden_dims:
            x = nn.Dense(dim, kernel_init=self.kernel_init)(x)
            x = parse_activation_fn(self.activation)(x)
        return nn.Dense(
            1, kernel_init=self.kernel_init, bias_init=nn.initializers.constant(self.bias_init)
        )(x)


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
    num_state_tokens: int = 1
    sem_group_size: int = 0
    sem_temperature: float = 1.0
    sem_readout_only: bool = False
    latent_rms_norm: bool = False
    input_injection: bool = False
    residual_scaling: bool = False
    alpha_1_init: float = 0.75
    alpha_2_init: float = 0.25
    block_init: str = "orthogonal"
    halting_bias_init: float = 0.0
    action_logit_accum: str = "last"
    action_logit_discount: float = 1.0
    attention_dropout: float = 0.0

    @nn.compact
    def __call__(
        self, carry: Tuple[chex.Array, ...], xs: Tuple[chex.Array, chex.Array, chex.Array]
    ) -> Tuple[Tuple[chex.Array, ...], None]:
        # `pos` is the scratchpad position. The first `num_state_tokens - 1`
        # positions only prefill the KV-cache with state tokens (no halting).
        # The CoT step `step_idx` starts at the last state token.
        # `inject_token` is the token CoT step 0 runs on, repeated per position.
        pos, state_token, inject_token = xs
        num_prefill = self.num_state_tokens - 1
        num_positions = self.max_steps + num_prefill
        step_idx = pos - num_prefill
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
            dropout_key,
        ) = carry
        # Prefill positions are never "running": they cannot halt, count as
        # compute, or add halting log prob.
        all_running = still_running
        still_running = still_running & (step_idx >= 0)
        hist_idx = jnp.maximum(step_idx, 0)  # prefill writes are overwritten at step 0

        pos_embedding = self.param(
            "pos_embedding",
            normal(stddev=0.02),
            (num_positions + 1, self.hidden_dim),
        )
        blocks = [
            TransformerBlock(
                self.hidden_dim,
                self.num_heads,
                self.mlp_dim,
                self.activation,
                _block_kernel_init(self.block_init, self.kernel_init),
                self.use_sandwich_norm,
                self.use_rmsnorm,
                self.qkv_dim,
                self.attention_dropout,
            )
            for _ in range(self.num_layers)
        ]
        halting_head = HaltingHead(
            self.halting_hidden_dims, self.activation, self.kernel_init, self.halting_bias_init
        )

        # Coupled residual scaling: learned per-dim a1 (block residuals) and a2 (input mixing), with b1/b2 tied to
        # them so the per-step map stays contractive. Without input injection
        # b2 = 1 - a1^(2L), which gives b1 = 1 - a1 (a convex residual).
        res_scale = None
        a2, b2 = 1.0, 1.0
        if self.residual_scaling:
            a1 = nn.sigmoid(
                self.param(
                    "alpha_1_logit",
                    nn.initializers.constant(_logit(self.alpha_1_init)),
                    (self.hidden_dim,),
                )
            )
            a1_pow = a1 ** (2 * self.num_layers)
            if self.input_injection:
                a2 = nn.sigmoid(
                    self.param(
                        "alpha_2_logit",
                        nn.initializers.constant(_logit(self.alpha_2_init)),
                        (self.hidden_dim,),
                    )
                )
                b2 = 1.0 - a2 * a1_pow
            else:
                b2 = 1.0 - a1_pow
            b1 = b2 * (1.0 - a1) / (1.0 - a1_pow + 1e-5)
            res_scale = (a1, b1)

        token = jnp.where(pos < self.num_state_tokens, state_token, current_token)
        if self.input_injection:
            # Re-inject the observation into every fed-back thought
            # (`a2 * thought + b2 * s`). Step 0 already *is* that token.
            token = jnp.where(pos >= self.num_state_tokens, a2 * token + b2 * inject_token, token)
        x = token + pos_embedding[pos]
        new_cached_keys = []
        new_cached_values = []
        for layer_idx, block in enumerate(blocks):
            layer_key = None
            if dropout_key is not None:  # one key stream per layer
                layer_key = jax.vmap(jax.random.fold_in, (0, None))(
                    dropout_key.reshape(-1, 2), layer_idx + 1
                ).reshape(dropout_key.shape)
            x, k, v = block.step(
                x, cached_keys[layer_idx], cached_values[layer_idx], pos, num_positions,
                res_scale, layer_key,
            )
            new_cached_keys.append(k)
            new_cached_values.append(v)
        # The latent goes through (RMSNorm then) SEM before it is fed back or
        # read by the halting/action heads.
        state, fed_back = _readout_and_feedback(
            x, self.latent_rms_norm, self.sem_group_size, self.sem_temperature,
            self.sem_readout_only,
        )
        cached_keys = new_cached_keys
        cached_values = new_cached_values
        if self.replaying:
            # Only ever populated when replaying - see
            # `TransformerChainOfThoughtTorso`'s docstring for `states_history`.
            states_history = states_history.at[..., hist_idx, :].set(state)

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
            per_step_halting_log_prob = per_step_halting_log_prob.at[..., hist_idx].set(
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
            per_step_halting_entropy = per_step_halting_entropy.at[..., hist_idx].set(
                jnp.where(still_running & (~is_forced_step), halting_entropy_this_step, 0.0)
            )
        num_steps_taken = num_steps_taken + still_running.astype(jnp.float32)
        if self.action_logit_accum == "last":
            final_state = jnp.where(halts_this_step[..., None], state, final_state)
        else:  # running (discounted) sum of the states of the steps taken
            gamma = self.action_logit_discount if self.action_logit_accum == "discount" else 1.0
            final_state = jnp.where(still_running[..., None], gamma * final_state + state, final_state)

        still_running = all_running & (~halts_this_step)
        current_token = fed_back

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
            dropout_key,
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

    `num_state_tokens` (default `1`) is N: one `Dense` maps the observation
    to N tokens of width `hidden_dim`. Tokens 0..N-2 only prefill the
    KV-cache. CoT step 0 runs on token N-1. `N = 1` is the original torso.

    `sem_group_size` (default `0`, off) applies `simplicial_embedding` (groups
    of this size, softmax temperature `sem_temperature`) to each state token
    before the transformer, and to each latent the transformer makes, before
    it is fed back or read by the halting/action heads. `sem_readout_only`
    (default `False`) uses SEM only on what the heads read (action, halting,
    and the value head of a `CoTCriticTorso`); state tokens and fed-back
    thoughts skip it.

    `input_is_tokens` (default `False`) takes an input of shape `(*batch, N,
    F)`, e.g. from `CNNTorso(channel_tokens=True)`. N comes from the input and
    `num_state_tokens` is not used. `concat_input_tokens` (default `False`)
    instead gives each of the N input tokens its own `Dense(F -> hidden_dim /
    N)` and concatenates the N outputs into 1 state token of width `hidden_dim`.

    Three options keep the weight-tied loop stable, so more CoT steps refine the latent
    instead of compounding its scale. All are off by default:

      - `latent_rms_norm`: parameter-free RMSNorm on every latent (state
        tokens and each step's output) before SEM. The heads read
        `SEM(rms_norm(x))`, and the fed-back token is `sqrt(sem_group_size) *
        SEM(rms_norm(x))`. Without it, SEM gets the raw residual stream and
        saturates. With SEM off, the latent is just `rms_norm(x)`.
      - `input_injection`: each fed-back thought (CoT step >= 1) becomes `a2 *
        thought + b2 * s`, where `s` is the token CoT step 0 runs on (the last
        state token). This keeps the observation in every step's input rather
        than only in the attention cache. Without `residual_scaling`, `a2 = b2
        = 1`.
      - `residual_scaling`: every block residual in the loop becomes `a1 * h +
        b1 * sublayer(norm(h))`, with learned per-dim `a1 = sigmoid(.)`
        (initialized to `alpha_1_init`) and `a2` (initialized to
        `alpha_2_init`), and `b2 = 1 - a2 a1^(2L)`, `b1 = b2 (1 - a1) / (1 -
        a1^(2L))`, which keeps the per-step map contractive. Without injection, `b2 = 1 - a1^(2L)`.

    `block_init` (default `"orthogonal"`) sets the kernel init of every
    `TransformerBlock` projection (Q/K/V, attention out, both MLP layers).
    `"orthogonal"` uses `kernel_init` (orthogonal(sqrt(2))); its sqrt(2) gain
    on the residual-branch outputs grows the residual stream every block.
    `"trunc_normal"` uses `truncated_normal(1/sqrt(fan_in))` (see
    `trunc_normal_kernel_init`). The observation projection and halting head keep
    `kernel_init` either way.

    `action_logit_accum` (default `"last"`) sets the embedding the action head
    reads: `"last"` is the state at the halt step, `"sum"`/`"mean"` the sum/mean
    of the states of all steps taken, `"discount"` the discounted sum
    `gamma^(t-1) s_1 + ... + gamma s_(t-1) + s_t` with `gamma = action_logit_discount`.
    With a linear action head, `"sum"` equals adding each step's logits (the bias
    is added once, not T times), so every step gets a direct gradient from the
    action loss.
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
    halting_bias_init: float = 0.0
    use_sandwich_norm: bool = False
    use_rmsnorm: bool = False
    qkv_dim: Optional[int] = None
    num_state_tokens: int = 1
    sem_group_size: int = 0
    sem_temperature: float = 1.0
    sem_readout_only: bool = False
    input_is_tokens: bool = False
    concat_input_tokens: bool = False
    latent_rms_norm: bool = False
    input_injection: bool = False
    residual_scaling: bool = False
    alpha_1_init: float = 0.75
    alpha_2_init: float = 0.25
    block_init: str = "orthogonal"
    action_logit_accum: str = "last"
    action_logit_discount: float = 1.0
    attention_dropout: float = 0.0

    @nn.compact
    def __call__(
        self,
        observation: chex.Array,
        rng: Optional[chex.PRNGKey] = None,
        target_compute_time: Optional[chex.Array] = None,
        deterministic: bool = False,
        dropout_key: Optional[chex.Array] = None,
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
            dropout_key: per-example keys `(*batch, 2)`. With `attention_dropout
                > 0`, drops attention weights with a mask set by (key, layer,
                position), so the same keys give the same mask in rollout and
                replay. `None` (e.g. evaluation, critic) means no dropout.

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

        # The first N scratchpad tokens are the observation projected into N
        # tokens of the model width.
        if self.input_is_tokens:
            # `observation` is already `(*batch, N, F)` (e.g. CNN kernel
            # features). One shared Dense maps each token to `hidden_dim`.
            batch_shape, n = observation.shape[:-2], observation.shape[-2]
            if self.concat_input_tokens:
                # Token i gets its own Dense(F -> hidden_dim / N); the N outputs make 1 token.
                assert self.hidden_dim % n == 0, f"hidden_dim ({self.hidden_dim}) % N ({n}) != 0"
                per_token_dense = nn.vmap(
                    nn.Dense, in_axes=-2, out_axes=-2,
                    variable_axes={"params": 0}, split_rngs={"params": True},
                )
                state_tokens = per_token_dense(self.hidden_dim // n, kernel_init=self.kernel_init)(
                    observation
                ).reshape(batch_shape + (1, self.hidden_dim))
                n = 1
            else:
                state_tokens = nn.Dense(self.hidden_dim, kernel_init=self.kernel_init)(observation)
        else:
            n = self.num_state_tokens
            state_tokens = nn.Dense(n * self.hidden_dim, kernel_init=self.kernel_init)(observation)
            state_tokens = state_tokens.reshape(batch_shape + (n, self.hidden_dim))
        if self.use_input_layer_norm:
            state_tokens = _norm_cls(self.use_rmsnorm)()(state_tokens)
        # State tokens enter the residual stream exactly like fed-back thoughts.
        _, state_tokens = _readout_and_feedback(
            state_tokens, self.latent_rms_norm, self.sem_group_size, self.sem_temperature,
            self.sem_readout_only,
        )
        initial_token = state_tokens[..., -1, :]
        num_positions = self.max_steps + n - 1
        # Per-position scan input: state token p at position p < N, else unused.
        step_inputs = jnp.concatenate(
            [state_tokens, jnp.zeros(batch_shape + (self.max_steps - 1, self.hidden_dim))],
            axis=-2,
        )
        step_inputs = jnp.moveaxis(step_inputs, -2, 0)
        inject_inputs = jnp.broadcast_to(initial_token, (num_positions,) + initial_token.shape)

        # Pre-allocate each layer's KV-cache; positions written by
        # `_CoTStep`/`TransformerBlock.step` are set incrementally, and
        # not-yet-written positions are masked out of attention (see
        # `TransformerBlock.step`), never read.
        head_dim = qkv_dim // self.num_heads
        cache_shape = batch_shape + (num_positions + 1, self.num_heads, head_dim)
        cached_keys = [jnp.zeros(cache_shape) for _ in range(self.num_layers)]
        cached_values = [jnp.zeros(cache_shape) for _ in range(self.num_layers)]

        still_running = jnp.ones(batch_shape, dtype=bool)
        num_steps_taken = jnp.zeros(batch_shape)
        halting_log_prob = jnp.zeros(batch_shape)
        assert self.action_logit_accum in ("last", "sum", "mean", "discount"), self.action_logit_accum
        final_state = initial_token if self.action_logit_accum == "last" else jnp.zeros_like(initial_token)
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
            n,
            self.sem_group_size,
            self.sem_temperature,
            sem_readout_only=self.sem_readout_only,
            latent_rms_norm=self.latent_rms_norm,
            input_injection=self.input_injection,
            residual_scaling=self.residual_scaling,
            alpha_1_init=self.alpha_1_init,
            alpha_2_init=self.alpha_2_init,
            block_init=self.block_init,
            halting_bias_init=self.halting_bias_init,
            action_logit_accum=self.action_logit_accum,
            action_logit_discount=self.action_logit_discount,
            attention_dropout=self.attention_dropout,
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
            dropout_key,
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
            _,
        ), _ = cot_step(initial_carry, (jnp.arange(num_positions), step_inputs, inject_inputs))
        if self.action_logit_accum == "mean":
            final_state = final_state / num_steps_taken[..., None]

        if replaying:
            return (
                final_state,
                halting_log_prob,
                states_history,
                per_step_halting_log_prob,
                per_step_halting_entropy,
            )
        return final_state, num_steps_taken, first_convergence_step, num_close_steps


class CoTCriticTorso(nn.Module):
    """Critic torso: runs `torso` (a `TransformerChainOfThoughtTorso`, usually with
    `min_steps == max_steps`) greedily and returns only its embedding."""

    torso: nn.Module

    @nn.compact
    def __call__(self, observation: chex.Array) -> chex.Array:
        return self.torso(observation, deterministic=True)[0]
