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

from typing import Any, Callable, Mapping, Optional, Tuple

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


# Name under which `MixtureOfExpertsMLP` sows its routing statistics.
MOE_STATS_KEY = "moe_stats"


class MixtureOfExpertsMLP(nn.Module):
    """A top-k routed mixture of `num_experts` two-layer MLPs - a drop-in
    replacement for `TransformerBlock`'s dense MLP (`Dense(mlp_dim) ->
    activation -> Dense(out_dim)`), with each expert shaped exactly like it.

    A linear router maps each token to `num_experts` logits; the token is
    routed to its `num_experts_per_token` highest-scoring experts, whose
    outputs are mixed with weights given by a softmax over just those
    selected logits (Mixtral-style), so the mixing weights sum to 1 and
    the router gets gradient through the selected experts' weights.
    `num_experts_per_token == num_experts` makes this a dense, softly-gated
    mixture (every expert used, weights = full softmax).

    Works on any leading batch/sequence shape `(..., in_dim)`, so it serves
    both `TransformerBlock.__call__` (full sequence) and
    `TransformerBlock.step` (single token).

    Every expert is evaluated for every token and the unselected ones are
    zero-weighted, rather than dispatching tokens to experts sparsely. At
    the model sizes used here that is simpler and faster on accelerators
    than gather/scatter dispatch. The cost is that per-token FLOPs scale
    with `num_experts` and not with `num_experts_per_token`, but the routing
    (and the function computed) is identical to a sparse implementation.

    Load balancing: every call sows (into the `"intermediates"` collection,
    under `MOE_STATS_KEY`) the routing statistics the Switch Transformer
    load-balancing loss (Fedus et al., 2021) needs - the number of tokens,
    how many of them were dispatched to each expert, and the summed router
    probability per expert - restricted to tokens where `token_mask` is
    `True` (all tokens if `None`). `sow` is a no-op unless the caller applies
    the model with `mutable=["intermediates"]`, so rollout/eval are
    untouched; `moe_load_balancing_loss` turns the sown statistics into the
    loss, and `apply_with_moe_load_balancing_loss` does both in one call.
    Statistics are sums (not means) so they aggregate correctly across the
    CoT steps of a scan and across calls before being normalized.
    """

    num_experts: int
    num_experts_per_token: int
    mlp_dim: int
    out_dim: int
    activation: str = "relu"
    kernel_init: Initializer = orthogonal(np.sqrt(2.0))

    @nn.compact
    def __call__(self, x: chex.Array, token_mask: Optional[chex.Array] = None) -> chex.Array:
        """
        x: `(..., in_dim)`.
        token_mask: optional `(...)` bool - which tokens count towards the
            sown load-balancing statistics (e.g. excluding CoT steps past an
            example's halt). Doesn't affect the output.
        """
        if not (1 <= self.num_experts_per_token <= self.num_experts):
            raise ValueError(
                f"num_experts_per_token must be between 1 and num_experts ({self.num_experts}), "
                f"got num_experts_per_token={self.num_experts_per_token}."
            )
        in_dim = x.shape[-1]

        router_logits = nn.Dense(self.num_experts, name="router")(x)
        top_logits, top_idx = jax.lax.top_k(router_logits, self.num_experts_per_token)
        top_gates = jax.nn.softmax(top_logits, axis=-1)
        # Scatter the k gates back to a dense `(..., num_experts)` weight
        # vector, zero for every unselected expert.
        top_one_hot = jax.nn.one_hot(top_idx, self.num_experts, dtype=x.dtype)
        gates = jnp.sum(top_one_hot * top_gates[..., None], axis=-2)

        token_weight = (
            jnp.ones(x.shape[:-1], dtype=x.dtype)
            if token_mask is None
            else token_mask.astype(x.dtype)
        )
        token_axes = tuple(range(x.ndim - 1))
        self.sow(
            "intermediates",
            MOE_STATS_KEY,
            {
                "num_tokens": jnp.sum(token_weight),
                # `(num_experts,)` - tokens routed to each expert (each token
                # counts once per selected expert, so this sums to
                # `num_experts_per_token * num_tokens`). Non-differentiable,
                # as in the Switch Transformer loss.
                "dispatch": jnp.sum(
                    jnp.sum(top_one_hot, axis=-2) * token_weight[..., None], axis=token_axes
                ),
                # `(num_experts,)` - summed full-softmax router probability
                # per expert; the loss's only gradient path into the router.
                "router_prob": jnp.sum(
                    jax.nn.softmax(router_logits, axis=-1) * token_weight[..., None],
                    axis=token_axes,
                ),
            },
        )

        expert_init = _expert_kernel_init(self.kernel_init)
        w0 = self.param("w0", expert_init, (self.num_experts, in_dim, self.mlp_dim))
        b0 = self.param("b0", nn.initializers.zeros, (self.num_experts, self.mlp_dim))
        w1 = self.param("w1", expert_init, (self.num_experts, self.mlp_dim, self.out_dim))
        b1 = self.param("b1", nn.initializers.zeros, (self.num_experts, self.out_dim))

        h = jnp.einsum("...d,edm->...em", x, w0) + b0
        h = parse_activation_fn(self.activation)(h)
        y = jnp.einsum("...em,emo->...eo", h, w1) + b1
        return jnp.einsum("...e,...eo->...o", gates, y)


def _expert_kernel_init(kernel_init: Initializer) -> Initializer:
    """Initializes a stacked `(num_experts, in, out)` kernel one expert at a
    time with `kernel_init` (not one init over the whole 3D array), so every
    expert starts out exactly like a standalone `nn.Dense(kernel_init=...)`."""

    def init(key: chex.PRNGKey, shape: Tuple[int, ...], dtype: Any = jnp.float32) -> chex.Array:
        keys = jax.random.split(key, shape[0])
        return jax.vmap(lambda k: kernel_init(k, shape[1:], dtype))(keys)

    return init


MOE_TYPES = ("topk", "soft")


def init_soft_moe_cache(
    batch_shape: Tuple[int, ...], max_steps: int, hidden_dim: int, moe_type: str, num_experts: int
) -> Optional[chex.Array]:
    """Per-layer cache of past MLP inputs for `SoftMoEMLP.step` (shape
    `(*batch, max_steps + 1, hidden_dim)`, the soft-MoE analogue of the
    per-layer KV-cache), or `None` when the block has no soft MoE - pass the
    result as `TransformerBlock.step`'s `cached_moe_inputs` either way."""
    if num_experts > 0 and moe_type == "soft":
        return jnp.zeros(batch_shape + (max_steps + 1, hidden_dim))
    return None


class SoftMoEMLP(nn.Module):
    """Causal Soft MoE (Puigcerver et al., 2023, "From Sparse to Soft
    Mixtures of Experts") over the CoT scratchpad - a replacement for
    `TransformerBlock`'s dense MLP.

    Soft MoE: with tokens `X` (`m x d`) and learned slot parameters `Phi`
    (`d x (num_experts * slots_per_expert)`), logits `L = X Phi`; dispatch
    weights `D = softmax(L)` over *tokens* (per slot) build each slot's input
    as a weighted average of tokens, `X~ = D^T X`; expert `i` processes its
    `slots_per_expert` slots, `Y~_j = f_{j // slots_per_expert}(X~_j)`; and
    combine weights `C = softmax(L)` over *slots* (per token) mix the slot
    outputs back into one output per token, `Y = C Y~`. Fully
    differentiable - no routing, no dropped tokens, no load-balancing loss
    needed (nothing is sown for `moe_load_balancing_loss`).

    Causal: the token axis here is the CoT scratchpad, so - unlike the
    original, which mixes every token of an image/sequence - the slots seen
    by the token at position `t` mix only positions `<= t` of the *same*
    example (a causally-masked dispatch softmax). That keeps each position a
    function of its own past only, matching the causal attention, so step-
    by-step decoding (`step`, reading a cache of past MLP inputs like the
    KV-cache) gives exactly what the full-sequence pass (`__call__`) gives,
    and rollout/replay score the same policy. Each position therefore has
    its own slot inputs, so the experts run once per (position, slot):
    `num_experts * slots_per_expert` expert evaluations per token.

    Each expert is shaped like the dense MLP it replaces (`Dense(mlp_dim) ->
    activation -> Dense(out_dim)`). `normalize` (default `False`) applies
    Puigcerver et al.'s l2 normalization - tokens l2-normalized along the
    feature axis and `Phi` along its input axis, times a learned scalar
    (init 1) - when computing the logits; they recommend it for large model
    widths and found it makes little difference for small ones.
    """

    num_experts: int
    slots_per_expert: int
    mlp_dim: int
    out_dim: int
    activation: str = "relu"
    kernel_init: Initializer = orthogonal(np.sqrt(2.0))
    normalize: bool = False

    @nn.compact
    def _mix(self, queries: chex.Array, keys: chex.Array, key_mask: chex.Array) -> chex.Array:
        """queries: `(..., Q, d)` - the tokens to produce outputs for.
        keys: `(..., K, d)` - the tokens slots are built from.
        key_mask: broadcastable to `(..., Q, K)` - which keys each query's
            slots may mix (its causal past).
        Returns `(..., Q, out_dim)`."""
        in_dim = keys.shape[-1]
        num_slots = self.num_experts * self.slots_per_expert
        phi = self.param(
            "slot_params", nn.initializers.lecun_normal(), (in_dim, num_slots)
        )
        if self.normalize:
            scale = self.param("scale", nn.initializers.ones, ())
            phi = scale * _l2_normalize(phi, axis=0)
            logit_queries, logit_keys = _l2_normalize(queries), _l2_normalize(keys)
        else:
            logit_queries, logit_keys = queries, keys

        # Dispatch: per query and slot, a softmax over that query's causal
        # past -> each slot's input is a weighted average of past tokens.
        key_logits = jnp.einsum("...kd,ds->...ks", logit_keys, phi)
        dispatch_logits = jnp.where(
            key_mask[..., None], key_logits[..., None, :, :], _NEG_INF
        )  # (..., Q, K, S)
        dispatch = jax.nn.softmax(dispatch_logits, axis=-2)
        slots = jnp.einsum("...qks,...kd->...qsd", dispatch, keys)

        w0 = self.param(
            "w0", _expert_kernel_init(self.kernel_init), (self.num_experts, in_dim, self.mlp_dim)
        )
        b0 = self.param("b0", nn.initializers.zeros, (self.num_experts, self.mlp_dim))
        w1 = self.param(
            "w1",
            _expert_kernel_init(self.kernel_init),
            (self.num_experts, self.mlp_dim, self.out_dim),
        )
        b1 = self.param("b1", nn.initializers.zeros, (self.num_experts, self.out_dim))
        # Slot `j` belongs to expert `j // slots_per_expert`.
        slots = slots.reshape(*slots.shape[:-2], self.num_experts, self.slots_per_expert, in_dim)
        h = jnp.einsum("...epd,edm->...epm", slots, w0) + b0[:, None, :]
        h = parse_activation_fn(self.activation)(h)
        slot_out = jnp.einsum("...epm,emo->...epo", h, w1) + b1[:, None, :]
        slot_out = slot_out.reshape(*slot_out.shape[:-3], num_slots, self.out_dim)

        # Combine: per query, a softmax over all slots.
        combine = jax.nn.softmax(jnp.einsum("...qd,ds->...qs", logit_queries, phi), axis=-1)
        return jnp.einsum("...qs,...qso->...qo", combine, slot_out)

    def __call__(self, x: chex.Array) -> chex.Array:
        """Full-sequence, causal: `x` is `(..., seq_len, d)`."""
        seq_len = x.shape[-2]
        causal = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
        return self._mix(x, x, causal)

    def step(
        self, x: chex.Array, cached_inputs: chex.Array, step_idx: chex.Array, max_steps: int
    ) -> Tuple[chex.Array, chex.Array]:
        """Incremental counterpart to `__call__` for one new token `x`
        (`(..., d)`) at position `step_idx`: writes it into `cached_inputs`
        (`(..., max_steps + 1, d)`, see `init_soft_moe_cache`) and mixes only
        positions written so far. Returns `(output, updated_cache)`."""
        cached_inputs = cached_inputs.at[..., step_idx, :].set(x)
        key_mask = (jnp.arange(max_steps + 1) <= step_idx)[None, :]
        out = self._mix(x[..., None, :], cached_inputs, key_mask)
        return out[..., 0, :], cached_inputs


def _l2_normalize(x: chex.Array, axis: int = -1, eps: float = 1e-6) -> chex.Array:
    return x * jax.lax.rsqrt(jnp.sum(x * x, axis=axis, keepdims=True) + eps)


def moe_load_balancing_loss(intermediates: Mapping[str, Any]) -> chex.Array:
    """Switch Transformer load-balancing loss from the statistics every
    `MixtureOfExpertsMLP` sowed into `intermediates` (the `"intermediates"`
    collection returned by an `apply(..., mutable=["intermediates"])`).

    Per MoE layer: `num_experts * sum_e f_e * P_e`, where `f_e` is the
    fraction of (masked-in) routing assignments sent to expert `e` and `P_e`
    the mean router probability of `e`, pooled over every call of that layer
    (all CoT steps, all batch elements). It equals 1 when routing is
    perfectly uniform and grows as it concentrates. Averaged over layers;
    `0.0` if no MoE layer ran (e.g. `num_experts=0`), so it's safe to add
    unconditionally.
    """
    layer_losses = []

    def visit(tree: Any) -> None:
        if not isinstance(tree, Mapping):
            return
        for key, value in tree.items():
            if key != MOE_STATS_KEY:
                visit(value)
                continue
            # `value` is the tuple `sow` appends to, one dict per call; under
            # `nn.scan` each leaf also has a leading step axis - summing over
            # everything but the expert axis handles both.
            num_experts = value[0]["dispatch"].shape[-1]
            num_tokens = sum(jnp.sum(stats["num_tokens"]) for stats in value)
            dispatch = sum(
                stats["dispatch"].reshape(-1, num_experts).sum(axis=0) for stats in value
            )
            router_prob = sum(
                stats["router_prob"].reshape(-1, num_experts).sum(axis=0) for stats in value
            )
            dispatch_fraction = dispatch / jnp.maximum(jnp.sum(dispatch), 1.0)
            mean_router_prob = router_prob / jnp.maximum(num_tokens, 1.0)
            layer_losses.append(num_experts * jnp.sum(dispatch_fraction * mean_router_prob))

    visit(intermediates)
    if not layer_losses:
        return jnp.zeros(())
    return sum(layer_losses) / len(layer_losses)


def apply_with_moe_load_balancing_loss(
    apply_fn: Callable, enabled: bool, *args: Any, **kwargs: Any
) -> Tuple[Any, chex.Array]:
    """`apply_fn(*args, **kwargs)` plus its MoE load-balancing loss (see
    `moe_load_balancing_loss`). With `enabled=False` it's exactly the plain
    call - no `mutable` collection, nothing sown - and the loss is `0.0`."""
    if not enabled:
        return apply_fn(*args, **kwargs), jnp.zeros(())
    outputs, state = apply_fn(*args, mutable=["intermediates"], **kwargs)
    return outputs, moe_load_balancing_loss(state.get("intermediates", {}))


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

    `num_experts` (default `0`, meaning a plain dense MLP) replaces the MLP
    sub-layer with a `MixtureOfExpertsMLP` of that many experts (each
    `mlp_dim` wide), routing each token to its top `num_experts_per_token`
    experts. Only the MLP changes: attention, norms and the residual are
    untouched. With `num_experts=0` the parameter tree is exactly what it
    was before, so existing checkpoints and configs are unaffected.

    `moe_type` picks the kind of mixture when `num_experts > 0`: `"topk"`
    (default, `MixtureOfExpertsMLP`, uses `num_experts_per_token`) or
    `"soft"` (`SoftMoEMLP`, causal Soft MoE over the scratchpad, uses
    `soft_moe_slots_per_expert`/`soft_moe_normalize`). Soft MoE's `step`
    needs a cache of past MLP inputs (`cached_moe_inputs`, allocate with
    `init_soft_moe_cache`); for every other MLP kind it's `None` and passed
    straight through.
    """

    hidden_dim: int
    num_heads: int
    mlp_dim: int
    activation: str = "relu"
    kernel_init: Initializer = orthogonal(np.sqrt(2.0))
    use_sandwich_norm: bool = False
    use_rmsnorm: bool = False
    qkv_dim: Optional[int] = None
    num_experts: int = 0
    num_experts_per_token: int = 1
    moe_type: str = "topk"
    soft_moe_slots_per_expert: int = 1
    soft_moe_normalize: bool = False

    @property
    def _uses_soft_moe(self) -> bool:
        return self.num_experts > 0 and self.moe_type == "soft"

    def setup(self) -> None:
        if self.moe_type not in MOE_TYPES:
            raise ValueError(f"moe_type must be one of {MOE_TYPES}, got {self.moe_type!r}.")
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
        if self._uses_soft_moe:
            self.mlp_moe = SoftMoEMLP(
                self.num_experts,
                self.soft_moe_slots_per_expert,
                self.mlp_dim,
                self.hidden_dim,
                self.activation,
                self.kernel_init,
                self.soft_moe_normalize,
            )
        elif self.num_experts > 0:
            self.mlp_moe = MixtureOfExpertsMLP(
                self.num_experts,
                self.num_experts_per_token,
                self.mlp_dim,
                self.hidden_dim,
                self.activation,
                self.kernel_init,
            )
        else:
            self.mlp_dense_0 = nn.Dense(self.mlp_dim, kernel_init=self.kernel_init)
            self.mlp_dense_1 = nn.Dense(self.hidden_dim, kernel_init=self.kernel_init)
        if self.use_sandwich_norm:
            self.attn_post_norm = norm_cls()
            self.mlp_post_norm = norm_cls()

    def _mlp_out(self, y: chex.Array, token_mask: Optional[chex.Array]) -> chex.Array:
        """The (non-soft-MoE) MLP applied to already-normalized `y`."""
        if self.num_experts > 0:
            return self.mlp_moe(y, token_mask)
        y = self.mlp_dense_0(y)
        y = parse_activation_fn(self.activation)(y)
        return self.mlp_dense_1(y)

    def _mlp_residual(self, tokens: chex.Array, mlp_out: chex.Array) -> chex.Array:
        y = tokens + mlp_out
        return self.mlp_post_norm(y) if self.use_sandwich_norm else y

    def __call__(
        self,
        tokens: chex.Array,
        mask: Optional[chex.Array] = None,
        token_mask: Optional[chex.Array] = None,
    ) -> chex.Array:
        """`token_mask` (optional, `(*batch, seq_len)` bool) only selects
        which tokens count towards the MoE load-balancing statistics (see
        `MixtureOfExpertsMLP`); it never changes the output."""
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
        y = self.mlp_norm(tokens)
        # Soft MoE is always causal over `seq_len` (see `SoftMoEMLP`) - the
        # only way this block is ever run over a full sequence here.
        mlp_out = self.mlp_moe(y) if self._uses_soft_moe else self._mlp_out(y, token_mask)
        return self._mlp_residual(tokens, mlp_out)

    def step(
        self,
        token: chex.Array,
        cached_keys: chex.Array,
        cached_values: chex.Array,
        step_idx: chex.Array,
        max_steps: int,
        token_mask: Optional[chex.Array] = None,
        cached_moe_inputs: Optional[chex.Array] = None,
    ) -> Tuple[chex.Array, chex.Array, chex.Array, Optional[chex.Array]]:
        """
        token: `(*batch, hidden_dim)` - this layer's input at this step.
        cached_keys, cached_values: `(*batch, max_steps + 1, num_heads,
            head_dim)` - this layer's cache (positions > step_idx are
            not-yet-written).
        step_idx: traced scalar - the position to write/attend through.
        token_mask: optional `(*batch,)` bool - see `__call__`.
        cached_moe_inputs: `(*batch, max_steps + 1, hidden_dim)` cache of
            past MLP inputs if this block uses soft MoE, else `None` - see
            `init_soft_moe_cache`.

        Returns `(new_token, updated_keys, updated_values,
        updated_moe_inputs)` (the last is `None` without soft MoE).
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
        y = self.mlp_norm(token)
        if self._uses_soft_moe:
            mlp_out, cached_moe_inputs = self.mlp_moe.step(
                y, cached_moe_inputs, step_idx, max_steps
            )
        else:
            mlp_out = self._mlp_out(y, token_mask)
        return self._mlp_residual(token, mlp_out), cached_keys, cached_values, cached_moe_inputs


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
    num_experts: int = 0
    num_experts_per_token: int = 1
    moe_type: str = "topk"
    soft_moe_slots_per_expert: int = 1
    soft_moe_normalize: bool = False

    @nn.compact
    def __call__(
        self, carry: Tuple[chex.Array, ...], step_idx: chex.Array
    ) -> Tuple[Tuple[chex.Array, ...], None]:
        (
            current_token,
            cached_keys,
            cached_values,
            cached_moe_inputs,
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
                self.num_experts,
                self.num_experts_per_token,
                self.moe_type,
                self.soft_moe_slots_per_expert,
                self.soft_moe_normalize,
            )
            for _ in range(self.num_layers)
        ]
        halting_head = HaltingHead(
            self.halting_hidden_dims, self.activation, self.kernel_init
        )

        x = current_token + pos_embedding[step_idx]
        new_cached_keys = []
        new_cached_values = []
        new_cached_moe_inputs = []
        for layer_idx, block in enumerate(blocks):
            # `still_running` (pre-update) masks this step out of the MoE
            # load-balancing statistics for examples that already halted.
            x, k, v, m = block.step(
                x,
                cached_keys[layer_idx],
                cached_values[layer_idx],
                step_idx,
                self.max_steps,
                token_mask=still_running,
                cached_moe_inputs=cached_moe_inputs[layer_idx],
            )
            new_cached_keys.append(k)
            new_cached_values.append(v)
            new_cached_moe_inputs.append(m)
        state = x
        cached_keys = new_cached_keys
        cached_values = new_cached_values
        cached_moe_inputs = new_cached_moe_inputs
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
            cached_moe_inputs,
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

    `num_experts` (default `0`, a plain dense MLP) and
    `num_experts_per_token` (default `1`) are forwarded to every shared
    `TransformerBlock`: a positive `num_experts` turns each block's MLP into
    a top-`num_experts_per_token` routed `MixtureOfExpertsMLP`.
    `moe_type="soft"` makes it a causal `SoftMoEMLP` instead, with
    `soft_moe_slots_per_expert` slots per expert and optional
    `soft_moe_normalize` - see `TransformerBlock`.
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
    num_experts: int = 0
    num_experts_per_token: int = 1
    moe_type: str = "topk"
    soft_moe_slots_per_expert: int = 1
    soft_moe_normalize: bool = False

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
        cached_moe_inputs = [
            init_soft_moe_cache(
                batch_shape, self.max_steps, self.hidden_dim, self.moe_type, self.num_experts
            )
            for _ in range(self.num_layers)
        ]

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
            # MoE load-balancing statistics sown per step (see
            # `MixtureOfExpertsMLP`) are stacked along a leading step axis.
            variable_axes={"intermediates": 0},
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
            self.num_experts,
            self.num_experts_per_token,
            self.moe_type,
            self.soft_moe_slots_per_expert,
            self.soft_moe_normalize,
        )

        initial_carry = (
            initial_token,  # current_token
            cached_keys,
            cached_values,
            cached_moe_inputs,
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
