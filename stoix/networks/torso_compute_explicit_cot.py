"""A transformer torso whose adaptive computation is an *explicit* chain of thought.

Unlike `stoix.networks.torso_compute_transformer.TransformerChainOfThoughtTorso`
(latent CoT: the scratchpad entry fed back in at each step is just the raw
continuous hidden state), `TransformerExplicitCoTTorso` forces every "thought"
through a discrete vocabulary bottleneck: at every step it samples a thought
*token id* from a learned `vocab_size`-way categorical distribution - a
genuine output of the policy, in exactly the same sense as the environment
action - and it is that token's embedding, not the raw hidden state, which
gets appended to the scratchpad and attended over at the next step. This
makes the reasoning trace literally inspectable (a sequence of token ids that
could be decoded/rendered), rather than an opaque vector.

There is no separate halt predictor. The categorical distribution at each
step has one extra class beyond the `vocab_size` thought tokens - "predict
the environment action now" - and *choosing that class is what halts*.
Halting and "which thought to think next" are therefore a single discrete
choice, trained the same way as the environment action itself: via the
score-function (REINFORCE) estimator, using the log-probability of the
sampled trajectory. This torso therefore has the same rollout/replay/
deterministic modes as `AdaptiveComputationTimeTorso`, except replaying also
needs the actual sequence of emitted token ids (not just how many steps were
taken), since a different token would have led the model to think something
different at the next step:

  - Rollout mode (`target_tokens=None`): samples the token trajectory
    (thoughts and, eventually, the halting "act now" token) and returns
    `(embedding, compute_time, thought_tokens)`. `thought_tokens` has shape
    `(*batch, max_steps)`, one entry per step including the halting one;
    entries past `compute_time` are never read back in replay mode, since
    replaying halts at the same step that produced them (the first "act now"
    token in the sequence).
  - Replay mode (`target_tokens=<array>`): deterministically replays exactly
    that token trajectory (no rng) and returns `(embedding, log_prob,
    per_step_log_prob)`: `log_prob` is the log-probability, under the
    current parameters, of the whole trajectory (thought choices and the
    halting choice together, i.e. `per_step_log_prob.sum(-1)`);
    `per_step_log_prob` (shape `(*batch, max_steps)`) is that same quantity
    left unsummed, one entry per step, zeroed past the step the trajectory
    actually halted at - callers that need a per-decision (rather than
    per-trajectory) PPO ratio, e.g. to clip each step's importance ratio
    individually instead of one joint ratio over the summed log-prob, use
    this instead of `log_prob`. Attention within a
    step is causally masked (see `TransformerBlock`), so - unlike the
    latent-CoT torso, where the scratchpad entry fed back in is the model's
    own hidden state and therefore only knowable by actually running the
    steps - every scratchpad entry here is just the embedding of an
    already-known token, so replay doesn't need the unrolled step-by-step
    loop at all: the whole scratchpad can be built up front and scored in
    one causal-masked pass, reading every step's logits off in parallel
    instead of recomputing the growing prefix from scratch `max_steps`
    times. (This stops being true under `use_latent_feedback=True` - see
    below.)
  - Deterministic mode (`deterministic=True`): at every step, picks the
    highest-probability class - a thought token or "act now" - instead of
    sampling, for greedy evaluation.

The environment action head still reads off the final continuous hidden
state (not a token embedding) - the discrete tokens are a communicative side
channel the policy can use to "show its work", not a bottleneck on how much
information reaches the action itself.

`use_latent_feedback=True` additionally implements "latent feedback
decoding" from the Full-Bandwidth Transformer (arXiv:2608.08888). In a
standard transformer, the only thing carried from step t to step t+1's input
is the sampled token's embedding - the top-layer hidden state that produced
it (itself a function of every layer below) is discarded. Latent feedback
decoding instead fuses that hidden state with the new token's embedding via
a dimension-preserving gated linear unit, and feeds the fused vector back
into the scratchpad in place of the bare embedding:

    e_t (X) h_{t-1} = W^U h_{t-1} * sigmoid(W^G e_t)          (paper Eq. 4)

The asymmetry is deliberate (see the paper's sec. 3.1): the hidden state
`h_{t-1}` occupies the multiplicative *value* pathway and the new token
embedding `e_t` only *gates* it, so the model cannot cheat by learning to
ignore `h_{t-1}` and recover plain token-only feedback - discarding the
value pathway discards the input itself. The very first scratchpad entry
(the observation) is never gated, matching the paper's `C = e_0, e_1 (X)
h_0^L, ...`: there is no previous hidden state for it to fuse with.

This recurrence (step t+1's input depends on the *computed* hidden state at
step t, not just on which token was sampled) is exactly why generation is
already sequential in the paper's decoding setting, and it is why replay
mode here can no longer score a known token trajectory in a single
causal-masked pass when `use_latent_feedback=True`: the fused scratchpad
entries have to be built up one step at a time, just like rollout, even
though the tokens themselves are already known. See the `replaying` branch
below.
"""

from typing import Optional, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.linen.initializers import Initializer, normal, orthogonal

from stoix.networks.torso_compute_transformer import TransformerBlock
from stoix.networks.utils import parse_activation_fn

_NEG_INF = jnp.finfo(jnp.float32).min


class _ExplicitCoTBackbone(nn.Module):
    """The transformer blocks, token (un)embedding, positional embedding and
    (when `use_latent_feedback=True`) latent-feedback fusion - instantiated
    *once* in `TransformerExplicitCoTTorso.__call__` and reused, via its
    methods, by every code path that needs them: `run` for the parallel
    one-shot replay pass, and `step` for the scanned per-step body (`nn.scan`
    's functional form, see `TransformerExplicitCoTTorso.__call__`) used by
    rollout mode and (when `use_latent_feedback=True`) replay mode. `run`
    reprocesses a whole (sub)sequence in one call (used once, so O(seq_len)
    is fine); `step` is `TransformerBlock`'s KV-cached incremental mode,
    advancing one new token per call in O(1) instead of reprocessing
    everything so far - see `TransformerBlock` for why the two share weights.

    Sharing one `TransformerBlock` list between `run` and `step` - rather
    than, say, building separate blocks for each code path - is what makes
    those paths use the exact same weights instead of silently diverging into
    independently-initialized copies: a `setup()`-style module's submodules
    are created once (on first use) and namespaced under this module's own
    scope regardless of which method is called first, so a single `.init()`
    call (which only ever exercises *one* code path, whichever `replaying`/
    `use_latent_feedback` combination it's called with) still creates the
    complete parameter set every other path also needs - the layers here are
    only ever a function of `hidden_dim`/`num_heads`/`mlp_dim`/`vocab_size`,
    never of which method/sequence length a given call happens to use.
    """

    hidden_dim: int
    vocab_size: int
    num_heads: int
    num_layers: int
    mlp_dim: int
    max_steps: int
    activation: str
    kernel_init: Initializer
    use_latent_feedback: bool

    def setup(self) -> None:
        self.pos_embedding = self.param(
            "pos_embedding", normal(stddev=0.02), (self.max_steps + 1, self.hidden_dim)
        )
        # `token_embed` doubles as the unembedding ("token_head") via
        # `.attend()` (query @ embedding.T) - standard input/output
        # weight-tying. Its table has one row beyond `vocab_size` for
        # `act_token_id`: that row is only ever read through `.attend()` (to
        # produce the "act now" logit), never through `token_embed(...)` (the
        # embedding lookup fed back into the scratchpad), since choosing
        # `act_token_id` halts before another embedding is needed - see the
        # `act_token_id -> 0` substitutions at the `token_embed(...)` call
        # sites in `TransformerExplicitCoTTorso.__call__`.
        self.token_embed = nn.Embed(
            num_embeddings=self.vocab_size + 1, features=self.hidden_dim
        )
        self.blocks = [
            TransformerBlock(
                self.hidden_dim, self.num_heads, self.mlp_dim, self.activation, self.kernel_init
            )
            for _ in range(self.num_layers)
        ]
        if self.use_latent_feedback:
            # Paper Eq. 4: e_t (X) h_{t-1} = W^U h_{t-1} * sigmoid(W^G e_t).
            # No bias terms, matching the paper.
            self.latent_feedback_value = nn.Dense(
                self.hidden_dim, use_bias=False, kernel_init=self.kernel_init
            )
            self.latent_feedback_gate = nn.Dense(
                self.hidden_dim, use_bias=False, kernel_init=self.kernel_init
            )

    def token_head(self, state: chex.Array) -> chex.Array:
        return self.token_embed.attend(state)

    def run(self, scratchpad: chex.Array) -> chex.Array:
        """Runs the transformer over `scratchpad` (`(*batch, max_steps,
        hidden_dim)`) under a causal mask in one call and returns the
        per-position states - used only by the parallel one-shot replay pass
        (`TransformerExplicitCoTTorso.__call__`'s `replaying and not
        use_latent_feedback` branch), which already knows the whole token
        trajectory up front so has no per-step recurrence to cache against.
        The other paths use `step` instead (see class docstring)."""
        seq_len = scratchpad.shape[-2]
        batch_shape = scratchpad.shape[:-2]
        tokens_in = scratchpad + self.pos_embedding[:seq_len]
        causal_mask = nn.make_causal_mask(jnp.ones(batch_shape + (seq_len,)))
        for block in self.blocks:
            tokens_in = block(tokens_in, mask=causal_mask)
        return tokens_in

    def fuse_latent_feedback(self, state: chex.Array, token_emb: chex.Array) -> chex.Array:
        # `state` plays the role of h_{t-1} and `token_emb` the role of e_t,
        # evaluated one step "in the future" relative to the paper's
        # indexing (we fuse the state and token produced *at* step to build
        # the scratchpad entry consumed *after* step, whereas the paper
        # names the state as already "previous" - same relationship).
        return self.latent_feedback_value(state) * jax.nn.sigmoid(
            self.latent_feedback_gate(token_emb)
        )

    def step(
        self,
        token: chex.Array,
        cached_keys: list,
        cached_values: list,
        step_idx: chex.Array,
    ) -> Tuple[chex.Array, list, list]:
        """Incremental counterpart to `run`, used by the scanned per-step
        body (rollout mode and, when `use_latent_feedback=True`, replay
        mode): advances a single new token through every layer, reading/
        extending each layer's KV-cache (`TransformerBlock.step`) instead of
        reprocessing the whole scratchpad. Uses the same `self.blocks` as
        `run`, so its weights stay shared with the parallel replay pass (see
        class docstring)."""
        x = token + self.pos_embedding[step_idx]
        new_cached_keys = []
        new_cached_values = []
        for layer_idx, block in enumerate(self.blocks):
            x, k, v = block.step(
                x, cached_keys[layer_idx], cached_values[layer_idx], step_idx, self.max_steps
            )
            new_cached_keys.append(k)
            new_cached_values.append(v)
        return x, new_cached_keys, new_cached_values


class TransformerExplicitCoTTorso(nn.Module):
    """Explicit Chain-of-Thought transformer torso with adaptively-sampled
    halting and discrete, sampled thought tokens. See module docstring.

    `min_steps` forbids halting (voluntarily, in replay, or greedily) before
    that many steps have been taken - `is_final_step` still forces a halt at
    `max_steps` regardless. Setting `min_steps == max_steps` therefore removes
    adaptivity entirely: every example always takes exactly `max_steps` -
    useful as a fixed-budget baseline against the adaptive policy.

    `use_latent_feedback` switches the scratchpad feedback from the sampled
    token's bare embedding to a gated fusion of that embedding with the
    hidden state that produced it (the Full-Bandwidth Transformer's "latent
    feedback decoding", arXiv:2608.08888) - see the module docstring.
    """

    hidden_dim: int
    vocab_size: int
    num_heads: int = 4
    num_layers: int = 2
    mlp_dim: int = 512
    max_steps: int = 8
    min_steps: int = 1
    activation: str = "relu"
    kernel_init: Initializer = orthogonal(np.sqrt(2.0))
    use_input_layer_norm: bool = False
    use_latent_feedback: bool = False

    @nn.compact
    def __call__(
        self,
        observation: chex.Array,
        rng: Optional[chex.PRNGKey] = None,
        target_tokens: Optional[chex.Array] = None,
        deterministic: bool = False,
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """
        Args:
            observation: the input embedding to think about; becomes the
                first scratchpad token.
            rng: PRNG key used to sample the per-step token (thought or
                "act now") choices. Required unless replaying or
                `deterministic=True`.
            target_tokens: for replay mode, the exact per-step token
                trajectory (shape `(*batch, max_steps)`) to replay - see
                module docstring.
            deterministic: if True (and not replaying), pick the highest
                probability class at each step instead of sampling.

        Returns:
            `(embedding, compute_time, thought_tokens)` when not replaying,
            or `(embedding, log_prob, per_step_log_prob)` when replaying a
            known trajectory - see module docstring for `per_step_log_prob`.
        """
        batch_shape = observation.shape[:-1]
        replaying = target_tokens is not None
        if not replaying and not deterministic and rng is None:
            raise ValueError(
                "rng must be provided to TransformerExplicitCoTTorso when sampling "
                "(i.e. not replaying and deterministic=False)."
            )
        if not (1 <= self.min_steps <= self.max_steps):
            raise ValueError(
                f"min_steps must be between 1 and max_steps ({self.max_steps}), "
                f"got min_steps={self.min_steps}."
            )

        # The token vocabulary has one extra class beyond the `vocab_size`
        # thought tokens: choosing it means "predict the environment action
        # now" - i.e. it is the halting decision.
        act_token_id = self.vocab_size
        num_classes = self.vocab_size + 1

        initial_token = nn.Dense(self.hidden_dim, kernel_init=self.kernel_init)(observation)
        if self.use_input_layer_norm:
            initial_token = nn.LayerNorm()(initial_token)

        # See `_ExplicitCoTBackbone` for why a single shared instance (rather
        # than separately-constructed submodules per code path below) is
        # what keeps the parallel replay pass and the scanned rollout/
        # latent-feedback-replay path tied to the same weights.
        backbone = _ExplicitCoTBackbone(
            self.hidden_dim,
            self.vocab_size,
            self.num_heads,
            self.num_layers,
            self.mlp_dim,
            self.max_steps,
            self.activation,
            self.kernel_init,
            self.use_latent_feedback,
        )

        # Per-step "act now" legality, precomputed once as a constant boolean
        # mask over the step axis (shape `(max_steps, num_classes)`) instead
        # of as a Python-level `if` per loop iteration - used by both the
        # rollout loop below and the parallel replay path. Applied via
        # `jnp.where`/`log_softmax(..., where=...)` (value substitution)
        # rather than adding `_NEG_INF` to the logits, so an illegal class is
        # excluded regardless of how large `token_head`'s raw output is -
        # additive masking can in principle be defeated by a logit close to
        # float32's max magnitude cancelling out `_NEG_INF`.
        step_counts = np.arange(1, self.max_steps + 1)
        is_final_step = step_counts == self.max_steps
        can_halt = step_counts >= self.min_steps
        legal_mask = np.ones((self.max_steps, num_classes), dtype=bool)
        # A halt is forced at the final step regardless of the policy: mask
        # out every thought class so "act now" is the only one left. Its
        # log-probability under that mask is an inputs-independent constant
        # (zero), so no gradient flows through a step that was never really a
        # choice.
        legal_mask[is_final_step, :] = False
        legal_mask[is_final_step, act_token_id] = True
        # Before min_steps, "act now" isn't a legal choice yet.
        legal_mask[~can_halt, act_token_id] = False
        legal_mask = jnp.asarray(legal_mask)

        if replaying and not self.use_latent_feedback:
            # Attention is causally masked (see `TransformerBlock`), so every
            # scratchpad entry is determined by the known `target_tokens`
            # alone - the whole trajectory can be built and scored in one
            # pass instead of a step-by-step loop. Scratchpad position `step`
            # is the embedding of `target_tokens[..., step - 1]` (position 0
            # is the observation); `act_token_id` has no embedding, so
            # substitute a dummy id where it was emitted - those entries are
            # never read since nothing attends past its own (causally masked)
            # position. Only valid without latent feedback: with it, each
            # scratchpad entry depends on the *computed* hidden state of the
            # previous step (see module docstring), so it can no longer be
            # built up front from `target_tokens` alone - see the scanned
            # path below, shared with rollout mode.
            if self.max_steps > 1:
                # With `max_steps == 1` the single step is also the final,
                # forced-halt step, so no previous token is ever fed back -
                # `token_embed` is simply not called in that case (its params
                # still exist either way, created unconditionally by
                # `backbone`'s `setup()`).
                thought_id = jnp.where(target_tokens == act_token_id, 0, target_tokens)
                token_embeds = backbone.token_embed(thought_id)
                scratchpad = jnp.concatenate(
                    [initial_token[..., None, :], token_embeds[..., :-1, :]], axis=-2
                )
            else:
                scratchpad = initial_token[..., None, :]
            states = backbone.run(scratchpad)  # (*batch, max_steps, hidden_dim)

            # Same masked-logits expression as the scanned path below, so
            # replay scores the exact distribution rollout would have sampled
            # from at each state - required for the log-probs used in the
            # policy gradient to match the trajectory that was actually taken.
            token_logits = jnp.where(legal_mask, backbone.token_head(states), _NEG_INF)
            log_token_probs = jax.nn.log_softmax(token_logits, axis=-1)
            token_log_prob = jnp.take_along_axis(
                log_token_probs, target_tokens[..., None], axis=-1
            ).squeeze(axis=-1)

            # A step only counts if no earlier step already halted - i.e. its
            # exclusive running count of "act now" tokens so far is zero.
            halted = target_tokens == act_token_id
            earlier_halts = jnp.cumsum(halted.astype(jnp.int32), axis=-1) - halted.astype(
                jnp.int32
            )
            still_running = earlier_halts == 0
            per_step_log_prob = jnp.where(still_running, token_log_prob, 0.0)
            log_prob = jnp.sum(per_step_log_prob, axis=-1)

            # The final state is read off at the first "act now" token (the
            # forced halt at max_steps guarantees there always is one).
            halt_step = jnp.argmax(halted, axis=-1)
            final_state = jnp.take_along_axis(
                states, halt_step[..., None, None], axis=-2
            ).squeeze(axis=-2)

            return final_state, log_prob, per_step_log_prob

        # KV-cached step-by-step build, shared by rollout mode and (when
        # `use_latent_feedback=True`) replay mode: with latent feedback, each
        # scratchpad entry depends on the *computed* hidden state of the
        # previous step (see module docstring), so - unlike the parallel path
        # above - it can't be built up front even when the token trajectory
        # (`target_tokens`) is already known. The two modes differ only in
        # where `token_id` comes from (sampled/argmax vs. read off
        # `target_tokens`) and what they accumulate (compute_time/emitted
        # tokens vs. log_prob) - everything else, including the cache update,
        # is identical.
        assert self.hidden_dim % self.num_heads == 0, (
            f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads})."
        )
        head_dim = self.hidden_dim // self.num_heads
        cache_shape = batch_shape + (self.max_steps + 1, self.num_heads, head_dim)
        cached_keys = [jnp.zeros(cache_shape) for _ in range(self.num_layers)]
        cached_values = [jnp.zeros(cache_shape) for _ in range(self.num_layers)]

        still_running = jnp.ones(batch_shape, dtype=bool)
        final_state = initial_token
        num_steps_taken = jnp.zeros(batch_shape)
        emitted_tokens = jnp.zeros(batch_shape + (self.max_steps,), dtype=jnp.int32)
        log_prob = jnp.zeros(batch_shape)
        # Only ever written when replaying (see `step_fn`); carried
        # regardless of mode since `nn.scan` needs a fixed carry structure.
        per_step_log_prob = jnp.zeros(batch_shape + (self.max_steps,))
        # Unused (never read) unless sampling, but must still be a concrete
        # array: it's carried through every scan step regardless of mode.
        step_rng = rng if rng is not None else jax.random.PRNGKey(0)

        def step_fn(
            backbone: _ExplicitCoTBackbone, carry: Tuple[chex.Array, ...], step_idx: chex.Array
        ) -> Tuple[Tuple[chex.Array, ...], None]:
            (
                current_token,
                cached_keys,
                cached_values,
                still_running,
                num_steps_taken,
                emitted_tokens,
                log_prob,
                per_step_log_prob,
                final_state,
                rng,
            ) = carry

            state, cached_keys, cached_values = backbone.step(
                current_token, cached_keys, cached_values, step_idx
            )
            token_logits = jnp.where(legal_mask[step_idx], backbone.token_head(state), _NEG_INF)

            if replaying:
                token_id = target_tokens[..., step_idx]
                log_token_probs = jax.nn.log_softmax(token_logits, axis=-1)
                token_log_prob = jnp.take_along_axis(
                    log_token_probs, token_id[..., None], axis=-1
                ).squeeze(axis=-1)
                # Zeroed (not merely left unwritten) past the step the
                # trajectory actually halted at, matching the parallel
                # (`run`-based) replay path's masking - see module docstring.
                step_log_prob = jnp.where(still_running, token_log_prob, 0.0)
                log_prob = log_prob + step_log_prob
                per_step_log_prob = per_step_log_prob.at[..., step_idx].set(step_log_prob)
            elif deterministic:
                token_id = jnp.argmax(token_logits, axis=-1)
            else:
                rng, token_rng = jax.random.split(rng)
                token_id = jax.random.categorical(token_rng, token_logits)

            halts_this_step = still_running & (token_id == act_token_id)
            final_state = jnp.where(halts_this_step[..., None], state, final_state)
            if not replaying:
                num_steps_taken = num_steps_taken + still_running.astype(jnp.float32)
                emitted_tokens = emitted_tokens.at[..., step_idx].set(token_id)
            still_running = still_running & (~halts_this_step)

            # `act_token_id`'s embedding row is never read back through this
            # lookup (see `_ExplicitCoTBackbone.setup`); substitute a dummy
            # id where it was chosen - those entries are never read back,
            # since once an example has halted, its `current_token` keeps
            # being computed (every example runs the same fixed number of
            # scan iterations) but is discarded.
            thought_id = jnp.where(token_id == act_token_id, 0, token_id)
            token_emb = backbone.token_embed(thought_id)
            current_token = (
                backbone.fuse_latent_feedback(state, token_emb)
                if self.use_latent_feedback
                else token_emb
            )

            new_carry = (
                current_token,
                cached_keys,
                cached_values,
                still_running,
                num_steps_taken,
                emitted_tokens,
                log_prob,
                per_step_log_prob,
                final_state,
                rng,
            )
            return new_carry, None

        # Functional form of `nn.scan` (rather than wrapping a fresh
        # `nn.Module` class): `backbone` is instantiated once, above, and
        # passed through explicitly so its params stay shared with the
        # parallel replay path rather than becoming a second, independently
        # -initialized copy scoped under the scan - see `_ExplicitCoTBackbone`.
        scan_step = nn.scan(step_fn, variable_broadcast="params", split_rngs={"params": False})
        initial_carry = (
            initial_token,  # current_token
            cached_keys,
            cached_values,
            still_running,
            num_steps_taken,
            emitted_tokens,
            log_prob,
            per_step_log_prob,
            final_state,
            step_rng,
        )
        (
            _,
            _,
            _,
            _,
            num_steps_taken,
            emitted_tokens,
            log_prob,
            per_step_log_prob,
            final_state,
            _,
        ), _ = scan_step(backbone, initial_carry, jnp.arange(self.max_steps))

        if replaying:
            return final_state, log_prob, per_step_log_prob
        return final_state, num_steps_taken, emitted_tokens
