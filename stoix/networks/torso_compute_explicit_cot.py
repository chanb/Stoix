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
    that token trajectory (no rng) and returns `(embedding, log_prob)`: the
    log-probability, under the current parameters, of the whole trajectory
    (thought choices and the halting choice together). Attention within a
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

from typing import Optional, Tuple, Union

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.linen.initializers import Initializer, normal, orthogonal

from stoix.networks.torso_compute_transformer import TransformerBlock
from stoix.networks.utils import parse_activation_fn

_NEG_INF = jnp.finfo(jnp.float32).min


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
    ) -> Union[Tuple[chex.Array, chex.Array], Tuple[chex.Array, chex.Array, chex.Array]]:
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
            or `(embedding, log_prob)` when replaying a known trajectory.
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
            observation = nn.LayerNorm()(observation)

        pos_embedding = self.param(
            "pos_embedding", normal(stddev=0.02), (self.max_steps + 1, self.hidden_dim)
        )
        # `token_embed` doubles as the unembedding ("token_head") via
        # `.attend()` (query @ embedding.T) - standard input/output
        # weight-tying. Its table has one row beyond `vocab_size` for
        # `act_token_id`: that row is only ever read through `.attend()` (to
        # produce the "act now" logit), never through `token_embed(...)` (the
        # embedding lookup fed back into the scratchpad), since choosing
        # `act_token_id` halts before another embedding is needed - see the
        # `act_token_id -> 0` substitutions below/in the rollout loop.
        token_embed = nn.Embed(num_embeddings=num_classes, features=self.hidden_dim)
        token_head = token_embed.attend

        if self.use_latent_feedback:
            # Paper Eq. 4: e_t (X) h_{t-1} = W^U h_{t-1} * sigmoid(W^G e_t).
            # No bias terms, matching the paper; `state` plays the role of
            # h_{t-1} and `token_emb` the role of e_t, evaluated one step
            # "in the future" relative to the paper's indexing (we fuse the
            # state and token produced *at* step to build the scratchpad
            # entry consumed *after* step, whereas the paper names the state
            # as already "previous" - same relationship, see call sites).
            latent_feedback_value = nn.Dense(
                self.hidden_dim,
                use_bias=False,
                kernel_init=self.kernel_init,
                name="latent_feedback_value",
            )
            latent_feedback_gate = nn.Dense(
                self.hidden_dim,
                use_bias=False,
                kernel_init=self.kernel_init,
                name="latent_feedback_gate",
            )

            def fuse_latent_feedback(state: chex.Array, token_emb: chex.Array) -> chex.Array:
                return latent_feedback_value(state) * jax.nn.sigmoid(latent_feedback_gate(token_emb))

        blocks = [
            TransformerBlock(
                self.hidden_dim, self.num_heads, self.mlp_dim, self.activation, self.kernel_init
            )
            for _ in range(self.num_layers)
        ]

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

        def run_step(scratchpad: chex.Array, step: int) -> Tuple[chex.Array, chex.Array]:
            """Runs the transformer over `scratchpad[..., :step + 1, :]` and
            returns `(state, token_logits)` for that step - the shared core
            of both the rollout loop and (when `use_latent_feedback=True`)
            the sequential replay loop below, since both need to recompute
            the growing causal-masked prefix from scratch at every step
            (no KV cache)."""
            seq_len = step + 1
            tokens_in = scratchpad[..., :seq_len, :] + pos_embedding[:seq_len]
            causal_mask = nn.make_causal_mask(jnp.ones(batch_shape + (seq_len,)))
            for block in blocks:
                tokens_in = block(tokens_in, mask=causal_mask)
            state = tokens_in[..., -1, :]
            token_logits = jnp.where(legal_mask[step], token_head(state), _NEG_INF)
            return state, token_logits

        if replaying and self.use_latent_feedback:
            # Latent feedback makes each scratchpad entry depend on the
            # *computed* hidden state of the previous step (see module
            # docstring), so - unlike the parallel path below - the whole
            # scratchpad can no longer be built up front from `target_tokens`
            # alone. Replay a known token trajectory one step at a time
            # instead, mirroring the rollout loop but reading `target_tokens`
            # instead of sampling.
            scratchpad = jnp.zeros(batch_shape + (self.max_steps + 1, self.hidden_dim))
            scratchpad = scratchpad.at[..., 0, :].set(initial_token)

            still_running = jnp.ones(batch_shape, dtype=bool)
            final_state = initial_token
            log_prob = jnp.zeros(batch_shape)

            for step in range(self.max_steps):
                state, token_logits = run_step(scratchpad, step)
                log_token_probs = jax.nn.log_softmax(token_logits, axis=-1)
                token_id = target_tokens[..., step]
                token_log_prob = jnp.take_along_axis(
                    log_token_probs, token_id[..., None], axis=-1
                ).squeeze(axis=-1)
                log_prob = log_prob + jnp.where(still_running, token_log_prob, 0.0)

                halts_this_step = still_running & (token_id == act_token_id)
                final_state = jnp.where(halts_this_step[..., None], state, final_state)
                still_running = still_running & (~halts_this_step)

                if step_counts[step] != self.max_steps:
                    thought_id = jnp.where(token_id == act_token_id, 0, token_id)
                    token_emb = token_embed(thought_id)
                    fused = fuse_latent_feedback(state, token_emb)
                    scratchpad = scratchpad.at[..., step + 1, :].set(fused)

            return final_state, log_prob

        if replaying:
            # Attention is causally masked (see `TransformerBlock`), so every
            # scratchpad entry is determined by the known `target_tokens`
            # alone - the whole trajectory can be built and scored in one
            # pass instead of an unrolled loop. Scratchpad position `step` is
            # the embedding of `target_tokens[..., step - 1]` (position 0 is
            # the observation); `act_token_id` has no embedding, so substitute
            # a dummy id where it was emitted - those entries are never read
            # since nothing attends past its own (causally masked) position.
            if self.max_steps > 1:
                # Guarded the same way as the rollout loop below (which only
                # calls `token_embed` on non-final steps): with `max_steps ==
                # 1` the single step is also the final, forced-halt step, so
                # `token_embed` is never called there either - calling it
                # unconditionally here would create an "embedding" param at
                # apply-time that rollout-driven `init` never created.
                thought_id = jnp.where(target_tokens == act_token_id, 0, target_tokens)
                token_embeds = token_embed(thought_id)
                scratchpad = jnp.concatenate(
                    [initial_token[..., None, :], token_embeds[..., :-1, :]], axis=-2
                )
            else:
                scratchpad = initial_token[..., None, :]
            tokens_in = scratchpad + pos_embedding[: self.max_steps]

            causal_mask = nn.make_causal_mask(jnp.ones(batch_shape + (self.max_steps,)))
            for block in blocks:
                tokens_in = block(tokens_in, mask=causal_mask)
            states = tokens_in  # (*batch, max_steps, hidden_dim); one state per step.

            # Same masked-logits expression as the rollout loop below, so
            # replay scores the exact distribution rollout would have sampled
            # from at each state - required for the log-probs used in the
            # policy gradient to match the trajectory that was actually taken.
            token_logits = jnp.where(legal_mask, token_head(states), _NEG_INF)
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
            log_prob = jnp.sum(jnp.where(still_running, token_log_prob, 0.0), axis=-1)

            # The final state is read off at the first "act now" token (the
            # forced halt at max_steps guarantees there always is one).
            halt_step = jnp.argmax(halted, axis=-1)
            final_state = jnp.take_along_axis(
                states, halt_step[..., None, None], axis=-2
            ).squeeze(axis=-2)

            return final_state, log_prob

        scratchpad = jnp.zeros(batch_shape + (self.max_steps + 1, self.hidden_dim))
        scratchpad = scratchpad.at[..., 0, :].set(initial_token)

        still_running = jnp.ones(batch_shape, dtype=bool)
        num_steps_taken = jnp.zeros(batch_shape)
        final_state = initial_token
        emitted_tokens = jnp.zeros(batch_shape + (self.max_steps,), dtype=jnp.int32)

        for step in range(self.max_steps):
            state, token_logits = run_step(scratchpad, step)

            if deterministic:
                token_id = jnp.argmax(token_logits, axis=-1)
            else:
                rng, token_rng = jax.random.split(rng)
                token_id = jax.random.categorical(token_rng, token_logits)

            halts_this_step = still_running & (token_id == act_token_id)

            num_steps_taken = num_steps_taken + still_running.astype(jnp.float32)
            final_state = jnp.where(halts_this_step[..., None], state, final_state)
            emitted_tokens = emitted_tokens.at[..., step].set(token_id)

            still_running = still_running & (~halts_this_step)

            if step_counts[step] != self.max_steps:
                # `act_token_id` is out of range for `token_embed` (which only
                # knows the `vocab_size` thought tokens); substitute a dummy
                # id where it was chosen - those entries are never read back,
                # since scratchpad positions for already-halted examples are
                # only ever fed to blocks whose output gets discarded above.
                thought_id = jnp.where(token_id == act_token_id, 0, token_id)
                token_emb = token_embed(thought_id)
                next_entry = (
                    fuse_latent_feedback(state, token_emb)
                    if self.use_latent_feedback
                    else token_emb
                )
                scratchpad = scratchpad.at[..., step + 1, :].set(next_entry)

        return final_state, num_steps_taken, emitted_tokens
