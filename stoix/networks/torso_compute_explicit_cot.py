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
    already-known token. That means replay doesn't need the unrolled
    step-by-step loop at all: the whole scratchpad can be built up front and
    scored in one causal-masked pass, reading every step's logits off in
    parallel instead of recomputing the growing prefix from scratch
    `max_steps` times.
  - Deterministic mode (`deterministic=True`): at every step, picks the
    highest-probability class - a thought token or "act now" - instead of
    sampling, for greedy evaluation.

The environment action head still reads off the final continuous hidden
state (not a token embedding) - the discrete tokens are a communicative side
channel the policy can use to "show its work", not a bottleneck on how much
information reaches the action itself.
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
            seq_len = step + 1
            tokens_in = scratchpad[..., :seq_len, :] + pos_embedding[:seq_len]
            causal_mask = nn.make_causal_mask(jnp.ones(batch_shape + (seq_len,)))
            for block in blocks:
                tokens_in = block(tokens_in, mask=causal_mask)
            state = tokens_in[..., -1, :]

            token_logits = jnp.where(legal_mask[step], token_head(state), _NEG_INF)

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
                scratchpad = scratchpad.at[..., step + 1, :].set(token_embed(thought_id))

        return final_state, num_steps_taken, emitted_tokens
