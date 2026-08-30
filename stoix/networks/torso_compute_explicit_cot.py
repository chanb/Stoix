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
    (thought choices and the halting choice together).
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
        token_embed = nn.Embed(num_embeddings=self.vocab_size, features=self.hidden_dim)
        token_head = nn.Dense(num_classes, kernel_init=self.kernel_init)

        scratchpad = jnp.zeros(batch_shape + (self.max_steps + 1, self.hidden_dim))
        scratchpad = scratchpad.at[..., 0, :].set(initial_token)

        blocks = [
            TransformerBlock(
                self.hidden_dim, self.num_heads, self.mlp_dim, self.activation, self.kernel_init
            )
            for _ in range(self.num_layers)
        ]

        still_running = jnp.ones(batch_shape, dtype=bool)
        num_steps_taken = jnp.zeros(batch_shape)
        log_prob = jnp.zeros(batch_shape)
        final_state = initial_token
        emitted_tokens = jnp.zeros(batch_shape + (self.max_steps,), dtype=jnp.int32)

        for step in range(self.max_steps):
            seq_len = step + 1
            tokens_in = scratchpad[..., :seq_len, :] + pos_embedding[:seq_len]
            for block in blocks:
                tokens_in = block(tokens_in)
            state = tokens_in[..., -1, :]

            step_count = step + 1
            is_final_step = step_count == self.max_steps
            can_halt = step_count >= self.min_steps

            token_logits = token_head(state)
            # `step`/`is_final_step`/`can_halt` are Python-level (the loop is
            # unrolled), so this masking is static per iteration - no need to
            # broadcast over the batch.
            if is_final_step:
                # A halt is forced here regardless of the policy: mask out
                # every thought class so "act now" is the only one left. Its
                # log-probability under that mask is an inputs-independent
                # constant (zero), so no gradient flows through a step that
                # was never really a choice.
                mask = jnp.full((num_classes,), _NEG_INF).at[act_token_id].set(0.0)
                token_logits = token_logits + mask
            elif not can_halt:
                # Before min_steps, "act now" isn't a legal choice yet.
                token_logits = token_logits.at[..., act_token_id].set(_NEG_INF)

            log_token_probs = jax.nn.log_softmax(token_logits)

            if replaying:
                token_id = target_tokens[..., step]
            elif deterministic:
                token_id = jnp.argmax(token_logits, axis=-1)
            else:
                rng, token_rng = jax.random.split(rng)
                token_id = jax.random.categorical(token_rng, token_logits)

            token_log_prob = jnp.take_along_axis(
                log_token_probs, token_id[..., None], axis=-1
            ).squeeze(axis=-1)

            halts_this_step = still_running & (token_id == act_token_id)

            log_prob = log_prob + jnp.where(still_running, token_log_prob, 0.0)
            num_steps_taken = num_steps_taken + still_running.astype(jnp.float32)
            final_state = jnp.where(halts_this_step[..., None], state, final_state)
            emitted_tokens = emitted_tokens.at[..., step].set(token_id)

            still_running = still_running & (~halts_this_step)

            if not is_final_step:
                # `act_token_id` is out of range for `token_embed` (which only
                # knows the `vocab_size` thought tokens); substitute a dummy
                # id where it was chosen - those entries are never read back,
                # since scratchpad positions for already-halted examples are
                # only ever fed to blocks whose output gets discarded above.
                thought_id = jnp.where(token_id == act_token_id, 0, token_id)
                scratchpad = scratchpad.at[..., step + 1, :].set(token_embed(thought_id))

        if replaying:
            return final_state, log_prob
        return final_state, num_steps_taken, emitted_tokens
