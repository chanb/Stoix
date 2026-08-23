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

Both the halting decision and the thought-token choice are discrete sampled
actions, so both are trained the same way: via the score-function
(REINFORCE) estimator, using the log-probability of the sampled trajectory.
This torso therefore has the same rollout/replay/deterministic modes as
`AdaptiveComputationTimeTorso`, except replaying also needs the actual
sequence of emitted token ids (not just how many steps were taken), since a
different token would have led the model to think something different at the
next step:

  - Rollout mode (`target_compute_time=None`, `target_tokens=None`): samples
    both the halting trajectory and the emitted thought tokens, and returns
    `(embedding, compute_time, thought_tokens)`. `thought_tokens` has shape
    `(*batch, max_steps - 1)` - the maximum number of thoughts that could ever
    be emitted before a forced final halt - padded arbitrarily past
    `compute_time - 1` (those entries are never read back in replay mode,
    since replaying masks by the same halting trajectory that produced them).
  - Replay mode (`target_compute_time=<array>`, `target_tokens=<array>`):
    deterministically replays exactly that halting-and-token trajectory (no
    rng) and returns `(embedding, log_prob)`: the log-probability, under the
    current parameters, of the halting decisions *and* the thought tokens
    together.
  - Deterministic mode (`deterministic=True`): halts as soon as the halting
    probability crosses 0.5, and picks the highest-probability token instead
    of sampling - for greedy evaluation.

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

_PROB_EPS = 1e-6


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

    @nn.compact
    def __call__(
        self,
        observation: chex.Array,
        rng: Optional[chex.PRNGKey] = None,
        target_compute_time: Optional[chex.Array] = None,
        target_tokens: Optional[chex.Array] = None,
        deterministic: bool = False,
    ) -> Union[Tuple[chex.Array, chex.Array], Tuple[chex.Array, chex.Array, chex.Array]]:
        """
        Args:
            observation: the input embedding to think about; becomes the
                first scratchpad token.
            rng: PRNG key used to sample halting decisions and thought
                tokens. Required unless replaying or `deterministic=True`.
            target_compute_time: paired with `target_tokens` for replay mode
                - see module docstring.
            target_tokens: paired with `target_compute_time` for replay mode.
            deterministic: if True (and not replaying), halt as soon as the
                halting probability crosses 0.5 and pick the highest
                probability thought token, instead of sampling.

        Returns:
            `(embedding, compute_time, thought_tokens)` when not replaying,
            or `(embedding, log_prob)` when replaying a known trajectory.
        """
        batch_shape = observation.shape[:-1]
        replaying = target_compute_time is not None
        if replaying != (target_tokens is not None):
            raise ValueError(
                "target_compute_time and target_tokens must be provided together "
                "(replay mode) or not at all."
            )
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

        max_thoughts = self.max_steps - 1

        initial_token = parse_activation_fn(self.activation)(
            nn.Dense(self.hidden_dim, kernel_init=self.kernel_init)(observation)
        )

        pos_embedding = self.param(
            "pos_embedding", normal(stddev=0.02), (self.max_steps + 1, self.hidden_dim)
        )
        token_embed = nn.Embed(num_embeddings=self.vocab_size, features=self.hidden_dim)
        token_head = nn.Dense(self.vocab_size, kernel_init=self.kernel_init)
        halting_head = nn.Dense(1, kernel_init=self.kernel_init)

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
        emitted_tokens = jnp.zeros(batch_shape + (max_thoughts,), dtype=jnp.int32)

        for step in range(self.max_steps):
            seq_len = step + 1
            tokens_in = scratchpad[..., :seq_len, :] + pos_embedding[:seq_len]
            for block in blocks:
                tokens_in = block(tokens_in)
            state = tokens_in[..., -1, :]

            halting_prob = nn.sigmoid(halting_head(state))
            halting_prob = jnp.clip(halting_prob.squeeze(axis=-1), _PROB_EPS, 1.0 - _PROB_EPS)

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
                rng, halt_rng = jax.random.split(rng)
                sampled_halt = jax.random.bernoulli(halt_rng, halting_prob)
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
            log_prob = log_prob + jnp.where(still_running, step_log_prob, 0.0)
            num_steps_taken = num_steps_taken + still_running.astype(jnp.float32)
            final_state = jnp.where(halts_this_step[..., None], state, final_state)

            # `still_running` now means "continuing past this step" - exactly
            # the examples that will go on to emit a thought token below.
            still_running = still_running & (~halts_this_step)

            if not is_final_step:
                token_logits = token_head(state)
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
                log_prob = log_prob + jnp.where(still_running, token_log_prob, 0.0)

                emitted_tokens = emitted_tokens.at[..., step].set(token_id)
                scratchpad = scratchpad.at[..., step + 1, :].set(token_embed(token_id))

        if replaying:
            return final_state, log_prob
        return final_state, num_steps_taken, emitted_tokens
