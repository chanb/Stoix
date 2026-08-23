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
  1. Run `num_layers` shared transformer blocks over the scratchpad tokens
     produced so far (`t + 1` of them, including the initial token derived
     from the observation), with learned positional embeddings.
  2. Take the representation at the last (most recent) position as this
     step's "thought" - this is what the halting unit and, if halting, the
     action head see.
  3. Decide whether to halt (sampled / replayed / deterministic, same as
     `AdaptiveComputationTimeTorso`). If not halting, append the thought to
     the scratchpad and continue.

Because the whole scratchpad is visible via self-attention, a step's thought
can depend on every earlier thought, not just the immediately preceding one -
the natural inductive bias for a chain of thought, as opposed to the
Markovian single-hidden-state recurrence of the MLP ACT torso.

Since `max_steps` must be static for JAX, the scratchpad is a pre-allocated
buffer and the CoT loop is unrolled; each step re-attends over a growing
prefix (no KV-caching), so this is O(max_steps^2) rather than O(max_steps) -
fine for the small `max_steps` this is meant to be used with.
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


class TransformerBlock(nn.Module):
    """A single pre-norm transformer block: self-attention + MLP.

    Operates on `(*batch, seq_len, hidden_dim)`. Since callers only ever read
    off the representation at the last sequence position, no causal mask is
    applied - full self-attention over whatever prefix is passed in is
    equivalent, and simpler.
    """

    hidden_dim: int
    num_heads: int
    mlp_dim: int
    activation: str = "relu"
    kernel_init: Initializer = orthogonal(np.sqrt(2.0))

    @nn.compact
    def __call__(self, tokens: chex.Array) -> chex.Array:
        y = nn.LayerNorm()(tokens)
        y = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.hidden_dim,
            out_features=self.hidden_dim,
            kernel_init=self.kernel_init,
        )(y, y)
        tokens = tokens + y

        y = nn.LayerNorm()(tokens)
        y = nn.Dense(self.mlp_dim, kernel_init=self.kernel_init)(y)
        y = parse_activation_fn(self.activation)(y)
        y = nn.Dense(self.hidden_dim, kernel_init=self.kernel_init)(y)
        tokens = tokens + y
        return tokens


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
    """

    hidden_dim: int
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
        deterministic: bool = False,
    ) -> Tuple[chex.Array, chex.Array]:
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
            `(embedding, compute_time)` when `target_compute_time` is None,
            or `(embedding, halting_log_prob)` when replaying a known
            trajectory.
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

        # The first scratchpad token is the observation projected into the
        # model width.
        initial_token = parse_activation_fn(self.activation)(
            nn.Dense(self.hidden_dim, kernel_init=self.kernel_init)(observation)
        )

        pos_embedding = self.param(
            "pos_embedding",
            normal(stddev=0.02),
            (self.max_steps + 1, self.hidden_dim),
        )

        # Pre-allocate the CoT scratchpad; unwritten positions are never read
        # since we always slice to the valid prefix.
        scratchpad = jnp.zeros(batch_shape + (self.max_steps + 1, self.hidden_dim))
        scratchpad = scratchpad.at[..., 0, :].set(initial_token)

        blocks = [
            TransformerBlock(
                self.hidden_dim, self.num_heads, self.mlp_dim, self.activation, self.kernel_init
            )
            for _ in range(self.num_layers)
        ]
        halting_head = nn.Dense(1, kernel_init=self.kernel_init)

        still_running = jnp.ones(batch_shape, dtype=bool)
        num_steps_taken = jnp.zeros(batch_shape)
        halting_log_prob = jnp.zeros(batch_shape)
        final_state = initial_token

        for step in range(self.max_steps):
            seq_len = step + 1
            tokens = scratchpad[..., :seq_len, :] + pos_embedding[:seq_len]
            for block in blocks:
                tokens = block(tokens)
            state = tokens[..., -1, :]

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
            halting_log_prob = halting_log_prob + jnp.where(still_running, step_log_prob, 0.0)
            num_steps_taken = num_steps_taken + still_running.astype(jnp.float32)
            final_state = jnp.where(halts_this_step[..., None], state, final_state)

            still_running = still_running & (~halts_this_step)

            if not is_final_step:
                scratchpad = scratchpad.at[..., step + 1, :].set(state)

        if replaying:
            return final_state, halting_log_prob
        return final_state, num_steps_taken
