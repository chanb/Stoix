"""A transformer torso whose adaptive computation is an *explicit* chain of thought.

At each pondering step `TransformerExplicitCoTTorso` samples a token from a
learned categorical over `vocab_size` thought tokens plus `num_actions` action
tokens. A thought token's embedding is appended to the scratchpad and attended
over at the next step; choosing an action token both halts and selects the
environment action in the same draw, so there is no separate halting or action
head. Attention is causally masked.

  - Rollout mode (`target_tokens=None`): returns `(action, compute_time,
    thought_tokens)`, with `thought_tokens` of shape `(*batch, max_steps)`.
  - Replay mode (`target_tokens=<array>`): replays that token trajectory and
    returns `(log_prob, per_step_log_prob, per_step_entropy)`, the per-step
    terms zeroed past the halting step. Without latent feedback every
    scratchpad entry is a known token embedding, so replay scores all steps in
    one parallel causal-masked pass.
  - Deterministic mode (`deterministic=True`): greedy class at each step.

`use_latent_feedback=True` implements latent feedback decoding from the
Full-Bandwidth Transformer (arXiv:2608.08888): the previous step's top-layer
hidden state is fused with the new token's embedding via a gated linear unit,
`e_t (X) h_{t-1} = W^U h_{t-1} * sigmoid(W^G e_t)`, instead of feeding back
the bare embedding. Replay then needs the step-by-step (KV-cached) path.
"""

from typing import Optional, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.linen.initializers import Initializer, normal, orthogonal

from stoix.networks.torso_compute_transformer import TransformerBlock, _norm_cls, _resolve_qkv_dim
from stoix.networks.utils import parse_activation_fn

_NEG_INF = jnp.finfo(jnp.float32).min


def _categorical_entropy(log_probs: chex.Array) -> chex.Array:
    """`-sum(p * log(p))` over the last axis, safe against `legal_mask`'s
    illegal (zero-probability) classes."""
    probs = jnp.exp(log_probs)
    return jnp.sum(jnp.where(probs > 0, -probs * log_probs, 0.0), axis=-1)


class _ExplicitCoTBackbone(nn.Module):
    """The transformer blocks, token (un)embedding, positional embedding and
    (when `use_latent_feedback=True`) latent-feedback fusion. Instantiated
    once and shared, via its methods, by every code path in
    `TransformerExplicitCoTTorso.__call__`, so the parallel replay pass and
    the step-by-step path use the same weights. `token_embed` has
    `vocab_size + num_actions` rows.
    """

    hidden_dim: int
    vocab_size: int
    num_actions: int
    num_heads: int
    num_layers: int
    mlp_dim: int
    max_steps: int
    activation: str
    kernel_init: Initializer
    use_latent_feedback: bool
    use_sandwich_norm: bool = False
    use_rmsnorm: bool = False
    qkv_dim: Optional[int] = None

    def setup(self) -> None:
        self.pos_embedding = self.param(
            "pos_embedding", normal(stddev=0.02), (self.max_steps + 1, self.hidden_dim)
        )
        # `token_embed` doubles as the unembedding ("token_head") via
        # `.attend()` (query @ embedding.T) - standard input/output
        # weight-tying. Its table has `num_actions` rows beyond `vocab_size`,
        # one per environment action: those rows are only ever read through
        # `.attend()` (to produce that action's "act with this action" logit),
        # never through `token_embed(...)` (the embedding lookup fed back
        # into the scratchpad), since choosing one of them halts before
        # another embedding is needed - see the `token_id -> 0` substitutions
        # at the `token_embed(...)` call sites below.
        self.token_embed = nn.Embed(
            num_embeddings=self.vocab_size + self.num_actions, features=self.hidden_dim
        )
        self.blocks = [
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
        """Causal-masked pass over the whole scratchpad."""
        seq_len = scratchpad.shape[-2]
        batch_shape = scratchpad.shape[:-2]
        tokens_in = scratchpad + self.pos_embedding[:seq_len]
        causal_mask = nn.make_causal_mask(jnp.ones(batch_shape + (seq_len,)))
        for block in self.blocks:
            tokens_in = block(tokens_in, mask=causal_mask)
        return tokens_in

    def fuse_latent_feedback(self, state: chex.Array, token_emb: chex.Array) -> chex.Array:
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
        """One KV-cached step at position `step_idx`."""
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
    """Explicit Chain-of-Thought transformer torso whose halting decision and
    environment action are a single choice among `vocab_size + num_actions`
    classes - see module docstring.

    `min_steps` forbids halting before that many steps and `max_steps` forces a
    halt at the last step; `use_sandwich_norm`/`use_rmsnorm`/`qkv_dim` configure
    `TransformerBlock` (see `stoix.networks.torso_compute_transformer`).
    """

    hidden_dim: int
    vocab_size: int
    num_actions: int
    num_heads: int = 4
    num_layers: int = 2
    mlp_dim: int = 512
    max_steps: int = 8
    min_steps: int = 1
    activation: str = "relu"
    kernel_init: Initializer = orthogonal(np.sqrt(2.0))
    use_input_layer_norm: bool = False
    use_latent_feedback: bool = False
    use_sandwich_norm: bool = False
    use_rmsnorm: bool = False
    qkv_dim: Optional[int] = None

    @nn.compact
    def __call__(
        self,
        observation: chex.Array,
        rng: Optional[chex.PRNGKey] = None,
        target_tokens: Optional[chex.Array] = None,
        deterministic: bool = False,
    ) -> Tuple[chex.Array, ...]:
        """
        Args:
            observation: the input embedding to think about; becomes the
                first scratchpad token.
            rng: PRNG key used to sample the per-step token (thought or
                "act with action k") choices. Required unless replaying or
                `deterministic=True`.
            target_tokens: for replay mode, the exact per-step token
                trajectory (shape `(*batch, max_steps)`) to replay - see
                module docstring.
            deterministic: if True (and not replaying), pick the highest
                probability class at each step instead of sampling.

        Returns:
            `(action, compute_time, thought_tokens)` when not replaying, or
            `(log_prob, per_step_log_prob, per_step_entropy)` when replaying
            a known trajectory - see module docstring.
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

        # The token vocabulary has `num_actions` extra classes beyond the
        # `vocab_size` thought tokens: choosing class `vocab_size + a` means
        # "halt now, taking environment action `a`" - a class id `>=
        # vocab_size` is therefore both the halting signal and the resolved
        # action (minus `vocab_size`).
        num_classes = self.vocab_size + self.num_actions

        initial_token = nn.Dense(self.hidden_dim, kernel_init=self.kernel_init)(observation)
        if self.use_input_layer_norm:
            initial_token = _norm_cls(self.use_rmsnorm)()(initial_token)

        backbone = _ExplicitCoTBackbone(
            self.hidden_dim,
            self.vocab_size,
            self.num_actions,
            self.num_heads,
            self.num_layers,
            self.mlp_dim,
            self.max_steps,
            self.activation,
            self.kernel_init,
            self.use_latent_feedback,
            self.use_sandwich_norm,
            self.use_rmsnorm,
            self.qkv_dim,
        )

        # Per-step legality mask (shape `(max_steps, num_classes)`): all
        # `num_actions` action columns are gated by the same
        # `can_halt`/`is_final_step` logic.
        step_counts = np.arange(1, self.max_steps + 1)
        is_final_step = step_counts == self.max_steps
        can_halt = step_counts >= self.min_steps
        legal_mask = np.ones((self.max_steps, num_classes), dtype=bool)
        legal_mask[is_final_step, :] = False
        legal_mask[is_final_step, self.vocab_size :] = True
        legal_mask[~can_halt, self.vocab_size :] = False
        legal_mask = jnp.asarray(legal_mask)

        if replaying and not self.use_latent_feedback:
            # Parallel replay pass: every scratchpad entry is a known token
            # embedding, so all steps are scored in one causal-masked pass.
            # A token `>= self.vocab_size` is an action (halting) token.
            if self.max_steps > 1:
                thought_id = jnp.where(target_tokens >= self.vocab_size, 0, target_tokens)
                token_embeds = backbone.token_embed(thought_id)
                scratchpad = jnp.concatenate(
                    [initial_token[..., None, :], token_embeds[..., :-1, :]], axis=-2
                )
            else:
                scratchpad = initial_token[..., None, :]
            states = backbone.run(scratchpad)  # (*batch, max_steps, hidden_dim)

            token_logits = jnp.where(legal_mask, backbone.token_head(states), _NEG_INF)
            log_token_probs = jax.nn.log_softmax(token_logits, axis=-1)
            token_log_prob = jnp.take_along_axis(
                log_token_probs, target_tokens[..., None], axis=-1
            ).squeeze(axis=-1)

            halted = target_tokens >= self.vocab_size
            earlier_halts = jnp.cumsum(halted.astype(jnp.int32), axis=-1) - halted.astype(
                jnp.int32
            )
            still_running = earlier_halts == 0
            per_step_log_prob = jnp.where(still_running, token_log_prob, 0.0)
            log_prob = jnp.sum(per_step_log_prob, axis=-1)
            per_step_entropy = jnp.where(
                still_running, _categorical_entropy(log_token_probs), 0.0
            )

            return log_prob, per_step_log_prob, per_step_entropy

        # KV-cached step-by-step build, shared by rollout mode and (when
        # `use_latent_feedback=True`) replay mode.
        qkv_dim = _resolve_qkv_dim(self.hidden_dim, self.qkv_dim)
        assert qkv_dim % self.num_heads == 0, (
            f"qkv_dim ({qkv_dim}) must be divisible by num_heads ({self.num_heads})."
        )
        head_dim = qkv_dim // self.num_heads
        cache_shape = batch_shape + (self.max_steps + 1, self.num_heads, head_dim)
        cached_keys = [jnp.zeros(cache_shape) for _ in range(self.num_layers)]
        cached_values = [jnp.zeros(cache_shape) for _ in range(self.num_layers)]

        still_running = jnp.ones(batch_shape, dtype=bool)
        num_steps_taken = jnp.zeros(batch_shape)
        emitted_tokens = jnp.zeros(batch_shape + (self.max_steps,), dtype=jnp.int32)
        log_prob = jnp.zeros(batch_shape)
        per_step_log_prob = jnp.zeros(batch_shape + (self.max_steps,))
        per_step_entropy = jnp.zeros(batch_shape + (self.max_steps,))
        step_rng = rng if rng is not None else jax.random.PRNGKey(0)

        def step_fn(
            backbone: _ExplicitCoTBackbone,
            carry: Tuple[chex.Array, ...],
            step_idx: chex.Array,
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
                per_step_entropy,
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
                step_log_prob = jnp.where(still_running, token_log_prob, 0.0)
                log_prob = log_prob + step_log_prob
                per_step_log_prob = per_step_log_prob.at[..., step_idx].set(step_log_prob)
                per_step_entropy = per_step_entropy.at[..., step_idx].set(
                    jnp.where(still_running, _categorical_entropy(log_token_probs), 0.0)
                )
            elif deterministic:
                token_id = jnp.argmax(token_logits, axis=-1)
            else:
                rng, token_rng = jax.random.split(rng)
                token_id = jax.random.categorical(token_rng, token_logits)

            halts_this_step = still_running & (token_id >= self.vocab_size)
            if not replaying:
                num_steps_taken = num_steps_taken + still_running.astype(jnp.float32)
                emitted_tokens = emitted_tokens.at[..., step_idx].set(token_id)
            still_running = still_running & (~halts_this_step)

            # An action class's embedding row is never read back through this
            # lookup (see `_ExplicitCoTBackbone.setup`); substitute a
            # dummy id where one was chosen - those entries are never read
            # back, since once an example has halted, its `current_token`
            # keeps being computed (every example runs the same fixed number
            # of scan iterations) but is discarded.
            thought_id = jnp.where(token_id >= self.vocab_size, 0, token_id)
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
                per_step_entropy,
                rng,
            )
            return new_carry, None

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
            per_step_entropy,
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
            per_step_entropy,
            _,
        ), _ = scan_step(backbone, initial_carry, jnp.arange(self.max_steps))

        if replaying:
            return log_prob, per_step_log_prob, per_step_entropy

        # The environment action is whichever class halted (the forced halt
        # at max_steps guarantees one always did), minus `vocab_size` - no
        # downstream head needed, the choice already *is* the action.
        halted = emitted_tokens >= self.vocab_size
        halt_step = jnp.argmax(halted, axis=-1)
        action = (
            jnp.take_along_axis(emitted_tokens, halt_step[..., None], axis=-1).squeeze(axis=-1)
            - self.vocab_size
        )
        return action, num_steps_taken, emitted_tokens
