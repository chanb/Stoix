"""A transformer torso whose adaptive computation follows the State-Prediction
Separation (SPS) design, with Coconut-style latent (rather than token)
feedback.

"The State-Prediction Separation Hypothesis" (Monea et al., arXiv:2607.01218,
code at https://github.com/lil-lab/sps) observes that a standard Transformer's
hidden state at position `i` is forced to do two jobs at once: predict the
next token, and prepare the key/value entries later positions will read from
(the persistent "state"). Its fix is architectural: interleave every real
input token `x_i` with a dedicated, learned `<predict>` token `\\rho_i`
immediately after it. `x_i` is the *state* stream - it attends causally to
every earlier `x_k`, and its key/value entries persist in the cache for
every later position to read. `\\rho_i` is the *prediction* stream - besides
attending to those same persistent `x_k`, it can only see a small sliding
window of `w` recent `\\rho_k` entries, and its own key/value entries are
never read beyond that window. Both use the *same* transformer weights;
they differ only in which token feeds them and which cache entries they may
attend to (paper Eq. 5):

    A_SPS(i, x_i) = {x_k : k <= i} \\cup {\\rho_k : i - w <= k <  i}
    A_SPS(i, \\rho_i) = {x_k : k <= i} \\cup {\\rho_k : i - w <= k <= i}

In the paper's language-modelling setting, `\\rho_i`'s hidden state is read
by the LM head to predict `x_{i+1}` - a discrete next token, whose embedding
then becomes the next real input. This module adapts SPS into an
adaptively-halting policy torso (in the style of
`stoix.networks.torso_compute_transformer.TransformerChainOfThoughtTorso`):
`x_0` is the encoded environment observation, and each step's `\\rho_t`
plays the role of a "thought" - instead of an LM head, it feeds a halting
unit (sampled/replayed/deterministic, exactly like this codebase's other
adaptive-computation-time torsos). If it doesn't halt, `\\rho_t`'s hidden
state becomes `x_{t+1}`, the next step's persistent-stream input.

Crucially, that last step is *not* what the SPS paper does. There, `\\rho_i`
only ever proposes a distribution over discrete tokens; what actually
becomes `x_{i+1}` is the embedding of one sampled/argmax token id
`\\hat{x}_i`, so the continuous hidden state that produced it is thrown
away, exactly like `stoix.networks.torso_compute_explicit_cot.
TransformerExplicitCoTTorso` without `use_latent_feedback`. This module
instead takes `\\rho_t`'s raw hidden state itself as `x_{t+1}` - the same
substitution Coconut ("Training Large Language Models to Reason in a
Continuous Latent Space", Hao et al., arXiv:2412.06769) makes over standard
per-token autoregressive decoding: feed the last hidden state back in
directly instead of round-tripping it through an unembedding/embedding
pair. Concretely, this makes `SPSChainOfThoughtTorso`'s state stream
identical in spirit to `TransformerChainOfThoughtTorso`'s single growing
scratchpad (latent, not discretised, feedback) - the SPS-specific
contribution this module adds on top is the *prediction* stream: instead of
reading the halting decision off the same position that carries the
persistent state forward (as `TransformerChainOfThoughtTorso` does), it is
read off a dedicated `\\rho_t` position that can see the persistent state
but whose own representation never persists beyond the sliding window -
i.e. the halting computation gets a place to happen that is architecturally
prevented from also being asked to carry state for later steps, mirroring
the paper's Eq. 2 argument for why conflating the two hurts.

Two simplifications relative to the reference implementation, neither of
which changes the attention pattern above:
  - The reference uses RoPE; this module uses additive learned positional
    embeddings, matching every other transformer torso in this codebase
    (`TransformerChainOfThoughtTorso`, `TransformerExplicitCoTTorso`). As
    in the paper (Sec. 3, "The two tokens x_i and rho_i at index i share the
    same position encoding"), `x_t` and `\\rho_t` are given the *same*
    positional embedding, indexed by the pondering step `t`.
  - The reference physically evicts `\\rho` cache entries once they leave
    the window, for memory efficiency at LM sequence lengths. Since
    `max_steps` here is a small pondering budget rather than a document
    length, this module instead always allocates the full `(max_steps,
    hidden_dim)` `\\rho` cache and enforces the window purely through the
    attention mask - semantically identical, just without the memory
    optimisation, matching how `min_steps`/`max_steps` masking already
    works in the rest of this codebase's ACT torsos.

Since `max_steps` must be static for JAX, the pondering loop runs as an
`nn.scan` (a weight-tied `jax.lax.scan`), exactly like
`TransformerChainOfThoughtTorso` - see that module's docstring for why, and
`stoix.networks.torso_compute_transformer.TransformerBlock.step_joint` for
the joint-softmax attention primitive this needs that `TransformerBlock.step`
doesn't provide (one cache, one mask).
"""

from typing import Optional, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.linen.initializers import Initializer, normal, orthogonal

from stoix.networks.torso_compute_transformer import TransformerBlock

_PROB_EPS = 1e-6


class _SPSStep(nn.Module):
    """One pondering step, meant to be lifted into a weight-tied loop via
    `nn.scan` - see `TransformerChainOfThoughtTorso`'s `_CoTStep` for the
    same pattern. Each step processes *two* positions through the shared
    transformer blocks, in order: the persistent-stream token `x_t` (written
    into `x_cached_keys`/`x_cached_values`), then the prediction-stream
    token `\\rho_t` (written into `rho_cached_keys`/`rho_cached_values`) -
    see module docstring for the attention pattern each uses and why the
    order matters (`\\rho_t` must see the just-written `x_t`).
    """

    hidden_dim: int
    num_heads: int
    num_layers: int
    mlp_dim: int
    max_steps: int
    min_steps: int
    window_size: int
    activation: str
    kernel_init: Initializer
    convergence_threshold: float
    replaying: bool
    deterministic: bool
    stop_gradient_halting_input: bool = False

    @nn.compact
    def __call__(
        self, carry: Tuple[chex.Array, ...], step_idx: chex.Array
    ) -> Tuple[Tuple[chex.Array, ...], None]:
        (
            current_token,
            x_cached_keys,
            x_cached_values,
            rho_cached_keys,
            rho_cached_values,
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
        batch_shape = current_token.shape[:-1]

        pos_embedding = self.param(
            "pos_embedding",
            normal(stddev=0.02),
            (self.max_steps, self.hidden_dim),
        )
        predict_embedding = self.param(
            "predict_embedding", normal(stddev=0.02), (self.hidden_dim,)
        )
        blocks = [
            TransformerBlock(
                self.hidden_dim, self.num_heads, self.mlp_dim, self.activation, self.kernel_init
            )
            for _ in range(self.num_layers)
        ]
        halting_head = nn.Dense(1, kernel_init=self.kernel_init)

        # Positions already written (< step_idx) or about to be written
        # (== step_idx) this step; positions > step_idx are still zero and
        # must never be attended to. `x_visible`/`rho_window` are shared by
        # both sub-steps below (only whether rho's own not-yet-written
        # slot - itself - is included differs, per Eq. 5).
        step_positions = jnp.arange(self.max_steps)
        x_visible = step_positions <= step_idx
        rho_in_window = step_positions >= step_idx - self.window_size

        x_cached_keys = list(x_cached_keys)
        x_cached_values = list(x_cached_values)
        rho_cached_keys = list(rho_cached_keys)
        rho_cached_values = list(rho_cached_values)

        # --- Persistent stream: x_t. Sees every earlier x_k (causal) and
        # rho_k strictly within the window (rho_t itself doesn't exist yet).
        x_layer_input = current_token + pos_embedding[step_idx]
        x_rho_mask = rho_in_window & (step_positions < step_idx)
        for layer_idx, block in enumerate(blocks):
            x_layer_input, k, v = block.step_joint(
                x_layer_input,
                [x_cached_keys[layer_idx], rho_cached_keys[layer_idx]],
                [x_cached_values[layer_idx], rho_cached_values[layer_idx]],
                [x_visible, x_rho_mask],
            )
            x_cached_keys[layer_idx] = x_cached_keys[layer_idx].at[..., step_idx, :, :].set(k)
            x_cached_values[layer_idx] = (
                x_cached_values[layer_idx].at[..., step_idx, :, :].set(v)
            )
        x_state = x_layer_input

        # --- Prediction stream: rho_t. Sees every x_k up to and including
        # the x_t just written above, and rho_k within the window including
        # itself (self-attention, same as any causal position).
        rho_layer_input = jnp.broadcast_to(
            predict_embedding + pos_embedding[step_idx], batch_shape + (self.hidden_dim,)
        )
        rho_rho_mask = rho_in_window & (step_positions <= step_idx)
        for layer_idx, block in enumerate(blocks):
            rho_layer_input, k, v = block.step_joint(
                rho_layer_input,
                [x_cached_keys[layer_idx], rho_cached_keys[layer_idx]],
                [x_cached_values[layer_idx], rho_cached_values[layer_idx]],
                [x_visible, rho_rho_mask],
            )
            rho_cached_keys[layer_idx] = (
                rho_cached_keys[layer_idx].at[..., step_idx, :, :].set(k)
            )
            rho_cached_values[layer_idx] = (
                rho_cached_values[layer_idx].at[..., step_idx, :, :].set(v)
            )
        state = rho_layer_input
        if self.replaying:
            # Only ever populated when replaying - see
            # `SPSChainOfThoughtTorso`'s docstring for `states_history`.
            states_history = states_history.at[..., step_idx, :].set(state)

        halting_input = state
        if self.stop_gradient_halting_input:
            halting_input = jax.lax.stop_gradient(halting_input)
        halting_prob = nn.sigmoid(halting_head(halting_input))
        halting_prob = jnp.clip(halting_prob.squeeze(axis=-1), _PROB_EPS, 1.0 - _PROB_EPS)

        step_count = step_idx + 1
        is_final_step = step_count == self.max_steps
        can_halt = step_count >= self.min_steps

        if not self.replaying:
            # Only meaningful from the second step on (step 0 has no
            # previous "thought" to compare against - gated via
            # `step_idx > 0` rather than skipped, since `step_idx` is
            # traced under scan), and only while still running - once an
            # example has halted, later thoughts keep being computed (every
            # example runs the same fixed number of scan iterations) but
            # are discarded, so they say nothing about the example's real
            # trajectory.
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
        # Coconut-style latent feedback (see module docstring): the next
        # step's persistent-stream input is rho_t's raw hidden state
        # directly, not the embedding of a sampled/predicted token.
        current_token = state

        new_carry = (
            current_token,
            x_cached_keys,
            x_cached_values,
            rho_cached_keys,
            rho_cached_values,
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


class SPSChainOfThoughtTorso(nn.Module):
    """State-Prediction-Separation transformer torso with adaptively-sampled
    halting and Coconut-style latent feedback. See module docstring for the
    full mechanism and how it relates to the SPS paper (arXiv:2607.01218)
    and Coconut (arXiv:2412.06769).

    Has the same three-mode interface as `AdaptiveComputationTimeTorso`/
    `TransformerChainOfThoughtTorso`:

      - Rollout mode (`target_compute_time=None`, `rng` given): samples a
        halting trajectory and returns `(embedding, compute_time,
        first_convergence_step, num_close_steps)`.
      - Replay mode (`target_compute_time=<array>`): deterministically
        replays exactly that many pondering steps (no rng) and returns
        `(embedding, halting_log_prob, states_history,
        per_step_halting_log_prob, per_step_halting_entropy)` - use
        `halting_log_prob` in a REINFORCE-style loss; `states_history`
        (shape `(*batch, max_steps, hidden_dim)`) is the prediction-stream
        "thought" `\\rho_t` at every step, including steps past the one the
        trajectory actually halted at (mask by `step_idx < compute_time`
        before using it, same as the convergence diagnostics below already
        implicitly do). `per_step_halting_log_prob`/`per_step_halting_entropy`
        (same shape) are `halting_log_prob` and the halting Bernoulli's
        entropy left unsummed, one entry per step, zeroed at forced steps or
        past the actual halt - see `TransformerChainOfThoughtTorso`'s
        docstring and `ff_ppo.py` for how these get used.
      - Deterministic mode (`deterministic=True`): halts as soon as the
        halting probability crosses 0.5, for greedy evaluation.

    `min_steps` forbids halting (voluntarily, in replay, or greedily) before
    that many pondering steps have been taken - `is_final_step` still forces
    a halt at `max_steps` regardless. Setting `min_steps == max_steps`
    therefore removes adaptivity entirely: every example always takes
    exactly `max_steps` - useful as a fixed-budget baseline against the
    adaptive policy.

    `window_size` is `w` in the module docstring's attention pattern: how
    many recent prediction-stream (`\\rho`) steps a query may attend to,
    besides the always-fully-visible persistent (`x`) stream. `window_size
    >= max_steps - 1` makes the window non-binding (every `\\rho` step stays
    visible for the whole trajectory); the SPS paper finds a small but
    non-zero window (their default `w=64`) works best, and that `window_size
    = 0` (no `\\rho`-to-`\\rho` attention at all) noticeably hurts - see
    `Reverse SPS`/Figure 4 in the paper for the ablation this mirrors.

    `use_input_layer_norm` normalizes the raw `observation` before it becomes
    `x_0` (i.e. before its projection into `hidden_dim`).

    Also tracks, per example, how quickly the "thought" (`\\rho_t`'s hidden
    state) settles: the L2 distance between consecutive steps' thoughts
    (step `t` vs `t - 1`, starting at `t = 2` since there's no step 0 to
    compare step 1 against) is compared against `convergence_threshold`.
    This gives two diagnostics, only while the example hasn't halted yet:

      - `first_convergence_step`: the (1-indexed) step count `t` at which
        that distance first drops below `convergence_threshold`, or `-1` if
        it never does within the steps actually taken.
      - `num_close_steps`: how many steps (not necessarily consecutive) had
        a distance below `convergence_threshold`.

    `stop_gradient_halting_input` (default `False`) detaches `\\rho_t`'s
    state before it's read by the halting head's `Dense(1)`, mirroring
    `TransformerChainOfThoughtTorso`'s knob of the same name: the halting
    head's own weights still get trained via REINFORCE, but that gradient
    can no longer backprop into the shared transformer blocks that also
    produce the action head's representation (since `\\rho_t`'s state
    becomes `x_{t+1}` via the Coconut-style feedback, that representation
    is exactly what later steps - and, at the final step, the action head -
    read). A no-op when `min_steps == max_steps` (every step is already
    forced, so no halting-loss gradient reaches the torso regardless).
    """

    hidden_dim: int
    num_heads: int = 4
    num_layers: int = 2
    mlp_dim: int = 512
    max_steps: int = 8
    min_steps: int = 1
    window_size: int = 64
    activation: str = "relu"
    kernel_init: Initializer = orthogonal(np.sqrt(2.0))
    use_input_layer_norm: bool = False
    convergence_threshold: float = 0.1
    stop_gradient_halting_input: bool = False

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
            observation: the input embedding to think about; becomes `x_0`.
            rng: PRNG key used to sample halting decisions. Required unless
                `target_compute_time` is given or `deterministic=True`.
            target_compute_time: if given (e.g. `Transition.compute_time`
                from a prior rollout), halting is *not* sampled - instead
                pondering is replayed deterministically to halt at exactly
                this many steps per example, and the second output is
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
            replaying a known trajectory (see class docstring).
        """
        batch_shape = observation.shape[:-1]
        replaying = target_compute_time is not None
        if not replaying and not deterministic and rng is None:
            raise ValueError(
                "rng must be provided to SPSChainOfThoughtTorso when sampling "
                "(i.e. target_compute_time is None and deterministic=False)."
            )
        if not (1 <= self.min_steps <= self.max_steps):
            raise ValueError(
                f"min_steps must be between 1 and max_steps ({self.max_steps}), "
                f"got min_steps={self.min_steps}."
            )
        assert self.hidden_dim % self.num_heads == 0, (
            f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads})."
        )

        # x_0: the observation projected into the model width.
        initial_token = nn.Dense(self.hidden_dim, kernel_init=self.kernel_init)(observation)
        if self.use_input_layer_norm:
            initial_token = nn.LayerNorm()(initial_token)

        # Pre-allocate every layer's persistent (x) and ephemeral (rho)
        # caches at `max_steps` (one slot per pondering step); not-yet-
        # written slots are masked out of attention by `_SPSStep`, never
        # read regardless of `window_size` (see module docstring for why
        # the rho cache isn't physically evicted the way the reference
        # implementation's is).
        head_dim = self.hidden_dim // self.num_heads
        cache_shape = batch_shape + (self.max_steps, self.num_heads, head_dim)
        x_cached_keys = [jnp.zeros(cache_shape) for _ in range(self.num_layers)]
        x_cached_values = [jnp.zeros(cache_shape) for _ in range(self.num_layers)]
        rho_cached_keys = [jnp.zeros(cache_shape) for _ in range(self.num_layers)]
        rho_cached_values = [jnp.zeros(cache_shape) for _ in range(self.num_layers)]

        still_running = jnp.ones(batch_shape, dtype=bool)
        num_steps_taken = jnp.zeros(batch_shape)
        halting_log_prob = jnp.zeros(batch_shape)
        final_state = initial_token
        first_convergence_step = jnp.full(batch_shape, -1.0)
        num_close_steps = jnp.zeros(batch_shape)
        # Only ever written when replaying (see `_SPSStep`); carried
        # regardless of mode since `nn.scan` needs a fixed carry structure.
        states_history = jnp.zeros(batch_shape + (self.max_steps, self.hidden_dim))
        per_step_halting_log_prob = jnp.zeros(batch_shape + (self.max_steps,))
        per_step_halting_entropy = jnp.zeros(batch_shape + (self.max_steps,))
        # Unused (never read) unless sampling, but must still be a concrete
        # array: it's carried through every scan step regardless of mode.
        step_rng = rng if rng is not None else jax.random.PRNGKey(0)

        sps_step = nn.scan(
            _SPSStep,
            variable_broadcast="params",
            split_rngs={"params": False},
        )(
            self.hidden_dim,
            self.num_heads,
            self.num_layers,
            self.mlp_dim,
            self.max_steps,
            self.min_steps,
            self.window_size,
            self.activation,
            self.kernel_init,
            self.convergence_threshold,
            replaying,
            deterministic,
            self.stop_gradient_halting_input,
        )

        initial_carry = (
            initial_token,  # current_token
            x_cached_keys,
            x_cached_values,
            rho_cached_keys,
            rho_cached_values,
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
        ), _ = sps_step(initial_carry, jnp.arange(self.max_steps))

        if replaying:
            return (
                final_state,
                halting_log_prob,
                states_history,
                per_step_halting_log_prob,
                per_step_halting_entropy,
            )
        return final_state, num_steps_taken, first_convergence_step, num_close_steps
