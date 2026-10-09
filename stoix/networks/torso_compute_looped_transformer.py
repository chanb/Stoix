"""A looped (weight-tied, recurrent-in-depth) transformer torso over grid cells,
with a learned halting predictor.

`LoopedTransformerTorso` is the looped counterpart of
`stoix.networks.torso_compute_transformer.TransformerChainOfThoughtTorso`:
instead of growing a scratchpad of "thought" tokens from a single observation
embedding, it tokenizes the grid observation per cell (as in the LoopedActor
codebase's plane tokenizer) and repeatedly applies the same transformer
blocks to *every* token, so the whole board representation is refined at
each step:

    tokens  x = [READOUT, CELL_00, ..., CELL_(H-1,W-1)]
              CELL_ij = Dense([obs channels at (i, j); geometry(i, j)])
              geometry = normalized row/col in [-1, 1] + 4 boundary flags
    z_0 = x
    step t: z_{t+1} = Blocks(Inject(z_t, x))     (bidirectional attention)
            thought_t = z_{t+1}[READOUT]

After every step a `HaltingHead` reads the READOUT token's latent and the
halting decision is sampled (rollout), replayed (`target_compute_time`) or
taken greedily (`deterministic`) - exactly the halting semantics of
`TransformerChainOfThoughtTorso` (forced steps before `min_steps` /
`forced_min_steps` and at `max_steps`, per-step log-probs and entropies,
convergence diagnostics). The READOUT latent at the halting step is the
returned embedding, read by the action (and, with a shared torso, critic)
head. The interface - inputs, modes and outputs - is identical to
`TransformerChainOfThoughtTorso`, so it is a drop-in `pre_torso` for
`stoix.networks.base_compute.FeedForwardActorWithComputeTime` /
`FeedForwardActorCriticWithComputeTime` and `stoix.systems.ramdp_vpg.ff_ppo`.

The torso consumes the raw `(*batch, H, W, C)` grid itself, so the actor
network should have no `input_layer` (identity) - see
`stoix/configs/network/looped_transformer_compute*.yaml` - and run on a grid
scenario (e.g. `env=jumanji/sokoban_grid`).

Unlike the CoT torso there is no KV-cache: every step re-reads all tokens,
which have all changed, so a step costs a full pass over the `1 + H*W`
tokens. All examples run `max_steps` steps (a fixed-length `nn.scan`);
an example's embedding/log-probs are taken at its own halting step and later
steps are discarded, so they get no gradient. `remat` (default on)
rematerializes each step in the backward pass, keeping activation memory at
one step regardless of `max_steps`.

`input_injection` re-injects the input tokens `x` into every step's input
(`stoix.networks.torso_compute_transformer.InputInjection`): `"none"` is a
plain loop (`x` enters only as `z_0`), `"add"` adds a zero-initialized
projection of `x`, `"concat"` uses a `Dense([z; x])` adapter. `use_grid_conv`
(default off) additionally applies a residual depthwise 3x3 convolution to
the cell tokens at the start of every step (the FPRM core's local spatial
mixing). `use_step_embedding` (default off) adds a learned per-step
embedding to every token; it ties the parameters to `max_steps`.

MoE: `num_experts > 0` with `moe_type="switch"` gives every block a
`SwitchMoEMLP`; its expert capacity, if any, is per example over the
`1 + H*W` tokens, and its sown per-"step" statistics are per token position
here. Soft MoE is not supported (`SoftMoEMLP` is causal over the sequence,
which here would be the cell tokens).
"""

from typing import Optional, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.linen.initializers import Initializer, normal, orthogonal

from stoix.networks.torso_compute_transformer import (
    INPUT_INJECTION_MODES,
    MOE_TYPES,
    HaltingHead,
    InputInjection,
    TransformerBlock,
    _norm_cls,
    _resolve_qkv_dim,
)

_PROB_EPS = 1e-6


def grid_geometry(height: int, width: int) -> chex.Array:
    """`(height * width, 6)` per-cell geometry features, row-major: normalized
    row/col in [-1, 1] and top/bottom/left/right boundary flags."""
    rows = jnp.arange(height * width) // width
    cols = jnp.arange(height * width) % width
    norm_row = 2.0 * rows / max(height - 1, 1) - 1.0 if height > 1 else jnp.zeros_like(rows, float)
    norm_col = 2.0 * cols / max(width - 1, 1) - 1.0 if width > 1 else jnp.zeros_like(cols, float)
    flags = [rows == 0, rows == height - 1, cols == 0, cols == width - 1]
    return jnp.stack([norm_row, norm_col] + [f.astype(jnp.float32) for f in flags], axis=-1)


class _LoopedStep(nn.Module):
    """One loop step, lifted into a weight-tied loop via `nn.scan` (shared
    params across steps, as in `torso_compute_transformer._CoTStep`).

    Carry (all `(N, ...)` with a flattened batch):
        z: `(N, L, d)` current latent of every token.
        x: `(N, L, d)` input tokens (constant; for input injection).
        ...: the halting bookkeeping of `_CoTStep` (see that class).
    """

    hidden_dim: int
    num_heads: int
    num_layers: int
    mlp_dim: int
    max_steps: int
    grid_shape: Tuple[int, int]
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
    moe_type: str = "switch"
    switch_capacity_factor: Optional[float] = None
    switch_init_scale: Optional[float] = None
    halting_input_norm: bool = False
    input_injection: str = "none"
    use_grid_conv: bool = False
    use_step_embedding: bool = False

    @nn.compact
    def __call__(
        self, carry: Tuple[chex.Array, ...], step_idx: chex.Array
    ) -> Tuple[Tuple[chex.Array, ...], None]:
        (
            z,
            x,
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
            min_steps_per_example,
        ) = carry

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
                moe_type=self.moe_type,
                switch_capacity_factor=self.switch_capacity_factor,
                switch_init_scale=self.switch_init_scale,
            )
            for _ in range(self.num_layers)
        ]
        halting_head = HaltingHead(
            self.halting_hidden_dims,
            self.activation,
            self.kernel_init,
            input_norm=self.halting_input_norm,
            use_rmsnorm=self.use_rmsnorm,
        )

        h = z
        if self.input_injection != "none":
            h = InputInjection(
                self.hidden_dim, self.input_injection, self.kernel_init, name="input_injection"
            )(h, x)
        if self.use_step_embedding:
            step_embedding = self.param(
                "step_embedding", normal(stddev=0.02), (self.max_steps, self.hidden_dim)
            )
            h = h + step_embedding[step_idx]
        if self.use_grid_conv:
            # Residual depthwise convolution over the cell tokens (READOUT untouched).
            height, width = self.grid_shape
            cells = h[:, 1:].reshape(h.shape[0], height, width, self.hidden_dim)
            cells = cells + nn.Conv(
                self.hidden_dim,
                (3, 3),
                padding="SAME",
                feature_group_count=self.hidden_dim,
                use_bias=False,
                name="grid_conv",
            )(cells)
            h = jnp.concatenate([h[:, :1], cells.reshape(h.shape[0], -1, self.hidden_dim)], axis=1)

        # Steps past an example's halt are computed but discarded - keep them
        # out of the MoE statistics.
        token_mask = jnp.broadcast_to(still_running[:, None], h.shape[:2])
        for block in blocks:
            h = block(h, mask=None, token_mask=token_mask)
        z = h
        state = h[:, 0]  # READOUT latent: this step's "thought"
        if self.replaying:
            states_history = states_history.at[:, step_idx, :].set(state)

        # --- Halting: identical semantics to torso_compute_transformer._CoTStep ---
        halting_input = state
        if self.stop_gradient_halting_input:
            halting_input = jax.lax.stop_gradient(halting_input)
        halting_logit = halting_head(halting_input)
        halting_prob = nn.sigmoid(halting_logit / self.halting_temperature)
        halting_prob = jnp.clip(halting_prob.squeeze(axis=-1), _PROB_EPS, 1.0 - _PROB_EPS)

        step_count = step_idx + 1
        is_final_step = step_count == self.max_steps
        can_halt = step_count >= min_steps_per_example

        if not self.replaying:
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
            halts_this_step = still_running & (((halting_prob >= 0.5) & can_halt) | is_final_step)
        else:
            rng, step_rng = jax.random.split(rng)
            sampled_halt = jax.random.bernoulli(step_rng, halting_prob)
            halts_this_step = still_running & ((sampled_halt & can_halt) | is_final_step)

        # Forced outcomes (before the floor, or the halt at max_steps) are not
        # policy choices: no gradient and no log-prob/entropy contribution.
        is_forced_step = (step_count < min_steps_per_example) | is_final_step
        halting_prob_for_log = jnp.where(
            is_forced_step, jax.lax.stop_gradient(halting_prob), halting_prob
        )
        step_log_prob = jnp.where(
            halts_this_step, jnp.log(halting_prob_for_log), jnp.log(1.0 - halting_prob_for_log)
        )
        step_contribution = jnp.where(still_running & (~is_forced_step), step_log_prob, 0.0)
        halting_log_prob = halting_log_prob + step_contribution
        if self.replaying:
            per_step_halting_log_prob = per_step_halting_log_prob.at[:, step_idx].set(
                step_contribution
            )
            halting_entropy_this_step = -(
                halting_prob * jnp.log(halting_prob)
                + (1.0 - halting_prob) * jnp.log(1.0 - halting_prob)
            )
            per_step_halting_entropy = per_step_halting_entropy.at[:, step_idx].set(
                jnp.where(still_running & (~is_forced_step), halting_entropy_this_step, 0.0)
            )
        num_steps_taken = num_steps_taken + still_running.astype(jnp.float32)
        final_state = jnp.where(halts_this_step[:, None], state, final_state)
        still_running = still_running & (~halts_this_step)

        new_carry = (
            z,
            x,
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
            min_steps_per_example,
        )
        return new_carry, None


class LoopedTransformerTorso(nn.Module):
    """Looped transformer over grid-cell tokens with adaptively-sampled
    halting - see the module docstring. Same three-mode interface (rollout /
    replay / deterministic), `forced_min_steps`, and outputs as
    `TransformerChainOfThoughtTorso`; the shared hyperparameters (norms,
    `qkv_dim`, Switch MoE, halting head options, `input_injection`,
    `action_input_norm`) mean the same thing there.

    `observation` is the raw `(*batch, H, W, C)` grid. `use_input_layer_norm`
    normalizes the projected input tokens. `states_history` holds the READOUT
    latent at every step.
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
    moe_type: str = "switch"
    switch_capacity_factor: Optional[float] = None
    switch_init_scale: Optional[float] = None
    halting_input_norm: bool = False
    action_input_norm: bool = False
    input_injection: str = "none"
    use_grid_conv: bool = False
    use_step_embedding: bool = False
    remat: bool = True

    @nn.compact
    def __call__(
        self,
        observation: chex.Array,
        rng: Optional[chex.PRNGKey] = None,
        target_compute_time: Optional[chex.Array] = None,
        deterministic: bool = False,
        forced_min_steps: Optional[chex.Array] = None,
    ) -> Tuple[chex.Array, ...]:
        """See `TransformerChainOfThoughtTorso.__call__` for the arguments and
        returns; `observation` is the `(*batch, H, W, C)` grid."""
        if observation.ndim < 4:
            raise ValueError(
                "LoopedTransformerTorso expects a (*batch, H, W, C) grid observation (use a grid "
                f"scenario and no actor input_layer), got shape {observation.shape}."
            )
        replaying = target_compute_time is not None
        if not replaying and not deterministic and rng is None:
            raise ValueError(
                "rng must be provided to LoopedTransformerTorso when sampling "
                "(i.e. target_compute_time is None and deterministic=False)."
            )
        if not (1 <= self.min_steps <= self.max_steps):
            raise ValueError(
                f"min_steps must be between 1 and max_steps ({self.max_steps}), "
                f"got min_steps={self.min_steps}."
            )
        if self.input_injection not in INPUT_INJECTION_MODES:
            raise ValueError(
                f"input_injection must be one of {INPUT_INJECTION_MODES}, "
                f"got {self.input_injection!r}."
            )
        if self.moe_type not in MOE_TYPES:
            raise ValueError(f"moe_type must be one of {MOE_TYPES}, got {self.moe_type!r}.")
        if self.num_experts > 0 and self.moe_type == "soft":
            raise ValueError(
                "LoopedTransformerTorso does not support soft MoE (SoftMoEMLP is causal over the "
                "sequence, here the grid cells); use moe_type='switch'."
            )
        qkv_dim = _resolve_qkv_dim(self.hidden_dim, self.qkv_dim)
        assert qkv_dim % self.num_heads == 0, (
            f"qkv_dim ({qkv_dim}) must be divisible by num_heads ({self.num_heads})."
        )

        batch_shape = observation.shape[:-3]
        height, width, channels = observation.shape[-3:]
        grid = observation.reshape((-1, height * width, channels)).astype(jnp.float32)
        n = grid.shape[0]

        def flat(a: Optional[chex.Array]) -> Optional[chex.Array]:
            return None if a is None else jnp.broadcast_to(a, batch_shape).reshape(n)

        min_steps_per_example = jnp.full((n,), self.min_steps, dtype=jnp.float32)
        if forced_min_steps is not None:
            min_steps_per_example = jnp.maximum(
                min_steps_per_example, flat(jnp.asarray(forced_min_steps, jnp.float32))
            )

        # Tokenize: one token per cell from its channels + geometry, plus READOUT.
        geometry = jnp.broadcast_to(grid_geometry(height, width), (n, height * width, 6))
        cell_tokens = nn.Dense(self.hidden_dim, kernel_init=self.kernel_init, name="cell_proj")(
            jnp.concatenate([grid, geometry], axis=-1)
        )
        readout = self.param("readout_token", normal(stddev=0.02), (1, 1, self.hidden_dim))
        tokens = jnp.concatenate(
            [jnp.broadcast_to(readout, (n, 1, self.hidden_dim)), cell_tokens], axis=1
        )
        if self.use_input_layer_norm:
            tokens = _norm_cls(self.use_rmsnorm)(name="input_norm")(tokens)

        step_rng = rng if rng is not None else jax.random.PRNGKey(0)
        initial_carry = (
            tokens,  # z_0 = x
            tokens,  # x (constant)
            jnp.ones((n,), dtype=bool),  # still_running
            jnp.zeros((n,)),  # num_steps_taken
            jnp.zeros((n,)),  # halting_log_prob
            jnp.zeros((n, self.hidden_dim)),  # final_state (set at every example's halt)
            jnp.full((n,), -1.0),  # first_convergence_step
            jnp.zeros((n,)),  # num_close_steps
            tokens[:, 0],  # prev_state
            step_rng,
            flat(target_compute_time),
            jnp.zeros((n, self.max_steps, self.hidden_dim)),  # states_history
            jnp.zeros((n, self.max_steps)),  # per_step_halting_log_prob
            jnp.zeros((n, self.max_steps)),  # per_step_halting_entropy
            min_steps_per_example,
        )

        step_cls = nn.remat(_LoopedStep, prevent_cse=False) if self.remat else _LoopedStep
        loop_step = nn.scan(
            step_cls,
            variable_broadcast="params",
            variable_axes={"intermediates": 0},
            split_rngs={"params": False},
        )(
            self.hidden_dim,
            self.num_heads,
            self.num_layers,
            self.mlp_dim,
            self.max_steps,
            (height, width),
            self.activation,
            self.kernel_init,
            self.convergence_threshold,
            replaying,
            deterministic,
            self.stop_gradient_halting_input,
            self.halting_temperature,
            tuple(self.halting_hidden_dims),
            self.use_sandwich_norm,
            self.use_rmsnorm,
            self.qkv_dim,
            self.num_experts,
            self.moe_type,
            self.switch_capacity_factor,
            self.switch_init_scale,
            self.halting_input_norm,
            self.input_injection,
            self.use_grid_conv,
            self.use_step_embedding,
            name="loop_step",
        )
        (
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
        ), _ = loop_step(initial_carry, jnp.arange(self.max_steps))

        if self.action_input_norm:
            final_state = _norm_cls(self.use_rmsnorm)(name="action_input_norm")(final_state)

        def unflat(a: chex.Array) -> chex.Array:
            return a.reshape(batch_shape + a.shape[1:])

        if replaying:
            return (
                unflat(final_state),
                unflat(halting_log_prob),
                unflat(states_history),
                unflat(per_step_halting_log_prob),
                unflat(per_step_halting_entropy),
            )
        return (
            unflat(final_state),
            unflat(num_steps_taken),
            unflat(first_convergence_step),
            unflat(num_close_steps),
        )
