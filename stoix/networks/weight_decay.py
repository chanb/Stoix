"""Weight-decay masks for `optax.adamw(..., mask=...)`.

`make_weight_decay_mask` builds the actor's mask from two settings
(`config.system.actor_weight_decay_mask` and
`config.system.moe_router_weight_decay_exempt`):

  - `"all"`: decay every parameter - the original behaviour - except the
    Switch MoE router(s) when `exempt_moe_router` is set (see
    `stoix.networks.torso_compute_transformer.switch_router_weight_decay_mask`).
  - `"standard"`: the usual transformer convention (GPT-2/BERT/T5-style) -
    decay only the weight matrices, exempting every parameter that acts as an
    offset, a gain, a position lookup or a router:
      - biases (`bias`, and the MoE experts' `b0`/`b1`);
      - normalization parameters (LayerNorm/RMSNorm `scale`, LayerNorm
        `bias`; also Soft MoE's learned logit `scale`);
      - learned positional embeddings (`pos_embedding`);
      - the Switch MoE router(s), whatever `exempt_moe_router` says.
    Kernels, MoE expert weights (`w0`/`w1`), Soft MoE slot parameters and
    token embedding tables are still decayed.

Matching is by parameter name, so it applies to any actor (CNN/MLP input
layers, IRU/GRU/MLP/transformer torsos, heads), not just transformers.
"""

from typing import Any, Callable, Optional, Tuple

import jax

from stoix.networks.torso_compute_transformer import switch_router_weight_decay_mask

WEIGHT_DECAY_MASK_MODES = ("all", "standard")

# Parameter (leaf) names never decayed under "standard".
_STANDARD_EXEMPT_LEAF_NAMES = frozenset({"bias", "b0", "b1", "scale", "pos_embedding"})


def standard_weight_decay_mask(params: Any) -> Any:
    """`False` (no decay) for biases, normalization parameters, positional
    embeddings and Switch MoE routers; `True` for everything else - see the
    module docstring."""

    def _decay(path: Tuple[Any, ...], _leaf: Any) -> bool:
        name = getattr(path[-1], "key", None) if path else None
        return name not in _STANDARD_EXEMPT_LEAF_NAMES

    by_name = jax.tree_util.tree_map_with_path(_decay, params)
    return jax.tree_util.tree_map(
        lambda a, b: a and b, by_name, switch_router_weight_decay_mask(params)
    )


def make_weight_decay_mask(
    mode: str, exempt_moe_router: bool = False
) -> Optional[Callable[[Any], Any]]:
    """The `mask` to pass to `optax.adamw` for `mode` (see the module
    docstring), or `None` - decay everything, AdamW's default - for `"all"`
    without a router exemption."""
    if mode not in WEIGHT_DECAY_MASK_MODES:
        raise ValueError(
            f"weight-decay mask must be one of {WEIGHT_DECAY_MASK_MODES}, got {mode!r}."
        )
    if mode == "standard":
        return standard_weight_decay_mask
    return switch_router_weight_decay_mask if exempt_moe_router else None
