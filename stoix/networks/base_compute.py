"""Actor networks whose torso reports how much compute it used.

`FeedForwardActorWithComputeTime` mirrors `stoix.networks.base.FeedForwardActor`,
except its `torso` is expected to be an adaptively-halting torso (e.g.
`stoix.networks.torso_compute.AdaptiveComputationTimeTorso`,
`stoix.networks.torso_compute_transformer.TransformerChainOfThoughtTorso`, or
`stoix.networks.torso_compute_explicit_cot.TransformerExplicitCoTTorso`) that
returns `(embedding, *extra)`, where `extra` is one or more compute-related
outputs (a sampled `compute_time`, a `halting_log_prob` for training, and -
for the explicit-CoT torso - the emitted `thought_tokens`) depending on
`torso_kwargs` and which mode the torso is in - see the torso modules for
details. Whichever they are, they are threaded straight through to the
caller alongside the action distribution, rather than being consumed
internally: this wrapper doesn't need to know how many extra outputs there
are, or what they mean, only that the first output is the embedding to feed
the action head. This is a separate module so `stoix/networks/base.py` is
left untouched.
"""

from typing import Dict, Optional, Tuple

import chex
import distrax
import jax
import jax.numpy as jnp
from flax import linen as nn

from stoix.base_types import Observation
from stoix.networks.inputs import ArrayInput


class FeedForwardActorWithComputeTime(nn.Module):
    """Feedforward actor whose torso also reports the compute time it used."""

    action_head: nn.Module
    torso: nn.Module
    input_layer: nn.Module = ArrayInput()
    # Feature-wise dropout rate on the input_layer output. Applied only when `dropout_key`
    # (one key per example, shape (*batch, 2)) is given, so the same key gives the same mask
    # in rollout and in the PPO update.
    input_dropout: float = 0.0

    @nn.compact
    def __call__(
        self,
        observation: Observation,
        input_kwargs: Optional[Dict] = None,
        torso_kwargs: Optional[Dict] = None,
        head_kwargs: Optional[Dict] = None,
        dropout_key: Optional[chex.PRNGKey] = None,
    ) -> Tuple[distrax.DistributionLike, ...]:

        obs_embedding = self.input_layer(observation, **(input_kwargs or {}))
        if self.input_dropout > 0.0 and dropout_key is not None:
            keep_prob = 1.0 - self.input_dropout
            feature_shape = obs_embedding.shape[dropout_key.ndim - 1 :]
            keep = jax.vmap(lambda k: jax.random.bernoulli(k, keep_prob, feature_shape))(
                dropout_key.reshape(-1, 2)
            ).reshape(obs_embedding.shape)
            obs_embedding = jnp.where(keep, obs_embedding / keep_prob, 0.0)
        torso_kwargs = dict(torso_kwargs or {})
        if dropout_key is not None and (
            getattr(self.torso, "attention_dropout", 0.0) > 0.0
            or getattr(self.torso, "context_dropout", 0.0) > 0.0
        ):
            torso_kwargs["dropout_key"] = dropout_key  # see TransformerChainOfThoughtTorso
        obs_embedding, *torso_extra = self.torso(obs_embedding, **torso_kwargs)
        action_distribution = self.action_head(obs_embedding, **(head_kwargs or {}))

        return (action_distribution, *torso_extra)


class FeedForwardActorFromTorso(nn.Module):
    """Like `FeedForwardActorWithComputeTime`, but for a torso that is
    already a complete actor - e.g.
    `stoix.networks.torso_compute_explicit_cot.TransformerExplicitCoTTorso`,
    whose per-step categorical folds the environment action into the same
    vocabulary as the thought tokens, so choosing a class both halts and
    resolves the action in one draw. There is no continuous embedding left
    over for a head to turn into an action afterwards - the torso's first
    return value is already the resolved output (an `action` in rollout
    mode, a `log_prob` in replay mode) - so, unlike
    `FeedForwardActorWithComputeTime`, this has no `action_head` field at
    all; every torso output is threaded straight through to the caller.
    `input_layer` is kept (unlike `action_head`) since the CNN+transformer
    explicit-CoT network variants still need it ahead of the torso."""

    torso: nn.Module
    input_layer: nn.Module = ArrayInput()

    @nn.compact
    def __call__(
        self,
        observation: Observation,
        input_kwargs: Optional[Dict] = None,
        torso_kwargs: Optional[Dict] = None,
    ) -> Tuple:

        obs_embedding = self.input_layer(observation, **(input_kwargs or {}))
        return self.torso(obs_embedding, **(torso_kwargs or {}))
