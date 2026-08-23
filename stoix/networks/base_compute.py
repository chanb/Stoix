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

import distrax
from flax import linen as nn

from stoix.base_types import Observation
from stoix.networks.inputs import ArrayInput


class FeedForwardActorWithComputeTime(nn.Module):
    """Feedforward actor whose torso also reports the compute time it used."""

    action_head: nn.Module
    torso: nn.Module
    input_layer: nn.Module = ArrayInput()

    @nn.compact
    def __call__(
        self,
        observation: Observation,
        input_kwargs: Optional[Dict] = None,
        torso_kwargs: Optional[Dict] = None,
        head_kwargs: Optional[Dict] = None,
    ) -> Tuple[distrax.DistributionLike, ...]:

        obs_embedding = self.input_layer(observation, **(input_kwargs or {}))
        obs_embedding, *torso_extra = self.torso(obs_embedding, **(torso_kwargs or {}))
        action_distribution = self.action_head(obs_embedding, **(head_kwargs or {}))

        return (action_distribution, *torso_extra)
