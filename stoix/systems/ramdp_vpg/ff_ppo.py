"""Compute-time-aware PPO (RAMDP-PPO).

This is a variant of `stoix.systems.vpg.ff_reinforce`'s sibling
`stoix.systems.ramdp_vpg.ff_reinforce`/`ff_qac`, but trained with PPO's
clipped surrogate objective (several epochs of minibatch gradient steps per
rollout) instead of a single REINFORCE/QAC gradient step. As in those files,
compute time is folded into the discounting (Semi-MDP/RAMDP view - see
`ff_reinforce.py`'s module docstring for the full `G_h = gamma^(C_h - 1) *
(r_h + gamma * G_{h+1})` derivation) rather than the reward, and the actor's
halting trajectory (sampled by an Adaptive Computation Time torso, see
`stoix.networks.torso_compute.AdaptiveComputationTimeTorso`) is trained
jointly with the environment action from one combined log-probability
`log_prob = env_log_prob + halting_log_prob`.

`config.system.qac_variant` selects how the advantage used to train the
actor is computed - the same knob `ff_qac.py` uses, extended with three more
values:

  - "fac" / "naive": Q - V, exactly as in `ff_qac.py` (see that file's module
    docstring for the runtime-factorization identity "fac" relies on, and
    what "naive" learns instead). Advantage = Q(s_t, a_t, c_t) - V(s_t), with
    no n-step return or bootstrapping in the advantage itself.
  - "cond_naive" / "cond_fac": Q - V too, but with `compute_time` fed into
    the critic as an *input* - a single learned linear feature of `c` (see
    `stoix.networks.base_qac.ValueAndQCritic._q_input`) concatenated onto
    the shared embedding - rather than encoded in the output shape
    ("naive"'s `(num_actions, max_steps)` table) or assumed via an analytic
    scaling ("fac"'s `gamma ** (c - 1)`). Both use the exact same conditioned
    architecture and so have identical parameter counts, unlike "naive" vs
    "fac" - "cond_naive" reads the conditioned output directly as Q(s,·,c);
    "cond_fac" additionally scales it by `gamma ** (c - 1)`. Comparing the
    two isolates whether that analytic prior helps, holding capacity fixed,
    rather than conflating the prior with a parameter-count difference the
    way plain "naive" vs "fac" do.
  - "reinforce": G - V, REINFORCE-with-baseline exactly as in
    `ff_reinforce.py` - advantage = compute-discounted n-step Monte-Carlo
    return `G_h` minus V(s_t), with no GAE(lambda) term (this file does not
    reintroduce `stoix.systems.vpg.ff_ppo`'s GAE machinery; `G_h` is the same
    plain multistep return `ff_reinforce.py` computes). Uses
    `stoix.networks.base.FeedForwardCritic` (a plain V-only critic).

The four Q-V variants all use `stoix.networks.base_qac.ValueAndQCritic` (a
shared torso with a V head and a Q head).

Whichever variant, PPO's clipped ratio (`stoix.utils.loss.ppo_clip_loss`) is
applied to the *joint* log-probability, mirroring how the un-clipped
REINFORCE weight in `ff_reinforce.py`/`ff_qac.py` multiplies that same joint
log_prob: `ratio = exp((env_log_prob + halting_log_prob) - old_log_prob)`,
one clipped surrogate term, not separate ratios for the action and the
halting decisions. The critic (V, and Q for "naive"/"fac") is trained with
PPO's own clipped value loss (`stoix.utils.loss.clipped_value_loss`) against
the same `old` value/Q estimate recorded at rollout time by default -
consistent with this being a genuine multi-epoch PPO system - or with plain
L2 regression (`rlax.l2_loss`), as in `ff_reinforce.py`/`ff_qac.py`, when
`config.system.clip_value_loss=False`.

Because PPO needs a fixed "old" log-probability to compute the ratio against
across every epoch (unlike `ff_reinforce.py`/`ff_qac.py`, which only ever
evaluate at the current parameters, one gradient step per rollout), and the
Adaptive Computation Time torso's sampling pass does not itself report the
log-probability of the halting trajectory it just sampled (see
`AdaptiveComputationTimeTorso`'s two modes), each rollout step makes a
second actor forward pass in replay mode - `torso_kwargs={"target_compute_time":
compute_time}` using the compute time just sampled - purely to read off
`halting_log_prob` at the rollout-time parameters. This doubles the actor's
per-step forward cost during rollout (not during the epoch/minibatch update,
which already replays in this way); there's no cheaper alternative without
changing what the torso reports.

Like `ff_reinforce.py`/`ff_qac.py`, and unlike `stoix.systems.vpg.ff_ppo`,
this file does not implement truncation-aware bootstrapping (`done` is
`timestep.last()`, covering both termination and truncation) or observation
normalization - both are simplifications already baked into the RAMDP-VPG
family this extends, not something specific to adding PPO. It also drops
`ff_reinforce.py`/`ff_qac.py`'s optional "delightful" REINFORCE-weight gate:
that gate was defined in terms of an un-clipped REINFORCE weight, and PPO's
clipped surrogate already serves an analogous "don't overreact to one
sample" role built on the ratio instead - revisit if the two turn out to be
complementary rather than redundant.

This file intentionally duplicates most of `ff_reinforce.py`/`ff_qac.py`
rather than modifying them, so those systems are left untouched.
"""

import copy
import time
from typing import Any, Tuple

import chex
import flax
import hydra
import jax
import jax.numpy as jnp
import optax
import rlax
from colorama import Fore, Style
from flax.core.frozen_dict import FrozenDict
from omegaconf import DictConfig, OmegaConf
from stoa import Environment, get_final_step_metrics

from stoix.base_types import (
    ActorApply,
    ActorCriticOptStates,
    ActorCriticParams,
    AnakinExperimentOutput,
    CriticApply,
    LearnerFn,
)
from stoix.networks.base import FeedForwardCritic
from stoix.networks.base_compute import FeedForwardActorWithComputeTime as Actor
from stoix.networks.base_qac import SeparateValueAndQCritic, ValueAndQCritic
from stoix.systems.ramdp_vpg.evaluator import evaluator_setup_with_compute_time
from stoix.systems.ramdp_vpg.ff_reinforce import get_distribution_act_fn_with_compute_time
from stoix.systems.ramdp_vpg.ppo_types import PPOTransition
from stoix.systems.ramdp_vpg.ramdp_vpg_types import (
    RamdpOnPolicyLearnerState,
    update_discounted_return,
)
from stoix.utils import make_env as environments
from stoix.utils.checkpointing import Checkpointer
from stoix.utils.jax_utils import (
    merge_leading_dims,
    unreplicate_batch_dim,
    unreplicate_n_dims,
)
from stoix.utils.logger import LogEvent, StoixLogger
from stoix.utils.loss import clipped_value_loss, ppo_clip_loss
from stoix.utils.multistep import batch_discounted_returns
from stoix.utils.total_timestep_checker import check_total_timesteps
from stoix.utils.training import make_learning_rate


def get_learner_fn(
    env: Environment,
    apply_fns: Tuple[ActorApply, CriticApply],
    update_fns: Tuple[optax.TransformUpdateFn, optax.TransformUpdateFn],
    config: DictConfig,
) -> LearnerFn[RamdpOnPolicyLearnerState]:
    """Get the learner function."""

    # Get apply and update functions for actor and critic networks.
    actor_apply_fn, critic_apply_fn = apply_fns
    actor_update_fn, critic_update_fn = update_fns

    qac_variant = config.system.qac_variant
    assert qac_variant in (
        "naive",
        "fac",
        "cond_naive",
        "cond_fac",
        "reinforce",
    ), f"Unknown qac_variant: {qac_variant}"
    is_qac = qac_variant in ("naive", "fac", "cond_naive", "cond_fac")
    # "cond_naive"/"cond_fac" condition the critic's q_value on compute_time
    # as an extra input (see `ValueAndQCritic._q_input`), rather than
    # applying compute_time afterwards via table-indexing ("naive") or an
    # analytic scaling ("fac").
    condition_q_on_compute_time = qac_variant in ("cond_naive", "cond_fac")

    def _q_output(
        critic_params: FrozenDict, obs: chex.Array, compute_time: chex.Array
    ) -> chex.Array:
        """Get the critic's raw `q_value` output. "cond_naive"/"cond_fac"
        condition the network on `compute_time` as an extra input;
        "naive"/"fac" don't - `compute_time` is applied afterwards instead,
        by indexing a table ("naive") or an analytic scaling ("fac"), see
        `_q_at_action_and_compute_time`."""
        if condition_q_on_compute_time:
            return critic_apply_fn(
                critic_params, obs, method="q_value", compute_time=compute_time
            )
        return critic_apply_fn(critic_params, obs, method="q_value")

    def _q_at_action_and_compute_time(
        q_output: chex.Array, action: chex.Array, compute_time: chex.Array
    ) -> chex.Array:
        """Read Q(s, a, c) off `_q_output`'s raw output, however it's
        parameterised:

          - "naive": `q_output` is a `(..., num_actions, max_steps)` table -
            index by both `action` and `compute_time`.
          - "fac": `q_output` is `(..., num_actions)`, representing Q(s,·,1) -
            index by `action` only, then scale by `gamma ** (compute_time - 1)`
            to recover Q(s,a,c), per the runtime-factorization identity.
          - "cond_naive": `q_output` (from `_q_output`, already conditioned
            on `compute_time`) is `(..., num_actions)`, representing Q(s,·,c)
            directly - index by `action` only, no further scaling.
          - "cond_fac": same conditioned shape as "cond_naive", but also
            scaled by `gamma ** (compute_time - 1)` - see module docstring.
        """
        if qac_variant == "naive":
            compute_time_idx = (compute_time - 1).astype(jnp.int32)
            q_at_c = jnp.take_along_axis(
                q_output, compute_time_idx[..., jnp.newaxis, jnp.newaxis], axis=-1
            ).squeeze(-1)
            return jnp.take_along_axis(q_at_c, action[..., jnp.newaxis], axis=-1).squeeze(-1)
        q_sa = jnp.take_along_axis(q_output, action[..., jnp.newaxis], axis=-1).squeeze(-1)
        if qac_variant in ("fac", "cond_fac"):
            return config.system.gamma ** (compute_time - 1) * q_sa
        return q_sa  # "cond_naive"

    def _value_loss_fn(
        pred: chex.Array, behavior: chex.Array, targets: chex.Array
    ) -> chex.Array:
        """The critic's (V, and Q for "naive") regression loss - either PPO's
        clipped value loss (clipping `pred` against the `behavior` estimate
        recorded at rollout time) or plain L2 to `targets`, per
        `config.system.clip_value_loss` - see module docstring."""
        if config.system.clip_value_loss:
            return clipped_value_loss(pred, behavior, targets, config.system.clip_eps)
        return rlax.l2_loss(pred, targets).mean()

    def _update_step(
        learner_state: RamdpOnPolicyLearnerState, _: Any
    ) -> Tuple[RamdpOnPolicyLearnerState, Tuple]:
        def _env_step(
            learner_state: RamdpOnPolicyLearnerState, _: Any
        ) -> Tuple[RamdpOnPolicyLearnerState, PPOTransition]:
            """Step the environment."""
            (
                params,
                opt_states,
                key,
                env_state,
                last_timestep,
                running_cum_compute_time,
                running_discounted_return,
                episode_discounted_return,
            ) = learner_state

            # SELECT ACTION
            key, policy_key, halting_key = jax.random.split(key, 3)
            actor_policy, compute_time, first_convergence_step, num_close_steps = actor_apply_fn(
                params.actor_params,
                last_timestep.observation,
                torso_kwargs={"rng": halting_key},
            )
            action = actor_policy.sample(seed=policy_key)
            env_log_prob = actor_policy.log_prob(action)

            # The sampling pass above doesn't report the log-probability of the
            # halting trajectory it just sampled (see `AdaptiveComputationTimeTorso`'s
            # two modes) - get it with a second, replay-mode forward pass at these
            # same (rollout-time) parameters, so PPO has a fixed "old" joint
            # log_prob to compute its ratio against across every later epoch.
            _, halting_log_prob = actor_apply_fn(
                params.actor_params,
                last_timestep.observation,
                torso_kwargs={"target_compute_time": compute_time},
            )
            log_prob = env_log_prob + halting_log_prob

            if is_qac:
                value = critic_apply_fn(
                    params.critic_params, last_timestep.observation, method="value"
                )
                q_output = _q_output(
                    params.critic_params, last_timestep.observation, compute_time
                )
                q_value = _q_at_action_and_compute_time(q_output, action, compute_time)
            else:  # "reinforce"
                value = critic_apply_fn(params.critic_params, last_timestep.observation)
                q_value = jnp.zeros_like(value)

            # STEP ENVIRONMENT
            env_state, timestep = env.step(env_state, action)

            # LOG EPISODE METRICS
            done = timestep.last().reshape(-1)
            (
                running_cum_compute_time,
                running_discounted_return,
                episode_discounted_return,
            ) = update_discounted_return(
                running_cum_compute_time,
                running_discounted_return,
                episode_discounted_return,
                compute_time,
                timestep.reward,
                done,
                config.system.gamma,
            )
            info = {
                **timestep.extras["episode_metrics"],
                "episode_discounted_return": episode_discounted_return,
                "compute_time": compute_time,
            }

            transition = PPOTransition(
                done,
                action,
                value,
                q_value,
                timestep.reward,
                last_timestep.observation,
                info,
                compute_time,
                first_convergence_step,
                num_close_steps,
                log_prob,
            )
            learner_state = RamdpOnPolicyLearnerState(
                params,
                opt_states,
                key,
                env_state,
                timestep,
                running_cum_compute_time,
                running_discounted_return,
                episode_discounted_return,
            )
            return learner_state, transition

        # STEP ENVIRONMENT FOR ROLLOUT LENGTH
        learner_state, traj_batch = jax.lax.scan(
            _env_step, learner_state, None, config.system.rollout_length
        )

        # CALCULATE ADVANTAGE
        (
            params,
            opt_states,
            key,
            env_state,
            last_timestep,
            running_cum_compute_time,
            running_discounted_return,
            episode_discounted_return,
        ) = learner_state
        if is_qac:
            last_val = critic_apply_fn(
                params.critic_params, last_timestep.observation, method="value"
            )
        else:
            last_val = critic_apply_fn(params.critic_params, last_timestep.observation)

        # Swap the batch and time axes.
        traj_batch = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), traj_batch)

        # RETURN TARGET for V (and, for "naive", for Q too): the same n-step,
        # compute-discounted return as `ff_reinforce.py`'s critic - see its
        # module docstring for the full derivation of
        # `G_h = gamma^(C_h - 1) * (r_h + gamma * G_{h+1})`.
        compute_time = traj_batch.compute_time
        r_t = traj_batch.reward * config.system.gamma ** (compute_time - 1)
        v_t = jnp.concatenate([traj_batch.value, last_val[..., jnp.newaxis]], axis=-1)[:, 1:]
        not_done = 1.0 - traj_batch.done.astype(jnp.float32)
        d_t = (not_done * config.system.gamma**compute_time).astype(jnp.float32)
        g_targets = batch_discounted_returns(r_t, d_t, v_t, True, False)

        if is_qac:
            # Q - V: directly Q(s_t, a_t, c_t) - V(s_t), no n-step/Monte-Carlo
            # return involved at all (see `ff_qac.py`). `g_targets` here is
            # used only to train the critic, never the advantage.
            advantages = traj_batch.q_value - traj_batch.value
        else:
            # G - V: REINFORCE-with-baseline, as in `ff_reinforce.py`.
            advantages = g_targets - traj_batch.value
        targets = g_targets

        if config.system.standardize_advantages:
            advantages = jax.nn.standardize(advantages, axis=(0, 1))

        def _update_epoch(update_state: Tuple, _: Any) -> Tuple:
            """Update the network for a single epoch."""

            def _update_minibatch(train_state: Tuple, batch_info: Tuple) -> Tuple:
                """Update the network for a single minibatch."""

                # UNPACK TRAIN STATE AND BATCH INFO
                params, opt_states = train_state
                traj_batch, advantages, targets = batch_info

                def _actor_loss_fn(
                    actor_params: FrozenDict,
                    traj_batch: PPOTransition,
                    advantage: chex.Array,
                ) -> Tuple:
                    """Calculate the actor loss."""
                    # RERUN NETWORK. Replay (rather than re-sample) the halting
                    # trajectory that was actually taken during rollout
                    # (`traj_batch.compute_time`), mirroring how
                    # `log_prob(traj_batch.action)` below evaluates the stored
                    # action rather than drawing a fresh sample.
                    actor_policy, halting_log_prob = actor_apply_fn(
                        actor_params,
                        traj_batch.obs,
                        torso_kwargs={"target_compute_time": traj_batch.compute_time},
                    )
                    env_log_prob = actor_policy.log_prob(traj_batch.action)
                    # The halting decisions and the environment action they led
                    # to are trained jointly, from one combined ratio - see
                    # module docstring.
                    log_prob = env_log_prob + halting_log_prob

                    # CALCULATE ACTOR LOSS
                    loss_actor = ppo_clip_loss(
                        log_prob, traj_batch.log_prob, advantage, config.system.clip_eps
                    )
                    entropy = actor_policy.entropy().mean()

                    total_loss_actor = loss_actor - config.system.ent_coef * entropy
                    loss_info = {
                        "actor_loss": loss_actor,
                        "entropy": entropy,
                        "advantages": advantage,
                        "compute_time": traj_batch.compute_time,
                        "first_convergence_step": traj_batch.first_convergence_step,
                        "num_close_steps": traj_batch.num_close_steps,
                    }
                    return total_loss_actor, loss_info

                def _critic_loss_fn(
                    critic_params: FrozenDict,
                    traj_batch: PPOTransition,
                    targets: chex.Array,
                ) -> Tuple:
                    """Calculate the critic loss."""
                    if is_qac:
                        # RERUN NETWORK
                        value = critic_apply_fn(critic_params, traj_batch.obs, method="value")
                        value_loss = _value_loss_fn(value, traj_batch.value, targets)

                        q_output = _q_output(
                            critic_params, traj_batch.obs, traj_batch.compute_time
                        )
                        if qac_variant == "fac":
                            # Regress the raw (unscaled) Q(s,·,1) prediction
                            # against a rescaled target, rather than scaling
                            # the prediction up to the target's natural scale
                            # - keeps the critic's gradient in the same
                            # (small, c=1) space regardless of the realised
                            # compute time, avoiding a gamma^(2(c-1))
                            # implicit down-weighting of large-c samples that
                            # scaling the prediction instead would introduce.
                            q_pred = jnp.take_along_axis(
                                q_output, traj_batch.action[..., jnp.newaxis], axis=-1
                            ).squeeze(-1)
                            q_targets = targets / config.system.gamma ** (
                                traj_batch.compute_time - 1
                            )
                        else:  # "naive", "cond_naive", "cond_fac"
                            # Already in the true Q(s,a,c) scale (table
                            # lookup, or a c-conditioned prediction possibly
                            # scaled by `_q_at_action_and_compute_time`) - no
                            # prediction/target scale mismatch, so regress
                            # directly against `targets`.
                            q_pred = _q_at_action_and_compute_time(
                                q_output, traj_batch.action, traj_batch.compute_time
                            )
                            q_targets = targets
                        q_loss = _value_loss_fn(q_pred, traj_batch.q_value, q_targets)

                        critic_total_loss = config.system.vf_coef * (value_loss + q_loss)
                        loss_info = {
                            "value_loss": value_loss,
                            "q_loss": q_loss,
                        }
                    else:  # "reinforce"
                        # RERUN NETWORK
                        value = critic_apply_fn(critic_params, traj_batch.obs)

                        # CALCULATE VALUE LOSS
                        value_loss = _value_loss_fn(value, traj_batch.value, targets)

                        critic_total_loss = config.system.vf_coef * value_loss
                        loss_info = {
                            "value_loss": value_loss,
                        }
                    return critic_total_loss, loss_info

                # CALCULATE ACTOR LOSS
                actor_grad_fn = jax.grad(_actor_loss_fn, has_aux=True)
                actor_grads, actor_loss_info = actor_grad_fn(
                    params.actor_params, traj_batch, advantages
                )

                # CALCULATE CRITIC LOSS
                critic_grad_fn = jax.grad(_critic_loss_fn, has_aux=True)
                critic_grads, critic_loss_info = critic_grad_fn(
                    params.critic_params, traj_batch, targets
                )

                # Compute the parallel mean (pmean) over the batch.
                # This calculation is inspired by the Anakin architecture demo notebook.
                # available at https://tinyurl.com/26tdzs5x
                # This pmean could be a regular mean as the batch axis is on the same device.
                actor_grads, actor_loss_info, critic_grads, critic_loss_info = jax.lax.pmean(
                    (actor_grads, actor_loss_info, critic_grads, critic_loss_info),
                    axis_name="batch",
                )
                # pmean over devices.
                actor_grads, actor_loss_info, critic_grads, critic_loss_info = jax.lax.pmean(
                    (actor_grads, actor_loss_info, critic_grads, critic_loss_info),
                    axis_name="device",
                )

                # UPDATE ACTOR PARAMS AND OPTIMISER STATE
                actor_updates, actor_new_opt_state = actor_update_fn(
                    actor_grads, opt_states.actor_opt_state
                )
                actor_new_params = optax.apply_updates(params.actor_params, actor_updates)

                # UPDATE CRITIC PARAMS AND OPTIMISER STATE
                critic_updates, critic_new_opt_state = critic_update_fn(
                    critic_grads, opt_states.critic_opt_state
                )
                critic_new_params = optax.apply_updates(params.critic_params, critic_updates)

                # PACK NEW PARAMS AND OPTIMISER STATE
                new_params = ActorCriticParams(actor_new_params, critic_new_params)
                new_opt_state = ActorCriticOptStates(actor_new_opt_state, critic_new_opt_state)

                # PACK LOSS INFO
                loss_info = {
                    **actor_loss_info,
                    **critic_loss_info,
                }
                return (new_params, new_opt_state), loss_info

            (
                params,
                opt_states,
                traj_batch,
                advantages,
                targets,
                key,
            ) = update_state
            key, shuffle_key = jax.random.split(key)

            # SHUFFLE MINIBATCHES
            batch_size = config.system.rollout_length * config.arch.num_envs
            permutation = jax.random.permutation(shuffle_key, batch_size)
            batch = (traj_batch, advantages, targets)
            batch = jax.tree_util.tree_map(lambda x: merge_leading_dims(x, 2), batch)
            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0), batch
            )
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, [config.system.num_minibatches, -1] + list(x.shape[1:])),
                shuffled_batch,
            )

            # UPDATE MINIBATCHES
            (params, opt_states), loss_info = jax.lax.scan(
                _update_minibatch, (params, opt_states), minibatches
            )

            update_state = (
                params,
                opt_states,
                traj_batch,
                advantages,
                targets,
                key,
            )
            return update_state, loss_info

        update_state = (
            params,
            opt_states,
            traj_batch,
            advantages,
            targets,
            key,
        )

        # UPDATE EPOCHS
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config.system.epochs
        )

        params, opt_states, traj_batch, advantages, targets, key = update_state
        learner_state = RamdpOnPolicyLearnerState(
            params,
            opt_states,
            key,
            env_state,
            last_timestep,
            running_cum_compute_time,
            running_discounted_return,
            episode_discounted_return,
        )
        metric = traj_batch.info
        return learner_state, (metric, loss_info)

    def learner_fn(
        learner_state: RamdpOnPolicyLearnerState,
    ) -> AnakinExperimentOutput[RamdpOnPolicyLearnerState]:
        """Learner function.

        This function represents the learner, it updates the network parameters
        by iteratively applying the `_update_step` function for a fixed number of
        updates. The `_update_step` function is vectorized over a batch of inputs.
        """

        batched_update_step = jax.vmap(_update_step, in_axes=(0, None), axis_name="batch")

        learner_state, (episode_info, loss_info) = jax.lax.scan(
            batched_update_step, learner_state, None, config.arch.num_updates_per_eval
        )
        return AnakinExperimentOutput(
            learner_state=learner_state,
            episode_metrics=episode_info,
            train_metrics=loss_info,
        )

    return learner_fn


def learner_setup(
    env: Environment, keys: chex.Array, config: DictConfig
) -> Tuple[LearnerFn[RamdpOnPolicyLearnerState], Actor, RamdpOnPolicyLearnerState]:
    """Initialise learner_fn, network, optimiser, environment and states."""
    # Get available TPU cores.
    n_devices = len(jax.devices())

    # Get number/dimension of actions.
    num_actions = int(env.action_space().num_values)
    config.system.action_dim = num_actions

    # PRNG keys.
    key, actor_net_key, critic_net_key = keys

    qac_variant = config.system.qac_variant
    assert qac_variant in (
        "naive",
        "fac",
        "cond_naive",
        "cond_fac",
        "reinforce",
    ), f"Unknown qac_variant: {qac_variant}"

    # Define network and optimiser.
    actor_torso = hydra.utils.instantiate(config.network.actor_network.pre_torso)
    actor_action_head = hydra.utils.instantiate(
        config.network.actor_network.action_head, action_dim=num_actions
    )
    # A `value_pre_torso`/`q_pre_torso` pair (instead of a single shared
    # `pre_torso`) selects `SeparateValueAndQCritic`: V and Q each get their
    # own torso, so the two heads share no parameters at all (see
    # `base_qac.py`'s module docstring for the shared- vs separate-torso
    # tradeoff).
    separate_qv_torsos = "value_pre_torso" in config.network.critic_network
    if separate_qv_torsos:
        value_torso = hydra.utils.instantiate(config.network.critic_network.value_pre_torso)
        q_torso = hydra.utils.instantiate(config.network.critic_network.q_pre_torso)
    else:
        critic_torso = hydra.utils.instantiate(config.network.critic_network.pre_torso)

    # input_layer is optional (e.g. a CNNTorso encoding a raw grid observation into
    # a vector before it reaches the pondering torso, which itself needs a plain
    # vector) - defaults to Actor/Critic's own ArrayInput() (identity) when absent.
    actor_kwargs = {}
    if "input_layer" in config.network.actor_network:
        actor_kwargs["input_layer"] = hydra.utils.instantiate(
            config.network.actor_network.input_layer
        )
    critic_kwargs = {}
    if separate_qv_torsos:
        if "value_input_layer" in config.network.critic_network:
            critic_kwargs["value_input_layer"] = hydra.utils.instantiate(
                config.network.critic_network.value_input_layer
            )
        if "q_input_layer" in config.network.critic_network:
            critic_kwargs["q_input_layer"] = hydra.utils.instantiate(
                config.network.critic_network.q_input_layer
            )
    elif "input_layer" in config.network.critic_network:
        critic_kwargs["input_layer"] = hydra.utils.instantiate(
            config.network.critic_network.input_layer
        )

    actor_network = Actor(torso=actor_torso, action_head=actor_action_head, **actor_kwargs)

    if qac_variant in ("naive", "fac", "cond_naive", "cond_fac"):
        value_head = hydra.utils.instantiate(config.network.critic_network.value_head)
        # The Q head's output shape depends on `qac_variant` (see module
        # docstring): "naive" learns a full (action, compute_time) table, so
        # needs `max_steps` (read off the actor's compute-aware torso
        # config, which bounds the range of compute times the actor can ever
        # realise); "fac"/"cond_naive"/"cond_fac" only ever need one value
        # per action - "cond_naive"/"cond_fac" get their compute_time
        # dependence from an extra input instead (see
        # `ValueAndQCritic._q_input`), not the output shape.
        if qac_variant == "naive":
            max_steps = config.network.actor_network.pre_torso.max_steps
            q_head = hydra.utils.instantiate(
                config.network.critic_network.q_head,
                output_dim=max_steps,
                pre_shape=(num_actions,),
            )
        else:  # "fac", "cond_naive", "cond_fac"
            q_head = hydra.utils.instantiate(
                config.network.critic_network.q_head, output_dim=num_actions
            )
        if separate_qv_torsos:
            critic_network = SeparateValueAndQCritic(
                value_torso=value_torso,
                q_torso=q_torso,
                value_head=value_head,
                q_head=q_head,
                **critic_kwargs,
            )
        else:
            critic_network = ValueAndQCritic(
                torso=critic_torso, value_head=value_head, q_head=q_head, **critic_kwargs
            )
    else:  # "reinforce"
        critic_head = hydra.utils.instantiate(config.network.critic_network.critic_head)
        critic_network = FeedForwardCritic(
            torso=critic_torso, critic_head=critic_head, **critic_kwargs
        )

    actor_lr = make_learning_rate(
        config.system.actor_lr, config, config.system.epochs, config.system.num_minibatches
    )
    critic_lr = make_learning_rate(
        config.system.critic_lr, config, config.system.epochs, config.system.num_minibatches
    )

    actor_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(actor_lr, eps=1e-5),
    )
    critic_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(critic_lr, eps=1e-5),
    )

    # Initialise observation
    init_x = env.observation_space().generate_value()
    init_x = jax.tree_util.tree_map(lambda x: x[None, ...], init_x)

    # Initialise actor params and optimiser state.
    actor_params = actor_network.init(
        actor_net_key, init_x, torso_kwargs={"rng": actor_net_key}
    )
    actor_opt_state = actor_optim.init(actor_params)

    # Initialise critic params and optimiser state. "cond_naive"/"cond_fac"
    # need a representative `compute_time` at init time too, so `q_head`'s
    # input layer is sized to include the conditioning feature from the
    # start (see `ValueAndQCritic._q_input`).
    if qac_variant in ("cond_naive", "cond_fac"):
        dummy_compute_time = jnp.ones((1,), dtype=jnp.int32)
        critic_params = critic_network.init(
            critic_net_key, init_x, compute_time=dummy_compute_time
        )
    else:
        critic_params = critic_network.init(critic_net_key, init_x)
    critic_opt_state = critic_optim.init(critic_params)

    # Pack params.
    params = ActorCriticParams(actor_params, critic_params)

    actor_network_apply_fn = actor_network.apply
    critic_network_apply_fn = critic_network.apply

    # Pack apply and update functions.
    apply_fns = (actor_network_apply_fn, critic_network_apply_fn)
    update_fns = (actor_optim.update, critic_optim.update)

    # Get batched iterated update and replicate it to pmap it over cores.
    learn = get_learner_fn(env, apply_fns, update_fns, config)
    learn = jax.pmap(learn, axis_name="device")

    # Initialise environment states and timesteps: across devices and batches.
    key, *env_keys = jax.random.split(
        key, n_devices * config.arch.update_batch_size * config.arch.num_envs + 1
    )
    env_states, timesteps = env.reset(jnp.stack(env_keys))
    reshape_states = lambda x: x.reshape(
        (n_devices, config.arch.update_batch_size, config.arch.num_envs) + x.shape[1:]
    )
    # (devices, update batch size, num_envs, ...)
    env_states = jax.tree_util.tree_map(reshape_states, env_states)
    timesteps = jax.tree_util.tree_map(reshape_states, timesteps)

    # Load model from checkpoint if specified.
    if config.logger.checkpointing.load_model:
        loaded_checkpoint = Checkpointer(
            model_name=config.system.system_name,
            **config.logger.checkpointing.load_args,  # Other checkpoint args
        )
        # Restore the learner state from the checkpoint
        restored_params, _ = loaded_checkpoint.restore_params(input_params=params)
        # Update the params
        params = restored_params

    # Define params to be replicated across devices and batches.
    key, step_key = jax.random.split(key)
    step_keys = jax.random.split(step_key, n_devices * config.arch.update_batch_size)
    reshape_keys = lambda x: x.reshape((n_devices, config.arch.update_batch_size) + x.shape[1:])
    step_keys = reshape_keys(jnp.stack(step_keys))
    opt_states = ActorCriticOptStates(actor_opt_state, critic_opt_state)
    replicate_learner = (params, opt_states)

    # Duplicate learner for update_batch_size.
    broadcast = lambda x: jnp.broadcast_to(x, (config.arch.update_batch_size,) + x.shape)
    replicate_learner = jax.tree_util.tree_map(broadcast, replicate_learner)

    # Duplicate learner across devices.
    replicate_learner = flax.jax_utils.replicate(replicate_learner, devices=jax.devices())

    # Initialise learner state. running_cum_compute_time/running_discounted_return/
    # episode_discounted_return (see RamdpOnPolicyLearnerState) start at 0 for every
    # env - shape matches env_states/timesteps' leading (devices, update_batch_size,
    # num_envs) dims, since these are per-env running accumulators, not replicated
    # network state.
    params, opt_states = replicate_learner
    zeros_per_env = jnp.zeros(
        (n_devices, config.arch.update_batch_size, config.arch.num_envs), dtype=jnp.float32
    )
    init_learner_state = RamdpOnPolicyLearnerState(
        params,
        opt_states,
        step_keys,
        env_states,
        timesteps,
        zeros_per_env,
        zeros_per_env,
        zeros_per_env,
    )

    return learn, actor_network, init_learner_state


def run_experiment(_config: DictConfig) -> float:
    """Runs experiment."""
    config = copy.deepcopy(_config)

    # Calculate total timesteps.
    n_devices = len(jax.devices())
    config.num_devices = n_devices
    config = check_total_timesteps(config)
    assert (
        config.arch.num_updates >= config.arch.num_evaluation
    ), "Number of updates per evaluation must be less than total number of updates."

    # Create the environments for train and eval.
    env, eval_env = environments.make(config=config)

    # PRNG keys.
    key, key_e, actor_net_key, critic_net_key = jax.random.split(
        jax.random.PRNGKey(config.arch.seed), num=4
    )

    # Setup learner.
    learn, actor_network, learner_state = learner_setup(
        env, (key, actor_net_key, critic_net_key), config
    )

    # Setup evaluator.
    evaluator, absolute_metric_evaluator, (trained_params, eval_keys) = (
        evaluator_setup_with_compute_time(
            eval_env=eval_env,
            key_e=key_e,
            eval_act_fn=get_distribution_act_fn_with_compute_time(config, actor_network.apply),
            params=learner_state.params.actor_params,
            config=config,
        )
    )

    # Calculate environment steps per evaluation.
    steps_per_rollout = (
        n_devices
        * config.arch.num_updates_per_eval
        * config.system.rollout_length
        * config.arch.update_batch_size
        * config.arch.num_envs
    )

    # Logger setup
    logger = StoixLogger(config)
    logger.log_config(OmegaConf.to_container(config, resolve=True))
    print(f"{Fore.YELLOW}{Style.BRIGHT}JAX Global Devices {jax.devices()}{Style.RESET_ALL}")

    # Set up checkpointer
    save_checkpoint = config.logger.checkpointing.save_model
    if save_checkpoint:
        checkpointer = Checkpointer(
            metadata=config,  # Save all config as metadata in the checkpoint
            model_name=config.system.system_name,
            **config.logger.checkpointing.save_args,  # Checkpoint args
        )

    # Run experiment for a total number of evaluations.
    max_episode_return = -jnp.inf
    best_params = unreplicate_batch_dim(learner_state.params.actor_params)
    for eval_step in range(config.arch.num_evaluation):
        # Train.
        start_time = time.time()

        learner_output = learn(learner_state)
        jax.block_until_ready(learner_output)

        # Log the results of the training.
        elapsed_time = time.time() - start_time
        t = int(steps_per_rollout * (eval_step + 1))
        episode_metrics, ep_completed = get_final_step_metrics(learner_output.episode_metrics)
        episode_metrics["steps_per_second"] = steps_per_rollout / elapsed_time

        # Separately log timesteps, actoring metrics and training metrics.
        logger.log({"timestep": t}, t, eval_step, LogEvent.MISC)
        if ep_completed:  # only log episode metrics if an episode was completed in the rollout.
            logger.log(episode_metrics, t, eval_step, LogEvent.ACT)
        train_metrics = learner_output.train_metrics
        # Calculate the number of optimiser steps per second. Since gradients are aggregated
        # across the device and batch axis, we don't consider updates per device/batch as part of
        # the SPS for the learner.
        opt_steps_per_eval = config.arch.num_updates_per_eval * (
            config.system.epochs * config.system.num_minibatches
        )
        train_metrics["steps_per_second"] = opt_steps_per_eval / elapsed_time
        logger.log(train_metrics, t, eval_step, LogEvent.TRAIN)

        # Prepare for evaluation.
        start_time = time.time()
        trained_params = unreplicate_batch_dim(
            learner_output.learner_state.params.actor_params
        )  # Select only actor params
        key_e, *eval_keys = jax.random.split(key_e, n_devices + 1)
        eval_keys = jnp.stack(eval_keys)
        eval_keys = eval_keys.reshape(n_devices, -1)

        # Evaluate.
        evaluator_output = evaluator(trained_params, eval_keys)
        jax.block_until_ready(evaluator_output)

        # Log the results of the evaluation.
        elapsed_time = time.time() - start_time
        episode_return = jnp.mean(evaluator_output.episode_metrics["episode_return"])

        steps_per_eval = int(jnp.sum(evaluator_output.episode_metrics["episode_length"]))
        evaluator_output.episode_metrics["steps_per_second"] = steps_per_eval / elapsed_time
        logger.log(evaluator_output.episode_metrics, t, eval_step, LogEvent.EVAL)

        if save_checkpoint:
            # Save checkpoint of learner state
            checkpointer.save(
                timestep=int(steps_per_rollout * (eval_step + 1)),
                unreplicated_learner_state=unreplicate_n_dims(learner_output.learner_state),
                episode_return=episode_return,
            )

        if config.arch.absolute_metric and max_episode_return <= episode_return:
            best_params = copy.deepcopy(trained_params)
            max_episode_return = episode_return

        # Update runner state to continue training.
        learner_state = learner_output.learner_state

    # Measure absolute metric.
    if config.arch.absolute_metric:
        start_time = time.time()

        key_e, *eval_keys = jax.random.split(key_e, n_devices + 1)
        eval_keys = jnp.stack(eval_keys)
        eval_keys = eval_keys.reshape(n_devices, -1)

        evaluator_output = absolute_metric_evaluator(best_params, eval_keys)
        jax.block_until_ready(evaluator_output)

        elapsed_time = time.time() - start_time
        t = int(steps_per_rollout * (eval_step + 1))
        steps_per_eval = int(jnp.sum(evaluator_output.episode_metrics["episode_length"]))
        evaluator_output.episode_metrics["steps_per_second"] = steps_per_eval / elapsed_time
        logger.log(evaluator_output.episode_metrics, t, eval_step, LogEvent.ABSOLUTE)

    # Stop the logger.
    logger.stop()
    # Record the performance for the final evaluation run. If the absolute metric is not
    # calculated, this will be the final evaluation run.
    eval_performance = float(jnp.mean(evaluator_output.episode_metrics[config.env.eval_metric]))
    return eval_performance


@hydra.main(
    config_path="../../configs/default/anakin",
    config_name="default_ramdp_ff_ppo.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    # Allow dynamic attributes.
    OmegaConf.set_struct(cfg, False)

    # Run experiment.
    eval_performance = run_experiment(cfg)

    print(
        f"{Fore.CYAN}{Style.BRIGHT}Compute-time-aware PPO (RAMDP-PPO) experiment "
        f"completed{Style.RESET_ALL}"
    )
    return eval_performance


if __name__ == "__main__":
    hydra_entry_point()