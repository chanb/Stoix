"""Compute-time-aware PPO with an explicit chain of thought whose vocabulary
merges thought tokens and the environment action (RAMDP-PPO, explicit CoT).

The actor's torso is `TransformerMergedActionCoTTorso` (see
`stoix.networks.torso_compute_explicit_cot_merged`): at each pondering step it
samples a token from a vocabulary with `num_actions` extra classes beyond the
`vocab_size` thought classes, so choosing one of them both halts and selects
the environment action. There is therefore a single per-step categorical, one
per-token PPO clip and one entropy bonus (`config.system.ent_coef`). The
advantage and critic follow `ff_ppo.py` (G - V with a plain V-only critic).
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
from stoix.networks.base_compute import FeedForwardActorFromTorso as Actor
from stoix.systems.ramdp_vpg.evaluator import ComputeAwareActFn, evaluator_setup_with_compute_time
from stoix.systems.ramdp_vpg.explicit_cot_types import PPOMergedActionCoTTransition
from stoix.systems.ramdp_vpg.ramdp_vpg_types import (
    RamdpOnPolicyLearnerState,
    solved_episode_info,
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
from stoix.utils.loss import clipped_value_loss, dpo_surrogate, expectile_loss
from stoix.utils.multistep import batch_truncated_generalized_advantage_estimation
from stoix.utils.total_timestep_checker import check_total_timesteps
from stoix.utils.training import make_learning_rate


def get_merged_action_act_fn_with_compute_time(
    config: DictConfig,
    actor_apply: ActorApply,
) -> ComputeAwareActFn:
    """Act fn for `TransformerMergedActionCoTTorso`: the torso's rollout-mode output already *is*
    the resolved `action` - there is no distribution to call `.mode()`/`.sample()` on, since
    `deterministic` already picks the greedy class (thought or halt-with-action) at each step."""

    def act_fn(
        params: FrozenDict, observation: chex.Array, key: chex.PRNGKey
    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
        if config.arch.evaluation_greedy:
            action, compute_time, _thought_tokens = actor_apply(
                params, observation, torso_kwargs={"deterministic": True}
            )
        else:
            action, compute_time, _thought_tokens = actor_apply(
                params, observation, torso_kwargs={"rng": key}
            )
        first_convergence_step = -jnp.ones_like(compute_time)
        num_close_steps = jnp.zeros_like(compute_time)
        return action, compute_time, first_convergence_step, num_close_steps

    return act_fn


def get_learner_fn(
    env: Environment,
    apply_fns: Tuple[ActorApply, CriticApply],
    update_fns: Tuple[optax.TransformUpdateFn, optax.TransformUpdateFn],
    config: DictConfig,
) -> LearnerFn[RamdpOnPolicyLearnerState]:
    """Get the learner function."""

    actor_apply_fn, critic_apply_fn = apply_fns
    actor_update_fn, critic_update_fn = update_fns

    # Needed to build the "was this CoT step actually taken" mask used by
    # the per-step clip below - see `_actor_loss_fn`.
    max_steps = config.network.actor_network.pre_torso.max_steps

    def _value_loss_fn(
        pred: chex.Array, behavior: chex.Array, targets: chex.Array, use_expectile: bool = False
    ) -> chex.Array:
        """PPO's clipped value loss, plain L2, or expectile regression."""
        if use_expectile:
            return expectile_loss(pred, targets, config.system.expectile)
        if config.system.clip_value_loss:
            return clipped_value_loss(pred, behavior, targets, config.system.clip_eps)
        return rlax.l2_loss(pred, targets).mean()

    def _update_step(
        learner_state: RamdpOnPolicyLearnerState, _: Any
    ) -> Tuple[RamdpOnPolicyLearnerState, Tuple]:
        def _env_step(
            learner_state: RamdpOnPolicyLearnerState, _: Any
        ) -> Tuple[RamdpOnPolicyLearnerState, PPOMergedActionCoTTransition]:
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

            key, cot_key = jax.random.split(key)
            # The torso's rollout output already is the resolved action -
            # there is no separate `action_head`/`actor_policy.sample(...)`
            # step afterwards.
            action, compute_time, thought_tokens = actor_apply_fn(
                params.actor_params,
                last_timestep.observation,
                torso_kwargs={"rng": cot_key},
            )

            # Second replay-mode pass to get cot_log_prob at these same
            # (rollout-time) params, giving PPO a fixed "old" log_prob -
            # per-step (not summed), so each CoT step - including the one
            # that halted, which now also carries the action choice - can be
            # ratio/clipped individually rather than as one joint ratio.
            _, cot_log_prob, _ = actor_apply_fn(
                params.actor_params,
                last_timestep.observation,
                torso_kwargs={"target_tokens": thought_tokens},
            )

            value = critic_apply_fn(params.critic_params, last_timestep.observation)

            env_state, timestep = env.step(env_state, action)

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
                **solved_episode_info(config, timestep.reward, done),
                "compute_time": compute_time,
            }

            transition = PPOMergedActionCoTTransition(
                done,
                action,
                value,
                timestep.reward,
                last_timestep.observation,
                info,
                compute_time,
                thought_tokens,
                cot_log_prob,
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

        learner_state, traj_batch = jax.lax.scan(
            _env_step, learner_state, None, config.system.rollout_length
        )

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
        last_val = critic_apply_fn(params.critic_params, last_timestep.observation)

        traj_batch = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), traj_batch)

        # Return target for V: the n-step compute-discounted return `G_h`
        # (see `ff_ppo.py`'s module docstring).
        compute_time = traj_batch.compute_time
        r_t = traj_batch.reward * config.system.gamma ** (compute_time - 1)
        v_t = jnp.concatenate([traj_batch.value, last_val[..., jnp.newaxis]], axis=-1)[:, 1:]
        not_done = 1.0 - traj_batch.done.astype(jnp.float32)
        d_t = (not_done * config.system.gamma**compute_time).astype(jnp.float32)
        _, g_targets = batch_truncated_generalized_advantage_estimation(
            r_t, d_t, config.system.gae_lambda, v_tm1=traj_batch.value, v_t=v_t,
            stop_target_gradients=True,
        )

        advantages = g_targets - traj_batch.value
        targets = g_targets

        if config.system.standardize_advantages:
            advantages = jax.nn.standardize(advantages, axis=(0, 1))

        # --- config.system.critic_before_actor machinery - `_actor_loss_fn`/ `_critic_loss_fn`
        # below are exact copies of the ones nested inside the joint `_update_minibatch` further
        # down, hoisted to this scope so both the sequential path here and the joint path below can
        # each use their own copy without depending on one another.

        def _actor_loss_fn(
            actor_params: FrozenDict,
            traj_batch: PPOMergedActionCoTTransition,
            advantage: chex.Array,
        ) -> Tuple:
            """Calculate the actor loss (see the identical copy nested in
            the joint `_update_minibatch` below for the full explanation)."""
            # Replay the token trajectory actually taken during rollout - the
            # halting step's ratio/clip already covers the action choice, so
            # there is no separate `env_log_prob`/`action_loss` here.
            _, cot_log_prob, cot_entropy = actor_apply_fn(
                actor_params,
                traj_batch.obs,
                torso_kwargs={"target_tokens": traj_batch.thought_tokens},
            )

            step_idx = jnp.arange(max_steps)
            valid_step = (step_idx < traj_batch.compute_time[..., None]).astype(jnp.float32)
            num_valid_steps = jnp.maximum(jnp.sum(valid_step), 1.0)

            cot_ratio = jnp.exp(cot_log_prob - traj_batch.cot_log_prob)
            advantage_per_step = advantage[..., None]
            if config.system.use_dpo_loss:
                cot_per_step_loss = dpo_surrogate(
                    cot_log_prob,
                    traj_batch.cot_log_prob,
                    advantage_per_step,
                    config.system.dpo_alpha,
                    config.system.dpo_beta,
                )
            else:
                cot_surrogate1 = cot_ratio * advantage_per_step
                cot_surrogate2 = (
                    jnp.clip(cot_ratio, 1.0 - config.system.clip_eps, 1.0 + config.system.clip_eps)
                    * advantage_per_step
                )
                cot_per_step_loss = -jnp.minimum(cot_surrogate1, cot_surrogate2)
            loss_actor = jnp.sum(cot_per_step_loss * valid_step) / num_valid_steps
            clip_fraction = (
                jnp.sum(
                    (jnp.abs(cot_ratio - 1.0) > config.system.clip_eps).astype(jnp.float32)
                    * valid_step
                )
                / num_valid_steps
            )
            # Entropy bonus on the whole per-step categorical (thought
            # tokens and the halting-with-action classes together) - there's
            # no separate action distribution to regularize, so
            # `config.system.ent_coef` alone covers both the environment
            # action's exploration and the halting decision's, since they're
            # the same decision.
            entropy = jnp.sum(cot_entropy * valid_step) / num_valid_steps

            total_loss_actor = loss_actor - config.system.ent_coef * entropy
            loss_info = {
                "actor_loss": loss_actor,
                "entropy": entropy,
                "advantages": advantage,
                "compute_time": traj_batch.compute_time,
                "clip_fraction": clip_fraction,
            }
            return total_loss_actor, loss_info

        def _critic_loss_fn(
            critic_params: FrozenDict,
            traj_batch: PPOMergedActionCoTTransition,
            targets: chex.Array,
        ) -> Tuple:
            """Calculate the critic loss (see the identical copy nested in
            the joint `_update_minibatch` below for the full explanation)."""
            value = critic_apply_fn(critic_params, traj_batch.obs)
            value_loss = _value_loss_fn(
                value,
                traj_batch.value,
                targets,
                use_expectile=config.system.use_expectile_value_loss,
            )

            critic_total_loss = config.system.vf_coef * value_loss
            loss_info = {"value_loss": value_loss}
            return critic_total_loss, loss_info

        def _apply_actor_update(
            params: ActorCriticParams,
            opt_states: ActorCriticOptStates,
            traj_batch: PPOMergedActionCoTTransition,
            advantage: chex.Array,
        ) -> Tuple[ActorCriticParams, ActorCriticOptStates, dict]:
            """Actor-only minibatch update - critic params/opt_state pass
            through unchanged."""
            actor_grad_fn = jax.grad(_actor_loss_fn, has_aux=True)
            actor_grads, actor_loss_info = actor_grad_fn(
                params.actor_params, traj_batch, advantage
            )
            actor_grads, actor_loss_info = jax.lax.pmean(
                (actor_grads, actor_loss_info), axis_name="batch"
            )
            actor_grads, actor_loss_info = jax.lax.pmean(
                (actor_grads, actor_loss_info), axis_name="device"
            )
            actor_loss_info["actor_grad_norm"] = optax.global_norm(actor_grads)

            actor_updates, actor_new_opt_state = actor_update_fn(
                actor_grads, opt_states.actor_opt_state, params.actor_params
            )
            actor_new_params = optax.apply_updates(params.actor_params, actor_updates)
            actor_loss_info["actor_param_norm"] = optax.global_norm(actor_new_params)

            new_params = ActorCriticParams(actor_new_params, params.critic_params)
            new_opt_state = ActorCriticOptStates(actor_new_opt_state, opt_states.critic_opt_state)
            return new_params, new_opt_state, actor_loss_info

        def _apply_critic_update(
            params: ActorCriticParams,
            opt_states: ActorCriticOptStates,
            traj_batch: PPOMergedActionCoTTransition,
            targets: chex.Array,
        ) -> Tuple[ActorCriticParams, ActorCriticOptStates, dict]:
            """Critic-only minibatch update - actor params/opt_state pass
            through unchanged."""
            critic_grad_fn = jax.grad(_critic_loss_fn, has_aux=True)
            critic_grads, critic_loss_info = critic_grad_fn(
                params.critic_params, traj_batch, targets
            )
            critic_grads, critic_loss_info = jax.lax.pmean(
                (critic_grads, critic_loss_info), axis_name="batch"
            )
            critic_grads, critic_loss_info = jax.lax.pmean(
                (critic_grads, critic_loss_info), axis_name="device"
            )
            critic_loss_info["critic_grad_norm"] = optax.global_norm(critic_grads)

            critic_updates, critic_new_opt_state = critic_update_fn(
                critic_grads, opt_states.critic_opt_state, params.critic_params
            )
            critic_new_params = optax.apply_updates(params.critic_params, critic_updates)
            critic_loss_info["critic_param_norm"] = optax.global_norm(critic_new_params)

            new_params = ActorCriticParams(params.actor_params, critic_new_params)
            new_opt_state = ActorCriticOptStates(opt_states.actor_opt_state, critic_new_opt_state)
            return new_params, new_opt_state, critic_loss_info

        def _refresh_targets_and_advantages(
            critic_params: FrozenDict,
        ) -> Tuple[chex.Array, chex.Array]:
            """Recompute `targets`/`advantages` from `critic_params`."""
            new_value = critic_apply_fn(critic_params, traj_batch.obs)
            new_last_val = critic_apply_fn(critic_params, last_timestep.observation)

            v_t = jnp.concatenate([new_value, new_last_val[..., jnp.newaxis]], axis=-1)[:, 1:]
            _, refreshed_targets = batch_truncated_generalized_advantage_estimation(
                r_t, d_t, config.system.gae_lambda, v_tm1=new_value, v_t=v_t,
                stop_target_gradients=True,
            )

            refreshed_advantages = refreshed_targets - new_value

            if config.system.standardize_advantages:
                refreshed_advantages = jax.nn.standardize(refreshed_advantages, axis=(0, 1))
            return refreshed_targets, refreshed_advantages

        def _update_minibatch_critic_only(train_state: Tuple, batch_info: Tuple) -> Tuple:
            """`critic_before_actor`'s critic-only minibatch step."""
            params, opt_states = train_state
            mb_traj_batch, mb_targets = batch_info
            params, opt_states, critic_loss_info = _apply_critic_update(
                params, opt_states, mb_traj_batch, mb_targets
            )
            return (params, opt_states), critic_loss_info

        def _update_minibatch_actor_only(train_state: Tuple, batch_info: Tuple) -> Tuple:
            """`critic_before_actor`'s actor-only minibatch step."""
            params, opt_states = train_state
            mb_traj_batch, mb_advantages = batch_info
            params, opt_states, actor_loss_info = _apply_actor_update(
                params, opt_states, mb_traj_batch, mb_advantages
            )
            return (params, opt_states), actor_loss_info

        def _update_epoch_critic_only(update_state: Tuple, _: Any) -> Tuple:
            """`critic_before_actor`'s critic-only epoch: one full shuffled
            pass over the rollout's minibatches, critic params only."""
            params, opt_states, epoch_traj_batch, epoch_targets, key = update_state
            key, shuffle_key = jax.random.split(key)

            batch_size = config.system.rollout_length * config.arch.num_envs
            permutation = jax.random.permutation(shuffle_key, batch_size)
            batch = (epoch_traj_batch, epoch_targets)
            batch = jax.tree_util.tree_map(lambda x: merge_leading_dims(x, 2), batch)
            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0), batch
            )
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, [config.system.num_minibatches, -1] + list(x.shape[1:])),
                shuffled_batch,
            )

            (params, opt_states), loss_info = jax.lax.scan(
                _update_minibatch_critic_only, (params, opt_states), minibatches
            )

            if config.system.recompute_advantages:
                epoch_targets, _ = _refresh_targets_and_advantages(params.critic_params)

            update_state = (params, opt_states, epoch_traj_batch, epoch_targets, key)
            return update_state, loss_info

        def _update_epoch_actor_only(update_state: Tuple, _: Any) -> Tuple:
            """`critic_before_actor`'s actor-only epoch: one full shuffled
            pass over the rollout's minibatches, actor params only."""
            params, opt_states, epoch_traj_batch, epoch_advantages, key = update_state
            key, shuffle_key = jax.random.split(key)

            batch_size = config.system.rollout_length * config.arch.num_envs
            permutation = jax.random.permutation(shuffle_key, batch_size)
            batch = (epoch_traj_batch, epoch_advantages)
            batch = jax.tree_util.tree_map(lambda x: merge_leading_dims(x, 2), batch)
            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0), batch
            )
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, [config.system.num_minibatches, -1] + list(x.shape[1:])),
                shuffled_batch,
            )

            (params, opt_states), loss_info = jax.lax.scan(
                _update_minibatch_actor_only, (params, opt_states), minibatches
            )

            update_state = (params, opt_states, epoch_traj_batch, epoch_advantages, key)
            return update_state, loss_info

        # --- end config.system.critic_before_actor machinery ---

        def _update_epoch(update_state: Tuple, _: Any) -> Tuple:
            """Update the network for a single epoch."""

            def _update_minibatch(train_state: Tuple, batch_info: Tuple) -> Tuple:
                """Update the network for a single minibatch."""

                params, opt_states = train_state
                traj_batch, advantages, targets = batch_info

                def _actor_loss_fn(
                    actor_params: FrozenDict,
                    traj_batch: PPOMergedActionCoTTransition,
                    advantage: chex.Array,
                ) -> Tuple:
                    """Calculate the actor loss."""
                    # Replay the token trajectory actually taken during
                    # rollout, mirroring `_env_step`'s replay-mode pass.
                    _, cot_log_prob, cot_entropy = actor_apply_fn(
                        actor_params,
                        traj_batch.obs,
                        torso_kwargs={"target_tokens": traj_batch.thought_tokens},
                    )

                    # `cot_log_prob`/`traj_batch.cot_log_prob`: `(*batch, max_steps)`, zeroed past
                    # the step each example actually halted at (see
                    # `TransformerMergedActionCoTTorso`).
                    step_idx = jnp.arange(max_steps)
                    valid_step = (step_idx < traj_batch.compute_time[..., None]).astype(
                        jnp.float32
                    )
                    num_valid_steps = jnp.maximum(jnp.sum(valid_step), 1.0)

                    cot_ratio = jnp.exp(cot_log_prob - traj_batch.cot_log_prob)
                    advantage_per_step = advantage[..., None]
                    if config.system.use_dpo_loss:
                        cot_per_step_loss = dpo_surrogate(
                            cot_log_prob,
                            traj_batch.cot_log_prob,
                            advantage_per_step,
                            config.system.dpo_alpha,
                            config.system.dpo_beta,
                        )
                    else:
                        cot_surrogate1 = cot_ratio * advantage_per_step
                        cot_surrogate2 = (
                            jnp.clip(
                                cot_ratio,
                                1.0 - config.system.clip_eps,
                                1.0 + config.system.clip_eps,
                            )
                            * advantage_per_step
                        )
                        cot_per_step_loss = -jnp.minimum(cot_surrogate1, cot_surrogate2)
                    # Masked mean over every CoT step actually taken across
                    # the whole minibatch (not a per-example mean averaged
                    # over examples), so trajectories with more valid steps
                    # don't get down-weighted relative to shorter ones.
                    loss_actor = jnp.sum(cot_per_step_loss * valid_step) / num_valid_steps
                    clip_fraction = (
                        jnp.sum(
                            (jnp.abs(cot_ratio - 1.0) > config.system.clip_eps).astype(
                                jnp.float32
                            )
                            * valid_step
                        )
                        / num_valid_steps
                    )
                    # Entropy bonus on the whole per-step categorical
                    # (thought tokens and the halting-with-action classes
                    # together) - `config.system.ent_coef` alone now covers
                    # what used to need both `ent_coef` (on the environment
                    # action's own distribution) and `halting_ent_coef` (on
                    # the CoT-step distribution), since they're the same
                    # distribution now - see module docstring.
                    entropy = jnp.sum(cot_entropy * valid_step) / num_valid_steps

                    total_loss_actor = loss_actor - config.system.ent_coef * entropy
                    loss_info = {
                        "actor_loss": loss_actor,
                        "entropy": entropy,
                        "advantages": advantage,
                        "compute_time": traj_batch.compute_time,
                        "clip_fraction": clip_fraction,
                    }
                    return total_loss_actor, loss_info

                def _critic_loss_fn(
                    critic_params: FrozenDict,
                    traj_batch: PPOMergedActionCoTTransition,
                    targets: chex.Array,
                ) -> Tuple:
                    """Calculate the critic loss."""
                    value = critic_apply_fn(critic_params, traj_batch.obs)
                    value_loss = _value_loss_fn(
                        value,
                        traj_batch.value,
                        targets,
                        use_expectile=config.system.use_expectile_value_loss,
                    )

                    critic_total_loss = config.system.vf_coef * value_loss
                    loss_info = {
                        "value_loss": value_loss,
                    }
                    return critic_total_loss, loss_info

                actor_grad_fn = jax.grad(_actor_loss_fn, has_aux=True)
                actor_grads, actor_loss_info = actor_grad_fn(
                    params.actor_params, traj_batch, advantages
                )

                critic_grad_fn = jax.grad(_critic_loss_fn, has_aux=True)
                critic_grads, critic_loss_info = critic_grad_fn(
                    params.critic_params, traj_batch, targets
                )

                # pmean over the batch axis, then over devices.
                actor_grads, actor_loss_info, critic_grads, critic_loss_info = jax.lax.pmean(
                    (actor_grads, actor_loss_info, critic_grads, critic_loss_info),
                    axis_name="batch",
                )
                actor_grads, actor_loss_info, critic_grads, critic_loss_info = jax.lax.pmean(
                    (actor_grads, actor_loss_info, critic_grads, critic_loss_info),
                    axis_name="device",
                )

                actor_loss_info["actor_grad_norm"] = optax.global_norm(actor_grads)
                critic_loss_info["critic_grad_norm"] = optax.global_norm(critic_grads)

                actor_updates, actor_new_opt_state = actor_update_fn(
                    actor_grads, opt_states.actor_opt_state, params.actor_params
                )
                actor_new_params = optax.apply_updates(params.actor_params, actor_updates)

                critic_updates, critic_new_opt_state = critic_update_fn(
                    critic_grads, opt_states.critic_opt_state, params.critic_params
                )
                critic_new_params = optax.apply_updates(params.critic_params, critic_updates)

                new_params = ActorCriticParams(actor_new_params, critic_new_params)
                new_opt_state = ActorCriticOptStates(actor_new_opt_state, critic_new_opt_state)

                actor_loss_info["actor_param_norm"] = optax.global_norm(actor_new_params)
                critic_loss_info["critic_param_norm"] = optax.global_norm(critic_new_params)

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

            (params, opt_states), loss_info = jax.lax.scan(
                _update_minibatch, (params, opt_states), minibatches
            )

            if config.system.recompute_advantages:
                new_value = critic_apply_fn(params.critic_params, traj_batch.obs)
                new_last_val = critic_apply_fn(params.critic_params, last_timestep.observation)

                v_t = jnp.concatenate(
                    [new_value, new_last_val[..., jnp.newaxis]], axis=-1
                )[:, 1:]
                _, targets = batch_truncated_generalized_advantage_estimation(
                    r_t, d_t, config.system.gae_lambda, v_tm1=new_value, v_t=v_t,
                    stop_target_gradients=True,
                )

                advantages = targets - new_value

                if config.system.standardize_advantages:
                    advantages = jax.nn.standardize(advantages, axis=(0, 1))

            update_state = (
                params,
                opt_states,
                traj_batch,
                advantages,
                targets,
                key,
            )
            return update_state, loss_info

        if config.system.critic_before_actor:
            # Phase 1: `epochs` epochs of critic-only updates.
            critic_update_state = (params, opt_states, traj_batch, targets, key)
            critic_update_state, critic_loss_info = jax.lax.scan(
                _update_epoch_critic_only, critic_update_state, None, config.system.epochs
            )
            params, opt_states, traj_batch, targets, key = critic_update_state

            targets, advantages = _refresh_targets_and_advantages(params.critic_params)

            # Phase 2: `epochs` epochs of actor-only updates against that
            # fixed advantage.
            actor_update_state = (params, opt_states, traj_batch, advantages, key)
            actor_update_state, actor_loss_info = jax.lax.scan(
                _update_epoch_actor_only, actor_update_state, None, config.system.epochs
            )
            params, opt_states, traj_batch, advantages, key = actor_update_state

            loss_info = {**critic_loss_info, **actor_loss_info}
        else:
            update_state = (
                params,
                opt_states,
                traj_batch,
                advantages,
                targets,
                key,
            )

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
    n_devices = len(jax.devices())

    num_actions = int(env.action_space().num_values)
    config.system.action_dim = num_actions

    key, actor_net_key, critic_net_key = keys

    # `num_actions` sizes the torso's merged vocabulary directly (no separate
    # `action_head` to instantiate with it - see
    # `TransformerMergedActionCoTTorso`/`FeedForwardActorFromTorso`).
    actor_torso = hydra.utils.instantiate(
        config.network.actor_network.pre_torso, num_actions=num_actions
    )
    critic_torso = hydra.utils.instantiate(config.network.critic_network.pre_torso)

    # input_layer is optional - defaults to identity when absent.
    actor_kwargs = {}
    if "input_layer" in config.network.actor_network:
        actor_kwargs["input_layer"] = hydra.utils.instantiate(
            config.network.actor_network.input_layer
        )
    critic_kwargs = {}
    if "input_layer" in config.network.critic_network:
        critic_kwargs["input_layer"] = hydra.utils.instantiate(
            config.network.critic_network.input_layer
        )

    actor_network = Actor(torso=actor_torso, **actor_kwargs)

    critic_head = hydra.utils.instantiate(config.network.critic_network.critic_head)
    critic_network = FeedForwardCritic(torso=critic_torso, critic_head=critic_head, **critic_kwargs)

    actor_lr = make_learning_rate(
        config.system.actor_lr, config, config.system.epochs, config.system.num_minibatches
    )
    critic_lr = make_learning_rate(
        config.system.critic_lr, config, config.system.epochs, config.system.num_minibatches
    )

    actor_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adamw(actor_lr, eps=1e-5, weight_decay=config.system.actor_weight_decay),
    )
    critic_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adamw(critic_lr, eps=1e-5, weight_decay=config.system.critic_weight_decay),
    )

    init_x = env.observation_space().generate_value()
    init_x = jax.tree_util.tree_map(lambda x: x[None, ...], init_x)

    actor_params = actor_network.init(
        actor_net_key, init_x, torso_kwargs={"rng": actor_net_key}
    )
    actor_opt_state = actor_optim.init(actor_params)

    critic_params = critic_network.init(critic_net_key, init_x)
    critic_opt_state = critic_optim.init(critic_params)

    params = ActorCriticParams(actor_params, critic_params)

    actor_network_apply_fn = actor_network.apply
    critic_network_apply_fn = critic_network.apply

    apply_fns = (actor_network_apply_fn, critic_network_apply_fn)
    update_fns = (actor_optim.update, critic_optim.update)

    learn = get_learner_fn(env, apply_fns, update_fns, config)
    learn = jax.pmap(learn, axis_name="device")

    key, *env_keys = jax.random.split(
        key, n_devices * config.arch.update_batch_size * config.arch.num_envs + 1
    )
    env_states, timesteps = env.reset(jnp.stack(env_keys))
    reshape_states = lambda x: x.reshape(
        (n_devices, config.arch.update_batch_size, config.arch.num_envs) + x.shape[1:]
    )
    env_states = jax.tree_util.tree_map(reshape_states, env_states)
    timesteps = jax.tree_util.tree_map(reshape_states, timesteps)

    if config.logger.checkpointing.load_model:
        loaded_checkpoint = Checkpointer(
            model_name=config.system.system_name,
            **config.logger.checkpointing.load_args,
        )
        restored_params, _ = loaded_checkpoint.restore_params(input_params=params)
        params = restored_params

    key, step_key = jax.random.split(key)
    step_keys = jax.random.split(step_key, n_devices * config.arch.update_batch_size)
    reshape_keys = lambda x: x.reshape((n_devices, config.arch.update_batch_size) + x.shape[1:])
    step_keys = reshape_keys(jnp.stack(step_keys))
    opt_states = ActorCriticOptStates(actor_opt_state, critic_opt_state)
    replicate_learner = (params, opt_states)

    broadcast = lambda x: jnp.broadcast_to(x, (config.arch.update_batch_size,) + x.shape)
    replicate_learner = jax.tree_util.tree_map(broadcast, replicate_learner)
    replicate_learner = flax.jax_utils.replicate(replicate_learner, devices=jax.devices())

    # running_cum_compute_time/running_discounted_return/episode_discounted_return
    # start at 0 for every env, shaped like env_states/timesteps' leading dims.
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

    n_devices = len(jax.devices())
    config.num_devices = n_devices
    config = check_total_timesteps(config)
    assert (
        config.arch.num_updates >= config.arch.num_evaluation
    ), "Number of updates per evaluation must be less than total number of updates."

    env, eval_env = environments.make(config=config)

    key, key_e, actor_net_key, critic_net_key = jax.random.split(
        jax.random.PRNGKey(config.arch.seed), num=4
    )

    learn, actor_network, learner_state = learner_setup(
        env, (key, actor_net_key, critic_net_key), config
    )

    evaluator, absolute_metric_evaluator, (trained_params, eval_keys) = (
        evaluator_setup_with_compute_time(
            eval_env=eval_env,
            key_e=key_e,
            eval_act_fn=get_merged_action_act_fn_with_compute_time(config, actor_network.apply),
            params=learner_state.params.actor_params,
            config=config,
        )
    )

    steps_per_rollout = (
        n_devices
        * config.arch.num_updates_per_eval
        * config.system.rollout_length
        * config.arch.update_batch_size
        * config.arch.num_envs
    )

    logger = StoixLogger(config)
    logger.log_config(OmegaConf.to_container(config, resolve=True))
    print(f"{Fore.YELLOW}{Style.BRIGHT}JAX Global Devices {jax.devices()}{Style.RESET_ALL}")

    save_checkpoint = config.logger.checkpointing.save_model
    if save_checkpoint:
        checkpointer = Checkpointer(
            metadata=config,
            model_name=config.system.system_name,
            **config.logger.checkpointing.save_args,
        )

    max_episode_return = -jnp.inf
    best_params = unreplicate_batch_dim(learner_state.params.actor_params)
    for eval_step in range(config.arch.num_evaluation):
        start_time = time.time()

        learner_output = learn(learner_state)
        jax.block_until_ready(learner_output)

        elapsed_time = time.time() - start_time
        t = int(steps_per_rollout * (eval_step + 1))
        episode_metrics, ep_completed = get_final_step_metrics(learner_output.episode_metrics)
        episode_metrics["steps_per_second"] = steps_per_rollout / elapsed_time

        logger.log({"timestep": t}, t, eval_step, LogEvent.MISC)
        if ep_completed:
            logger.log(episode_metrics, t, eval_step, LogEvent.ACT)
        train_metrics = learner_output.train_metrics
        opt_steps_per_eval = config.arch.num_updates_per_eval * (
            config.system.epochs * config.system.num_minibatches
        )
        train_metrics["steps_per_second"] = opt_steps_per_eval / elapsed_time
        logger.log(train_metrics, t, eval_step, LogEvent.TRAIN)

        start_time = time.time()
        trained_params = unreplicate_batch_dim(learner_output.learner_state.params.actor_params)
        key_e, *eval_keys = jax.random.split(key_e, n_devices + 1)
        eval_keys = jnp.stack(eval_keys)
        eval_keys = eval_keys.reshape(n_devices, -1)

        evaluator_output = evaluator(trained_params, eval_keys)
        jax.block_until_ready(evaluator_output)

        elapsed_time = time.time() - start_time
        episode_return = jnp.mean(evaluator_output.episode_metrics["episode_return"])

        steps_per_eval = int(jnp.sum(evaluator_output.episode_metrics["episode_length"]))
        evaluator_output.episode_metrics["steps_per_second"] = steps_per_eval / elapsed_time
        logger.log(evaluator_output.episode_metrics, t, eval_step, LogEvent.EVAL)

        if save_checkpoint:
            checkpointer.save(
                timestep=int(steps_per_rollout * (eval_step + 1)),
                unreplicated_learner_state=unreplicate_n_dims(learner_output.learner_state),
                episode_return=episode_return,
            )

        if config.arch.absolute_metric and max_episode_return <= episode_return:
            best_params = copy.deepcopy(trained_params)
            max_episode_return = episode_return

        learner_state = learner_output.learner_state

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

    logger.stop()
    eval_performance = float(jnp.mean(evaluator_output.episode_metrics[config.env.eval_metric]))
    return eval_performance


@hydra.main(
    config_path="../../configs/default/anakin",
    config_name="default_ramdp_ff_ppo_explicit_cot.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    OmegaConf.set_struct(cfg, False)

    eval_performance = run_experiment(cfg)

    print(
        f"{Fore.CYAN}{Style.BRIGHT}Compute-time-aware PPO with merged-action explicit CoT "
        f"(RAMDP-PPO, merged-action explicit CoT) experiment completed{Style.RESET_ALL}"
    )
    return eval_performance


if __name__ == "__main__":
    hydra_entry_point()
