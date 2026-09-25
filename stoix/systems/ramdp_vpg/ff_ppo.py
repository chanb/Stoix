"""Compute-time-aware PPO (RAMDP-PPO).

The actor's torso is an Adaptive Computation Time torso (see
`stoix.networks.torso_compute`) that samples a "halt now?" decision at each
pondering step, so `compute_time` is the sampled number of steps taken. The
halting trajectory is trained alongside the environment action, each with its
own per-decision PPO ratio/clip.

Compute cost is folded into the discounting rather than the reward: a
transition that took `C_h` pondering steps is treated as `C_h` elapsed time
steps, so the return recursion is

    G_h = gamma^(C_h - 1) * (r_h + gamma * G_{h+1})

The advantage is G - V, with `G` computed via GAE(lambda)
(`config.system.gae_lambda`) and a plain V-only critic.
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
from stoix.systems.ramdp_vpg.evaluator import ComputeAwareActFn, evaluator_setup_with_compute_time
from stoix.systems.ramdp_vpg.ppo_types import PPOTransition
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
from stoix.utils.loss import (
    clipped_value_loss,
    dpo_loss,
    dpo_surrogate,
    expectile_loss,
    ppo_clip_loss,
)
from stoix.utils.multistep import batch_truncated_generalized_advantage_estimation
from stoix.utils.total_timestep_checker import check_total_timesteps
from stoix.utils.training import make_learning_rate


def get_distribution_act_fn_with_compute_time(
    config: DictConfig,
    actor_apply: ActorApply,
) -> ComputeAwareActFn:
    """Like `stoix.evaluator.get_distribution_act_fn`, but for actor networks
    whose torso samples a halting trajectory and so returns
    `(action_distribution, compute_time, first_convergence_step,
    num_close_steps)` instead of just the action distribution."""

    def act_fn(
        params: FrozenDict, observation: chex.Array, key: chex.PRNGKey
    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
        if config.arch.evaluation_greedy:
            pi, compute_time, first_convergence_step, num_close_steps = actor_apply(
                params, observation, torso_kwargs={"deterministic": True}
            )
            action = pi.mode()
        else:
            halting_key, action_key = jax.random.split(key)
            pi, compute_time, first_convergence_step, num_close_steps = actor_apply(
                params, observation, torso_kwargs={"rng": halting_key}
            )
            action = pi.sample(seed=action_key)
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

    # Needed to build the "was this pondering step actually taken" mask used
    # by the optional latent KL penalty below - see `_actor_loss_fn`.
    max_steps = config.network.actor_network.pre_torso.max_steps

    def _value_loss_fn(
        pred: chex.Array, behavior: chex.Array, targets: chex.Array, use_expectile: bool = False
    ) -> chex.Array:
        """PPO's clipped value loss against `behavior`, plain L2 to
        `targets`, or (when `use_expectile=True`, see
        `config.system.use_expectile_value_loss` and the module docstring)
        expectile regression to `targets`
        (`stoix.utils.loss.expectile_loss`) at `config.system.expectile`."""
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
        ) -> Tuple[RamdpOnPolicyLearnerState, PPOTransition]:
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

            key, policy_key, halting_key = jax.random.split(key, 3)
            actor_policy, compute_time, first_convergence_step, num_close_steps = actor_apply_fn(
                params.actor_params,
                last_timestep.observation,
                torso_kwargs={"rng": halting_key},
            )
            action = actor_policy.sample(seed=policy_key)
            env_log_prob = actor_policy.log_prob(action)

            # Second replay-mode pass to get halting_log_prob at these same (rollout-time) params,
            # giving PPO a fixed "old" per-step log_prob (the summed scalar is discarded - PPO clips
            # each step individually, see module docstring).
            _, _, old_latent_states, halting_log_prob, _ = actor_apply_fn(
                params.actor_params,
                last_timestep.observation,
                torso_kwargs={"target_compute_time": compute_time},
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

            transition = PPOTransition(
                done,
                action,
                value,
                timestep.reward,
                last_timestep.observation,
                info,
                compute_time,
                first_convergence_step,
                num_close_steps,
                env_log_prob,
                halting_log_prob,
                old_latent_states,
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
        # (see module docstring).
        compute_time = traj_batch.compute_time
        r_t = traj_batch.reward * config.system.gamma ** (compute_time - 1)
        v_t = jnp.concatenate([traj_batch.value, last_val[..., jnp.newaxis]], axis=-1)[:, 1:]
        not_done = 1.0 - traj_batch.done.astype(jnp.float32)
        d_t = (not_done * config.system.gamma**compute_time).astype(jnp.float32)
        _, g_targets = batch_truncated_generalized_advantage_estimation(
            r_t, d_t, config.system.gae_lambda, v_tm1=traj_batch.value, v_t=v_t,
            stop_target_gradients=True,
        )

        advantages = g_targets - traj_batch.value  # G - V
        targets = g_targets

        if config.system.standardize_advantages:
            advantages = jax.nn.standardize(advantages, axis=(0, 1))

        # --- config.system.critic_before_actor machinery (see module docstring) -
        # `_actor_loss_fn`/`_critic_loss_fn` below are exact copies of the ones nested inside the
        # joint `_update_minibatch` further down, hoisted to this scope so both the sequential path
        # here and the joint path below can each use their own copy without the two paths depending
        # on one another.

        def _actor_loss_fn(
            actor_params: FrozenDict,
            traj_batch: PPOTransition,
            advantage: chex.Array,
        ) -> Tuple:
            """Calculate the actor loss (see the identical copy nested in
            the joint `_update_minibatch` below for the full explanation)."""
            (
                actor_policy,
                _,
                new_latent_states,
                halting_log_prob,
                per_step_halting_entropy,
            ) = actor_apply_fn(
                actor_params,
                traj_batch.obs,
                torso_kwargs={"target_compute_time": traj_batch.compute_time},
            )
            env_log_prob = actor_policy.log_prob(traj_batch.action)

            if config.system.use_dpo_loss:
                action_loss = dpo_loss(
                    env_log_prob,
                    traj_batch.env_log_prob,
                    advantage,
                    config.system.dpo_alpha,
                    config.system.dpo_beta,
                )
            else:
                action_loss = ppo_clip_loss(
                    env_log_prob, traj_batch.env_log_prob, advantage, config.system.clip_eps
                )
            action_ratio = jnp.exp(env_log_prob - traj_batch.env_log_prob)
            action_clip_fraction = jnp.mean(
                (jnp.abs(action_ratio - 1.0) > config.system.clip_eps).astype(jnp.float32)
            )
            entropy = actor_policy.entropy().mean()

            step_idx = jnp.arange(max_steps)
            valid_step = (step_idx < traj_batch.compute_time[..., None]).astype(jnp.float32)
            num_valid_steps = jnp.maximum(jnp.sum(valid_step), 1.0)

            halting_ratio = jnp.exp(halting_log_prob - traj_batch.halting_log_prob)
            advantage_per_step = advantage[..., None]
            if config.system.use_dpo_loss:
                halting_per_step_loss = dpo_surrogate(
                    halting_log_prob,
                    traj_batch.halting_log_prob,
                    advantage_per_step,
                    config.system.dpo_alpha,
                    config.system.dpo_beta,
                )
            else:
                halting_surrogate1 = halting_ratio * advantage_per_step
                halting_surrogate2 = (
                    jnp.clip(
                        halting_ratio, 1.0 - config.system.clip_eps, 1.0 + config.system.clip_eps
                    )
                    * advantage_per_step
                )
                halting_per_step_loss = -jnp.minimum(halting_surrogate1, halting_surrogate2)
            halting_loss = jnp.sum(halting_per_step_loss * valid_step) / num_valid_steps
            halting_clip_fraction = (
                jnp.sum(
                    (jnp.abs(halting_ratio - 1.0) > config.system.clip_eps).astype(jnp.float32)
                    * valid_step
                )
                / num_valid_steps
            )
            # Entropy bonus on the halting decision itself, mirroring `entropy` above for the
            # environment action - `ent_coef` alone never reaches the halting head
            # (`actor_policy.entropy()` is only the environment action's distribution), so without
            # this nothing keeps the halting policy from collapsing to a degenerate, non-adaptive
            # compute-time before it discovers any genuine per-example structure.
            halting_entropy = jnp.sum(per_step_halting_entropy * valid_step) / num_valid_steps

            loss_actor = action_loss + halting_loss

            sq_dist = jnp.sum(
                (new_latent_states - traj_batch.old_latent_states) ** 2, axis=-1
            )
            latent_kl_penalty = jnp.sum(sq_dist * valid_step) / num_valid_steps

            total_loss_actor = (
                loss_actor
                - config.system.ent_coef * entropy
                - config.system.halting_ent_coef * halting_entropy
                + config.system.latent_kl_coef * latent_kl_penalty
            )
            loss_info = {
                "actor_loss": loss_actor,
                "action_loss": action_loss,
                "halting_loss": halting_loss,
                "entropy": entropy,
                "halting_entropy": halting_entropy,
                "advantages": advantage,
                "compute_time": traj_batch.compute_time,
                "first_convergence_step": traj_batch.first_convergence_step,
                "num_close_steps": traj_batch.num_close_steps,
                "action_clip_fraction": action_clip_fraction,
                "halting_clip_fraction": halting_clip_fraction,
                "latent_kl_penalty": latent_kl_penalty,
            }
            return total_loss_actor, loss_info

        def _critic_loss_fn(
            critic_params: FrozenDict,
            traj_batch: PPOTransition,
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
            traj_batch: PPOTransition,
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
            traj_batch: PPOTransition,
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
            """Recompute `targets`/`advantages` from `critic_params`, mirroring the joint path's
            `recompute_advantages` refresh further down."""
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
                # Mirrors the joint path's per-epoch refresh further down -
                # only `targets` actually feeds the critic loss here;
                # `advantages` is unused until the actor-only phase but
                # cheap/harmless to keep refreshed too, for consistency.
                epoch_targets, _ = _refresh_targets_and_advantages(params.critic_params)

            update_state = (params, opt_states, epoch_traj_batch, epoch_targets, key)
            return update_state, loss_info

        def _update_epoch_actor_only(update_state: Tuple, _: Any) -> Tuple:
            """`critic_before_actor`'s actor-only epoch: one full shuffled pass over the
            rollout's minibatches, actor params only."""
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
                    traj_batch: PPOTransition,
                    advantage: chex.Array,
                ) -> Tuple:
                    """Calculate the actor loss."""
                    # Replay the halting trajectory actually taken during rollout, mirroring
                    # log_prob(traj_batch.action) below.
                    (
                        actor_policy,
                        _,
                        new_latent_states,
                        halting_log_prob,
                        per_step_halting_entropy,
                    ) = actor_apply_fn(
                        actor_params,
                        traj_batch.obs,
                        torso_kwargs={"target_compute_time": traj_batch.compute_time},
                    )
                    env_log_prob = actor_policy.log_prob(traj_batch.action)

                    if config.system.use_dpo_loss:
                        action_loss = dpo_loss(
                            env_log_prob,
                            traj_batch.env_log_prob,
                            advantage,
                            config.system.dpo_alpha,
                            config.system.dpo_beta,
                        )
                    else:
                        action_loss = ppo_clip_loss(
                            env_log_prob, traj_batch.env_log_prob, advantage, config.system.clip_eps
                        )
                    action_ratio = jnp.exp(env_log_prob - traj_batch.env_log_prob)
                    action_clip_fraction = jnp.mean(
                        (jnp.abs(action_ratio - 1.0) > config.system.clip_eps).astype(
                            jnp.float32
                        )
                    )
                    entropy = actor_policy.entropy().mean()

                    # `halting_log_prob`/`traj_batch.halting_log_prob`: `(*batch, max_steps)`,
                    # zeroed at forced steps or past the step each example actually halted at.
                    step_idx = jnp.arange(max_steps)
                    valid_step = (step_idx < traj_batch.compute_time[..., None]).astype(
                        jnp.float32
                    )
                    num_valid_steps = jnp.maximum(jnp.sum(valid_step), 1.0)

                    halting_ratio = jnp.exp(halting_log_prob - traj_batch.halting_log_prob)
                    advantage_per_step = advantage[..., None]
                    if config.system.use_dpo_loss:
                        halting_per_step_loss = dpo_surrogate(
                            halting_log_prob,
                            traj_batch.halting_log_prob,
                            advantage_per_step,
                            config.system.dpo_alpha,
                            config.system.dpo_beta,
                        )
                    else:
                        halting_surrogate1 = halting_ratio * advantage_per_step
                        halting_surrogate2 = (
                            jnp.clip(
                                halting_ratio,
                                1.0 - config.system.clip_eps,
                                1.0 + config.system.clip_eps,
                            )
                            * advantage_per_step
                        )
                        halting_per_step_loss = -jnp.minimum(
                            halting_surrogate1, halting_surrogate2
                        )
                    # Masked mean over every pondering step actually taken
                    # across the whole minibatch (not a per-example mean
                    # averaged over examples), so trajectories with more
                    # valid steps don't get down-weighted relative to
                    # shorter ones.
                    halting_loss = jnp.sum(halting_per_step_loss * valid_step) / num_valid_steps
                    halting_clip_fraction = (
                        jnp.sum(
                            (jnp.abs(halting_ratio - 1.0) > config.system.clip_eps).astype(
                                jnp.float32
                            )
                            * valid_step
                        )
                        / num_valid_steps
                    )
                    # Entropy bonus on the halting decision itself, mirroring `entropy` above for
                    # the environment action - `ent_coef` alone never reaches the halting head
                    # (`actor_policy.entropy()` is only the environment action's distribution), so
                    # without this nothing keeps the halting policy from collapsing to a degenerate,
                    # non-adaptive compute-time before it discovers any genuine per-example
                    # structure.
                    halting_entropy = (
                        jnp.sum(per_step_halting_entropy * valid_step) / num_valid_steps
                    )

                    loss_actor = action_loss + halting_loss

                    # Optional latent trust-region penalty (see module docstring): KL(N(new, sigma^2
                    # I) || N(old, sigma^2 I)) = ||new - old||^2 / (2 sigma^2), i.e. a plain L2
                    # penalty with latent_kl_coef playing the role of 1/(2 sigma^2).
                    sq_dist = jnp.sum(
                        (new_latent_states - traj_batch.old_latent_states) ** 2, axis=-1
                    )
                    latent_kl_penalty = jnp.sum(sq_dist * valid_step) / num_valid_steps

                    total_loss_actor = (
                        loss_actor
                        - config.system.ent_coef * entropy
                        - config.system.halting_ent_coef * halting_entropy
                        + config.system.latent_kl_coef * latent_kl_penalty
                    )
                    loss_info = {
                        "actor_loss": loss_actor,
                        "action_loss": action_loss,
                        "halting_loss": halting_loss,
                        "entropy": entropy,
                        "halting_entropy": halting_entropy,
                        "advantages": advantage,
                        "compute_time": traj_batch.compute_time,
                        "first_convergence_step": traj_batch.first_convergence_step,
                        "num_close_steps": traj_batch.num_close_steps,
                        "action_clip_fraction": action_clip_fraction,
                        "halting_clip_fraction": halting_clip_fraction,
                        "latent_kl_penalty": latent_kl_penalty,
                    }
                    return total_loss_actor, loss_info

                def _critic_loss_fn(
                    critic_params: FrozenDict,
                    traj_batch: PPOTransition,
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

                # Norm of the gradient actually applied by the optimizer,
                # i.e. after averaging but before clipping.
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
                # `targets` (the n-step return that trains the critic in `_critic_loss_fn`)
                # bootstraps off the value estimate at the tail of each n-step window (`v_t` below,
                # mirroring the pre-epoch-loop computation above) - holding it fixed at its
                # rollout-time bootstrap across every epoch would leave the critic regressing
                # towards a target built from an increasingly stale value function as the critic
                # itself moves epoch to epoch.
                new_value = critic_apply_fn(params.critic_params, traj_batch.obs)
                new_last_val = critic_apply_fn(params.critic_params, last_timestep.observation)

                v_t = jnp.concatenate(
                    [new_value, new_last_val[..., jnp.newaxis]], axis=-1
                )[:, 1:]
                _, targets = batch_truncated_generalized_advantage_estimation(
                    r_t, d_t, config.system.gae_lambda, v_tm1=new_value, v_t=v_t,
                    stop_target_gradients=True,
                )

                # Both terms of G - V are this epoch's fresh critic output:
                # `new_value` for V, and `targets` above (itself bootstrapped
                # off `new_value`/`new_last_val`) for G.
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
            # Phase 1: `epochs` epochs of critic-only updates (see module docstring).
            critic_update_state = (params, opt_states, traj_batch, targets, key)
            critic_update_state, critic_loss_info = jax.lax.scan(
                _update_epoch_critic_only, critic_update_state, None, config.system.epochs
            )
            params, opt_states, traj_batch, targets, key = critic_update_state

            # Unconditional refresh at the phase boundary, regardless of `recompute_advantages`: the
            # actor-only phase must see an advantage informed by the now-trained critic, not the
            # stale rollout-time one - otherwise this option would do nothing for the actor.
            targets, advantages = _refresh_targets_and_advantages(params.critic_params)

            # Phase 2: `epochs` epochs of actor-only updates against that
            # fixed advantage - the critic no longer moves, so there is
            # nothing to recompute epoch-to-epoch here.
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


def _label_actor_params_by_halting_head(params: Any) -> Any:
    """Labels every actor param leaf "halting_head" if it sits inside a submodule named
    `HaltingHead*` (currently only `stoix.networks.torso_ compute_transformer.HaltingHead`,
    used by `TransformerChainOfThoughtTorso`), or "default" otherwise - for
    `optax.multi_transform` to give the halting head's own weights a different
    gradient-clipping and/or learning-rate/ weight-decay treatment than the rest of the actor
    (see `config.system.clip_halting_head`/`halting_lr`/`halting_weight_decay` in
    `learner_setup`)."""

    def _label(path: Tuple[Any, ...], _leaf: Any) -> str:
        keys = (getattr(entry, "key", None) for entry in path)
        if any(isinstance(key, str) and key.startswith("HaltingHead") for key in keys):
            return "halting_head"
        return "default"

    labels = jax.tree_util.tree_map_with_path(_label, params)
    if not any(label == "halting_head" for label in jax.tree_util.tree_leaves(labels)):
        raise ValueError(
            "config.system.clip_halting_head=False (and/or halting_lr/halting_weight_decay "
            "were set) but no HaltingHead parameters were found in the actor - this "
            "architecture's torso doesn't expose a separately-named halting head to single "
            "out in the first place (only the transformer implicit-CoT torso, "
            "TransformerChainOfThoughtTorso, currently does)."
        )
    return labels


def learner_setup(
    env: Environment, keys: chex.Array, config: DictConfig
) -> Tuple[LearnerFn[RamdpOnPolicyLearnerState], Actor, RamdpOnPolicyLearnerState]:
    """Initialise learner_fn, network, optimiser, environment and states."""
    n_devices = len(jax.devices())

    num_actions = int(env.action_space().num_values)
    config.system.action_dim = num_actions

    key, actor_net_key, critic_net_key = keys

    actor_torso = hydra.utils.instantiate(config.network.actor_network.pre_torso)
    actor_action_head = hydra.utils.instantiate(
        config.network.actor_network.action_head, action_dim=num_actions
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

    actor_network = Actor(torso=actor_torso, action_head=actor_action_head, **actor_kwargs)

    critic_head = hydra.utils.instantiate(config.network.critic_network.critic_head)
    critic_network = FeedForwardCritic(torso=critic_torso, critic_head=critic_head, **critic_kwargs)

    actor_lr = make_learning_rate(
        config.system.actor_lr, config, config.system.epochs, config.system.num_minibatches
    )
    critic_lr = make_learning_rate(
        config.system.critic_lr, config, config.system.epochs, config.system.num_minibatches
    )

    # `halting_lr`/`halting_weight_decay` (both None by default) let the halting
    # head train with its own learning rate/weight decay instead of inheriting
    # the rest of the actor's - None falls back to actor_lr/actor_weight_decay
    # exactly, recovering the original behaviour.
    halting_lr_cfg = config.system.get("halting_lr", None)
    halting_weight_decay_cfg = config.system.get("halting_weight_decay", None)
    needs_halting_split = (
        not config.system.clip_halting_head
        or halting_lr_cfg is not None
        or halting_weight_decay_cfg is not None
    )

    if not needs_halting_split:
        actor_optim = optax.chain(
            optax.clip_by_global_norm(config.system.max_grad_norm),
            optax.adamw(actor_lr, eps=1e-5, weight_decay=config.system.actor_weight_decay),
        )
    else:
        halting_lr = (
            actor_lr
            if halting_lr_cfg is None
            else make_learning_rate(
                halting_lr_cfg, config, config.system.epochs, config.system.num_minibatches
            )
        )
        halting_weight_decay = (
            config.system.actor_weight_decay
            if halting_weight_decay_cfg is None
            else halting_weight_decay_cfg
        )
        # Each branch's clip_by_global_norm (when present) is computed only over
        # that branch's own leaves, not jointly over the whole actor - see
        # _label_actor_params_by_halting_head.
        halting_components = []
        if config.system.clip_halting_head:
            halting_components.append(optax.clip_by_global_norm(config.system.max_grad_norm))
        halting_components.append(
            optax.adamw(halting_lr, eps=1e-5, weight_decay=halting_weight_decay)
        )
        actor_optim = optax.multi_transform(
            {
                "default": optax.chain(
                    optax.clip_by_global_norm(config.system.max_grad_norm),
                    optax.adamw(
                        actor_lr, eps=1e-5, weight_decay=config.system.actor_weight_decay
                    ),
                ),
                "halting_head": optax.chain(*halting_components),
            },
            _label_actor_params_by_halting_head,
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
            eval_act_fn=get_distribution_act_fn_with_compute_time(config, actor_network.apply),
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
    config_name="default_ramdp_ff_ppo.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    OmegaConf.set_struct(cfg, False)

    eval_performance = run_experiment(cfg)

    print(
        f"{Fore.CYAN}{Style.BRIGHT}Compute-time-aware PPO (RAMDP-PPO) experiment "
        f"completed{Style.RESET_ALL}"
    )
    return eval_performance


if __name__ == "__main__":
    hydra_entry_point()