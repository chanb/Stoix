"""Compute-time-aware Q actor-critic (RAMDP-QAC).

This is a variant of `stoix.systems.ramdp_vpg.ff_reinforce` where the
advantage used to train the actor comes from a learned Q-function instead of
an n-step Monte-Carlo return. `c` (`compute_time`) is treated as part of what
the policy outputs, exactly like the environment action `a`, so the critic
(`stoix.networks.base_qac.ValueAndQCritic`) has two heads sharing one torso:

  - a state-value head V(s), trained by ordinary regression to the same
    n-step, compute-discounted return target as `ff_reinforce.py`'s critic
    (`G_h = gamma^(C_h - 1) * (r_h + gamma * G_{h+1})`, see that file for the
    full derivation) - this is *not* derived from Q, since deriving a proper
    V(s) would need marginalising Q over the actor's full joint (a, c)
    distribution, which needs the torso to expose its per-step halting
    probabilities;
  - a state-action-value head Q(s, a, c), read off at the realised action and
    compute time. The advantage is then

        A(s_t, a_t, c_t) = Q(s_t, a_t, c_t) - V(s_t)

    with no n-step return or bootstrapping in the advantage itself - only V's
    own regression target ever needs bootstrapping.

`config.system.qac_variant` selects how Q(s, a, c) is obtained:

  - "naive": Q is a genuinely learned `(num_actions, max_steps)` table,
    regressed directly to the same n-step return G used for V.
  - "fac" (runtime-factorized, the default): since compute time enters the
    RAMDP return *only* through the `gamma^(c-1)` prefactor (see the
    derivation above), Q(s,a,c) = gamma^(c-1) * Q(s,a,1) is not just a
    convenient approximation but the value implied by that same structure.
    So Q only has to learn the much smaller `(num_actions,)` table Q(s,·,1),
    regressed to B_h = G_h / gamma^(C_h - 1) - i.e. the same return target
    with its own compute-time scaling divided back out, so that
    gamma^(c-1) * B_h recovers G_h exactly.

The actor itself is unchanged from `ff_reinforce.py`: an Adaptive
Computation Time torso whose sampled halting trajectory is trained via the
score-function estimator, jointly with the environment action, from this
Q-based advantage.

This file intentionally duplicates most of `ff_reinforce.py` rather than
modifying it, so the existing REINFORCE-with-baseline RAMDP-VPG system is
left untouched. It reuses `ff_reinforce.py`'s evaluation act_fn as-is, since
evaluation only ever needs the actor.
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
    OnPolicyLearnerState,
)
from stoix.networks.base_compute import FeedForwardActorWithComputeTime as Actor
from stoix.networks.base_qac import ValueAndQCritic as Critic
from stoix.systems.ramdp_vpg.evaluator import evaluator_setup_with_compute_time
from stoix.systems.ramdp_vpg.ff_reinforce import get_distribution_act_fn_with_compute_time
from stoix.systems.ramdp_vpg.qac_types import Transition
from stoix.utils import make_env as environments
from stoix.utils.checkpointing import Checkpointer
from stoix.utils.jax_utils import unreplicate_batch_dim, unreplicate_n_dims
from stoix.utils.logger import LogEvent, StoixLogger
from stoix.utils.multistep import batch_discounted_returns
from stoix.utils.total_timestep_checker import check_total_timesteps
from stoix.utils.training import make_learning_rate


def get_learner_fn(
    env: Environment,
    apply_fns: Tuple[ActorApply, CriticApply],
    update_fns: Tuple[optax.TransformUpdateFn, optax.TransformUpdateFn],
    config: DictConfig,
) -> LearnerFn[OnPolicyLearnerState]:
    """Get the learner function."""

    # Get apply and update functions for actor and critic networks.
    actor_apply_fn, critic_apply_fn = apply_fns
    actor_update_fn, critic_update_fn = update_fns

    qac_variant = config.system.qac_variant
    assert qac_variant in ("naive", "fac"), f"Unknown qac_variant: {qac_variant}"

    def _q_at_action_and_compute_time(
        q_output: chex.Array, action: chex.Array, compute_time: chex.Array
    ) -> chex.Array:
        """Read Q(s, a, c) off the critic's raw `q_value` output, however it's
        parameterised:

          - "naive": `q_output` is a `(..., num_actions, max_steps)` table -
            index by both `action` and `compute_time`.
          - "fac": `q_output` is `(..., num_actions)`, representing Q(s,·,1) -
            index by `action` only, then scale by `gamma ** (compute_time - 1)`
            to recover Q(s,a,c), per the runtime-factorization identity.
        """
        if qac_variant == "naive":
            compute_time_idx = (compute_time - 1).astype(jnp.int32)
            q_at_c = jnp.take_along_axis(
                q_output, compute_time_idx[..., jnp.newaxis, jnp.newaxis], axis=-1
            ).squeeze(-1)
            return jnp.take_along_axis(q_at_c, action[..., jnp.newaxis], axis=-1).squeeze(-1)
        else:  # "fac"
            q1_sa = jnp.take_along_axis(q_output, action[..., jnp.newaxis], axis=-1).squeeze(-1)
            return config.system.gamma ** (compute_time - 1) * q1_sa

    def _update_step(
        learner_state: OnPolicyLearnerState, _: Any
    ) -> Tuple[OnPolicyLearnerState, Tuple]:
        def _env_step(
            learner_state: OnPolicyLearnerState, _: Any
        ) -> Tuple[OnPolicyLearnerState, Transition]:
            """Step the environment."""
            params, opt_states, key, env_state, last_timestep = learner_state

            # SELECT ACTION
            key, policy_key, halting_key = jax.random.split(key, 3)
            actor_policy, compute_time = actor_apply_fn(
                params.actor_params,
                last_timestep.observation,
                torso_kwargs={"rng": halting_key},
            )
            value = critic_apply_fn(
                params.critic_params, last_timestep.observation, method="value"
            )
            q_output = critic_apply_fn(
                params.critic_params, last_timestep.observation, method="q_value"
            )
            action = actor_policy.sample(seed=policy_key)
            q_value = _q_at_action_and_compute_time(q_output, action, compute_time)

            # STEP ENVIRONMENT
            env_state, timestep = env.step(env_state, action)

            # LOG EPISODE METRICS
            done = timestep.last().reshape(-1)
            info = timestep.extras["episode_metrics"]

            transition = Transition(
                done,
                action,
                value,
                q_value,
                timestep.reward,
                last_timestep.observation,
                info,
                compute_time,
            )
            learner_state = OnPolicyLearnerState(params, opt_states, key, env_state, timestep)
            return learner_state, transition

        # STEP ENVIRONMENT FOR ROLLOUT LENGTH
        learner_state, traj_batch = jax.lax.scan(
            _env_step, learner_state, None, config.system.rollout_length
        )

        # BOOTSTRAP V(s) AT THE TRUNCATION BOUNDARY, exactly as in `ff_reinforce.py`.
        params, opt_states, key, env_state, last_timestep = learner_state
        last_val = critic_apply_fn(params.critic_params, last_timestep.observation, method="value")

        # Swap the batch and time axes.
        traj_batch = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), traj_batch)

        # ADVANTAGE: directly Q(s_t, a_t, c_t) - V(s_t), no n-step/Monte-Carlo
        # return involved at all.
        advantage = traj_batch.q_value - traj_batch.value

        # RETURN TARGET for V (and, for "naive", for Q too): the same n-step,
        # compute-discounted return as `ff_reinforce.py`'s critic - see its
        # module docstring for the full derivation of
        # `G_h = gamma^(C_h - 1) * (r_h + gamma * G_{h+1})`. This target is
        # used purely to train the critic; the advantage above never touches it.
        compute_time = traj_batch.compute_time
        r_t = traj_batch.reward * config.system.gamma ** (compute_time - 1)
        v_t = jnp.concatenate([traj_batch.value, last_val[..., jnp.newaxis]], axis=-1)[:, 1:]
        not_done = 1.0 - traj_batch.done.astype(jnp.float32)
        d_t = (not_done * config.system.gamma**compute_time).astype(jnp.float32)
        g_targets = batch_discounted_returns(r_t, d_t, v_t, True, False)

        def _actor_loss_fn(
            actor_params: FrozenDict,
            observations: chex.Array,
            actions: chex.Array,
            advantage: chex.Array,
            compute_times: chex.Array,
        ) -> Tuple:
            """Calculate the actor loss."""
            # RERUN NETWORK. Replay (rather than re-sample) the halting trajectory
            # that was actually taken during rollout (`compute_times`, from the
            # stored transition), mirroring how `log_prob(actions)` below evaluates
            # the stored action rather than drawing a fresh sample.
            actor_policy, halting_log_prob = actor_apply_fn(
                actor_params, observations, torso_kwargs={"target_compute_time": compute_times}
            )
            env_log_prob = actor_policy.log_prob(actions)
            # The halting decisions and the environment action they led to are
            # trained jointly, from the same Q-based advantage.
            log_prob = env_log_prob + halting_log_prob
            # CALCULATE ACTOR LOSS
            loss_actor = -advantage * log_prob
            entropy = actor_policy.entropy().mean()

            total_loss_actor = loss_actor.mean() - config.system.ent_coef * entropy
            loss_info = {
                "actor_loss": loss_actor,
                "entropy": entropy,
                "compute_time": compute_times,
            }
            return total_loss_actor, loss_info

        def _critic_loss_fn(
            critic_params: FrozenDict,
            observations: chex.Array,
            actions: chex.Array,
            compute_times: chex.Array,
            targets: chex.Array,
        ) -> Tuple:
            """Calculate the critic loss: V regressed to the return target, and
            Q regressed to that same target - divided back down to a `c=1`
            equivalent target for the "fac" variant, since Q there only
            predicts Q(s,·,1)."""
            # RERUN NETWORK
            value = critic_apply_fn(critic_params, observations, method="value")
            value_loss = rlax.l2_loss(value, targets).mean()

            q_output = critic_apply_fn(critic_params, observations, method="q_value")
            if qac_variant == "naive":
                q_pred = _q_at_action_and_compute_time(q_output, actions, compute_times)
                q_targets = targets
            else:  # "fac"
                q_pred = jnp.take_along_axis(
                    q_output, actions[..., jnp.newaxis], axis=-1
                ).squeeze(-1)
                q_targets = targets / config.system.gamma ** (compute_times - 1)
            q_loss = rlax.l2_loss(q_pred, q_targets).mean()

            critic_total_loss = config.system.vf_coef * (value_loss + q_loss)
            loss_info = {
                "value_loss": value_loss,
                "q_loss": q_loss,
            }
            return critic_total_loss, loss_info

        # CALCULATE ACTOR LOSS
        actor_grad_fn = jax.grad(_actor_loss_fn, has_aux=True)
        actor_grads, actor_loss_info = actor_grad_fn(
            params.actor_params,
            traj_batch.obs,
            traj_batch.action,
            advantage,
            traj_batch.compute_time,
        )

        # CALCULATE CRITIC LOSS
        critic_grad_fn = jax.grad(_critic_loss_fn, has_aux=True)
        critic_grads, critic_loss_info = critic_grad_fn(
            params.critic_params,
            traj_batch.obs,
            traj_batch.action,
            traj_batch.compute_time,
            g_targets,
        )

        # Compute the parallel mean (pmean) over the batch.
        # This calculation is inspired by the Anakin architecture demo notebook.
        # available at https://tinyurl.com/26tdzs5x
        # This pmean could be a regular mean as the batch axis is on the same device.
        actor_grads, actor_loss_info = jax.lax.pmean(
            (actor_grads, actor_loss_info), axis_name="batch"
        )
        # pmean over devices.
        actor_grads, actor_loss_info = jax.lax.pmean(
            (actor_grads, actor_loss_info), axis_name="device"
        )

        critic_grads, critic_loss_info = jax.lax.pmean(
            (critic_grads, critic_loss_info), axis_name="batch"
        )
        # pmean over devices.
        critic_grads, critic_loss_info = jax.lax.pmean(
            (critic_grads, critic_loss_info), axis_name="device"
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

        learner_state = OnPolicyLearnerState(
            new_params, new_opt_state, key, env_state, last_timestep
        )
        metric = traj_batch.info
        return learner_state, (metric, loss_info)

    def learner_fn(
        learner_state: OnPolicyLearnerState,
    ) -> AnakinExperimentOutput[OnPolicyLearnerState]:

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
) -> Tuple[LearnerFn[OnPolicyLearnerState], Actor, OnPolicyLearnerState]:
    """Initialise learner_fn, network, optimiser, environment and states."""
    # Get available TPU cores.
    n_devices = len(jax.devices())

    # Get number/dimension of actions.
    num_actions = int(env.action_space().num_values)
    config.system.action_dim = num_actions

    # PRNG keys.
    key, actor_net_key, critic_net_key = keys

    # Define network and optimiser.
    actor_torso = hydra.utils.instantiate(config.network.actor_network.pre_torso)
    actor_action_head = hydra.utils.instantiate(
        config.network.actor_network.action_head, action_dim=num_actions
    )
    critic_torso = hydra.utils.instantiate(config.network.critic_network.pre_torso)
    value_head = hydra.utils.instantiate(config.network.critic_network.value_head)
    # The Q head's output shape depends on `qac_variant` (see module docstring):
    # "naive" learns a full (action, compute_time) table, so needs `max_steps`
    # (read off the actor's compute-aware torso config, which bounds the range
    # of compute times the actor can ever realise); "fac" only ever needs
    # Q(s,·,1), one value per action.
    qac_variant = config.system.qac_variant
    assert qac_variant in ("naive", "fac"), f"Unknown qac_variant: {qac_variant}"
    if qac_variant == "naive":
        max_steps = config.network.actor_network.pre_torso.max_steps
        q_head = hydra.utils.instantiate(
            config.network.critic_network.q_head,
            output_dim=max_steps,
            pre_shape=(num_actions,),
        )
    else:  # "fac"
        q_head = hydra.utils.instantiate(
            config.network.critic_network.q_head, output_dim=num_actions
        )

    actor_network = Actor(torso=actor_torso, action_head=actor_action_head)
    critic_network = Critic(torso=critic_torso, value_head=value_head, q_head=q_head)

    actor_lr = make_learning_rate(config.system.actor_lr, config, 1, 1)
    critic_lr = make_learning_rate(config.system.critic_lr, config, 1, 1)

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

    # Initialise critic params and optimiser state.
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

    # Initialise learner state.
    params, opt_states = replicate_learner
    init_learner_state = OnPolicyLearnerState(params, opt_states, step_keys, env_states, timesteps)

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

    # Calculate number of updates per evaluation.
    config.arch.num_updates_per_eval = config.arch.num_updates // config.arch.num_evaluation
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
        opt_steps_per_eval = config.arch.num_updates_per_eval
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
    config_name="default_ramdp_ff_qac.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    # Allow dynamic attributes.
    OmegaConf.set_struct(cfg, False)

    # Run experiment.
    eval_performance = run_experiment(cfg)

    print(
        f"{Fore.CYAN}{Style.BRIGHT}Compute-time-aware Q actor-critic (RAMDP-QAC) experiment "
        f"completed{Style.RESET_ALL}"
    )
    return eval_performance


if __name__ == "__main__":
    hydra_entry_point()
