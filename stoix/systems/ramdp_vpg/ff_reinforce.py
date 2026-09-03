"""Compute-time-aware REINFORCE (RAMDP-VPG).

Variant of `stoix.systems.vpg.ff_reinforce` where the actor's compute time is
part of what's being optimised for (a "resource-augmented" MDP, RAMDP). The
actor's torso is an Adaptive Computation Time torso (see
`stoix.networks.torso_compute`): it repeatedly applies a shared computation
step, sampling a "halt now?" decision at each step, so `compute_time` is the
actual sampled number of steps taken, not a value predicted by a separate
head.

Halting is trained the same way REINFORCE trains the environment action: via
the score-function estimator, using the log-probability of the sampled
halting trajectory (`AdaptiveComputationTimeTorso`'s replay mode).

Compute cost is folded into the *discounting*, not the reward: a transition
at step `h` that took `C_h` pondering steps is treated as if `C_h` elementary
environment time steps had elapsed while the agent was thinking, so the
discounted return recursion becomes

    G_h = gamma^(C_h - 1) * (r_h + gamma * G_{h+1})

instead of the usual `G_h = r_h + gamma * G_{h+1}`. Both the environment
action and the halting decisions that produced it are trained from the
resulting advantage.

Optionally (`config.system.delightful`), the REINFORCE weight is gated by a
stop-gradient'd sigmoid of `advantage * surprisal / eta`, where
`surprisal = -log_prob`: this pushes reinforcement towards samples that were
both good *and* surprising. It never changes where gradient flows - only
through `log_prob`, same as without it.

This file intentionally duplicates most of `stoix.systems.vpg.ff_reinforce`
rather than modifying it, so the existing VPG system is left untouched.
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
from stoix.networks.base import FeedForwardCritic as Critic
from stoix.networks.base_compute import FeedForwardActorWithComputeTime as Actor
from stoix.systems.ramdp_vpg.evaluator import ComputeAwareActFn, evaluator_setup_with_compute_time
from stoix.systems.ramdp_vpg.ramdp_vpg_types import (
    RamdpOnPolicyLearnerState,
    Transition,
    update_discounted_return,
)
from stoix.utils import make_env as environments
from stoix.utils.checkpointing import Checkpointer
from stoix.utils.jax_utils import unreplicate_batch_dim, unreplicate_n_dims
from stoix.utils.logger import LogEvent, StoixLogger
from stoix.utils.multistep import batch_discounted_returns
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

    def _update_step(
        learner_state: RamdpOnPolicyLearnerState, _: Any
    ) -> Tuple[RamdpOnPolicyLearnerState, Tuple]:
        def _env_step(
            learner_state: RamdpOnPolicyLearnerState, _: Any
        ) -> Tuple[RamdpOnPolicyLearnerState, Transition]:
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
            value = critic_apply_fn(params.critic_params, last_timestep.observation)
            action = actor_policy.sample(seed=policy_key)

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
            }

            transition = Transition(
                done,
                action,
                value,
                timestep.reward,
                last_timestep.observation,
                info,
                compute_time,
                first_convergence_step,
                num_close_steps,
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

        # Fold compute cost into the discounting: a transition that took
        # `compute_time` pondering steps is treated as that many elementary
        # environment time steps having elapsed, giving
        # G_h = gamma^(C_h - 1) * (r_h + gamma * G_{h+1}) - see module
        # docstring.
        compute_time = traj_batch.compute_time
        r_t = traj_batch.reward * config.system.gamma ** (compute_time - 1)
        v_t = jnp.concatenate([traj_batch.value, last_val[..., jnp.newaxis]], axis=-1)[:, 1:]
        not_done = 1.0 - traj_batch.done.astype(jnp.float32)
        d_t = (not_done * config.system.gamma**compute_time).astype(jnp.float32)
        monte_carlo_returns = batch_discounted_returns(r_t, d_t, v_t, True, False)

        def _actor_loss_fn(
            actor_params: FrozenDict,
            observations: chex.Array,
            actions: chex.Array,
            monte_carlo_returns: chex.Array,
            value_predictions: chex.Array,
            compute_times: chex.Array,
            first_convergence_steps: chex.Array,
            num_close_steps: chex.Array,
        ) -> Tuple:
            """Calculate the actor loss."""
            # Replay the halting trajectory actually taken during rollout,
            # mirroring log_prob(actions) evaluating the stored action.
            actor_policy, halting_log_prob = actor_apply_fn(
                actor_params, observations, torso_kwargs={"target_compute_time": compute_times}
            )
            env_log_prob = actor_policy.log_prob(actions)
            log_prob = env_log_prob + halting_log_prob
            advantage = monte_carlo_returns - value_predictions

            # "Delightful" policy gradient: gate the REINFORCE weight by how
            # surprising the sampled trajectory was, via stop-gradient'd
            # quantities only - never changes where gradient flows.
            weight = advantage
            if config.system.delightful:
                surprisal = -jax.lax.stop_gradient(log_prob)
                gate = jax.nn.sigmoid(advantage * surprisal / config.system.delightful_eta)
                weight = gate * advantage

            loss_actor = -weight * log_prob
            entropy = actor_policy.entropy().mean()

            total_loss_actor = loss_actor.mean() - config.system.ent_coef * entropy
            loss_info = {
                "actor_loss": loss_actor,
                "entropy": entropy,
                "compute_time": compute_times,
                "first_convergence_step": first_convergence_steps,
                "num_close_steps": num_close_steps,
                "advantage": advantage.mean(),
            }
            if config.system.delightful:
                loss_info["delightful_gate"] = gate
            return total_loss_actor, loss_info

        def _critic_loss_fn(
            critic_params: FrozenDict,
            observations: chex.Array,
            targets: chex.Array,
        ) -> Tuple:
            """Calculate the critic loss."""
            value = critic_apply_fn(critic_params, observations)
            value_loss = rlax.l2_loss(value, targets).mean()

            critic_total_loss = config.system.vf_coef * value_loss
            loss_info = {
                "value_loss": value_loss,
            }
            return critic_total_loss, loss_info

        actor_grad_fn = jax.grad(_actor_loss_fn, has_aux=True)
        actor_grads, actor_loss_info = actor_grad_fn(
            params.actor_params,
            traj_batch.obs,
            traj_batch.action,
            monte_carlo_returns,
            traj_batch.value,
            traj_batch.compute_time,
            traj_batch.first_convergence_step,
            traj_batch.num_close_steps,
        )

        critic_grad_fn = jax.grad(_critic_loss_fn, has_aux=True)
        critic_grads, critic_loss_info = critic_grad_fn(
            params.critic_params, traj_batch.obs, monte_carlo_returns
        )

        # pmean over the batch axis, then over devices.
        actor_grads, actor_loss_info = jax.lax.pmean(
            (actor_grads, actor_loss_info), axis_name="batch"
        )
        actor_grads, actor_loss_info = jax.lax.pmean(
            (actor_grads, actor_loss_info), axis_name="device"
        )

        critic_grads, critic_loss_info = jax.lax.pmean(
            (critic_grads, critic_loss_info), axis_name="batch"
        )
        critic_grads, critic_loss_info = jax.lax.pmean(
            (critic_grads, critic_loss_info), axis_name="device"
        )

        actor_updates, actor_new_opt_state = actor_update_fn(
            actor_grads, opt_states.actor_opt_state
        )
        actor_new_params = optax.apply_updates(params.actor_params, actor_updates)

        critic_updates, critic_new_opt_state = critic_update_fn(
            critic_grads, opt_states.critic_opt_state
        )
        critic_new_params = optax.apply_updates(params.critic_params, critic_updates)

        new_params = ActorCriticParams(actor_new_params, critic_new_params)
        new_opt_state = ActorCriticOptStates(actor_new_opt_state, critic_new_opt_state)

        loss_info = {
            **actor_loss_info,
            **critic_loss_info,
        }

        learner_state = RamdpOnPolicyLearnerState(
            new_params,
            new_opt_state,
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

    actor_torso = hydra.utils.instantiate(config.network.actor_network.pre_torso)
    actor_action_head = hydra.utils.instantiate(
        config.network.actor_network.action_head, action_dim=num_actions
    )
    critic_torso = hydra.utils.instantiate(config.network.critic_network.pre_torso)
    critic_head = hydra.utils.instantiate(config.network.critic_network.critic_head)

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
    critic_network = Critic(torso=critic_torso, critic_head=critic_head, **critic_kwargs)

    actor_lr = make_learning_rate(config.system.actor_lr, config, 1, 1)
    critic_lr = make_learning_rate(config.system.critic_lr, config, 1, 1)

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
            eval_act_fn=get_distribution_act_fn_with_compute_time(config, actor_network.apply),
            params=learner_state.params.actor_params,
            config=config,
        )
    )

    config.arch.num_updates_per_eval = config.arch.num_updates // config.arch.num_evaluation
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
        opt_steps_per_eval = config.arch.num_updates_per_eval
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
    config_name="default_ramdp_ff_reinforce.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    OmegaConf.set_struct(cfg, False)

    eval_performance = run_experiment(cfg)

    print(
        f"{Fore.CYAN}{Style.BRIGHT}Compute-time-aware REINFORCE (RAMDP-VPG) experiment "
        f"completed{Style.RESET_ALL}"
    )
    return eval_performance


if __name__ == "__main__":
    hydra_entry_point()
