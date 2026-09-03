"""Compute-time-aware REINFORCE with an *explicit* chain of thought.

Variant of `stoix.systems.ramdp_vpg.ff_reinforce` whose actor's torso is
`stoix.networks.torso_compute_explicit_cot.TransformerExplicitCoTTorso`
instead of the latent-CoT `AdaptiveComputationTimeTorso`/
`TransformerChainOfThoughtTorso`: at every pondering step, the actor samples a
token from a learned vocabulary that has one extra class meaning "predict the
environment action now" - so halting *is* emitting that token, in the same
sense as any other thought token or the environment action itself. A thought
token's embedding (not a raw hidden state) is fed back in as context for the
next step, so the resulting chain of thought is a literal, inspectable
sequence of token ids, not an opaque vector.

The halting decision and the thought tokens are trained via the
score-function (REINFORCE) estimator, using the log-probability of the
sampled trajectory (`TransformerExplicitCoTTorso`'s replay mode, which needs
the realised `thought_tokens` to replay the exact trajectory taken - the
halting step is recoverable from where the "act now" token appears in it).
Everything else - the compute-discounted return (see `ff_reinforce.py` for
the full derivation: `G_h = gamma^(C_h - 1) * (r_h + gamma * G_{h+1})`), the
critic, the actor loss shape, and the optional `config.system.delightful`
gate on the REINFORCE weight - is identical to the latent-CoT version; only
the Transition (which now also stores `thought_tokens`) and the torso calls
differ.

This file intentionally duplicates most of `ff_reinforce.py` rather than
modifying it, so both the plain VPG system and the latent-CoT RAMDP-VPG
system are left untouched.
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
from stoix.systems.ramdp_vpg.explicit_cot_types import ExplicitCoTTransition as Transition
from stoix.systems.ramdp_vpg.ramdp_vpg_types import (
    RamdpOnPolicyLearnerState,
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
    """Like the equivalent in `stoix.systems.ramdp_vpg.ff_reinforce`, but for
    an actor whose torso also emits explicit thought tokens, i.e. returns
    `(action_distribution, compute_time, thought_tokens)`. The emitted tokens
    aren't needed for evaluation metrics, so they're dropped here.

    `TransformerExplicitCoTTorso` has no latent-convergence diagnostics (its
    "thoughts" are discrete token ids, not a continuous state), so this always
    reports the sentinel `first_convergence_step=-1`, `num_close_steps=0`,
    matching `evaluator.py`'s shared `ComputeAwareActFn` interface."""

    def act_fn(
        params: FrozenDict, observation: chex.Array, key: chex.PRNGKey
    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
        if config.arch.evaluation_greedy:
            pi, compute_time, _thought_tokens = actor_apply(
                params, observation, torso_kwargs={"deterministic": True}
            )
            action = pi.mode()
        else:
            halting_key, action_key = jax.random.split(key)
            pi, compute_time, _thought_tokens = actor_apply(
                params, observation, torso_kwargs={"rng": halting_key}
            )
            action = pi.sample(seed=action_key)
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
            actor_policy, compute_time, thought_tokens = actor_apply_fn(
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
                thought_tokens,
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

        # Fold compute cost into the discounting - see `ff_reinforce.py` for
        # the full derivation of `G_h = gamma^(C_h - 1) * (r_h + gamma * G_{h+1})`.
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
            thought_tokens: chex.Array,
        ) -> Tuple:
            """Calculate the actor loss."""
            # Replay the halting-and-token trajectory actually taken during
            # rollout, mirroring log_prob(actions) evaluating the stored action.
            # Plain REINFORCE wants the joint trajectory log-prob (`log_prob`);
            # the per-step breakdown is only needed for per-decision PPO
            # clipping (see `ff_ppo_explicit_cot.py`), so it's ignored here.
            actor_policy, cot_log_prob, _ = actor_apply_fn(
                actor_params,
                observations,
                torso_kwargs={"target_tokens": thought_tokens},
            )
            env_log_prob = actor_policy.log_prob(actions)
            log_prob = env_log_prob + cot_log_prob
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
            traj_batch.thought_tokens,
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
            actor_grads, opt_states.actor_opt_state, params.actor_params
        )
        actor_new_params = optax.apply_updates(params.actor_params, actor_updates)

        # UPDATE CRITIC PARAMS AND OPTIMISER STATE
        critic_updates, critic_new_opt_state = critic_update_fn(
            critic_grads, opt_states.critic_opt_state, params.critic_params
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

    actor_network = Actor(torso=actor_torso, action_head=actor_action_head)
    critic_network = Critic(torso=critic_torso, critic_head=critic_head)

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
    config_name="default_ramdp_ff_reinforce_explicit_cot.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    OmegaConf.set_struct(cfg, False)

    eval_performance = run_experiment(cfg)

    print(
        f"{Fore.CYAN}{Style.BRIGHT}Compute-time-aware REINFORCE with explicit CoT "
        f"experiment completed{Style.RESET_ALL}"
    )
    return eval_performance


if __name__ == "__main__":
    hydra_entry_point()
