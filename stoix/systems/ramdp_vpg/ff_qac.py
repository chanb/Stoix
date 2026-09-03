"""Compute-time-aware Q actor-critic (RAMDP-QAC).

Variant of `stoix.systems.ramdp_vpg.ff_reinforce` where the advantage used to
train the actor comes from a learned Q-function instead of an n-step
Monte-Carlo return. `c` (`compute_time`) is treated as part of what the
policy outputs, exactly like the environment action `a`, so the critic
(`stoix.networks.base_qac.ValueAndQCritic`) has two heads sharing one torso:

  - a state-value head V(s), trained by ordinary regression to the same
    n-step, compute-discounted return target as `ff_reinforce.py`'s critic
    (`G_h = gamma^(C_h - 1) * (r_h + gamma * G_{h+1})`) - this is *not*
    derived from Q, since a proper V(s) would need marginalising Q over the
    actor's full joint (a, c) distribution;
  - a state-action-value head Q(s, a, c), read off at the realised action and
    compute time. The advantage is then

        A(s_t, a_t, c_t) = Q(s_t, a_t, c_t) - V(s_t)

    with no n-step return or bootstrapping in the advantage itself - only V's
    own regression target ever needs bootstrapping.

`config.system.qac_variant` selects how Q(s, a, c) is obtained:

  - "naive": Q is a genuinely learned `(num_actions, max_steps)` table,
    regressed directly to the same n-step return G used for V.
  - "fac" (runtime-factorized, the default): since compute time enters the
    return *only* through the `gamma^(c-1)` prefactor, Q(s,a,c) =
    gamma^(c-1) * Q(s,a,1) is not just a convenient approximation but the
    value implied by that same structure. So Q only has to learn the much
    smaller `(num_actions,)` table Q(s,·,1), regressed to
    B_h = G_h / gamma^(C_h - 1) - the same return target with its own
    compute-time scaling divided back out, so gamma^(c-1) * B_h recovers
    G_h exactly.

The actor itself is unchanged from `ff_reinforce.py`: an Adaptive
Computation Time torso whose sampled halting trajectory is trained via the
score-function estimator, jointly with the environment action, from this
Q-based advantage. It also supports the same optional
`config.system.delightful` gate on the REINFORCE weight.

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
)
from stoix.networks.base_compute import FeedForwardActorWithComputeTime as Actor
from stoix.networks.base_qac import SeparateValueAndQCritic
from stoix.networks.base_qac import ValueAndQCritic as Critic
from stoix.systems.ramdp_vpg.evaluator import evaluator_setup_with_compute_time
from stoix.systems.ramdp_vpg.ff_reinforce import get_distribution_act_fn_with_compute_time
from stoix.systems.ramdp_vpg.qac_types import Transition
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


def get_learner_fn(
    env: Environment,
    apply_fns: Tuple[ActorApply, CriticApply],
    update_fns: Tuple[optax.TransformUpdateFn, optax.TransformUpdateFn],
    config: DictConfig,
) -> LearnerFn[RamdpOnPolicyLearnerState]:
    """Get the learner function."""

    actor_apply_fn, critic_apply_fn = apply_fns
    actor_update_fn, critic_update_fn = update_fns

    qac_variant = config.system.qac_variant
    assert qac_variant in ("naive", "fac"), f"Unknown qac_variant: {qac_variant}"

    def _q_at_action_and_compute_time(
        q_output: chex.Array, action: chex.Array, compute_time: chex.Array
    ) -> chex.Array:
        """Read Q(s, a, c) off the critic's raw `q_value` output: "naive"
        indexes a `(num_actions, max_steps)` table by both `action` and
        `compute_time`; "fac" indexes Q(s,·,1) by `action` then scales by
        `gamma ** (compute_time - 1)` to recover Q(s,a,c)."""
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
            value = critic_apply_fn(
                params.critic_params, last_timestep.observation, method="value"
            )
            q_output = critic_apply_fn(
                params.critic_params, last_timestep.observation, method="q_value"
            )
            action = actor_policy.sample(seed=policy_key)
            q_value = _q_at_action_and_compute_time(q_output, action, compute_time)

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
                q_value,
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
        last_val = critic_apply_fn(params.critic_params, last_timestep.observation, method="value")

        traj_batch = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), traj_batch)

        # Q(s_t, a_t, c_t) - V(s_t) directly, no n-step/Monte-Carlo return.
        advantage = traj_batch.q_value - traj_batch.value

        # Return target for V (and, for "naive", Q too): the same n-step,
        # compute-discounted return as `ff_reinforce.py`'s critic. Used purely
        # to train the critic; the advantage above never touches it.
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
            first_convergence_steps: chex.Array,
            num_close_steps: chex.Array,
        ) -> Tuple:
            """Calculate the actor loss."""
            # Replay the halting trajectory actually taken during rollout,
            # mirroring log_prob(actions) evaluating the stored action.
            # `states_history`/`per_step_halting_log_prob` (only needed for
            # PPO's latent trust-region penalty/per-step clip, see
            # `ff_ppo.py`) are ignored here.
            actor_policy, halting_log_prob, _, _ = actor_apply_fn(
                actor_params, observations, torso_kwargs={"target_compute_time": compute_times}
            )
            env_log_prob = actor_policy.log_prob(actions)
            log_prob = env_log_prob + halting_log_prob

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
            actions: chex.Array,
            compute_times: chex.Array,
            targets: chex.Array,
        ) -> Tuple:
            """Calculate the critic loss: V regressed to the return target, and
            Q regressed to that same target - divided back down to a `c=1`
            equivalent target for the "fac" variant, since Q there only
            predicts Q(s,·,1)."""
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

        actor_grad_fn = jax.grad(_actor_loss_fn, has_aux=True)
        actor_grads, actor_loss_info = actor_grad_fn(
            params.actor_params,
            traj_batch.obs,
            traj_batch.action,
            advantage,
            traj_batch.compute_time,
            traj_batch.first_convergence_step,
            traj_batch.num_close_steps,
        )

        critic_grad_fn = jax.grad(_critic_loss_fn, has_aux=True)
        critic_grads, critic_loss_info = critic_grad_fn(
            params.critic_params,
            traj_batch.obs,
            traj_batch.action,
            traj_batch.compute_time,
            g_targets,
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
    # A `value_pre_torso`/`q_pre_torso` pair selects `SeparateValueAndQCritic`:
    # V and Q each get their own torso and share no parameters.
    separate_qv_torsos = "value_pre_torso" in config.network.critic_network
    if separate_qv_torsos:
        value_torso = hydra.utils.instantiate(config.network.critic_network.value_pre_torso)
        q_torso = hydra.utils.instantiate(config.network.critic_network.q_pre_torso)
    else:
        critic_torso = hydra.utils.instantiate(config.network.critic_network.pre_torso)
    value_head = hydra.utils.instantiate(config.network.critic_network.value_head)
    # "naive" learns a full (action, compute_time) table, so needs max_steps;
    # "fac" only ever needs Q(s,·,1), one value per action.
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

    # input_layer is optional - defaults to identity when absent.
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
    if separate_qv_torsos:
        critic_network = SeparateValueAndQCritic(
            value_torso=value_torso,
            q_torso=q_torso,
            value_head=value_head,
            q_head=q_head,
            **critic_kwargs,
        )
    else:
        critic_network = Critic(
            torso=critic_torso, value_head=value_head, q_head=q_head, **critic_kwargs
        )

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
    config_name="default_ramdp_ff_qac.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    OmegaConf.set_struct(cfg, False)

    eval_performance = run_experiment(cfg)

    print(
        f"{Fore.CYAN}{Style.BRIGHT}Compute-time-aware Q actor-critic (RAMDP-QAC) experiment "
        f"completed{Style.RESET_ALL}"
    )
    return eval_performance


if __name__ == "__main__":
    hydra_entry_point()
