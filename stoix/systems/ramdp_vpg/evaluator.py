"""Compute-time-aware evaluator utilities.

Mirrors the feedforward parts of `stoix.evaluator` (which is left unmodified)
for actor networks whose torso reports a `compute_time` (see
`stoix.networks.torso_compute.AdaptiveComputationTimeTorso`), so that
evaluation - not just training - reports how much the actor "thinks" per
action. `stoix.evaluator.get_ff_evaluator_fn`'s `act_fn` returns a bare
action, which is why it can't be reused directly here: our `act_fn`
(`stoix.systems.ramdp_vpg.ff_reinforce.get_distribution_act_fn_with_compute_time`)
also returns the realised `compute_time`, which needs to be threaded through
episode accumulation and into the reported metrics.
"""

from typing import Callable, Dict, Optional, Tuple

import chex
import hydra
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from omegaconf import DictConfig
from stoa import Environment, TimeStep
from typing_extensions import NamedTuple

from stoix.base_types import EvalFn, EvalResetFn, EvaluationOutput, State
from stoix.evaluator import make_random_initial_eval_reset_fn
from stoix.utils.jax_utils import unreplicate_batch_dim
from stoix.utils.running_statistics import RunningStatisticsState, normalize

# Returns (action, compute_time) rather than just an action.
ComputeAwareActFn = Callable[[FrozenDict, chex.Array, chex.PRNGKey], Tuple[chex.Array, chex.Array]]


class RamdpEvalState(NamedTuple):
    """Like `stoix.base_types.EvalState`, but additionally accumulates the
    actor's compute time over the episode."""

    key: chex.PRNGKey
    env_state: State
    timestep: TimeStep
    step_count: chex.Array
    episode_return: chex.Array
    episode_discounted_return: chex.Array
    discount_factor: chex.Array
    episode_compute_time: chex.Array


def get_ff_evaluator_fn_with_compute_time(
    env: Environment,
    act_fn: ComputeAwareActFn,
    config: DictConfig,
    eval_reset_fn: EvalResetFn,
    log_solve_rate: bool = False,
    eval_multiplier: int = 1,
) -> EvalFn:
    """Like `stoix.evaluator.get_ff_evaluator_fn`, but for a compute-time-aware
    `act_fn` and reporting the actor's mean compute time per action as the
    `compute_time` episode metric."""

    def eval_one_episode(
        params: FrozenDict,
        init_eval_state: RamdpEvalState,
        running_statistics: Optional[RunningStatisticsState] = None,
    ) -> Dict:
        """Evaluate one episode. It is vectorized over the number of evaluation episodes."""

        def _env_step(eval_state: RamdpEvalState) -> RamdpEvalState:
            """Step the environment."""
            (
                key,
                env_state,
                last_timestep,
                step_count,
                episode_return,
                episode_discounted_return,
                discount_factor,
                episode_compute_time,
            ) = eval_state

            # Select action.
            key, policy_key = jax.random.split(key)

            # Normalize observation if needed
            observation = last_timestep.observation
            if running_statistics is not None:
                observation = normalize(observation, running_statistics)

            action, compute_time = act_fn(
                params,
                jax.tree_util.tree_map(lambda x: x[jnp.newaxis, ...], observation),
                policy_key,
            )

            # Step environment.
            env_state, timestep = env.step(env_state, action.squeeze(0))

            # Log episode metrics. `episode_discounted_return` follows the same
            # compute-adjusted discounting the actor is trained with (see
            # `ff_reinforce.get_learner_fn`'s `G_h = gamma^(C_h - 1) * (r_h +
            # gamma * G_{h+1})`), accumulated forward from the start of the
            # episode via a running `discount_factor` rather than backward
            # like the training-time `batch_discounted_returns` recursion.
            episode_return += timestep.reward
            episode_discounted_return += (
                discount_factor * config.system.gamma ** (compute_time - 1) * timestep.reward
            )
            discount_factor *= config.system.gamma**compute_time
            episode_compute_time += compute_time
            step_count += 1
            eval_state = RamdpEvalState(
                key,
                env_state,
                timestep,
                step_count,
                episode_return,
                episode_discounted_return,
                discount_factor,
                episode_compute_time,
            )
            return eval_state

        def not_done(carry: Tuple) -> bool:
            """Check if the episode is done."""
            timestep = carry[2]
            is_not_done: bool = ~timestep.last()
            return is_not_done

        final_state = jax.lax.while_loop(not_done, _env_step, init_eval_state)

        eval_metrics = {
            "episode_return": final_state.episode_return,
            "episode_discounted_return": final_state.episode_discounted_return,
            "episode_length": final_state.step_count,
            "compute_time": final_state.episode_compute_time / final_state.step_count,
        }
        # Log solved episode if solve rate is required.
        if log_solve_rate:
            eval_metrics["solved_episode"] = jnp.all(
                final_state.episode_return >= config.env.solved_return_threshold
            ).astype(int)

        return eval_metrics

    def evaluator_fn(
        trained_params: FrozenDict,
        key: chex.PRNGKey,
        running_statistics: Optional[RunningStatisticsState] = None,
    ) -> EvaluationOutput[RamdpEvalState]:
        """Evaluator function."""

        # Initialise environment states and timesteps.
        n_devices = len(jax.devices())

        eval_batch = (config.arch.num_eval_episodes // n_devices) * eval_multiplier

        # Get initial states
        key, reset_key = jax.random.split(key)
        env_states, timesteps = eval_reset_fn(reset_key, eval_batch)

        # Split keys for each core.
        key, *step_keys = jax.random.split(key, eval_batch + 1)
        # Add dimension to pmap over.
        step_keys = jnp.stack(step_keys).reshape(eval_batch, -1)

        eval_state = RamdpEvalState(
            key=step_keys,
            env_state=env_states,
            timestep=timesteps,
            step_count=jnp.zeros((eval_batch, 1)),
            episode_return=jnp.zeros_like(timesteps.reward),
            episode_discounted_return=jnp.zeros((eval_batch, 1)),
            discount_factor=jnp.ones((eval_batch, 1)),
            episode_compute_time=jnp.zeros((eval_batch, 1)),
        )

        eval_metrics = jax.vmap(
            eval_one_episode,
            in_axes=(None, 0, None),
            axis_name="eval_batch",
        )(trained_params, eval_state, running_statistics)

        return EvaluationOutput(
            learner_state=eval_state,
            episode_metrics=eval_metrics,
        )

    return evaluator_fn


def evaluator_setup_with_compute_time(
    eval_env: Environment,
    key_e: chex.PRNGKey,
    eval_act_fn: ComputeAwareActFn,
    params: FrozenDict,
    config: DictConfig,
) -> Tuple[EvalFn, EvalFn, Tuple[FrozenDict, chex.Array]]:
    """Like `stoix.evaluator.evaluator_setup`, but wires up
    `get_ff_evaluator_fn_with_compute_time` so evaluation also reports the
    actor's mean compute time per action. Only supports feedforward networks
    (all `ramdp_vpg` uses)."""
    n_devices = len(jax.devices())
    log_solve_rate = hasattr(config.env, "solved_return_threshold")

    if "eval_reset_fn" in config.env and config.env.eval_reset_fn is not None:
        eval_reset_fn = hydra.utils.instantiate(config.env.eval_reset_fn, config, eval_env)
    else:
        eval_reset_fn = make_random_initial_eval_reset_fn(config, eval_env)

    evaluator = get_ff_evaluator_fn_with_compute_time(
        eval_env, eval_act_fn, config, eval_reset_fn, log_solve_rate
    )
    absolute_metric_evaluator = get_ff_evaluator_fn_with_compute_time(
        eval_env, eval_act_fn, config, eval_reset_fn, log_solve_rate, 10
    )

    # Pmap the evaluator functions
    evaluator_fn = jax.pmap(evaluator, axis_name="device")
    absolute_metric_evaluator_fn = jax.pmap(absolute_metric_evaluator, axis_name="device")

    # Broadcast trained params to cores and split keys for each core.
    trained_params = unreplicate_batch_dim(params)
    key_e, *eval_keys = jax.random.split(key_e, n_devices + 1)
    eval_keys = jnp.stack(eval_keys).reshape(n_devices, -1)

    return evaluator_fn, absolute_metric_evaluator_fn, (trained_params, eval_keys)
