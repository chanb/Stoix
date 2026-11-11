import jax
import jax.numpy as jnp
import numpy as np

from stoix.networks.base_compute import FeedForwardActorWithComputeTime
from stoix.networks.heads import CategoricalHead
from stoix.networks.torso_compute_transformer import TransformerChainOfThoughtTorso
from stoix.systems.ramdp_vpg.ff_ppo import halting_entropy_reward, targets_and_advantages

B, A, MAX_STEPS = 7, 4, 5


def test_entropy_reward_is_path_entropy_at_gamma_1():
    """Chain rule: E_N[sum of visited halting entropies] = H(N), with forced steps masked."""
    for min_steps in (1, 3):
        actor = FeedForwardActorWithComputeTime(
            torso=TransformerChainOfThoughtTorso(
                hidden_dim=16, num_heads=2, mlp_dim=32, max_steps=MAX_STEPS, min_steps=min_steps
            ),
            action_head=CategoricalHead(action_dim=A),
        )
        obs = jax.random.normal(jax.random.PRNGKey(0), (B, 10))
        params = actor.init(jax.random.PRNGKey(1), obs, torso_kwargs={"deterministic": True})
        log_p, r_ent = [], []
        for n in range(min_steps, MAX_STEPS + 1):
            _, _, _, step_lp, step_ent = actor.apply(
                params, obs, torso_kwargs={"target_compute_time": jnp.full(B, n, jnp.float32)}
            )
            log_p.append(step_lp.sum(-1))
            r_ent.append(halting_entropy_reward(step_ent, 1.0))
        p = jnp.exp(jnp.stack(log_p))
        np.testing.assert_allclose(p.sum(0), 1.0, atol=1e-5)
        path_entropy = -(p * jnp.stack(log_p)).sum(0)
        np.testing.assert_allclose((p * jnp.stack(r_ent)).sum(0), path_entropy, atol=1e-5)
        assert jnp.all(path_entropy > 0)


def test_halting_advantage_adds_entropy_advantage():
    T = 6
    k = jax.random.split(jax.random.PRNGKey(2), 6)
    r_t, r_ent = jax.random.uniform(k[0], (B, T)), jax.random.uniform(k[1], (B, T))
    d_t = 0.9 * (jax.random.uniform(k[2], (B, T)) > 0.2)
    value = jax.random.normal(k[3], (B, T, 2))
    last_val = jax.random.normal(k[4], (B, 2))
    coef = 0.3
    tgt, adv = targets_and_advantages(r_t, d_t, value, last_val, 0.95, False, r_ent, coef)
    plain_tgt, plain_adv = targets_and_advantages(r_t, d_t, value[..., 0], last_val[..., 0], 0.95, False)
    ent_tgt, ent_adv = targets_and_advantages(r_ent, d_t, value[..., 1], last_val[..., 1], 0.95, False)
    np.testing.assert_allclose(tgt[..., 0], plain_tgt, rtol=1e-6)
    np.testing.assert_allclose(tgt[..., 1], ent_tgt, rtol=1e-6)
    np.testing.assert_allclose(adv[..., 0], plain_adv, rtol=1e-6)  # the env action keeps A
    np.testing.assert_allclose(adv[..., 1], plain_adv + coef * ent_adv, rtol=1e-5)  # halting: A + coef A_ent
