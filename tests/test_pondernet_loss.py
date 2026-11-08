import jax
import jax.numpy as jnp
import numpy as np

from stoix.networks.base_compute import FeedForwardActorWithComputeTime
from stoix.networks.heads import CategoricalHead
from stoix.networks.torso_compute_transformer import TransformerChainOfThoughtTorso
from stoix.systems.ramdp_vpg.ff_ppo import act_policy, pondernet_mixture

B, A, MAX_STEPS = 7, 4, 5


def _setup(min_steps=1):
    actor = FeedForwardActorWithComputeTime(
        torso=TransformerChainOfThoughtTorso(
            hidden_dim=16, num_heads=2, mlp_dim=32, max_steps=MAX_STEPS, min_steps=min_steps
        ),
        action_head=CategoricalHead(action_dim=A),
    )
    obs = jax.random.normal(jax.random.PRNGKey(0), (B, 10))
    params = actor.init(jax.random.PRNGKey(1), obs, torso_kwargs={"deterministic": True})
    return actor, obs, params


def _mix(actor, params, obs, action):
    return pondernet_mixture(actor.apply, params, obs, action, MAX_STEPS, prior_lambda=0.3)


def test_mixture_is_a_distribution():
    for min_steps in (1, 3):
        actor, obs, params = _setup(min_steps)
        probs = sum(
            jnp.exp(_mix(actor, params, obs, jnp.full(B, a, jnp.int32))[0]) for a in range(A)
        )
        np.testing.assert_allclose(probs, 1.0, atol=1e-5)
        kl = _mix(actor, params, obs, jnp.zeros(B, jnp.int32))[4]
        assert jnp.all(jnp.isfinite(kl) & (kl >= 0.0))


def test_halting_head_gets_finite_nonzero_gradient():
    actor, obs, params = _setup(min_steps=2)
    action = jnp.arange(B) % A

    def loss(p):
        log_prob, entropy, _, _, kl = _mix(actor, p, obs, action)
        return -log_prob.mean() + kl.mean() - entropy.mean()

    _halt_grad_is_finite_and_nonzero(loss, params)


def _halt_grad_is_finite_and_nonzero(loss, params):
    grads = jax.grad(loss)(params)
    assert all(jnp.all(jnp.isfinite(g)) for g in jax.tree_util.tree_leaves(grads))
    halt = grads["params"]["torso"]["Scan_CoTStep_0"]["HaltingHead_0"]
    assert sum(jnp.abs(g).sum() for g in jax.tree_util.tree_leaves(halt)) > 0.0


def test_act_halts_where_cumulative_prob_reaches_threshold():
    for min_steps in (1, 3):
        actor, obs, params = _setup(min_steps)
        _, n, cost, _, _ = act_policy(actor.apply, params, obs, MAX_STEPS, eps=0.01)
        assert jnp.all((n >= min_steps) & (n <= MAX_STEPS))
        assert jnp.all((cost > n) & (cost <= n + 1.0 + 1e-5))  # remainder R in (0, 1]
    actor, obs, params = _setup(min_steps=2)
    action = jnp.arange(B) % A

    def loss(p):
        pi, _, cost, _, _ = act_policy(actor.apply, p, obs, MAX_STEPS, eps=0.01)
        return -pi.log_prob(action).mean() + 0.01 * cost.mean()

    _halt_grad_is_finite_and_nonzero(loss, params)


def test_negative_halting_bias_makes_act_ponder_longer():
    actor = FeedForwardActorWithComputeTime(
        torso=TransformerChainOfThoughtTorso(
            hidden_dim=16, num_heads=2, mlp_dim=32, max_steps=MAX_STEPS, halting_bias_init=-4.0
        ),
        action_head=CategoricalHead(action_dim=A),
    )
    obs = jax.random.normal(jax.random.PRNGKey(0), (B, 10))
    params = actor.init(jax.random.PRNGKey(1), obs, torso_kwargs={"deterministic": True})
    params["params"]["torso"]["Scan_CoTStep_0"]["HaltingHead_0"]["Dense_0"]["kernel"] *= 0.0
    _, n, _, _, _ = act_policy(actor.apply, params, obs, MAX_STEPS, eps=0.01)
    np.testing.assert_array_equal(n, MAX_STEPS)  # sigmoid(-4) * 4 < 0.99
