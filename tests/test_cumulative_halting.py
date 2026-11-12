import jax
import jax.numpy as jnp
import numpy as np

from stoix.networks.base_compute import FeedForwardActorWithComputeTime
from stoix.networks.heads import CategoricalHead
from stoix.networks.torso_compute_transformer import TransformerChainOfThoughtTorso
from stoix.systems.ramdp_vpg.ff_ppo import pondernet_mixture

B, A, MAX_STEPS = 5, 4, 8


def _setup(h=None, min_steps=1, param="cumulative"):
    """h: constant P(c = n) per step (halting kernel zeroed), else a random init."""
    bias = 0.0 if h is None else float(np.log(h / (1 - h)))
    actor = FeedForwardActorWithComputeTime(
        torso=TransformerChainOfThoughtTorso(
            hidden_dim=16, num_heads=2, mlp_dim=32, max_steps=MAX_STEPS, min_steps=min_steps,
            halting_bias_init=bias, halting_param=param, cumulative_halting_eps=0.01,
        ),
        action_head=CategoricalHead(action_dim=A),
    )
    obs = jax.random.normal(jax.random.PRNGKey(0), (B, 10))
    params = actor.init(jax.random.PRNGKey(1), obs, torso_kwargs={"deterministic": True})
    if h is not None:
        params["params"]["torso"]["Scan_CoTStep_0"]["HaltingHead_0"]["Dense_0"]["kernel"] *= 0.0
    return actor, obs, params


def _p_c(actor, params, obs):
    """P(c = n) for n = 1..MAX_STEPS from replaying each target compute time."""
    return jnp.stack([
        jnp.exp(actor.apply(params, obs, torso_kwargs={"target_compute_time": jnp.full(B, n, jnp.float32)})[3].sum(-1))
        for n in range(1, MAX_STEPS + 1)
    ], -1)


def test_constant_h_halts_when_mass_runs_out():
    actor, obs, params = _setup(h=0.3)
    expected = [0.3, 0.3, 0.3, 0.1, 0, 0, 0, 0]  # H_3 + h_4 = 1.2 >= 0.99: remainder 0.1 at step 4
    np.testing.assert_allclose(_p_c(actor, params, obs), np.tile(expected, (B, 1)), atol=1e-4)
    for seed in range(20):
        c = actor.apply(params, obs, torso_kwargs={"rng": jax.random.PRNGKey(seed)})[1]
        assert jnp.all((c >= 1) & (c <= 4))


def test_halt_distribution_sums_to_one():
    for min_steps in (1, 3):
        actor, obs, params = _setup(min_steps=min_steps)
        # Targets below min_steps cannot halt there (replay just runs on), so skip them.
        p = _p_c(actor, params, obs)[:, min_steps - 1 :]
        np.testing.assert_allclose(p.sum(-1), 1.0, atol=1e-4)


def test_pondernet_mixture_uses_cumulative_h():
    actor, obs, params = _setup(h=0.3)
    probs = sum(
        jnp.exp(pondernet_mixture(actor.apply, params, obs, jnp.full(B, a, jnp.int32), MAX_STEPS, 0.3)[0])
        for a in range(A)
    )
    np.testing.assert_allclose(probs, 1.0, atol=1e-4)


def test_halting_head_gets_finite_nonzero_gradient():
    actor, obs, params = _setup(min_steps=2)
    target = jnp.arange(B, dtype=jnp.float32) % MAX_STEPS + 1

    def loss(p):
        return -actor.apply(p, obs, torso_kwargs={"target_compute_time": target})[3].sum()

    grads = jax.grad(loss)(params)
    assert all(jnp.all(jnp.isfinite(g)) for g in jax.tree_util.tree_leaves(grads))
    halt = grads["params"]["torso"]["Scan_CoTStep_0"]["HaltingHead_0"]
    assert sum(jnp.abs(g).sum() for g in jax.tree_util.tree_leaves(halt)) > 0.0
