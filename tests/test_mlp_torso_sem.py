import jax
import numpy as np

from stoix.networks.torso import MLPTorso


def test_critic_mlp_sem_output_on_simplex():
    obs = jax.random.normal(jax.random.PRNGKey(0), (5, 10))
    torso = MLPTorso(layer_sizes=[32, 32], activation="gelu", sem_group_size=8)
    out = torso.apply(torso.init(jax.random.PRNGKey(1), obs), obs)

    assert out.shape == (5, 32)
    np.testing.assert_allclose(out.reshape(5, -1, 8).sum(-1), 1.0, atol=1e-5)
    assert (out >= 0).all()


def test_mlp_sem_off_by_default():
    obs = jax.random.normal(jax.random.PRNGKey(0), (5, 10))
    plain = MLPTorso(layer_sizes=[32], activation="gelu")
    params = plain.init(jax.random.PRNGKey(1), obs)
    out = plain.apply(params, obs)
    sem = MLPTorso(layer_sizes=[32], activation="gelu", sem_group_size=8).apply(params, obs)

    assert not np.allclose(out.reshape(5, -1, 8).sum(-1), 1.0)
    assert not np.allclose(out, sem)
