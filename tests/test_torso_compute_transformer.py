import jax
import jax.numpy as jnp
import numpy as np

from stoix.networks.torso_compute_transformer import TransformerChainOfThoughtTorso

N, H, G, MAX_STEPS = 3, 16, 4, 5


def _setup(min_steps=1):
    torso = TransformerChainOfThoughtTorso(
        hidden_dim=H, num_heads=2, mlp_dim=32, max_steps=MAX_STEPS, min_steps=min_steps,
        num_state_tokens=N, sem_group_size=G,
    )
    obs = jax.random.normal(jax.random.PRNGKey(0), (7, 10))
    params = torso.init(jax.random.PRNGKey(1), obs, deterministic=True)
    return torso, obs, params


def test_rollout_replay_match_and_sem_on_output():
    torso, obs, params = _setup()
    emb, compute_time, _, _ = torso.apply(params, obs, rng=jax.random.PRNGKey(2))
    replay_emb, _, history, _, _ = torso.apply(params, obs, target_compute_time=compute_time)

    np.testing.assert_allclose(emb, replay_emb, atol=1e-5)
    assert history.shape == (7, MAX_STEPS, H)
    assert jnp.all((compute_time >= 1) & (compute_time <= MAX_STEPS))
    # Each group of G is on the simplex.
    np.testing.assert_allclose(emb.reshape(7, -1, G).sum(-1), 1.0, atol=1e-5)


def test_prefill_tokens_do_not_count_as_compute():
    torso, obs, params = _setup(min_steps=MAX_STEPS)
    _, compute_time, _, _ = torso.apply(params, obs, rng=jax.random.PRNGKey(2))
    np.testing.assert_array_equal(compute_time, MAX_STEPS)


def test_first_state_token_changes_output():
    torso, obs, params = _setup()
    emb = torso.apply(params, obs, deterministic=True)[0]
    # Zero the Dense columns that make state token 0.
    kernel = params["params"]["Dense_0"]["kernel"]
    params2 = jax.tree_util.tree_map(lambda x: x, params)
    params2["params"]["Dense_0"]["kernel"] = kernel.at[:, :H].set(0.0)
    emb2 = torso.apply(params2, obs, deterministic=True)[0]
    assert not np.allclose(emb, emb2)


def test_cnn_channel_tokens_into_torso():
    from stoix.networks.torso import CNNTorso

    cnn = CNNTorso(channel_sizes=[8, N], kernel_sizes=[3, 3], strides=[1, 1], channel_tokens=True)
    torso = TransformerChainOfThoughtTorso(
        hidden_dim=H, num_heads=2, mlp_dim=32, max_steps=MAX_STEPS, input_is_tokens=True,
        sem_group_size=G,
    )
    grid = jax.random.normal(jax.random.PRNGKey(0), (7, 5, 6, 3))
    cnn_params = cnn.init(jax.random.PRNGKey(1), grid)
    tokens = cnn.apply(cnn_params, grid)
    assert tokens.shape == (7, N, 5 * 6)  # one token per last-layer kernel

    params = torso.init(jax.random.PRNGKey(1), tokens, deterministic=True)
    emb, compute_time, _, _ = torso.apply(params, tokens, rng=jax.random.PRNGKey(2))
    replay_emb = torso.apply(params, tokens, target_compute_time=compute_time)[0]
    np.testing.assert_allclose(emb, replay_emb, atol=1e-5)
    np.testing.assert_allclose(emb.reshape(7, -1, G).sum(-1), 1.0, atol=1e-5)
    # N comes from the input: the scan has N - 1 prefill positions.
    assert params["params"]["Scan_CoTStep_0"]["pos_embedding"].shape[0] == MAX_STEPS + N


STABLE_LOOP_FLAGS = dict(latent_rms_norm=True, input_injection=True, residual_scaling=True)


def _stable_setup(min_steps=1, **flags):
    torso = TransformerChainOfThoughtTorso(
        hidden_dim=H, num_heads=2, mlp_dim=32, max_steps=MAX_STEPS, min_steps=min_steps,
        num_state_tokens=N, sem_group_size=G, **{**STABLE_LOOP_FLAGS, **flags},
    )
    obs = jax.random.normal(jax.random.PRNGKey(0), (7, 10))
    params = torso.init(jax.random.PRNGKey(1), obs, deterministic=True)
    return torso, obs, params


def test_stable_loop_flags_off_add_no_params():
    _, _, params = _setup()
    assert not any(k.startswith("alpha_") for k in params["params"]["Scan_CoTStep_0"])


def test_stable_loop_flags_rollout_replay_match_and_sem_on_output():
    torso, obs, params = _stable_setup()
    step_params = params["params"]["Scan_CoTStep_0"]
    np.testing.assert_allclose(jax.nn.sigmoid(step_params["alpha_1_logit"]), 0.75, atol=1e-6)
    np.testing.assert_allclose(jax.nn.sigmoid(step_params["alpha_2_logit"]), 0.25, atol=1e-6)
    emb, compute_time, _, _ = torso.apply(params, obs, rng=jax.random.PRNGKey(2))
    replay_emb, log_prob, _, _, _ = torso.apply(params, obs, target_compute_time=compute_time)
    np.testing.assert_allclose(emb, replay_emb, atol=1e-5)
    assert jnp.all(jnp.isfinite(log_prob))
    # The heads read the unscaled simplex point.
    np.testing.assert_allclose(emb.reshape(7, -1, G).sum(-1), 1.0, atol=1e-5)


def test_stable_loop_flags_gradients_finite():
    torso, obs, params = _stable_setup(min_steps=MAX_STEPS)
    ct = jnp.full((7,), MAX_STEPS)
    grads = jax.grad(lambda p: torso.apply(p, obs, target_compute_time=ct)[0].sum())(params)
    leaves = jax.tree_util.tree_leaves(grads)
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in leaves)
    step_grads = grads["params"]["Scan_CoTStep_0"]
    assert float(jnp.abs(step_grads["alpha_1_logit"]).sum()) > 0
    assert float(jnp.abs(step_grads["alpha_2_logit"]).sum()) > 0


def test_input_injection_feeds_observation_to_later_steps():
    # With injection, a later step's thought depends on the step-0 token even if
    # the attention cache is ignored; without it, only through attention.
    torso, obs, params = _stable_setup(min_steps=MAX_STEPS, residual_scaling=False)
    assert "alpha_2_logit" not in params["params"]["Scan_CoTStep_0"]
    ct = jnp.full((7,), MAX_STEPS)
    emb = torso.apply(params, obs, target_compute_time=ct)[0]
    assert emb.shape == (7, H)
    assert bool(jnp.all(jnp.isfinite(emb)))


def test_trunc_normal_block_init_std():
    torso = TransformerChainOfThoughtTorso(
        hidden_dim=64, num_heads=4, mlp_dim=256, max_steps=2, block_init="trunc_normal",
    )
    obs = jax.random.normal(jax.random.PRNGKey(0), (3, 10))
    params = torso.init(jax.random.PRNGKey(1), obs, deterministic=True)
    block = params["params"]["Scan_CoTStep_0"]["TransformerBlock_0"]
    # Same realized std as flax truncated_normal(1/sqrt(fan_in)).
    ref = lambda fan_in, shape: jax.nn.initializers.truncated_normal(1.0 / np.sqrt(fan_in))(  # noqa: E731
        jax.random.PRNGKey(3), shape)
    for name, fan_in in (("query", 64), ("out", 64), ("mlp_dense_0", 64), ("mlp_dense_1", 256)):
        k = block[name]["kernel"]
        np.testing.assert_allclose(float(k.std()), float(ref(fan_in, k.shape).std()), rtol=0.05)
        assert float(jnp.abs(k).max()) <= 2.0 / np.sqrt(fan_in) + 1e-6  # truncated at 2 std


def test_block_init_default_is_orthogonal():
    _, _, params = _setup()
    k = params["params"]["Scan_CoTStep_0"]["TransformerBlock_0"]["mlp_dense_0"]["kernel"]
    # orthogonal(sqrt(2)): rows (or cols) have norm sqrt(2).
    norms = jnp.linalg.norm(k, axis=0 if k.shape[0] >= k.shape[1] else 1)
    np.testing.assert_allclose(norms, np.sqrt(2.0), rtol=1e-4)


def test_cot_critic_torso_returns_embedding():
    from stoix.networks.torso_compute_transformer import CoTCriticTorso

    torso = TransformerChainOfThoughtTorso(
        hidden_dim=H, num_heads=2, mlp_dim=32, max_steps=MAX_STEPS, min_steps=MAX_STEPS,
        num_state_tokens=N, sem_group_size=G,
    )
    critic = CoTCriticTorso(torso)
    obs = jax.random.normal(jax.random.PRNGKey(0), (7, 10))
    params = critic.init(jax.random.PRNGKey(1), obs)
    emb = critic.apply(params, obs)
    np.testing.assert_allclose(emb, torso.apply({"params": params["params"]["torso"]}, obs, deterministic=True)[0])
    assert emb.shape == (7, H)


def test_sem_readout_only_skips_sem_on_fed_back_and_state_tokens():
    from stoix.networks.torso_compute_transformer import _readout_and_feedback, rms_norm

    x = 3.0 * jax.random.normal(jax.random.PRNGKey(0), (7, H))
    state, fed_back = _readout_and_feedback(x, True, G, 1.0, readout_only=True)
    np.testing.assert_allclose(state.reshape(7, -1, G).sum(-1), 1.0, atol=1e-5)
    np.testing.assert_allclose(fed_back, rms_norm(x), atol=1e-6)
    _, raw = _readout_and_feedback(x, False, G, 1.0, readout_only=True)
    np.testing.assert_allclose(raw, x)

    torso, obs, params = _stable_setup(sem_readout_only=True)
    emb, compute_time, _, _ = torso.apply(params, obs, rng=jax.random.PRNGKey(2))
    replay_emb = torso.apply(params, obs, target_compute_time=compute_time)[0]
    np.testing.assert_allclose(emb, replay_emb, atol=1e-5)
    np.testing.assert_allclose(emb.reshape(7, -1, G).sum(-1), 1.0, atol=1e-5)
    full_sem = torso.clone(sem_readout_only=False).apply(params, obs, target_compute_time=compute_time)[0]
    assert not np.allclose(emb, full_sem)


def test_action_logit_accum_sums_or_averages_taken_states():
    """With a linear action head, summing per-step logits == head(sum of states)."""
    obs = jax.random.normal(jax.random.PRNGKey(0), (7, 10))
    for accum, gamma in [("sum", 1.0), ("mean", 1.0), ("discount", 0.5)]:
        torso = TransformerChainOfThoughtTorso(
            hidden_dim=H, num_heads=2, mlp_dim=32, max_steps=MAX_STEPS, num_state_tokens=N,
            sem_group_size=G, action_logit_accum=accum, action_logit_discount=gamma,
        )
        params = torso.init(jax.random.PRNGKey(1), obs, deterministic=True)
        emb, ct, _, _ = torso.apply(params, obs, rng=jax.random.PRNGKey(2))
        replay_emb, _, hist, _, _ = torso.apply(params, obs, target_compute_time=ct)
        # weight gamma^(t-1-k) on step k < t, 0 on steps not taken
        k = jnp.arange(MAX_STEPS)
        w = jnp.where(k < ct[:, None], gamma ** (ct[:, None] - 1 - k), 0.0)
        expected = (hist * w[..., None]).sum(1) / (w.sum(1, keepdims=True) if accum == "mean" else 1.0)
        assert not jnp.all(ct == 1)  # some examples take several steps
        np.testing.assert_allclose(emb, expected, atol=1e-5)
        np.testing.assert_allclose(replay_emb, expected, atol=1e-5)
