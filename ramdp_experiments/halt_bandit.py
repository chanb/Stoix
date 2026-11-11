"""Toy multichain compute bandit: the first test for any halting-predictor loss.

Root state s (one of S) gets a correct chain c ~ U{0..M-1}, drawn fresh each episode. At the
root, the agent halts after N0 compute steps, then a LEARNED action head picks a chain. The head
sees (s, c) only if N0 >= k(s) = s + 1; with less compute it sees one shared "unknown" input, so it
can do no better than 1/M (c is independent of everything it sees). So compute helps only once the action head has learned to use it. The agent then walks L
chain states (one action each, halting there too) and gets +1 at the end of the correct chain,
else +0. Compute is paid ONLY through the discount: every compute step multiplies by gamma.
Best: N0 = k(s) (when gamma^k > gamma / M), the correct chain, and N = 1 on chain states.

Two advantage modes (the action head and halting head both use the same advantage):
  --rollout 0 : Monte Carlo. Whole episodes, G_t = gamma^(compute from t to the end) * r,
                advantage = G - (per-row batch mean).
  --rollout T : bootstrapped, as in Stoix ff_ppo. Continuing env, rollouts cut after T env
                steps (episodes are L+1 steps, so most are cut). Tabular critic V, GAE(lambda)
                with r_t = r gamma^(C-1), d_t = gamma^C (1 - done), bootstrap V(s_T) at the cut.
                "G" for the ratchet losses is the lambda-return A + V.

Usage: python ramdp_experiments/halt_bandit.py [--rollout 2 --gae-lambda 0.95] --out plots/x.png
"""
import argparse

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

S = C = 8  # root states, max compute steps; k(s) = s + 1
K = jnp.arange(1, S + 1)
STEPS = jnp.arange(1, C + 1)


def log_halt_dist(theta):
    """theta [R, C-1] -> log p(N = c), log P(N < c), log P(N > c), each [R, C]."""
    cont_lp = jax.nn.log_sigmoid(-theta)  # log (1 - lam_c)
    surv = jnp.concatenate([jnp.zeros_like(theta[:, :1]), jnp.cumsum(cont_lp, -1)], -1)  # log P(N >= c)
    log_p = surv + jnp.concatenate([jax.nn.log_sigmoid(theta), jnp.zeros_like(theta[:, :1])], -1)
    log_gt = jnp.concatenate([surv[:, 1:], jnp.full_like(surv[:, :1], -jnp.inf)], -1)
    log_lt = jnp.log(-jnp.expm1(jnp.minimum(surv, -1e-12)))
    return log_p, log_lt, log_gt


def objective(theta, row, n, g, adv, method, slack, rho):
    """Mean per-sample objective (to maximize) for halting at step n in halting-table row `row`."""
    log_p, log_lt, log_gt = log_halt_dist(theta)
    lp, lt, gt = log_p[row, n - 1], log_lt[row, n - 1], log_gt[row, n - 1]
    good = g > slack
    if method == "pg":
        return jnp.mean(adv * lp)
    if method == "as_written":  # (1(G>slack) G - 1(G<=slack)) log P(N < c_h)
        return jnp.mean(jnp.where(n > 1, jnp.where(good, g, -1.0) * lt, 0.0))
    # bounded: success -> G log P(N < c_h), failure -> rho log P(N > c_h)
    return jnp.mean(jnp.where(good, jnp.where(n > 1, g * lt, 0.0), jnp.where(n < C, rho * gt, 0.0)))


def exact_eval(params, a):
    """Exact E[G_root], P(correct chain), and mean action-head P(correct | enough compute)."""
    theta, phi = params[0], params[2]
    p = jnp.exp(log_halt_dist(theta)[0])
    M = a.chains
    pi = jax.nn.softmax(phi)
    knows = pi[:-1].reshape(S, M, M).diagonal(axis1=1, axis2=2).mean(-1)  # mean_c pi(c | s, c)
    guess = pi[-1].mean()  # mean_c pi(c | unknown) = 1/M
    p_ok = jnp.where(STEPS[None] >= K[:, None], knows[:, None], guess)  # [S, C]
    disc = a.gamma**STEPS
    g = (p[:S] * disc * p_ok).sum(-1).mean() * jnp.prod(p[S:] @ disc)
    return jnp.stack([g, (p[:S] * p_ok).sum(-1).mean(), knows.mean()])


def run(method, rho, a, seed, ent_mode="none", ent_coef=0.0):
    """ent_mode (bootstrapped mode only): "none"; "step" = Stoix halting_ent_coef (mean Bernoulli
    entropy over visited halting decisions); "value" = entropy return V_ent, with per-env-step reward
    r_ent = sum_{c<=N, c<C} gamma^(c-1) H(lam_c), discounted across env steps like the reward, its own
    critic, and halting advantage A + ent_coef * A_ent (plus the direct grad of r_ent)."""
    M, L, B = a.chains, a.chain_len, a.batch
    assert ent_mode == "none" or a.rollout, "entropy modes need --rollout > 0"
    # halting rows: S roots, L chain steps; V: S roots, (chain step, correct?); action rows: (s, c) pairs
    # + "unknown"; V_ent like V
    params = (jnp.zeros((S + L, C - 1)), jnp.zeros(S + 2 * L), jnp.zeros((S * M + 1, M)), jnp.zeros(S + 2 * L))
    opt = optax.adam(a.lr)

    def act(phi, key, s, n):
        k1, k2 = jax.random.split(key)
        c = jax.random.randint(k1, s.shape, 0, M)  # correct chain, fresh each episode
        arow = jnp.where(n >= K[s], s * M + c, S * M)  # the head sees (s, c) only with enough compute
        act = jax.random.categorical(k2, phi[arow])
        return arow, act, act == c

    def action_obj(phi, arow, act, adv):
        return jnp.mean(adv * jax.nn.log_softmax(phi)[arow, act])

    def mc_batch(params, key):  # whole episodes, Monte Carlo reward-to-go
        ks = jax.random.split(key, 4)
        lp = log_halt_dist(params[0])[0]
        s = jax.random.randint(ks[0], (B,), 0, S)
        n0 = jax.random.categorical(ks[1], lp[s]) + 1
        nc = jax.random.categorical(ks[2], lp[S:], shape=(B, L)) + 1
        arow, action, correct = act(params[2], ks[3], s, n0)
        to_go = jnp.cumsum(nc[:, ::-1], -1)[:, ::-1]
        g = jnp.concatenate([a.gamma ** (n0 + to_go[:, 0])[:, None], a.gamma**to_go], -1) * correct[:, None]
        row = jnp.concatenate([s[:, None], jnp.broadcast_to(S + jnp.arange(L), (B, L))], -1).ravel()
        n, g = jnp.concatenate([n0[:, None], nc], -1).ravel(), g.ravel()
        b = jax.ops.segment_sum(g, row, S + L) / jnp.maximum(jax.ops.segment_sum(jnp.ones_like(g), row, S + L), 1)
        adv = g - b[row]
        root_adv = adv.reshape(B, L + 1)[:, 0]
        return row, n, g, adv, action_obj(params[2], arow, action, root_adv), 0.0, 0.0

    def env_step(params, env, key):  # continuing env; env = (phase, s, correct); phase 0 = root
        phase, s, correct = env
        ks = jax.random.split(key, 3)
        row = jnp.where(phase == 0, s, S + phase - 1)
        vrow = jnp.where(phase == 0, s, S + 2 * (phase - 1) + correct)
        n = jax.random.categorical(ks[0], log_halt_dist(params[0])[0][row]) + 1
        arow, action, new_correct = act(params[2], ks[1], s, n)
        correct = jnp.where(phase == 0, new_correct, correct)
        done = phase == L
        reward = (done & correct).astype(jnp.float32)
        env = (jnp.where(done, 0, phase + 1), jnp.where(done, jax.random.randint(ks[2], s.shape, 0, S), s), correct)
        return env, (row, vrow, n, reward, done, phase == 0, arow, action)

    def td_batch(params, key, env):  # T-step rollout cut mid-episode, GAE with bootstrap
        env, (row, vrow, n, r, done, is_root, arow, action) = jax.lax.scan(
            lambda e, k: env_step(params, e, k), env, jax.random.split(key, a.rollout)
        )
        phase, s, correct = env
        last_vrow = jnp.where(phase == 0, s, S + 2 * (phase - 1) + correct)
        d_t = a.gamma**n * (1 - done)

        def gae(r_t, V):  # -> (advantage, critic loss)
            Vs = jax.lax.stop_gradient(V)
            v = Vs[vrow]
            delta = r_t + d_t * jnp.concatenate([v[1:], Vs[last_vrow][None]], 0) - v

            def f(adv, x):
                adv = x[0] + x[1] * a.gae_lambda * adv
                return adv, adv

            _, adv = jax.lax.scan(f, jnp.zeros_like(v[0]), (delta, d_t), reverse=True)
            return adv, jnp.mean((V[vrow] - (adv + v)) ** 2)

        adv, critic_loss = gae(r * a.gamma ** (n - 1), params[1])
        target = adv + params[1][vrow]
        x = params[0][row]  # halting logits of the visited rows [T, B, C-1]
        H = jax.nn.softplus(x) - x * jax.nn.sigmoid(x)  # Bernoulli entropy of each decision
        visited = jnp.arange(C - 1) < n[..., None]  # decisions at steps 1..min(N, C-1)
        ent_obj, halt_adv = 0.0, adv
        if ent_mode == "step":
            ent_obj = ent_coef * jnp.sum(H * visited) / jnp.maximum(visited.sum(), 1)
        if ent_mode == "value":
            r_ent = jnp.sum(a.gamma ** jnp.arange(C - 1) * H * visited, -1)
            adv_ent, ent_critic_loss = gae(jax.lax.stop_gradient(r_ent), params[3])
            halt_adv = adv + ent_coef * adv_ent  # the action head keeps the plain advantage
            ent_obj = ent_coef * jnp.mean(r_ent)  # direct gradient of this step's entropy reward
            critic_loss = critic_loss + ent_critic_loss
        a_obj = action_obj(params[2], arow, action, adv * is_root) * is_root.size / jnp.maximum(is_root.sum(), 1)
        return row.ravel(), n.ravel(), target.ravel(), halt_adv.ravel(), a_obj, ent_obj, critic_loss, env

    def loss(params, key, env):
        if a.rollout:
            row, n, g, adv, a_obj, ent_obj, critic_loss, env = td_batch(params, key, env)
        else:
            row, n, g, adv, a_obj, ent_obj, critic_loss = mc_batch(params, key)
        halt_obj = objective(params[0], row, n, g, adv, method, a.slack, rho)
        return -(halt_obj + a_obj + ent_obj) + critic_loss, env

    def step(carry, key):
        params, opt_state, env = carry
        grads, env = jax.grad(loss, has_aux=True)(params, key, env)
        upd, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, upd)
        return (params, opt_state, env), exact_eval(params, a)

    env = (jnp.zeros(B, int), jax.random.randint(jax.random.PRNGKey(seed + 1000), (B,), 0, S), jnp.zeros(B, bool))
    (params, _, _), hist = jax.lax.scan(
        step, (params, opt.init(params), env), jax.random.split(jax.random.PRNGKey(seed), a.iters)
    )
    en = np.array(jnp.exp(log_halt_dist(params[0])[0]) @ STEPS)
    return np.array(hist), en[:S], en[S:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma", type=float, default=0.9)  # discount per compute step
    ap.add_argument("--chains", type=int, default=4)  # M
    ap.add_argument("--chain-len", type=int, default=3)  # L
    ap.add_argument("--slack", type=float, default=0.0)
    ap.add_argument("--rollout", type=int, default=0)  # 0 = Monte Carlo; T > 0 = bootstrapped T-step rollouts
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=1024)  # episodes (MC) or parallel envs (bootstrapped)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--compare", choices=["loss", "ent"], default="loss")  # halting losses, or PG entropy terms
    ap.add_argument("--ent-coefs", type=float, nargs="+", default=[0.01, 0.1])
    ap.add_argument("--out", default="plots/halt_bandit.png")
    a = ap.parse_args()
    if a.compare == "loss":
        methods = [("pg", 0, "PG", "none", 0.0), ("as_written", 0, "as written (fail w=-1)", "none", 0.0),
                   ("bounded", 1.0, "bounded, rho=1", "none", 0.0), ("bounded", 9.0, "bounded, rho=9", "none", 0.0)]
    else:  # PG with halting entropy terms
        methods = [("pg", 0, "PG, no entropy", "none", 0.0)] + [
            ("pg", 0, f"PG + {mode} entropy {cf}", mode, cf) for mode in ("step", "value") for cf in a.ent_coefs
        ]

    p_ok = lambda n, k: np.where(n >= k, 1.0, 1 / a.chains)
    v = np.array([[a.gamma**n * p_ok(n, k) for n in range(1, C + 1)] for k in range(1, S + 1)])
    best_n = v.argmax(1) + 1
    best = [(v.max(1) * a.gamma**a.chain_len).mean(), np.mean([p_ok(n, k) for n, k in zip(best_n, range(1, S + 1))]), 1.0]

    fig, ax = plt.subplots(1, 5, figsize=(25, 4.5))
    for j, (m, rho, label, ent_mode, ent_coef) in enumerate(methods):
        res = [run(m, rho, a, sd, ent_mode, ent_coef) for sd in range(a.seeds)]
        print(f"{label:24s} per-seed E[N | k=8] = {np.round([r[1][-1] for r in res], 2)}")
        hist = np.mean([r[0] for r in res], 0)
        e0, ec = np.mean([r[1] for r in res], 0), np.mean([r[2] for r in res], 0)
        c = f"C{j}"
        ax[0].plot(K, e0, "o-", color=f"C{j}", label=label)
        ax[1].plot(np.arange(1, a.chain_len + 1), ec, "o-", color=c)
        for i in range(3):
            ax[2 + i].plot(hist[:, i], color=c)
        print(f"{label:24s} root E[N|s]={np.round(e0, 2)} chain E[N]={np.round(ec, 2)} "
              f"G={hist[-1, 0]:.3f} correct={hist[-1, 1]:.3f} pi(correct|enough)={hist[-1, 2]:.3f}")
    print(f"{'optimum':24s} root N={best_n} chain N=1 G={best[0]:.3f} correct={best[1]:.3f}")
    ax[0].plot(K, best_n, "k--", lw=1, label="optimum")
    ax[1].axhline(1, color="k", ls="--", lw=1)
    for i in range(3):
        ax[2 + i].axhline(best[i], color="k", ls="--", lw=1)
    ax[0].set(xlabel="needed compute k(s)", ylabel="final E[N | s]", title="root state")
    ax[1].set(xlabel="chain step", ylabel="final E[N]", title="chain states (compute cannot help)")
    ax[2].set(xlabel="iteration", ylabel="exact E[G_root]")
    ax[3].set(xlabel="iteration", ylabel="exact P(correct chain)")
    ax[4].set(xlabel="iteration", ylabel="action head: P(correct | enough compute)")
    ax[0].legend(fontsize=8)
    mode = f"bootstrapped T={a.rollout}, GAE lambda={a.gae_lambda}" if a.rollout else "Monte Carlo"
    fig.suptitle(f"Multichain compute bandit, learned action head ({mode}): {a.chains} chains of length "
                 f"{a.chain_len}, gamma={a.gamma}, {a.seeds} seeds")
    fig.tight_layout()
    fig.savefig(a.out, dpi=120)
    print("saved", a.out)


if __name__ == "__main__":
    main()
