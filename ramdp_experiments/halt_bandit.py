"""Toy multichain compute bandit: the first test for any halting-predictor loss.

Root state s (one of S) has one correct chain out of M and needs k(s) compute steps to find it.
At the root, the agent halts after N0 steps. If N0 >= k(s), it picks the correct chain (except with
prob eps it picks at random); else it guesses at random (correct with prob 1/M). It then walks L
chain states (one action each, halting there too) and gets +1 at the end of the correct chain,
else +0. Compute is paid ONLY through the discount: every compute step multiplies by gamma.
Best: N0 = argmax_N gamma^N P(correct | N), and N = 1 on chain states (compute cannot help).

Two advantage modes:
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


def exact_eval(theta, a):
    """Exact E[G_root] (gamma^(total compute) * r) and P(correct chain) of the halting tables."""
    p = jnp.exp(log_halt_dist(theta)[0])
    disc = a.gamma**STEPS
    p_ok = jnp.where(STEPS[None] >= K[:, None], 1 - a.eps + a.eps / a.chains, 1 / a.chains)
    root_g = (p[:S] * disc * p_ok).sum(-1)
    return root_g.mean() * jnp.prod(p[S:] @ disc), (p[:S] * p_ok).sum(-1).mean()


def run(method, rho, a, seed):
    M, L, B = a.chains, a.chain_len, a.batch
    params = (jnp.zeros((S + L, C - 1)), jnp.zeros(S + 2 * L))  # halting rows: S roots, L chain steps; V
    opt = optax.adam(a.lr)

    def pick_correct(key, s, n):
        k1, k2 = jax.random.split(key)
        knows = (n >= K[s]) & (jax.random.uniform(k1, s.shape) > a.eps)
        return knows | (jax.random.randint(k2, s.shape, 0, M) == 0)

    def mc_batch(params, key):  # whole episodes, Monte Carlo reward-to-go
        ks = jax.random.split(key, 4)
        lp = log_halt_dist(params[0])[0]
        s = jax.random.randint(ks[0], (B,), 0, S)
        n0 = jax.random.categorical(ks[1], lp[s]) + 1
        nc = jax.random.categorical(ks[2], lp[S:], shape=(B, L)) + 1
        correct = pick_correct(ks[3], s, n0)
        to_go = jnp.cumsum(nc[:, ::-1], -1)[:, ::-1]
        g = jnp.concatenate([a.gamma ** (n0 + to_go[:, 0])[:, None], a.gamma**to_go], -1) * correct[:, None]
        row = jnp.concatenate([s[:, None], jnp.broadcast_to(S + jnp.arange(L), (B, L))], -1).ravel()
        n, g = jnp.concatenate([n0[:, None], nc], -1).ravel(), g.ravel()
        b = jax.ops.segment_sum(g, row, S + L) / jnp.maximum(jax.ops.segment_sum(jnp.ones_like(g), row, S + L), 1)
        return row, n, g, g - b[row], 0.0

    def env_step(params, env, key):  # continuing env; env = (phase, s, correct); phase 0 = root
        phase, s, correct = env
        ks = jax.random.split(key, 3)
        row = jnp.where(phase == 0, s, S + phase - 1)
        vrow = jnp.where(phase == 0, s, S + 2 * (phase - 1) + correct)
        n = jax.random.categorical(ks[0], log_halt_dist(params[0])[0][row]) + 1
        correct = jnp.where(phase == 0, pick_correct(ks[1], s, n), correct)
        done = phase == L
        reward = (done & correct).astype(jnp.float32)
        env = (jnp.where(done, 0, phase + 1), jnp.where(done, jax.random.randint(ks[2], s.shape, 0, S), s), correct)
        return env, (row, vrow, n, reward, done)

    def td_batch(params, key, env):  # T-step rollout cut mid-episode, GAE with bootstrap
        env, (row, vrow, n, r, done) = jax.lax.scan(
            lambda e, k: env_step(params, e, k), env, jax.random.split(key, a.rollout)
        )
        phase, s, correct = env
        V = jax.lax.stop_gradient(params[1])
        last_v = V[jnp.where(phase == 0, s, S + 2 * (phase - 1) + correct)]
        v = V[vrow]
        r_t = r * a.gamma ** (n - 1)
        d_t = a.gamma**n * (1 - done)
        v_next = jnp.concatenate([v[1:], last_v[None]], 0)
        delta = r_t + d_t * v_next - v

        def gae(adv, x):
            delta, d = x
            adv = delta + d * a.gae_lambda * adv
            return adv, adv

        _, adv = jax.lax.scan(gae, jnp.zeros_like(last_v), (delta, d_t), reverse=True)
        target = adv + v
        critic_loss = jnp.mean((params[1][vrow] - target) ** 2)
        return row.ravel(), n.ravel(), target.ravel(), adv.ravel(), critic_loss, env

    def loss(params, key, env):
        if a.rollout:
            row, n, g, adv, critic_loss, env = td_batch(params, key, env)
        else:
            (row, n, g, adv, critic_loss) = mc_batch(params, key)
        return -objective(params[0], row, n, g, adv, method, a.slack, rho) + critic_loss, env

    def step(carry, key):
        params, opt_state, env = carry
        grads, env = jax.grad(loss, has_aux=True)(params, key, env)
        upd, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, upd)
        return (params, opt_state, env), jnp.stack(exact_eval(params[0], a))

    env = (jnp.zeros(B, int), jax.random.randint(jax.random.PRNGKey(seed + 1000), (B,), 0, S), jnp.zeros(B, bool))
    (params, _, _), hist = jax.lax.scan(
        step, (params, opt.init(params), env), jax.random.split(jax.random.PRNGKey(seed), a.iters)
    )
    en = np.array(jnp.exp(log_halt_dist(params[0])[0]) @ STEPS)
    return np.array(hist), en[:S], en[S:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma", type=float, default=0.9)  # discount per compute step
    ap.add_argument("--eps", type=float, default=0.1)  # prob of a random chain even with enough compute
    ap.add_argument("--chains", type=int, default=4)  # M
    ap.add_argument("--chain-len", type=int, default=3)  # L
    ap.add_argument("--slack", type=float, default=0.0)
    ap.add_argument("--rollout", type=int, default=0)  # 0 = Monte Carlo; T > 0 = bootstrapped T-step rollouts
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=1024)  # episodes (MC) or parallel envs (bootstrapped)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--out", default="plots/halt_bandit.png")
    a = ap.parse_args()
    methods = [("pg", 0, "PG"), ("as_written", 0, "as written (fail w=-1)"),
               ("bounded", 1.0, "bounded, rho=1"), ("bounded", 9.0, "bounded, rho=9")]

    p_ok = lambda n, k: np.where(n >= k, 1 - a.eps + a.eps / a.chains, 1 / a.chains)
    v = np.array([[a.gamma**n * p_ok(n, k) for n in range(1, C + 1)] for k in range(1, S + 1)])
    best_n = v.argmax(1) + 1
    best_g = (v.max(1) * a.gamma**a.chain_len).mean()
    best_ok = np.mean([p_ok(n, k) for n, k in zip(best_n, range(1, S + 1))])

    fig, ax = plt.subplots(1, 4, figsize=(20, 4.5))
    for j, (m, rho, label) in enumerate(methods):
        res = [run(m, rho, a, sd) for sd in range(a.seeds)]
        hist = np.mean([r[0] for r in res], 0)
        e0, ec = np.mean([r[1] for r in res], 0), np.mean([r[2] for r in res], 0)
        c = f"C{j}"
        ax[0].plot(K, e0, "o-", color=c, label=label)
        ax[1].plot(np.arange(1, a.chain_len + 1), ec, "o-", color=c)
        ax[2].plot(hist[:, 0], color=c)
        ax[3].plot(hist[:, 1], color=c)
        print(f"{label:24s} root E[N|s]={np.round(e0, 2)} chain E[N]={np.round(ec, 2)} "
              f"G={hist[-1, 0]:.3f} correct={hist[-1, 1]:.3f}")
    print(f"{'optimum':24s} root N={best_n} chain N=1 G={best_g:.3f} correct={best_ok:.3f}")
    ax[0].plot(K, best_n, "k--", lw=1, label="optimum")
    ax[1].axhline(1, color="k", ls="--", lw=1)
    ax[2].axhline(best_g, color="k", ls="--", lw=1)
    ax[3].axhline(best_ok, color="k", ls="--", lw=1)
    ax[0].set(xlabel="needed compute k(s)", ylabel="final E[N | s]", title="root state")
    ax[1].set(xlabel="chain step", ylabel="final E[N]", title="chain states (compute cannot help)")
    ax[2].set(xlabel="iteration", ylabel="exact E[G_root] of halting tables")
    ax[3].set(xlabel="iteration", ylabel="exact P(correct chain)")
    ax[0].legend(fontsize=8)
    mode = f"bootstrapped T={a.rollout}, GAE lambda={a.gae_lambda}" if a.rollout else "Monte Carlo"
    fig.suptitle(f"Multichain compute bandit ({mode}): {a.chains} chains of length {a.chain_len}, "
                 f"gamma={a.gamma}, eps={a.eps}, slack={a.slack}, {a.seeds} seeds")
    fig.tight_layout()
    fig.savefig(a.out, dpi=120)
    print("saved", a.out)


if __name__ == "__main__":
    main()
