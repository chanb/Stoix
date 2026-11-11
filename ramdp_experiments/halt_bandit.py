"""Toy multichain compute bandit: the first test for any halting-predictor loss.

Root state s (one of S) has one correct chain out of M and needs k(s) compute steps to find it.
At the root, the agent halts after N0 steps. If N0 >= k(s), it picks the correct chain (except with
prob eps it picks at random); else it guesses at random (correct with prob 1/M). It then walks L
chain states (one action each, halting there too) and gets +1 at the end of the correct chain,
else +0. Compute is paid ONLY through the discount: every compute step multiplies by gamma, so
G_t = gamma^(compute from t to the end) * r. Best: N0 = argmax_N gamma^N P(correct | N), and N = 1
on chain states (compute there cannot help).

Usage: python ramdp_experiments/halt_bandit.py --out plots/halt_bandit.png
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


def objective(theta, row, n, g, method, slack, rho):
    """Mean per-sample objective (to maximize) for halting at step n in table row `row`."""
    log_p, log_lt, log_gt = log_halt_dist(theta)
    lp, lt, gt = log_p[row, n - 1], log_lt[row, n - 1], log_gt[row, n - 1]
    good = g > slack
    if method == "pg":  # REINFORCE on the discounted return, per-row mean baseline
        R = theta.shape[0]
        b = jax.ops.segment_sum(g, row, R) / jnp.maximum(jax.ops.segment_sum(jnp.ones_like(g), row, R), 1)
        return jnp.mean((g - b[row]) * lp)
    if method == "as_written":  # (1(G>slack) G - 1(G<=slack)) log P(N < c_h)
        return jnp.mean(jnp.where(n > 1, jnp.where(good, g, -1.0) * lt, 0.0))
    # bounded: success -> G log P(N < c_h), failure -> rho log P(N > c_h)
    return jnp.mean(jnp.where(good, jnp.where(n > 1, g * lt, 0.0), jnp.where(n < C, rho * gt, 0.0)))


def run(method, slack, rho, a, seed):
    M, L = a.chains, a.chain_len
    params = (jnp.zeros((S, C - 1)), jnp.zeros((L, C - 1)))  # root table, chain table; lam = 0.5
    opt = optax.adam(a.lr)

    def loss(params, s, n0, nc, g0, gc):
        root = objective(params[0], s, n0, g0, method, slack, rho)
        rows = jnp.broadcast_to(jnp.arange(L), nc.shape).ravel()
        chain = objective(params[1], rows, nc.ravel(), gc.ravel(), method, slack, rho)
        return -(root + chain)

    def step(carry, key):
        params, opt_state = carry
        ks = jax.random.split(key, 5)
        s = jax.random.randint(ks[0], (a.batch,), 0, S)
        n0 = jax.random.categorical(ks[1], log_halt_dist(params[0])[0][s]) + 1
        nc = jax.random.categorical(ks[2], log_halt_dist(params[1])[0], shape=(a.batch, L)) + 1
        knows = (n0 >= K[s]) & (jax.random.uniform(ks[3], (a.batch,)) > a.eps)
        correct = knows | (jax.random.randint(ks[4], (a.batch,), 0, M) == 0)
        to_go = jnp.cumsum(nc[:, ::-1], -1)[:, ::-1]  # compute from chain step j to the end
        gc = a.gamma**to_go * correct[:, None]
        g0 = a.gamma ** (n0 + to_go[:, 0]) * correct
        grads = jax.grad(loss)(params, s, n0, nc, g0, gc)
        upd, opt_state = opt.update(grads, opt_state)
        return (optax.apply_updates(params, upd), opt_state), (g0.mean(), correct.mean(), nc.mean())

    (params, _), hist = jax.lax.scan(
        step, (params, opt.init(params)), jax.random.split(jax.random.PRNGKey(seed), a.iters)
    )
    en = lambda th: np.array(jnp.exp(log_halt_dist(th)[0]) @ STEPS)
    return np.stack(hist, -1), en(params[0]), en(params[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma", type=float, default=0.9)  # discount per compute step
    ap.add_argument("--eps", type=float, default=0.1)  # prob of a random chain even with enough compute
    ap.add_argument("--chains", type=int, default=4)  # M
    ap.add_argument("--chain-len", type=int, default=3)  # L
    ap.add_argument("--slack", type=float, default=0.0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--out", default="plots/halt_bandit.png")
    a = ap.parse_args()
    methods = [("pg", 0, "PG (REINFORCE on G)"), ("as_written", 0, "as written (fail w=-1)"),
               ("bounded", 1.0, "bounded, rho=1"), ("bounded", 9.0, "bounded, rho=9")]

    # Optimal root budget per state; chain states use N = 1.
    p_ok = lambda n, k: np.where(n >= k, 1 - a.eps + a.eps / a.chains, 1 / a.chains)
    v = np.array([[a.gamma**n * p_ok(n, k) for n in range(1, C + 1)] for k in range(1, S + 1)])
    best_n = v.argmax(1) + 1
    best_g = (v.max(1) * a.gamma**a.chain_len).mean()
    best_ok = np.mean([p_ok(n, k) for n, k in zip(best_n, range(1, S + 1))])

    fig, ax = plt.subplots(1, 4, figsize=(20, 4.5))
    sm = lambda x: np.convolve(x, np.ones(50) / 50, "valid")
    for j, (m, rho, label) in enumerate(methods):
        res = [run(m, a.slack, rho, a, sd) for sd in range(a.seeds)]
        hist = np.mean([r[0] for r in res], 0)
        e0, ec = np.mean([r[1] for r in res], 0), np.mean([r[2] for r in res], 0)
        c = f"C{j}"
        ax[0].plot(K, e0, "o-", color=c, label=label)
        ax[1].plot(np.arange(1, a.chain_len + 1), ec, "o-", color=c)
        ax[2].plot(sm(hist[:, 0]), color=c)
        ax[3].plot(sm(hist[:, 1]), color=c)
        print(f"{label:24s} root E[N|s]={np.round(e0, 2)} chain E[N]={np.round(ec, 2)} "
              f"G={hist[-200:, 0].mean():.3f} correct={hist[-200:, 1].mean():.3f}")
    print(f"{'optimum':24s} root N={best_n} chain N=1 G={best_g:.3f} correct={best_ok:.3f}")
    ax[0].plot(K, best_n, "k--", lw=1, label="optimum")
    ax[1].axhline(1, color="k", ls="--", lw=1)
    ax[2].axhline(best_g, color="k", ls="--", lw=1)
    ax[3].axhline(best_ok, color="k", ls="--", lw=1)
    ax[0].set(xlabel="needed compute k(s)", ylabel="final E[N | s]", title="root state")
    ax[1].set(xlabel="chain step", ylabel="final E[N]", title="chain states (compute cannot help)")
    ax[2].set(xlabel="iteration", ylabel="mean discounted return G (root)")
    ax[3].set(xlabel="iteration", ylabel="correct-chain rate")
    ax[0].legend(fontsize=8)
    fig.suptitle(f"Multichain compute bandit: {a.chains} chains of length {a.chain_len}, gamma={a.gamma}, "
                 f"eps={a.eps}, slack={a.slack}, {a.seeds} seeds")
    fig.tight_layout()
    fig.savefig(a.out, dpi=120)
    print("saved", a.out)


if __name__ == "__main__":
    main()
