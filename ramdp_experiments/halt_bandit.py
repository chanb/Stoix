"""Toy compute bandit: the first test for any halting-predictor loss.

State s needs k(s) compute steps. Halting at step N succeeds iff N >= k(s) (and an action-noise
coin with prob 1 - eps). Compute is paid ONLY through the discount: G = gamma^N * success.
So the best budget is N = k(s). A tabular halting head gives per-step hazards lam_c(s).

Usage: python ramdp_experiments/halt_bandit.py --out plots/halt_bandit.png
"""
import argparse

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

S = C = 8  # states, compute steps; k(s) = s + 1
K = jnp.arange(1, S + 1)
STEPS = jnp.arange(1, C + 1)


def log_halt_dist(theta):
    """theta [S, C-1] -> log p(N = c | s) [S, C], log P(N < c) [S, C], log P(N > c) [S, C]."""
    lam_lp = jax.nn.log_sigmoid(theta)  # log lam_c, c = 1..C-1
    cont_lp = jax.nn.log_sigmoid(-theta)  # log (1 - lam_c)
    surv = jnp.concatenate([jnp.zeros((S, 1)), jnp.cumsum(cont_lp, -1)], -1)  # log P(N >= c)
    log_p = surv + jnp.concatenate([lam_lp, jnp.zeros((S, 1))], -1)
    log_gt = jnp.concatenate([surv[:, 1:], jnp.full((S, 1), -jnp.inf)], -1)  # log P(N > c)
    log_lt = jnp.log(-jnp.expm1(jnp.minimum(surv, -1e-12)))  # log P(N < c) = log(1 - P(N >= c))
    return log_p, log_lt, log_gt


def make_loss(method, slack, rho):
    def loss(theta, s, n, g):
        log_p, log_lt, log_gt = log_halt_dist(theta)
        i = n - 1
        lt, gt, lp = log_lt[s, i], log_gt[s, i], log_p[s, i]
        good = g > slack
        if method == "pg":  # REINFORCE on the discounted return, per-state mean baseline
            b = jax.ops.segment_sum(g, s, S) / jnp.maximum(jax.ops.segment_sum(jnp.ones_like(g), s, S), 1)
            return -jnp.mean((g - b[s]) * lp)
        if method == "as_written":  # (1(G>slack) G - 1(G<=slack)) log P(N < c_h)
            w = jnp.where(good, g, -1.0)
            return -jnp.mean(jnp.where(n > 1, w * lt, 0.0))
        # bounded: success -> G log P(N < c_h), failure -> rho log P(N > c_h)
        obj = jnp.where(good, jnp.where(n > 1, g * lt, 0.0), jnp.where(n < C, rho * gt, 0.0))
        return -jnp.mean(obj)

    return loss


def run(method, slack, rho, gamma, eps, seed, iters, batch, lr):
    key = jax.random.PRNGKey(seed)
    theta = jnp.zeros((S, C - 1))  # lam = 0.5 at every step
    opt = optax.adam(lr)
    opt_state = opt.init(theta)
    loss = make_loss(method, slack, rho)

    @jax.jit
    def step(theta, opt_state, key):
        k1, k2, k3 = jax.random.split(key, 3)
        s = jax.random.randint(k1, (batch,), 0, S)
        log_p, _, _ = log_halt_dist(theta)
        n = jax.random.categorical(k2, log_p[s]) + 1
        ok = (n >= K[s]) & (jax.random.uniform(k3, (batch,)) > eps)
        g = gamma**n * ok
        grads = jax.grad(loss)(theta, s, n, g)
        upd, opt_state = opt.update(grads, opt_state)
        return optax.apply_updates(theta, upd), opt_state, (g.mean(), ok.mean(), n.mean())

    hist = []
    for key in jax.random.split(key, iters):
        theta, opt_state, m = step(theta, opt_state, key)
        hist.append(m)
    p = jnp.exp(log_halt_dist(theta)[0])
    return np.array(hist), np.array(p @ STEPS)  # history, E[N | s]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma", type=float, default=0.9)  # discount per compute step
    ap.add_argument("--eps", type=float, default=0.1)  # action noise: fail even with enough compute
    ap.add_argument("--slack-n", type=int, default=5)  # 2nd row: slack = gamma^slack_n
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--out", default="plots/halt_bandit.png")
    a = ap.parse_args()
    methods = [("pg", 0, "PG (REINFORCE on G)"), ("as_written", 0, "as written (fail w=-1)"),
               ("bounded", 1.0, "bounded, rho=1"), ("bounded", 9.0, "bounded, rho=9")]
    slacks = [(0.0, "slack = 0"), (a.gamma**a.slack_n, f"slack = gamma^{a.slack_n}")]
    opt_g = (a.gamma ** K * (1 - a.eps)).mean()  # E[G] when N = k(s)

    fig, ax = plt.subplots(len(slacks), 3, figsize=(15, 4.2 * len(slacks)), squeeze=False)
    for r, (slack, sname) in enumerate(slacks):
        for j, (m, rho, label) in enumerate(methods):
            res = [run(m, slack, rho, a.gamma, a.eps, sd, a.iters, a.batch, a.lr) for sd in range(a.seeds)]
            hist = np.mean([h for h, _ in res], 0)
            en = np.mean([e for _, e in res], 0)
            c = f"C{j}"
            ax[r, 0].plot(K, en, "o-", color=c, label=label)
            sm = lambda x: np.convolve(x, np.ones(50) / 50, "valid")
            ax[r, 1].plot(sm(hist[:, 0]), color=c)
            ax[r, 2].plot(sm(hist[:, 1]), color=c)
            print(f"{sname:14s} {label:24s} E[N|s]={np.round(en, 2)} G={hist[-200:, 0].mean():.3f} "
                  f"succ={hist[-200:, 1].mean():.3f}")
        ax[r, 0].plot(K, K, "k--", lw=1, label="best: N = k(s)")
        ax[r, 1].axhline(opt_g, color="k", ls="--", lw=1)
        ax[r, 2].axhline(1 - a.eps, color="k", ls="--", lw=1)
        ax[r, 0].set(xlabel="needed compute k(s)", ylabel="final E[N | s]", title=sname)
        ax[r, 1].set(xlabel="iteration", ylabel="mean discounted return G", title=sname)
        ax[r, 2].set(xlabel="iteration", ylabel="success rate", title=sname)
    ax[0, 0].legend(fontsize=8)
    fig.suptitle(f"Compute bandit: gamma={a.gamma}, action noise eps={a.eps}, {a.seeds} seeds")
    fig.tight_layout()
    fig.savefig(a.out, dpi=120)
    print("saved", a.out)


if __name__ == "__main__":
    main()
