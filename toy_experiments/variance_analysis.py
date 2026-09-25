"""Variance analysis of the three policy-evaluation estimators (paper Sec 4.1).

Setup follows Theorem 4: fix a pair (s, a) at the initial state, draw N
independent rollouts; rollout i takes first action (a, C_i) with C_i ~ rho
drawn i.i.d. from a sampling distribution rho over N+, follows pi thereafter,
and records the RAMDP return G_i, so that E[G_i | C_i = c] = Q_c =
gamma^(c-1) Q_1 (Lemma 4). The estimators of Q_c compared are:

  single :  one rollout with C_i = c            (single-sample rollout)
  sep    :  mean of G_i over {i : C_i = c}      (per-runtime)
  fac    :  gamma^(c-1) * mean_i gamma^(-(C_i-1)) G_i   (factorized, pools all N)

Because runtime enters the return only through the multiplicative factor
gamma^(C-1) (the computation is priced by discount alone and does not affect
the environment), each rollout is simulated as G_i = gamma^(C_i - 1) * B_i,
where B_i = r_0 + gamma * (RAMDP return from s') is the runtime-1 return of an
independent env rollout with forced first action a, and C_i ~ rho independent
of B_i. This is exactly the generative process mu in Sec 4.2.

We replicate the batch M times and report, per c: empirical variance, bias
against ground truth Q_c (from a large pooled batch), MSE, and the fraction of
replications where the sep/single estimators are undefined (N_c = 0).
"""
import csv
import json
import numpy as np

from ramdp_frozenlake import FrozenLakeRAMDP, ramdp_returns
from pondernet_policy import PonderNetPolicy
from estimators import q_single, q_sep, q_fac


def load_policy(path, c_max=10000000, n_states=None):
    """Load a saved policy, inferring architecture and state encoding
    (one-hot vs scalar) from the parameter shapes."""
    from pondernet_policy import ScalarPonderNetPolicy
    data = np.load(path)
    n_actions, hidden = data["W_a"].shape
    d = data["W_s"].shape[1]
    if d == 1:  # scalar encoding: n_states needed only for normalization
        n_states = n_states or FrozenLakeRAMDP().n_states
        pol = ScalarPonderNetPolicy(n_states, n_actions, hidden, c_max)
    else:
        pol = PonderNetPolicy(d, n_actions, hidden, c_max)
    for k in pol.params:
        pol.params[k] = data[k] if np.ndim(pol.params[k]) else float(data[k])
    return pol


def policy_value_dp(env, policy, gamma, tol=1e-12, unroll_cap=100000):
    """Exact V^pi((s, 0)) of a fixed policy, via the runtime factorization.

    For each state, the deterministic PonderNet recurrence is unrolled to
    accumulate the discount-weighted joint action-runtime mass
        kappa[s, a] = sum_c P(c | s) gamma^(c-1) rho(a | x_c, s),
    (the never-halting residual earns zero reward, so truncating the unroll
    once the surviving discounted mass falls below tol is exact up to tol).
    Lemma 4 then gives V(s) = sum_a kappa[s, a] (rbar(s, a) + gamma E[V(s')]),
    a linear system solved exactly. Used to bootstrap truncated rollouts.
    """
    from pondernet_policy import _sigmoid, _softmax
    nS, nA = env.n_states, env.n_actions
    p = policy.params
    kappa = np.zeros((nS, nA))
    for s in range(nS):
        if env.terminal[s]:
            continue
        u_s = policy._state_input(s)
        x = np.zeros(policy.hidden)
        w, disc, i = 1.0, 1.0, 0  # survival prob, gamma^(c-1), step
        while True:
            i += 1
            x = np.tanh(p["W_x"] @ x + u_s + p["b"])
            lam = 1.0 if i >= policy.c_max else _sigmoid(p["w_h"] @ x + p["b_h"])
            kappa[s] += w * lam * disc * _softmax(p["W_a"] @ x + p["b_a"])
            w *= 1.0 - lam
            disc *= gamma
            if w * disc < tol or i >= unroll_cap:
                break
    R, M = np.zeros(nS), np.zeros((nS, nS))
    for s in range(nS):
        for a in range(nA):
            if kappa[s, a] == 0.0:
                continue
            for pr, s2, r, done in env.transitions[s][a]:
                R[s] += kappa[s, a] * pr * r
                M[s, s2] += gamma * kappa[s, a] * pr
    return np.linalg.solve(np.eye(nS) - M, R)


def sample_B(env, policy, a0, gamma, rng, V=None):
    """One rollout with forced first action a0; returns the runtime-1 return
    B = r_0 + gamma * G(s') (i.e., the sampled Q((s0,0),(a0,1))).

    Truncates when the cumulative runtime exceeds env.max_decisions and, if V
    is given, bootstraps the truncated tail with V[s_final]."""
    s = env.reset()
    s2, r0, done = env.step(s, a0, rng)
    runtimes, rewards = [], []
    t = 1
    s = s2
    while not done and t < env.max_decisions:
        a, c, _ = policy.sample(s, rng)
        s, r, done = env.step(s, a, rng)
        rewards.append(r)
        runtimes.append(c)
        t += c
    boot = V[s] if (V is not None and not done) else 0.0
    tail = ramdp_returns(rewards, runtimes, gamma, bootstrap=boot)[0] \
        if rewards else boot
    return r0 + gamma * tail


def runtime_sampler(kind, policy, s0, p_geom, c_max, rng, size):
    if kind == "geometric":
        c = rng.geometric(p_geom, size=size)
        return np.minimum(c, c_max)
    # policy's own halting distribution at s0
    return np.array([policy.sample_runtime(s0, rng) for _ in range(size)])


def main(policy_path="results/policy_fac_seed0.npz", out_dir="results",
         gamma=0.99, N=64, M=1000, n_truth=100000, cs=range(1, 9),
         rho="geometric", p_geom=0.35, seed=123, time_budget=None):
    """Resumable: caches ground-truth samples and replications to disk, so it
    can be re-invoked (optionally with a wall-clock time_budget in seconds)
    until it completes."""
    import os
    import pickle
    import time
    os.makedirs(out_dir, exist_ok=True)
    env = FrozenLakeRAMDP(slippery=True)  # should match the training env
    policy = load_policy(policy_path)
    s0 = env.reset()
    # evaluate the policy's most probable runtime-1 action at s0
    a0 = int(np.argmax(policy.action_probs(s0, 1)))
    cs = list(cs)
    t0 = time.perf_counter()

    # exact V^pi for bootstrapping truncated rollouts (+ a DP cross-check of
    # the Monte-Carlo ground truth below)
    V = policy_value_dp(env, policy, gamma)
    Q1_dp = sum(pr * (r + gamma * V[s2])
                for pr, s2, r, done in env.transitions[s0][a0])
    print(f"DP: V(s0)={V[s0]:.5f}, Q_1(s0,a0)={Q1_dp:.5f}", flush=True)

    def out_of_time():
        return time_budget and time.perf_counter() - t0 > time_budget

    # ---- ground truth: Q_1 from a large batch of B samples, Q_c = g^(c-1) Q_1
    truth_path = f"{out_dir}/var_B_truth.npy"
    B_truth = np.load(truth_path) if os.path.exists(truth_path) else np.empty(0)
    rng = np.random.default_rng(seed + B_truth.size)
    while B_truth.size < n_truth:
        n_new = min(5000, n_truth - B_truth.size)
        B_new = [sample_B(env, policy, a0, gamma, rng, V) for _ in range(n_new)]
        B_truth = np.concatenate([B_truth, B_new])
        np.save(truth_path, B_truth)
        if out_of_time():
            print(f"truth samples: {B_truth.size}/{n_truth} (budget hit)")
            return None
    Q1 = B_truth.mean()
    print(f"a0={a0}, Q_1 ~= {Q1:.5f} (n={n_truth}), Var(B)={B_truth.var():.5f}",
          flush=True)

    # ---- M replications of N-rollout batches
    rep_path = f"{out_dir}/var_reps.pkl"
    if os.path.exists(rep_path):
        with open(rep_path, "rb") as f:
            ck = pickle.load(f)
        est, m_done, rng_state = ck["est"], ck["m"], ck["rng_state"]
        rng = np.random.default_rng()
        rng.bit_generator.state = rng_state
    else:
        est = {name: {c: [] for c in cs} for name in ("single", "sep", "fac")}
        m_done = 0
        rng = np.random.default_rng(seed + 777)
    for m in range(m_done, M):
        B = np.array([sample_B(env, policy, a0, gamma, rng, V) for _ in range(N)])
        C = runtime_sampler(rho, policy, s0, p_geom, policy.c_max, rng, N)
        G = gamma ** (C - 1.0) * B
        for c in cs:
            est["single"][c].append(q_single(G, C, c, rng))
            est["sep"][c].append(q_sep(G, C, c))
            est["fac"][c].append(q_fac(G, C, c, gamma))
        if (m + 1) % 50 == 0 or m == M - 1:
            with open(rep_path, "wb") as f:
                pickle.dump(dict(est=est, m=m + 1,
                                 rng_state=rng.bit_generator.state), f)
            if out_of_time():
                print(f"replications: {m + 1}/{M} (budget hit)", flush=True)
                return None

    # ---- aggregate
    rows = []
    for c in cs:
        Qc = gamma ** (c - 1) * Q1
        row = {"c": c, "Q_true": Qc}
        for name in ("single", "sep", "fac"):
            x = np.array(est[name][c], dtype=float)
            ok = ~np.isnan(x)
            row[f"{name}_var"] = float(x[ok].var()) if ok.sum() > 1 else np.nan
            row[f"{name}_bias"] = float(x[ok].mean() - Qc) if ok.any() else np.nan
            row[f"{name}_mse"] = float(np.mean((x[ok] - Qc) ** 2)) if ok.any() else np.nan
            row[f"{name}_undefined_frac"] = float(np.mean(~ok))
        rows.append(row)

    keys = list(rows[0].keys())
    with open(f"{out_dir}/variance_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    with open(f"{out_dir}/variance_config.json", "w") as f:
        json.dump(dict(policy_path=policy_path, gamma=gamma, N=N, M=M,
                       n_truth=n_truth, rho=rho, p_geom=p_geom, a0=a0,
                       Q1=float(Q1), Q1_dp=float(Q1_dp), V_s0=float(V[s0]),
                       VarB=float(B_truth.var()), seed=seed,
                       slippery=True), f,
                  indent=2)
    for r in rows:
        print({k: (round(v, 6) if isinstance(v, float) else v)
               for k, v in r.items()}, flush=True)

    plot(rows, cs, N, M, rho, out_dir)
    return rows


def plot(rows, cs, N, M, rho, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import os
    import seaborn as sns
    sns.set_palette("colorblind")
    doc_width_pt = 452.9679

    def set_size(width_pt, fraction=1, subplots=(1, 1), use_golden_ratio=True):
        """
        Reference: https://jwalton.info/Matplotlib-latex-PGF/
        Set figure dimensions to sit nicely in our document.

        Parameters
        ----------
        width_pt: float
                Document width in points
        fraction: float, optional
                Fraction of the width which you wish the figure to occupy
        subplots: array-like, optional
                The number of rows and columns of subplots.
        Returns
        -------
        fig_dim: tuple
                Dimensions of figure in inches
        """
        # Width of figure (in pts)
        fig_width_pt = width_pt * fraction
        # Convert from pt to inches
        inches_per_pt = 1 / 72.27

        # Figure width in inches
        fig_width_in = fig_width_pt * inches_per_pt
        if use_golden_ratio:
            # Golden ratio to set aesthetic figure height
            golden_ratio = (5**0.5 - 1) / 2

            # Figure height in inches
            fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])
        else:
            fig_height_in = fig_width_in * (subplots[0] / subplots[1])

        return (fig_width_in, fig_height_in)

    pgf_with_latex = {  # setup matplotlib to use latex for output
        "pgf.texsystem": "pdflatex",  # change this if using xetex or lautex
        "text.usetex": True,  # use LaTeX to write all text
        "font.family": "serif",
        "font.serif": [],  # blank entries should cause plots to inherit fonts from the document
        "font.sans-serif": [],
        "font.monospace": [],
        "axes.labelsize": 10,  # LaTeX default is 10pt font.
        "font.size": 10,
        "legend.fontsize": 8,  # Make the legend/label fonts a little smaller
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pgf.rcfonts": False,  # don't setup fonts from rc parameters
    }

    if os.path.abspath(out_dir).startswith("/Users"):
        plt.rcParams.update(pgf_with_latex)

    names = [("single", "Single-sample rollout", "tab:red"),
             ("sep", r"$\hat{Q}^{\rm sep}_c$ (per-runtime)", "tab:blue"),
             ("fac", r"$\hat{Q}^{\rm fac}_c$ (factorized)", "tab:green")]
    
    num_rows = 1
    num_cols = 3
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=set_size(doc_width_pt, 0.95, (num_rows, num_cols), use_golden_ratio=False),
        layout="constrained",
    )

    for name, label, color in names:
        axes[0].plot(cs, [r[f"{name}_var"] for r in rows], "o-", color=color, label=label)
        axes[1].plot(cs, [r[f"{name}_mse"] for r in rows], "o-", color=color)
    axes[0].set_yscale("log")
    axes[0].set_title(f"Var$(Q_c)$")
    axes[0].set_ylabel("Variance")
    axes[1].set_yscale("log")
    axes[1].set_title("MSE against $Q_c$")
    axes[1].set_ylabel("MSE")
    for name, label, color in names[:2]:
        axes[2].plot(cs, [r[f"{name}_undefined_frac"] for r in rows], "o-",
                     color=color)
    axes[2].set_title("Unseen $c$ in batch")
    axes[2].set_ylabel("Frac. of batches")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.supxlabel("runtime $c$", y=0.1)
    # fig.suptitle(f"Critic estimators on FrozenLake "
    #              f"(runtime dist. $\\rho$: {rho}, $N$={N}, $M$={M})")
    fig.legend(
        bbox_to_anchor=(0.0, 1.0, 1.0, 0.0),
        loc="lower center",
        ncols=3,
        borderaxespad=0.0,
        frameon=True,
        fontsize="8", 
    )
    fig.tight_layout()
    fig.savefig(f"{out_dir}/variance_analysis.pdf", dpi=600, format="pdf", bbox_inches="tight")
    print(f"saved {out_dir}/variance_analysis.pdf")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", type=str, default="results/policy_fac_seed0.npz")
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--M", type=int, default=1000)
    ap.add_argument("--n-truth", type=int, default=100000)
    ap.add_argument("--rho", type=str, default="geometric",
                    choices=["geometric", "policy"])
    ap.add_argument("--out", type=str, default="results")
    ap.add_argument("--time-budget", type=float, default=None)
    ap.add_argument("--gamma", type=float, default=0.99)
    args = ap.parse_args()
    main(policy_path=args.policy, N=args.N, M=args.M, n_truth=args.n_truth,
         rho=args.rho, out_dir=args.out, time_budget=args.time_budget,
         gamma=args.gamma)
