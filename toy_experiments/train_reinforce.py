"""Learning experiment: REINFORCE variants on the FrozenLake RAMDP.

Three variants (paper Sec 4.2), differing only in the score coefficient
Psi_h multiplying grad log pi(a_h, C_h | s_h):

  none  :  Psi_h = G_h                      (REINFORCE, no baseline)
  naive :  Psi_h = Q_hat_sep(s_h, a_h, C_h) - V_hat(s_h)
           (paper's g_sep: per-runtime critic + baseline)
  fac   :  Psi_h = gamma^(C_h-1) Q_hat_1(s_h, a_h) - V_hat(s_h)
           (paper's g_fac: runtime-factorized critic + baseline)

where G_h is the RAMDP return-to-go (a sample of Q((s_h,0),(a_h,C_h))) and the
baseline V_hat is an EMA over previous batches. Both critics are the paper's
batch estimators computed over the current batch: Q_hat_sep(s, a, c) is the
mean of G over visits to exactly (s, a, c) (Sec 4.1 separate estimator), while
Q_hat_1(s, a) pools the runtime-1 samples B = gamma^(-(C-1)) G over ALL visits
to (s, a) regardless of runtime (Sec 4.1 factorized estimator) -- so naive and
fac differ only in whether samples are shared across runtimes. Each falls back
to an EMA critic from previous batches, then to the sampled return G_h, when
unvisited.

Metrics tracked per update (mean over the batch): wall-clock time, number of
decisions (env steps), runtime per decision (mean C and total compute), and
undiscounted / RAMDP-discounted returns.
"""
import csv
import json
import time
import numpy as np

from ramdp_frozenlake import FrozenLakeRAMDP, ramdp_returns_q1
from pondernet_policy import PonderNetPolicy, ScalarPonderNetPolicy, Adam
from estimators import TabularCritic

VARIANTS = ("fac", "naive", "none", "no_ponder")
POLICIES = {"onehot": PonderNetPolicy, "scalar": ScalarPonderNetPolicy}


def rollout(env, policy, rng):
    """One episode, truncated when the cumulative runtime exceeds
    env.max_decisions. Returns (states, actions, runtimes, rewards, caches,
    s_final, truncated): s_final is the state after the last transition and
    truncated is True when the episode was cut by the runtime limit rather
    than terminated by the environment -- the return should then bootstrap
    with a value estimate of s_final."""
    s = env.reset()
    states, actions, runtimes, rewards, caches = [], [], [], [], []
    done, t = False, 0
    while not done and t < env.max_decisions:
        a, c, cache = policy.sample(s, rng)
        s2, r, done = env.step(s, a, rng)
        states.append(s)
        actions.append(a)
        runtimes.append(c)
        rewards.append(r)
        caches.append(cache)
        s, t = s2, t + c

    return states, actions, runtimes, rewards, caches, s, not done


def train(
    variant,
    seed,
    n_updates=300,
    batch_size=64,
    gamma=0.99,
    hidden=32,
    c_max=1000,
    lr=0.01,
    ema=0.01,
    log_every=10,
    quiet=True,
    checkpoint=None,
    checkpoint_every=100,
    time_budget=None,
    state_encoding="onehot"):
    assert variant in VARIANTS
    import os
    import pickle
    if variant == "no_ponder":
        c_max = 1
    env = FrozenLakeRAMDP(slippery=True)
    policy_cls = POLICIES[state_encoding]
    policy = policy_cls(env.n_states, env.n_actions, hidden, c_max, seed)
    opt = Adam(policy.params, lr=lr)
    critic = TabularCritic(env.n_states, env.n_actions, ema=ema)
    rng = np.random.default_rng(seed)

    metrics = {k: [] for k in ("update", "wall_time", "undisc_return",
                               "disc_return", "n_decisions", "mean_runtime",
                               "total_compute")}
    start_u, elapsed = 0, 0.0
    if checkpoint and os.path.exists(checkpoint):
        with open(checkpoint, "rb") as f:
            ck = pickle.load(f)
        policy.params, critic, metrics = ck["params"], ck["critic"], ck["metrics"]
        opt.m, opt.v, opt.t = ck["opt_m"], ck["opt_v"], ck["opt_t"]
        rng.bit_generator.state = ck["rng_state"]
        start_u, elapsed = ck["u"] + 1, ck["elapsed"]
    t0 = time.perf_counter() - elapsed

    def save_ck(u):
        if not checkpoint:
            return
        with open(checkpoint + ".tmp", "wb") as f:
            pickle.dump(dict(params=policy.params, critic=critic,
                             metrics=metrics, opt_m=opt.m, opt_v=opt.v,
                             opt_t=opt.t, rng_state=rng.bit_generator.state,
                             u=u, elapsed=time.perf_counter() - t0), f)
        os.replace(checkpoint + ".tmp", checkpoint)

    for u in range(start_u, n_updates):
        grads = {k: np.zeros_like(np.asarray(v, dtype=float))
                 for k, v in policy.params.items()}
        # ---- first pass: collect the batch
        batch, stats, all_caches = [], [], []
        for _ in range(batch_size):
            (states, actions, runtimes, rewards, caches,
             s_final, truncated) = rollout(env, policy, rng)
            # bootstrap truncated episodes with the value of the final state
            # (V_hat is maintained for every variant)
            boot = critic.baseline(s_final) if truncated else 0.0
            Gs, Bs = ramdp_returns_q1(rewards, runtimes, gamma, bootstrap=boot)
            batch.append((states, actions, runtimes, Gs, Bs))
            all_caches.append(caches)
            # metrics report the environment-only discounted return (the
            # bootstrap term gamma^(sum C) * boot is removed)
            disc_env = (Gs[0] - boot * gamma ** float(np.sum(runtimes))
                        if len(Gs) else 0.0)
            stats.append((sum(rewards), disc_env,
                          len(states), float(np.mean(runtimes)), int(np.sum(runtimes))))

        # batch-pooled critics. fac: paper's Q_hat_fac, pooling the runtime-1
        # samples B = gamma^(-(C-1)) G at (s, a) across ALL runtimes.
        # naive/sep: paper's Q_hat_sep, pooling G only at the exact (s, a, c).
        if variant == "fac":
            q1_sum = np.zeros((env.n_states, env.n_actions))
            q1_cnt = np.zeros((env.n_states, env.n_actions), dtype=int)
            for states, actions, runtimes, Gs, Bs in batch:
                for s, a, B in zip(states, actions, Bs):
                    q1_sum[s, a] += B
                    q1_cnt[s, a] += 1
        elif variant == "naive" or variant == "no_ponder":
            sep_sum, sep_cnt = {}, {}
            for states, actions, runtimes, Gs, Bs in batch:
                for s, a, c, G in zip(states, actions, runtimes, Gs):
                    key = (s, a, c)
                    sep_sum[key] = sep_sum.get(key, 0.0) + G
                    sep_cnt[key] = sep_cnt.get(key, 0) + 1

        # ---- second pass: score coefficients and gradients
        # Including the own episode in the pooled critics keeps the estimators
        # unbiased -- each is a convex combination of the MC return G_h and an
        # independent-data critic, both unbiased coefficients.
        for j, ((states, actions, runtimes, Gs, Bs), caches) in enumerate(
                zip(batch, all_caches)):
            for h, cache in enumerate(caches):
                s, a, c = states[h], actions[h], runtimes[h]
                if variant == "none":
                    coeff = Gs[h]
                elif variant == "naive" or variant == "no_ponder":
                    # paper's g_sep: per-runtime batch estimator Q_hat_sep at
                    # exactly (s, a, c); EMA critic, then the sampled return,
                    # as fallbacks
                    key = (s, a, c)
                    if sep_cnt.get(key, 0) > 0:
                        qval = sep_sum[key] / sep_cnt[key]
                    else:
                        qval = critic.q_sep(s, a, c, fallback=Gs[h])
                    coeff = qval - critic.baseline(s)
                else:  # fac: paper's g_fac with batch-pooled factorized critic
                    if q1_cnt[s, a] > 0:
                        q1 = q1_sum[s, a] / q1_cnt[s, a]
                        qval = gamma ** (c - 1.0) * q1
                    else:  # EMA critic from previous batches, else raw return
                        qval = critic.q_fac(s, a, c, gamma, fallback=Gs[h])
                    coeff = qval - critic.baseline(s)
                if coeff != 0.0:
                    g = policy.grad_log_prob(cache)
                    for k in grads:
                        grads[k] += coeff * g[k]
        for k in grads:
            grads[k] /= batch_size
        opt.ascend(policy.params, grads)
        for states, actions, runtimes, Gs, Bs in batch:
            critic.update(states, actions, runtimes, Gs, gamma, q1s=Bs)

        st = np.array(stats)
        metrics["update"].append(u)
        metrics["wall_time"].append(time.perf_counter() - t0)
        metrics["undisc_return"].append(st[:, 0].mean())
        metrics["disc_return"].append(st[:, 1].mean())
        metrics["n_decisions"].append(st[:, 2].mean())
        metrics["mean_runtime"].append(st[:, 3].mean())
        metrics["total_compute"].append(st[:, 4].mean())
        if not quiet and (u + 1) % log_every == 0:
            print(f"[{variant} seed {seed}] update {u+1}/{n_updates} "
                  f"undisc {st[:, 0].mean():.3f} disc {st[:, 1].mean():.3f} "
                  f"mean_c {st[:, 3].mean():.2f}", flush=True)
        if (u + 1) % checkpoint_every == 0 or u == n_updates - 1:
            save_ck(u)
            if time_budget and time.perf_counter() - t0 > time_budget:
                raise TimeoutError(f"time budget hit at update {u + 1}")
    return metrics, policy


def run_single(variant, seed, n_updates, out_dir, state_encoding="onehot"):
    """Train one (variant, seed) run and cache its metrics to npz (resumable)."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{variant}_{seed}" if state_encoding == "onehot" \
        else f"{variant}_{state_encoding}_{seed}"
    path = f"{out_dir}/run_{tag}.npz"
    ck = f"{out_dir}/ck_{tag}.pkl"
    if os.path.exists(path):
        return dict(np.load(path))
    t0 = time.perf_counter()
    metrics, policy = train(variant, seed, n_updates=n_updates, checkpoint=ck,
                            state_encoding=state_encoding)
    dt = time.perf_counter() - t0
    np.savez(path, **{k: np.asarray(v) for k, v in metrics.items()})
    try:
        os.remove(ck)
    except OSError:
        pass
    if variant == "fac" and seed == 0:
        np.savez(f"{out_dir}/policy_fac_seed0.npz", **policy.params)
    print(f"done {variant} seed {seed} in {dt:.1f}s "
          f"(final undisc {np.mean(metrics['undisc_return'][-20:]):.3f}, "
          f"mean_c {np.mean(metrics['mean_runtime'][-20:]):.2f})", flush=True)
    return metrics


def main(n_seeds=5, n_updates=300, out_dir="results", state_encoding="onehot"):
    all_metrics = {}
    for variant in VARIANTS:
        for seed in range(n_seeds):
            all_metrics[(variant, seed)] = run_single(variant, seed, n_updates,
                                                      out_dir, state_encoding)

    # flat CSV
    with open(f"{out_dir}/learning_metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        keys = ["update", "wall_time", "undisc_return", "disc_return",
                "n_decisions", "mean_runtime", "total_compute"]
        w.writerow(["variant", "seed"] + keys)
        for (variant, seed), m in all_metrics.items():
            for row in zip(*(m[k] for k in keys)):
                w.writerow([variant, seed] + list(row))

    # summary (last 20% of training)
    summary = {}
    for variant in VARIANTS:
        tail = max(1, n_updates // 5)
        agg = {k: [np.mean(all_metrics[(variant, s)][k][-tail:]) for s in range(n_seeds)]
               for k in ("undisc_return", "disc_return", "n_decisions",
                         "mean_runtime", "total_compute")}
        agg["total_wall_time"] = [all_metrics[(variant, s)]["wall_time"][-1]
                                  for s in range(n_seeds)]
        summary[variant] = {k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                            for k, v in agg.items()}
    with open(f"{out_dir}/learning_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    plot(all_metrics, n_seeds, out_dir)
    return all_metrics


def plot(all_metrics, n_seeds, out_dir, smooth=10):
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

    def smoothed(x):
        x = np.asarray(x, dtype=float)
        k = np.ones(smooth) / smooth
        return np.convolve(x, k, mode="valid")

    panels = [("undisc_return", "Undisc. return"),
              ("disc_return", "Disc. return"),
              ("mean_runtime", "Compute time"),
              ("n_decisions", "Traj. length")]
    labels = {
        "none": "Var. REINFORCE",
        "naive": "Var. A2C w/ $\\hat{g}_{\\rm sep}$",
        "fac": "Var. A2C w/ $\\hat{g}_{\\rm fac}$",
        "no_ponder": "Unif. A2C",
    }

    num_rows = 1
    num_cols = 4
    # Keep each panel the same size it had in the original 2x2 layout (rather
    # than letting golden-ratio scaling crush a single row down to ~1in
    # tall, which crowds titles/ticks/legend into overlapping text).
    figsize = set_size(doc_width_pt, 0.95, (num_rows, num_cols), use_golden_ratio=False)
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=figsize,
        layout="constrained",
    )
    for ax_i, (ax, (key, title)) in enumerate(zip(axes.flat, panels)):
        for variant in VARIANTS:
            curves = np.array([smoothed(all_metrics[(variant, s)][key])
                               for s in range(n_seeds)])
            mu, se = curves.mean(0), curves.std(0) / np.sqrt(n_seeds)
            x = np.arange(len(mu))
            ax.plot(x, mu, label=labels[variant] if ax_i == 0 else "")
            ax.fill_between(x, mu - se, mu + se, alpha=0.2)
        ax.set_title(title)
        ax.grid(alpha=0.3)

        if "return" in key:
            ax.set_ylim(0.0, 1.0)

        if key == "mean_runtime":
            ax.set_ylim(0.0, 5.0)
    # fig.suptitle("REINFORCE variants on FrozenLake "
    #              f"({n_seeds} seeds, mean ± s.e., smoothed)")
    fig.supxlabel("Num. updates")
    # fig.tight_layout()
    fig.legend(
        bbox_to_anchor=(0.0, 1.0, 1.0, 0.0),
        loc="lower center",
        ncols=4,
        borderaxespad=0.0,
        frameon=True,
    )
    fig.savefig(f"{out_dir}/learning_curves.pdf", dpi=600, format="pdf", bbox_inches="tight")
    print(f"saved {out_dir}/learning_curves.pdf")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--updates", type=int, default=300)
    ap.add_argument("--out", type=str, default="results")
    ap.add_argument("--state-encoding", type=str, default="onehot",
                    choices=list(POLICIES))
    args = ap.parse_args()
    main(n_seeds=args.seeds, n_updates=args.updates, out_dir=args.out,
         state_encoding=args.state_encoding)
