"""Policy-evaluation estimators and tabular critics/baselines (paper Sec 4.1-4.2).

Given rollouts i = 1..N from (s, 0) with first action (a, C_i) and RAMDP return
G_i (so E[G_i | C_i = c] = Q_c = gamma^(c-1) Q_1 by the runtime factorization,
Lemma 4), the three policy-evaluation estimators of Q_c compared here are:

  single :  Q_hat = G_i for one rollout with C_i = c   (single-sample rollout)
  sep    :  Q_hat_c = mean_{i : C_i = c} G_i           (per-runtime estimator)
  fac    :  Q_hat_c = gamma^(c-1) * mean_i gamma^(-(C_i-1)) G_i  (factorized,
            pools all N samples through Q_c = gamma^(c-1) Q_1)

For learning, TabularCritic maintains EMA estimates of the baseline
b = V_hat(s) and of the factorized head Q_hat_1(s, a) = EMA of
gamma^(-(C-1)) G at (s, a).
"""
import numpy as np


# ------------------------------------------------------------ batch estimators
def q_single(returns, runtimes, c, rng):
    """Single-sample rollout estimate: G_i for one random rollout with C_i = c."""
    idx = np.flatnonzero(runtimes == c)
    if idx.size == 0:
        return np.nan
    return returns[rng.choice(idx)]


def q_sep(returns, runtimes, c):
    """Separate (per-runtime) estimator: mean of G_i over {i : C_i = c}."""
    mask = runtimes == c
    if not mask.any():
        return np.nan
    return returns[mask].mean()


def q_fac(returns, runtimes, c, gamma):
    """Factorized estimator: gamma^(c-1) * mean_i gamma^(-(C_i-1)) G_i."""
    return gamma ** (c - 1) * np.mean(gamma ** (-(runtimes - 1.0)) * returns)


# ------------------------------------------------------------- tabular critics
class TabularCritic:
    """EMA state baseline V_hat(s), factorized Q_hat_1(s, a), and per-runtime
    (separate) Q_hat_sep(s, a, c)."""

    def __init__(self, n_states, n_actions, ema=0.1):
        self.ema = ema
        self.V = np.zeros(n_states)
        self.V_count = np.zeros(n_states, dtype=int)
        self.Q1 = np.zeros((n_states, n_actions))
        self.Q1_count = np.zeros((n_states, n_actions), dtype=int)
        self.Qsep = {}        # (s, a, c) -> EMA of G (runtimes are unbounded)
        self.Qsep_count = {}

    def baseline(self, s):
        return self.V[s]

    def q_fac(self, s, a, c, gamma, fallback=None):
        """Factorized critic value gamma^(c-1) Q_hat_1(s, a).

        Falls back to `fallback` (e.g. the sampled return, unbiased) when no
        critic data has been observed for (s, a) yet.
        """
        if self.Q1_count[s, a] == 0 and fallback is not None:
            return fallback
        return gamma ** (c - 1) * self.Q1[s, a]

    def q_sep(self, s, a, c, fallback=None):
        """Separate (per-runtime) critic value Q_hat_sep(s, a, c).

        Falls back to `fallback` when (s, a, c) has never been observed --
        unlike the factorized critic, samples at other runtimes provide no
        estimate here (the empirical-support limitation of Sec 4.1).
        """
        key = (s, a, c)
        if key not in self.Qsep and fallback is not None:
            return fallback
        return self.Qsep.get(key, 0.0)

    def update(self, states, actions, runtimes, returns, gamma, q1s=None):
        """EMA update from one episode's (s_h, a_h, C_h, G_h) tuples.

        q1s optionally provides the runtime-1 samples B_h = r_h + gamma G_{h+1}
        (from ramdp_returns_q1); otherwise they are recovered from G_h in
        log-space, since gamma^(-(c-1)) overflows for large runtimes.
        """
        for h, (s, a, c, G) in enumerate(zip(states, actions, runtimes, returns)):
            self.V_count[s] += 1
            self.Q1_count[s, a] += 1
            # first observation initializes the EMA
            beta_v = 1.0 if self.V_count[s] == 1 else self.ema
            beta_q = 1.0 if self.Q1_count[s, a] == 1 else self.ema
            self.V[s] += beta_v * (G - self.V[s])
            if q1s is not None:
                g1 = q1s[h]
            elif G <= 0.0:
                g1 = 0.0  # sample of Q_1(s, a)
            else:
                g1 = np.exp(np.log(G) - (c - 1.0) * np.log(gamma))
            self.Q1[s, a] += beta_q * (g1 - self.Q1[s, a])
            key = (s, a, c)
            cnt = self.Qsep_count.get(key, 0) + 1
            self.Qsep_count[key] = cnt
            beta_s = 1.0 if cnt == 1 else self.ema
            self.Qsep[key] = self.Qsep.get(key, 0.0) + beta_s * (G - self.Qsep.get(key, 0.0))
