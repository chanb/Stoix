"""PonderNet-style runtime-aware policy (form (2) in policy_parameterization.md).

    pi(a, c | s) = rho(a | x_c, s) * lambda_c * prod_{i<c} (1 - lambda_i),

with deterministic recurrent computation x_i = tanh(W_x x_{i-1} + W_s s + b)
(one linear layer + nonlinearity as the intermediate layer), a linear action
head rho(a | x_c) = softmax(W_a x_c + b_a), and a linear halting head
lambda_i = sigmoid(w_h . x_i + b_h).

Halting is emergent (a stopping time adapted to the computation), the support
of c is unbounded up to the practical cap c_max, where truncation is modeled as
part of the policy via a forced halt (probability-1 halt, zero score
contribution), per the caveat in policy_parameterization.md.

Pure NumPy with manual backprop (BPTT through the recurrence); verified against
finite differences by grad_check().
"""
import numpy as np

PARAM_NAMES = ["W_x", "W_s", "b", "W_a", "b_a", "w_h", "b_h"]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


class PonderNetPolicy:
    def __init__(self, n_states, n_actions, hidden=32, c_max=20, seed=0,
                 halt_bias_init=-1.5):
        """halt_bias_init sets the initial halting logit: -1.5 gives
        lambda ~ 0.18, i.e. E[c] ~ 5.5 ponder steps, so the policy starts by
        over-thinking and must learn to halt earlier (runtimes have spread)."""
        self.n_states, self.n_actions = n_states, n_actions
        self.hidden, self.c_max = hidden, c_max
        rng = np.random.default_rng(seed)
        self.params = {
            "W_x": rng.normal(0, 1.0 / np.sqrt(hidden), (hidden, hidden)),
            "W_s": rng.normal(0, 0.5, (hidden, n_states)),
            "b": np.zeros(hidden),
            "W_a": np.zeros((n_actions, hidden)),  # uniform action dist at init
            "b_a": np.zeros(n_actions),
            "w_h": np.zeros(hidden),
            "b_h": float(halt_bias_init),
        }

    # ------------------------------------------------- state-encoding hooks
    # Subclasses can change how the state enters the recurrence by overriding
    # these two methods (forward contribution W_s phi(s) and its gradient).
    def _state_input(self, s):
        """W_s @ phi(s) with phi = one-hot encoding of the state."""
        return self.params["W_s"][:, s]

    def _grad_Ws(self, g, du, s):
        g["W_s"][:, s] += du

    # ---------------------------------------------------------------- forward
    def _core(self, s, c):
        """Run the recurrence for exactly c steps; return xs, lambdas."""
        p = self.params
        u_s = self._state_input(s)
        x = np.zeros(self.hidden)
        xs, lams = [x], []
        for _ in range(c):
            x = np.tanh(p["W_x"] @ x + u_s + p["b"])
            xs.append(x)
            lams.append(_sigmoid(p["w_h"] @ x + p["b_h"]))
        return xs, lams

    def sample(self, s, rng, force_action=None):
        """Sample (a, c) ~ pi(., . | s). Returns (a, c, cache)."""
        p = self.params
        u_s = self._state_input(s)
        x = np.zeros(self.hidden)
        xs, lams = [x], []
        c, forced = 0, False
        while True:
            c += 1
            x = np.tanh(p["W_x"] @ x + u_s + p["b"])
            xs.append(x)
            lam = _sigmoid(p["w_h"] @ x + p["b_h"])
            lams.append(lam)
            if c >= self.c_max:
                forced = True
                break
            if rng.random() < lam:
                break
        probs = _softmax(p["W_a"] @ x + p["b_a"])
        a = force_action if force_action is not None else rng.choice(self.n_actions, p=probs)
        cache = dict(s=s, a=a, c=c, xs=xs, lams=lams, probs=probs, forced=forced)
        return a, c, cache

    def sample_runtime(self, s, rng):
        """Sample only the runtime c (halting head), ignoring the action."""
        _, c, cache = self.sample(s, rng, force_action=0)
        return c

    def action_probs(self, s, c):
        xs, _ = self._core(s, c)
        return _softmax(self.params["W_a"] @ xs[-1] + self.params["b_a"])

    def log_prob(self, s, a, c, forced=False):
        """log pi(a, c | s) for a given (a, c). Used for finite-diff checks."""
        xs, lams = self._core(s, c)
        lams = np.clip(np.asarray(lams), 1e-12, 1 - 1e-12)
        probs = _softmax(self.params["W_a"] @ xs[-1] + self.params["b_a"])
        lp = np.log(probs[a]) + np.sum(np.log(1 - lams[:-1]))
        if not forced:
            lp += np.log(lams[-1])
        return lp

    # --------------------------------------------------------------- backward
    def grad_log_prob(self, cache):
        """Gradient of log pi(a, c | s) w.r.t. all parameters (manual BPTT)."""
        p = self.params
        s, a, c = cache["s"], cache["a"], cache["c"]
        xs, lams, probs, forced = cache["xs"], cache["lams"], cache["probs"], cache["forced"]
        g = {k: np.zeros_like(np.asarray(v, dtype=float)) for k, v in p.items()}

        # action head: d log softmax
        dz = -probs.copy()
        dz[a] += 1.0
        g["W_a"] += np.outer(dz, xs[c])
        g["b_a"] += dz

        # halting head coefficients: d log(lam_c)/d eta = 1 - lam_c (unless the
        # halt was forced at c_max, prob 1, zero score); d log(1-lam_i) = -lam_i
        dcoef = np.empty(c)
        dcoef[: c - 1] = -np.asarray(lams[: c - 1])
        dcoef[c - 1] = 0.0 if forced else (1.0 - lams[c - 1])
        for i in range(1, c + 1):
            g["w_h"] += dcoef[i - 1] * xs[i]
            g["b_h"] += dcoef[i - 1]

        # BPTT through x_i = tanh(W_x x_{i-1} + W_s s + b)
        dx = np.zeros(self.hidden)
        for i in range(c, 0, -1):
            dx = dx + dcoef[i - 1] * p["w_h"]
            if i == c:
                dx = dx + p["W_a"].T @ dz
            du = dx * (1.0 - xs[i] ** 2)
            g["W_x"] += np.outer(du, xs[i - 1])
            self._grad_Ws(g, du, s)
            g["b"] += du
            dx = p["W_x"].T @ du
        return g


class ScalarPonderNetPolicy(PonderNetPolicy):
    """PonderNet policy that takes the state as a scalar instead of one-hot.

    The state index s is fed to the recurrence as a single normalized scalar
    s / (n_states - 1) in [0, 1], so the state pathway is w_s * s_norm with
    w_s a single learned column: x_i = tanh(W_x x_{i-1} + w_s s_norm + b).
    Everything else (halting head, action head, BPTT) is inherited.

    Note this is a much harder representation: the network must separate all
    states along a single input dimension, and states with adjacent indices
    are entangled (generalization across neighboring indices, which need not
    be spatially meaningful in FrozenLake's row-major indexing).
    """

    def __init__(self, n_states, n_actions, hidden=32, c_max=20, seed=0,
                 halt_bias_init=-1.5):
        super().__init__(n_states, n_actions, hidden, c_max, seed,
                         halt_bias_init)
        rng = np.random.default_rng(seed)
        # replace the one-hot embedding with a single input column
        self.params["W_s"] = rng.normal(0, 1.0, (self.hidden, 1))

    def _s_norm(self, s):
        return s / max(1, self.n_states - 1)

    def _state_input(self, s):
        return self.params["W_s"][:, 0] * self._s_norm(s)

    def _grad_Ws(self, g, du, s):
        g["W_s"][:, 0] += du * self._s_norm(s)


class Adam:
    def __init__(self, params, lr=0.01, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(np.asarray(v, dtype=float)) for k, v in params.items()}
        self.v = {k: np.zeros_like(np.asarray(v, dtype=float)) for k, v in params.items()}
        self.t = 0

    def ascend(self, params, grads):
        self.t += 1
        for k in params:
            gk = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * gk
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * gk ** 2
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] = params[k] + self.lr * mhat / (np.sqrt(vhat) + self.eps)


def grad_check(seed=0, tol=1e-5, policy_cls=None):
    """Finite-difference check of grad_log_prob. Returns max relative error."""
    rng = np.random.default_rng(seed)
    policy_cls = policy_cls or PonderNetPolicy
    pol = policy_cls(16, 4, hidden=8, c_max=6, seed=seed)
    # random-ish params so heads are nonzero
    for k in pol.params:
        pol.params[k] = pol.params[k] + rng.normal(0, 0.3, np.shape(pol.params[k]))
    max_err = 0.0
    for trial in range(5):
        s = rng.integers(16)
        a, c, cache = pol.sample(s, rng)
        g = pol.grad_log_prob(cache)
        eps = 1e-6
        for k in PARAM_NAMES:
            arr = np.atleast_1d(np.asarray(pol.params[k], dtype=float))
            flat_idx = rng.integers(arr.size) if arr.size > 1 else 0
            idx = np.unravel_index(flat_idx, arr.shape)

            def perturbed(delta):
                orig = pol.params[k]
                pert = np.atleast_1d(np.asarray(orig, dtype=float)).copy()
                pert[idx] += delta
                pol.params[k] = pert.reshape(np.shape(orig)) if np.ndim(orig) else float(pert[0])
                lp = pol.log_prob(s, a, c, cache["forced"])
                pol.params[k] = orig
                return lp

            fd = (perturbed(eps) - perturbed(-eps)) / (2 * eps)
            an = np.atleast_1d(np.asarray(g[k]))[idx]
            err = abs(fd - an) / max(1.0, abs(fd), abs(an))
            max_err = max(max_err, err)
    assert max_err < tol, f"grad check failed: {max_err:.2e}"
    return max_err


if __name__ == "__main__":
    print(f"grad_check (one-hot) max rel err: {grad_check():.2e}")
    print(f"grad_check (scalar)  max rel err: "
          f"{grad_check(policy_cls=ScalarPonderNetPolicy):.2e}")
