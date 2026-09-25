"""FrozenLake 4x4 (slippery) as the base MDP of a RAMDP.

Base MDP rewards are in {0, 1} subset of [0, 1] as required. The RAMDP prices each
action's computation runtime c through the discount (see ramdp.md): the reward
collected at env decision h is discounted by gamma^(sum_{k<=h} C_k - 1) instead
of gamma^h. Equivalently, the sampled Q((s_h, 0), (a_h, C_h)) satisfies the
backward recursion G_h = gamma^(C_h - 1) * (r_h + gamma * G_{h+1}).
"""
import numpy as np

MAP_4X4 = ["SFFF", "FHFH", "FFFH", "HFFG"]
MAP_8X8 = [
    "SFFFFFFF",
    "FFFFFFFF",
    "FFFHFFFF",
    "FFFFFHFF",
    "FFFHFFFF",
    "FHHFFFHF",
    "FHFFHFHF",
    "FFFHFFFG",
]
LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3


class FrozenLakeRAMDP:
    """Native FrozenLake-v1 dynamics (4x4, slippery by default).

    States are indexed row-major (0..15). Under slippery dynamics the executed
    action is the intended one or either perpendicular one, each w.p. 1/3
    (gymnasium's convention). Reward 1 is given on transitioning into the goal.
    """

    n_actions = 4

    def __init__(self, slippery=True, max_decisions=100, desc_map=MAP_4X4):
        self.slippery = slippery
        self.max_decisions = max_decisions
        self.desc_map = desc_map
        self.desc = np.asarray([list(row) for row in desc_map])
        self.nrow, self.ncol = self.desc.shape[0], self.desc.shape[1]
        self.n_states = self.nrow * self.ncol
        self.terminal = np.array([c in "GH" for c in self.desc.flatten()])
        # transitions[s][a] = list of (prob, s_next, reward, done)
        self.transitions = [[self._build(s, a) for a in range(4)] for s in range(self.n_states)]

    def _move(self, s, a):
        row, col = divmod(s, self.ncol)
        if a == LEFT:
            col = max(col - 1, 0)
        elif a == DOWN:
            row = min(row + 1, self.nrow - 1)
        elif a == RIGHT:
            col = min(col + 1, self.ncol - 1)
        elif a == UP:
            row = max(row - 1, 0)
        return row * self.ncol + col

    def _build(self, s, a):
        if self.terminal[s]:
            return [(1.0, s, 0.0, True)]
        executed = [(a - 1) % 4, a, (a + 1) % 4] if self.slippery else [a]
        probs = [0.1, 0.1, 0.8]
        out = []
        for b, p in zip(executed, probs):
            s2 = self._move(s, b)
            done = bool(self.terminal[s2])
            r = 1.0 if self.desc[s2 // self.ncol][s2 % self.ncol] == "G" else 0.0
            out.append((p, s2, r, done))
        return out

    def reset(self):
        return 0

    def step(self, s, a, rng):
        outcomes = self.transitions[s][a]
        idx = rng.choice(len(outcomes), p=[o[0] for o in outcomes])
        _, s2, r, done = outcomes[idx]
        return s2, r, done


def ramdp_returns(rewards, runtimes, gamma, bootstrap=0.0):
    """RAMDP discounted returns-to-go.

    G_h = gamma^(C_h - 1) * (r_h + gamma * G_{h+1}) is a sample of
    Q((s_h, 0), (a_h, C_h)). G_0 is the RAMDP-discounted episode return.
    """
    return ramdp_returns_q1(rewards, runtimes, gamma, bootstrap)[0]


def ramdp_returns_q1(rewards, runtimes, gamma, bootstrap=0.0):
    """Returns-to-go G_h plus the runtime-1 samples B_h.

    B_h := r_h + gamma * G_{h+1} is a sample of Q((s_h, 0), (a_h, 1)) and
    satisfies B_h = gamma^(-(C_h - 1)) G_h -- computed here via the recursion
    directly, avoiding negative powers of gamma (which overflow for large
    runtimes). G_h = gamma^(C_h - 1) * B_h.

    bootstrap: value estimate V(s_H) of the state reached after the last
    transition, for episodes truncated by a step limit rather than terminated
    by the environment (Lemma 4 justifies bootstrapping with the fresh-runtime
    value V((s_H, 0)): cumulative runtime factors out of the continuation).
    Its total contribution to G_0 is gamma^(sum_h C_h) * bootstrap.
    """
    G = float(bootstrap)
    Gs, Bs = np.zeros(len(rewards)), np.zeros(len(rewards))
    for h in range(len(rewards) - 1, -1, -1):
        B = rewards[h] + gamma * G
        G = gamma ** (runtimes[h] - 1) * B
        Bs[h], Gs[h] = B, G
    return Gs, Bs
