"""
Illustration of the "thought MDP" parity example from:

"Necessity of thought actions in thought MDPs"

We compare:
1. Tabular Q-learning
2. Linear Q-learning

on two environments:
A. Original parity environment (no thought actions)
B. Thought-MDP parity environment (with thought actions)

Key phenomenon:
- Parity is NOT linearly separable.
- Therefore linear Q-learning fails in the original environment.
- But after augmenting the state with thought variables,
  the problem becomes linearly representable and linear Q-learning succeeds.

This script is self-contained.
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

from collections import defaultdict

random.seed(42)
np.random.seed(42)
max_eps = 5000

# ============================================================
# Utilities
# ============================================================

def parity(x):
    """Returns x1 XOR x2"""
    return int((x[0] + x[1]) % 2)


def epsilon_greedy(q_values, epsilon):
    if random.random() < epsilon:
        return random.randint(0, len(q_values) - 1)
    return int(np.argmax(q_values))


# ============================================================
# ENVIRONMENT 1:
# Original parity classification MDP
# ============================================================

class OriginalParityEnv:
    """
    State: s = (x1, x2) in {0,1}^2
    Actions: a in {0,1}

    Reward:
        +1 if action == parity(state)
         0 otherwise

    Episode terminates immediately after one action.
    """

    def reset(self):
        self.state = np.random.randint(0, 2, size=2)
        return tuple(self.state)

    def step(self, action):
        correct = parity(self.state)

        reward = 1.0 if action == correct else 0.0
        done = True

        return None, reward, done, {}


# ============================================================
# ENVIRONMENT 2:
# Thought-MDP parity environment
# ============================================================

class ThoughtParityEnv:
    """
    Implements the thought-MDP construction from the paper.

    Observation/state vector:
        s = [x1, x2, z1, z2, z3, z4]

    where:
        x1, x2 : original parity bits

        z1 : just received environment state?
        z2 : last thought action was 0?
        z3 : last thought action was 1?
        z4 : reward from thought action

    Actions:
        0 -> thought action "think 0"
        1 -> thought action "think 1"
        2 -> final answer 0
        3 -> final answer 1

    Idea:
        Thought actions reveal whether guessed parity matches true parity.

    Optimal strategy:
        1. Think 0
        2. If rewarded -> parity=0
           else -> parity=1
    """

    def reset(self):
        self.bits = np.random.randint(0, 2, size=2)

        # initial state
        self.state = np.array([
            self.bits[0],
            self.bits[1],
            1,  # z1
            0,  # z2
            0,  # z3
            0   # z4
        ], dtype=float)

        return self.state.copy()

    def step(self, action):

        true_parity = parity(self.bits)

        # Thought action 0
        if action == 0:
            reward_signal = 1 if true_parity == 0 else 0

            self.state = np.array([
                self.bits[0],
                self.bits[1],
                0,
                1,
                0,
                reward_signal
            ], dtype=float)

            return self.state.copy(), 0.0, False, {}

        # Thought action 1
        elif action == 1:
            reward_signal = 1 if true_parity == 1 else 0

            self.state = np.array([
                self.bits[0],
                self.bits[1],
                0,
                0,
                1,
                reward_signal
            ], dtype=float)

            return self.state.copy(), 0.0, False, {}

        # Final answer 0
        elif action == 2:
            reward = 1.0 if true_parity == 0 else 0.0
            return None, reward, True, {}

        # Final answer 1
        elif action == 3:
            reward = 1.0 if true_parity == 1 else 0.0
            return None, reward, True, {}

        else:
            raise ValueError("Invalid action")


# ============================================================
# TABULAR Q-LEARNING
# ============================================================

def train_tabular_q_learning(
    env,
    num_actions,
    episodes=5000,
    alpha=0.1,
    gamma=1.0,
    epsilon=0.01
):

    Q = defaultdict(lambda: np.zeros(num_actions))
    rewards = []

    for ep in range(episodes):

        s = env.reset()
        done = False
        total_reward = 0

        while not done:

            a = epsilon_greedy(Q[tuple(s)], epsilon)

            s2, r, done, _ = env.step(a)

            if done:
                td_target = r
            else:
                td_target = r + gamma * np.max(Q[tuple(s2)])

            Q[tuple(s)][a] += alpha * (td_target - Q[tuple(s)][a])

            total_reward += r

            if not done:
                s = tuple(s2)

        rewards.append(total_reward)

    return Q, rewards


# ============================================================
# LINEAR Q-LEARNING
# ============================================================

class LinearQ:
    """
    Q(s,a) = phi(s,a)^T w

    We use separate weights per action:
        Q(s,a) = w_a^T s
    """

    def __init__(self, state_dim, num_actions):
        self.W = np.zeros((num_actions, state_dim))

    def q_values(self, s):
        return self.W @ s

    def update(self, s, a, target, alpha):

        pred = self.W[a] @ s
        td_error = target - pred

        self.W[a] += alpha * td_error * s


def train_linear_q_learning(
    env,
    state_dim,
    num_actions,
    episodes=5000,
    alpha=0.05,
    gamma=1.0,
    epsilon=0.01
):

    agent = LinearQ(state_dim, num_actions)

    rewards = []

    for ep in range(episodes):

        s = np.array(env.reset(), dtype=float)
        done = False
        total_reward = 0

        while not done:

            qvals = agent.q_values(s)

            a = epsilon_greedy(qvals, epsilon)

            s2, r, done, _ = env.step(a)

            if done:
                target = r
            else:
                s2 = np.array(s2, dtype=float)
                target = r + gamma * np.max(agent.q_values(s2))

            agent.update(s, a, target, alpha)

            total_reward += r

            if not done:
                s = s2

        rewards.append(total_reward)

    return agent, rewards


# ============================================================
# EVALUATION
# ============================================================

def moving_average(x, k=100):
    x = np.array(x)
    return np.convolve(x, np.ones(k)/k, mode='valid')


# ============================================================
# EXPERIMENTS
# ============================================================

print("=" * 60)
print("1. ORIGINAL PARITY ENVIRONMENT")
print("=" * 60)

# ------------------------------------------------------------
# Tabular Q-learning on original parity
# ------------------------------------------------------------

env1 = OriginalParityEnv()

Q_tab_orig, rewards_tab_orig = train_tabular_q_learning(
    env1,
    num_actions=2,
    episodes=max_eps
)

# ------------------------------------------------------------
# Linear Q-learning on original parity
# ------------------------------------------------------------

agent_lin_orig, rewards_lin_orig = train_linear_q_learning(
    env1,
    state_dim=2,
    num_actions=2,
    episodes=max_eps
)

print("\nOriginal environment learned policies:")
print("State -> optimal parity")

states = [
    np.array([0,0]),
    np.array([0,1]),
    np.array([1,0]),
    np.array([1,1]),
]

for s in states:

    q_tab = Q_tab_orig[tuple(s)]
    a_tab = np.argmax(q_tab)

    q_lin = agent_lin_orig.q_values(s)
    a_lin = np.argmax(q_lin)

    print(f"\nState {s}")
    print(f" True parity : {parity(s)}")
    print(f" Tabular Q   : {a_tab}")
    print(f" Linear Q    : {a_lin}")


# ============================================================
# THOUGHT MDP
# ============================================================

print("\n" + "=" * 60)
print("2. THOUGHT-MDP PARITY ENVIRONMENT")
print("=" * 60)

env2 = ThoughtParityEnv()

# ------------------------------------------------------------
# Tabular
# ------------------------------------------------------------

Q_tab_thought, rewards_tab_thought = train_tabular_q_learning(
    env2,
    num_actions=4,
    episodes=max_eps
)

# ------------------------------------------------------------
# Linear
# ------------------------------------------------------------

agent_lin_thought, rewards_lin_thought = train_linear_q_learning(
    env2,
    state_dim=6,
    num_actions=4,
    episodes=max_eps
)

# ============================================================
# TEST LEARNED LINEAR POLICY
# ============================================================

print("\nTesting learned linear policy on thought MDP:\n")

test_states = [
    np.array([0,0]),
    np.array([0,1]),
    np.array([1,0]),
    np.array([1,1]),
]

for bits in test_states:

    # Initial state
    s0 = np.array([bits[0], bits[1], 1,0,0,0], dtype=float)

    q0 = agent_lin_thought.q_values(s0)
    a0 = np.argmax(q0)

    print(f"Bits={bits}, first action={a0}")

    # Simulate thought action
    parity_val = parity(bits)

    if a0 == 0:
        reward_signal = 1 if parity_val == 0 else 0
        s1 = np.array([bits[0], bits[1], 0,1,0,reward_signal])

    elif a0 == 1:
        reward_signal = 1 if parity_val == 1 else 0
        s1 = np.array([bits[0], bits[1], 0,0,1,reward_signal])

    else:
        print("Agent answered immediately")
        continue

    q1 = agent_lin_thought.q_values(s1)
    a1 = np.argmax(q1)

    final_answer = 0 if a1 == 2 else 1

    print(f"  reward signal={reward_signal}")
    print(f"  second action={a1}")
    print(f"  predicted parity={final_answer}")
    print(f"  true parity={parity_val}")
    print()


# ============================================================
# PLOTS
# ============================================================

sns.set_palette("colorblind")
# plt.style.use("seaborn-v0_8-colorblind")
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
    "axes.labelsize": 18,  # LaTeX default is 10pt font.
    "font.size": 10,
    "legend.fontsize": 10,  # Make the legend/label fonts a little smaller
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "pgf.rcfonts": False,  # don't setup fonts from rc parameters
}

plt.rcParams.update(pgf_with_latex)

num_rows = 1
num_cols = 1
fig, axes = plt.subplots(
    num_rows,
    num_cols,
    figsize=set_size(doc_width_pt, 0.45, (num_rows, num_cols), use_golden_ratio=False),
    layout="constrained",
)

k = 250
plt.plot(
    moving_average(rewards_tab_orig, k),
    label="Tabular $f$ (w/o comp. space)",
    color="red",
    linestyle=":",
)

plt.plot(
    moving_average(rewards_lin_orig, k),
    label="Linear $f$ (w/o comp. space)",
    color="blue",
    linestyle=":",
)

plt.plot(
    moving_average(rewards_tab_thought, k),
    label="Tabular $f$ (w/ comp. space)",
    color="red",
    linestyle="-",
)

plt.plot(
    moving_average(rewards_lin_thought, k),
    label="Linear $f$ (w/ comp. space)",
    color="blue",
    linestyle="-",
)

fig.supxlabel("Num. episodes")
fig.supylabel("Average reward")

fig.legend(
    bbox_to_anchor=(0.0, 1.0, 1.0, 0.0),
    loc="lower center",
    ncols=1,
    borderaxespad=0.0,
    frameon=True,
)
# fig.tight_layout()

plt.savefig("thought_mdp_parity_q.pdf", dpi=600, format="pdf", bbox_inches="tight")
