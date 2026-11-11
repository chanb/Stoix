"""Success rate vs hardness for each forced budget c (output of forced_budget.py), one row per training budget N.

Usage: python ramdp_experiments/plot_forced_budget.py OUT.pdf [N ...]   (default N: 1 4 8; 'var' = adaptive 1-8 model)
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analysis_utils as base

HERE = Path(__file__).resolve().parent
plt.rcParams.update(base.pgf_with_latex)

out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "analysis-forced_budget-cnn16x4.pdf"
ENVS = [("lightsout", "Lightsout"), ("slidingpuzzle", "Sliding puzzle")]
args = sys.argv[2:]
if args[:1] == ["--env"]:  # one model/env only, e.g. --env slidingpuzzle-bankfinal var
    ENVS = [(args[1], "Lightsout" if args[1].startswith("lightsout") else "Sliding puzzle")]
    args = args[2:]
Ns = args or ["1", "4", "8"]
w, h = base.set_size(base.doc_width_pt, fraction=0.95, subplots=(1, 2))
fig, axes = plt.subplots(len(ENVS), len(Ns), figsize=(w * max(len(Ns), 2) / 2, len(ENVS) * h), squeeze=False,
                         sharex="row", sharey=True)
cmap = plt.get_cmap("viridis")
# Per budget c (index 0 = own halting): dash pattern and marker, on top of colour.
LINESTYLES = ["-", "-", (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1)), "-", (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1))]
MARKERS = ["o", "o", "s", "^", "v", "D", "P", "X", "*"]
for col, N in zip(axes.T, Ns):  # rows: environment, columns: trained budget N
    for ax, (name, title) in zip(col, ENVS):
        for split in ["train", "test"]:  # same style; the grey band marks the test range
            df = pd.read_pickle(HERE / f"forced_budget-{name}-{split}-cnn16x4-N{N}-n50000.pkl")
            for c, g in df.groupby("c"):
                s = g.groupby("hardness")["solved"].agg(["mean", "size"])
                s = s[s["size"] >= 100]  # drop near-empty bins (e.g. sliding test length 31)
                half = 1.96 * np.sqrt(s["mean"] * (1 - s["mean"]) / s["size"])
                color = cmap((c - 1) / 7) if c else "black"  # c=0: the adaptive model's own halting
                label = (f"$c={c}$" if c else "own halting") if split == "train" else None
                ax.plot(s.index, s["mean"], ls=LINESTYLES[c], marker=MARKERS[c], ms=2.5, mew=0.6, mfc="none",
                        lw=2 if str(c) == N or not c else 1, color=color, label=label, zorder=3 if not c else 2)
                ax.fill_between(s.index, s["mean"] - half, s["mean"] + half, color=color, alpha=0.2, lw=0)
            if split == "test":  # test hardness range is disjoint from (above) train's
                ax.axvspan(df["hardness"].min() - 0.5, df["hardness"].max() + 0.5, color="gray", alpha=0.12, lw=0)
                ax.text(df["hardness"].min() - 0.3, 0.03, "test", fontsize=7, color="gray")
        ax.set_title(f"{title}, adaptive $N\\in[1,{df['c'].max()}]$" if N == "var" else f"{title}, trained $N={N}$")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Shortest path length")
for ax, (_, title) in zip(axes[:, 0], ENVS):
    ax.set_ylabel(f"{title}\nSuccess rate")
fig.tight_layout(pad=0.2, w_pad=1.0)
by_label = {}
for ax in axes[0]:
    for hh, ll in zip(*ax.get_legend_handles_labels()):
        by_label.setdefault(ll, hh)
ls_ = sorted(by_label, key=lambda l: (l != "own halting", l))
hs = [by_label[l] for l in ls_]
fig.legend(hs, ls_, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=len(hs), frameon=True,
           title="Forced compute budget $c$ (thick: $c=N$; grey band: test instances)")
fig.savefig(out, dpi=600, bbox_inches="tight")
print("Saved", out)
