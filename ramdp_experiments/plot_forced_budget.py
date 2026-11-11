"""Success rate vs hardness for each forced budget c (output of forced_budget.py), one row per training budget N.

Usage: python ramdp_experiments/plot_forced_budget.py OUT.pdf [N ...]   (default N: 1 4 8)
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
Ns = [int(n) for n in sys.argv[2:]] or [1, 4, 8]
w, h = base.set_size(base.doc_width_pt, fraction=0.95, subplots=(1, 2))
fig, axes = plt.subplots(len(Ns), 2, figsize=(w, h * len(Ns)), squeeze=False, sharex="col", sharey=True)
cmap = plt.get_cmap("viridis")
for row, N in zip(axes, Ns):
    for ax, (name, title) in zip(row, [("lightsout", "Lightsout"), ("slidingpuzzle", "Sliding puzzle")]):
        for split, ls in [("train", "-"), ("test", "--")]:
            df = pd.read_pickle(HERE / f"forced_budget-{name}-{split}-cnn16x4-N{N}-n50000.pkl")
            for c, g in df.groupby("c"):
                s = g.groupby("hardness")["solved"].agg(["mean", "size"])
                s = s[s["size"] >= 100]  # drop near-empty bins (e.g. sliding test length 31)
                half = 1.96 * np.sqrt(s["mean"] * (1 - s["mean"]) / s["size"])
                color = cmap((c - 1) / 7)
                ax.plot(s.index, s["mean"], ls=ls, marker="o", ms=1.5, lw=2 if c == N else 1, color=color,
                        label=f"$c={c}$" if split == "train" else None)
                ax.fill_between(s.index, s["mean"] - half, s["mean"] + half, color=color, alpha=0.2, lw=0)
            if split == "test":  # test hardness range is disjoint from (above) train's
                ax.axvspan(df["hardness"].min() - 0.5, df["hardness"].max() + 0.5, color="gray", alpha=0.12, lw=0)
                ax.text(df["hardness"].min() - 0.3, 0.03, "test", fontsize=7, color="gray")
        ax.set_title(f"{title}, trained $N={N}$")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
    row[0].set_ylabel("Success rate")
for ax in axes[-1]:
    ax.set_xlabel("Shortest path length")
fig.tight_layout(pad=0.2, w_pad=1.0)
hs, ls_ = axes[0, 0].get_legend_handles_labels()
fig.legend(hs, ls_, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=8, frameon=True,
           title="Forced compute budget $c$ (thick: $c=N$; dashed: test)")
fig.savefig(out, dpi=600, bbox_inches="tight")
print("Saved", out)
