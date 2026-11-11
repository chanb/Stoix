"""Success rate vs hardness for each forced budget c (output of forced_budget.py)."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analysis_utils as base

HERE = Path(__file__).resolve().parent
plt.rcParams.update(base.pgf_with_latex)

fig, axes = plt.subplots(1, 2, figsize=base.set_size(base.doc_width_pt, fraction=0.95, subplots=(1, 2)))
cmap = plt.get_cmap("viridis")
for ax, (name, title) in zip(axes, [("lightsout", "Lightsout"), ("slidingpuzzle", "Sliding puzzle")]):
    for split, ls in [("train", "-"), ("test", "--")]:
        df = pd.read_pickle(HERE / f"forced_budget-{name}-{split}-cnn16x4-N8-n50000.pkl")
        for c, g in df.groupby("c"):
            s = g.groupby("hardness")["solved"].agg(["mean", "size"])
            s = s[s["size"] >= 100]  # drop near-empty bins (e.g. sliding test length 31)
            half = 1.96 * np.sqrt(s["mean"] * (1 - s["mean"]) / s["size"])
            color = cmap((c - 1) / 7)
            ax.plot(s.index, s["mean"], ls=ls, marker="o", ms=1.5, lw=1, color=color,
                    label=f"$c={c}$" if split == "train" else None)
            ax.fill_between(s.index, s["mean"] - half, s["mean"] + half, color=color, alpha=0.2, lw=0)
        if split == "test":  # test hardness range is disjoint from (above) train's
            ax.axvspan(df["hardness"].min() - 0.5, df["hardness"].max() + 0.5, color="gray", alpha=0.12, lw=0)
            ax.text(df["hardness"].min() - 0.3, 0.03, "test", fontsize=7, color="gray")
    ax.set_title(title)
    ax.set_xlabel("Shortest path length")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
axes[0].set_ylabel("Success rate")
fig.tight_layout(pad=0.2, w_pad=1.0)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=8, frameon=True, title="Forced compute budget")
out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "analysis-forced_budget-cnn16x4.pdf"
fig.savefig(out, dpi=600, bbox_inches="tight")
print("Saved", out)
