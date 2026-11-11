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
FRAC = None
if args[:1] == ["--frac"]:  # whole figure FRAC * \linewidth wide (e.g. 0.3): small fonts, compact legend
    FRAC, args = float(args[1]), args[2:]
    plt.rcParams.update({"font.size": 6, "axes.titlesize": 6, "axes.labelsize": 6, "xtick.labelsize": 5,
                         "ytick.labelsize": 5, "legend.fontsize": 5, "legend.title_fontsize": 5})
Ns = args or ["1", "4", "8"]
w, h = base.set_size(base.doc_width_pt, fraction=0.95, subplots=(1, 2))
figsize = (w * max(len(Ns), 2) / 2, len(ENVS) * h)
if FRAC:
    figsize = base.set_size(base.doc_width_pt, fraction=FRAC, subplots=(len(ENVS), len(Ns)))
fig, axes = plt.subplots(len(ENVS), len(Ns), figsize=figsize, squeeze=False, sharex="row", sharey=True)
S = 0.5 if FRAC else 1.0  # line/marker scale
# Per budget c (index 0 = own halting): colour from the dataviz skill's validated 8-slot categorical palette in
# fixed order (viridis neighbours were indistinguishable), plus dash pattern and marker as secondary encoding.
COLORS = ["black", "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
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
                color = COLORS[c]  # c=0: the adaptive model's own halting
                label = (f"$c={c}$" if c else "own halting") if split == "train" else None
                if not c:  # wide, light reference line underneath, so it does not hide the budget lines
                    ax.plot(s.index, s["mean"], color="black", alpha=0.25, lw=2.4 * S, label=label, zorder=1,
                            solid_capstyle="round")
                    continue
                ax.plot(s.index, s["mean"], ls=LINESTYLES[c], marker=MARKERS[c], ms=2.5 * S, mew=0.5 * S, mfc="none",
                        markevery=2 if FRAC else 1, lw=(1.6 if str(c) == N else 0.8) * S, color=color, label=label,
                        zorder=2)
                if not FRAC:  # at 0.3 linewidth the CI bands smear the lines together
                    ax.fill_between(s.index, s["mean"] - half, s["mean"] + half, color=color, alpha=0.15, lw=0)
            if split == "test":  # test hardness range is disjoint from (above) train's
                ax.axvspan(df["hardness"].min() - 0.5, df["hardness"].max() + 0.5, color="gray", alpha=0.12, lw=0)
                ax.text(df["hardness"].min() - 0.3, 0.03, "test", fontsize=5 if FRAC else 7, color="gray")
        if FRAC:
            title = title.split()[0] if len(ENVS) > 1 else ""
        what = f"adaptive $N\\in[1,{df['c'].max()}]$" if N == "var" else f"trained $N={N}$"
        if not FRAC:  # the small single-panel figure carries no title (the caption names the model)
            ax.set_title(f"{title}, {what}" if title else what[0].upper() + what[1:])
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Shortest path length")
for ax, (_, title) in zip(axes[:, 0], ENVS):
    ax.set_ylabel("Success rate" if FRAC else f"{title}\nSuccess rate")
fig.tight_layout(pad=0.1 if FRAC else 0.2, w_pad=1.0)
by_label = {}
for ax in axes[0]:
    for hh, ll in zip(*ax.get_legend_handles_labels()):
        by_label.setdefault(ll, hh)
ls_ = sorted(by_label, key=lambda l: (l != "own halting", l))
hs = [by_label[l] for l in ls_]
fig.legend(hs, ls_, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=3 if FRAC else len(hs), frameon=True,
           title=None if FRAC else "Forced compute budget $c$ (thick: $c=N$; grey band: test instances)",
           **({"handlelength": 2.5, "columnspacing": 0.8, "borderpad": 0.3, "labelspacing": 0.2} if FRAC else {}))
if FRAC:  # shrink until the saved (tight) width is FRAC * \linewidth: labels can stick out of the layout box
    target = base.set_size(base.doc_width_pt, fraction=FRAC)[0] - 0.02  # minus 2 * pad_inches
    for _ in range(3):
        over = fig.get_tightbbox(fig.canvas.get_renderer()).width - target
        if over <= 0:
            break
        fig.set_size_inches(fig.get_figwidth() - over, fig.get_figheight())
        fig.tight_layout(pad=0.1, w_pad=1.0)
fig.savefig(out, dpi=600, bbox_inches="tight", pad_inches=0.01 if FRAC else 0.1)
print("Saved", out)
