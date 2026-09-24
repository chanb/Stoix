"""Builds a 2x5 grid of 3D t-SNE plots - row 1 Lights Out, row 2 sliding
puzzle, column i the latent state after the i'th Chain-of-Thought transformer
step (i = 1..MAX_STEPS=5 for both environments) - showing how each actor's
own representation of an instance evolves as it "thinks" for longer, coloured
throughout by that instance's (first-step) compute time.

Pure numpy/sklearn/matplotlib - no JAX/Flax/hydra/checkpoint restore needed.
Reads `plot_data-lightsout.pkl`/`plot_data-slidingpuzzle.pkl`, produced by the
"Export plot data for the standalone plotting scripts" cell near the end of
each notebook - run that cell (in each notebook) at least once before running
this script.
"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tqdm
from sklearn.manifold import TSNE

import analysis_utils as base

set_size = base.set_size
pgf_with_latex = base.pgf_with_latex
doc_width_pt = base.doc_width_pt
plt.rcParams.update(pgf_with_latex)

HERE = Path(__file__).resolve().parent

ENV_LABELS = {
    "lightsout-5x4": "Lightsout",
    "slidingtile-3x3": "Sliding puzzle",
}


def load_plot_data(name):
    path = HERE / f"plot_data-{name}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def perplexity_for(n):
    return min(50, max(5, n * 0.005))


def main():
    envs = [load_plot_data("lightsout"), load_plot_data("slidingpuzzle")]
    max_steps = envs[0]["max_steps"]
    assert all(env["max_steps"] == max_steps for env in envs), (
        "make_tsne_step_grid_plot.py assumes both environments share the same "
        "MAX_STEPS (both are 5 as of writing) - the grid has one column per step."
    )

    # One discrete colour per integer compute time (1..max_steps), shared
    # across every panel so a colour means the same compute time everywhere.
    step_colours = {
        k: plt.get_cmap("tab10")(i) for i, k in enumerate(range(1, max_steps + 1))
    }

    fig, axes = plt.subplots(
        2,
        max_steps,
        figsize=set_size(
            doc_width_pt, fraction=0.95, subplots=(2, max_steps), use_golden_ratio=False
        ),
        subplot_kw={"projection": "3d"},
    )

    for row, env in enumerate(envs):
        step_latents = env["step_latents"]  # (n, max_steps, hidden_dim)
        compute_time = env["tsne_sample_compute_time"]
        # states_history keeps recomputing a "thought" for every example at
        # every CoT step regardless of when it actually halted (see
        # TransformerChainOfThoughtTorso's docstring) - those post-halt steps
        # are a counterfactual continuation the real policy never acts on.
        # Clamp each example's step index to its own compute_time instead, so
        # column i shows the state after min(i, compute_time) real steps -
        # frozen at whatever it was when that example actually halted, for
        # every later column (column max_steps is then exactly final_state
        # for every example, since compute_time <= max_steps always).
        compute_time_int = np.rint(compute_time).astype(int)
        n = step_latents.shape[0]
        perplexity = perplexity_for(n)
        for col in tqdm.tqdm(range(max_steps)):
            step_count = col + 1
            step_idx = np.minimum(step_count, compute_time_int) - 1
            latents = step_latents[np.arange(n), step_idx, :]
            embedding = TSNE(
                n_components=3, perplexity=perplexity, random_state=0, init="pca"
            ).fit_transform(latents)
            ax = axes[row, col]
            for k, colour in step_colours.items():
                mask = compute_time_int == k
                if not mask.any():
                    continue
                ax.scatter(
                    embedding[mask, 0],
                    embedding[mask, 1],
                    embedding[mask, 2],
                    color=colour,
                    s=4,
                    alpha=0.9,
                )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            if row == 0:
                ax.set_title(f"Step {col + 1}", fontsize=8, y=1.0)

    # Reserve a right margin for the figure legend (unlike fig.colorbar, a
    # figure legend doesn't shrink the axes to make room for itself). 0.88 was
    # not enough room for the "First-step compute time" legend title, which
    # ended up overlapping the last column's 3D panel - a 3D axes' actual
    # drawn extent (the perspective cube) also overflows its own get_position()
    # bbox, so this needs a bigger margin than a 2D legend would.
    fig.subplots_adjust(right=0.8)
    legend_handles = [
        plt.Line2D(
            [], [], marker="o", linestyle="", markersize=4, color=colour, label=str(k)
        )
        for k, colour in step_colours.items()
    ]
    fig.legend(
        handles=legend_handles,
        title="Compute time",
        loc="center right",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=7,
        title_fontsize=8,
        frameon=False,
    )

    # Row labels via fig.text at each row's leftmost axis position, not
    # ax.set_ylabel: on a 3D axes, set_ylabel places a rotated label *inside*
    # the 3D box along its own y-axis (oriented by the current view angle),
    # not a conventional row label on the left margin like it would for a 2D
    # axes - it comes out diagonal and clipped against the next axes. Read
    # positions after layout is final.
    for row, env in enumerate(envs):
        pos = axes[row, 0].get_position()
        fig.text(
            pos.x0 - 0.02,
            (pos.y0 + pos.y1) / 2,
            ENV_LABELS[env["env_label"]],
            fontsize=9,
            rotation=90,
            ha="right",
            va="center",
        )

    # Figure-level suptitle, not per-axis titles for the row label: 3D axes'
    # titles/labels tend to clip against the figure edge, and bbox_inches=
    # "tight" mis-crops 3D axes' irregular bounding box - so it's omitted
    # below (see the notebooks' own single-panel 3D t-SNE cells for the same
    # two issues and fixes).
    fig.suptitle("t-SNE of CoT step latent states, coloured by compute step", y=0.98)
    out_path = HERE / "analysis-tsne_step_grid.pdf"
    fig.savefig(out_path, dpi=600, format="pdf")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
