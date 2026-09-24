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
from sklearn.manifold import TSNE

import analysis_utils as base

set_size = base.set_size
pgf_with_latex = base.pgf_with_latex
doc_width_pt = base.doc_width_pt
plt.rcParams.update(pgf_with_latex)

HERE = Path(__file__).resolve().parent


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

    compute_times = np.concatenate([env["tsne_sample_compute_time"] for env in envs])
    vmin, vmax = compute_times.min(), compute_times.max()

    fig, axes = plt.subplots(
        2,
        max_steps,
        figsize=set_size(
            doc_width_pt, fraction=1.3, subplots=(2, max_steps), use_golden_ratio=False
        ),
        subplot_kw={"projection": "3d"},
    )

    scatter = None
    for row, env in enumerate(envs):
        step_latents = env["step_latents"]  # (n, max_steps, hidden_dim)
        compute_time = env["tsne_sample_compute_time"]
        perplexity = perplexity_for(step_latents.shape[0])
        for col in range(max_steps):
            latents = step_latents[:, col, :]
            embedding = TSNE(
                n_components=3, perplexity=perplexity, random_state=0, init="pca"
            ).fit_transform(latents)
            ax = axes[row, col]
            scatter = ax.scatter(
                embedding[:, 0],
                embedding[:, 1],
                embedding[:, 2],
                c=compute_time,
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                s=4,
                alpha=0.7,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            if row == 0:
                ax.set_title(f"step {col + 1}", fontsize=8, y=1.0)

    fig.colorbar(
        scatter, ax=axes, label="First-step compute time", shrink=0.6, pad=0.05, aspect=30
    )

    # Row labels via fig.text at each row's leftmost axis position, not
    # ax.set_ylabel: on a 3D axes, set_ylabel places a rotated label *inside*
    # the 3D box along its own y-axis (oriented by the current view angle),
    # not a conventional row label on the left margin like it would for a 2D
    # axes - it comes out diagonal and clipped against the next axes. Read
    # positions after the colorbar call, which shrinks/repositions `axes`.
    for row, env in enumerate(envs):
        pos = axes[row, 0].get_position()
        fig.text(
            pos.x0 - 0.02,
            (pos.y0 + pos.y1) / 2,
            env["env_label"],
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
    fig.suptitle("t-SNE of CoT step latent states, coloured by compute time", y=0.98)
    out_path = HERE / "analysis-tsne_step_grid.pdf"
    fig.savefig(out_path, dpi=600, format="pdf")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
