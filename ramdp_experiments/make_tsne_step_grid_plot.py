"""Builds a 4x5 grid of 3D t-SNE plots of the actor's latent state after each
Chain-of-Thought step (columns) for Lights Out and sliding puzzle (row pairs),
coloured by compute time and by shortest-path hardness.
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
    # envs = [load_plot_data("lightsout"), load_plot_data("slidingpuzzle")]
    # envs = [load_plot_data("slidingpuzzle")]
    envs = [load_plot_data("lightsout")]
    max_steps = envs[0]["max_steps"]
    assert all(env["max_steps"] == max_steps for env in envs), (
        "make_tsne_step_grid_plot.py assumes both environments share the same "
        "MAX_STEPS (both are 5 as of writing) - the grid has one column per step."
    )

    nrows = 2 * len(envs)
    fig, axes = plt.subplots(
        nrows,
        max_steps,
        figsize=set_size(
            doc_width_pt, fraction=0.95, subplots=(nrows, max_steps), use_golden_ratio=False
        ),
        subplot_kw={"projection": "3d"},
    )

    row_scatter = {}  # row -> one representative scatter mappable for that row's colourbar
    for env_idx, env in enumerate(envs):
        compute_row = 2 * env_idx
        hardness_row = 2 * env_idx + 1

        step_latents = env["step_latents"]  # (n, max_steps, hidden_dim)
        compute_time = env["tsne_sample_compute_time"]
        hardness = env["tsne_sample_hardness"]
        # states_history keeps recomputing a "thought" for every example at every CoT step
        # regardless of when it actually halted (see TransformerChainOfThoughtTorso's docstring) -
        # those post-halt steps are a counterfactual continuation the real policy never acts on.
        compute_time_int = np.rint(compute_time).astype(int)
        n = step_latents.shape[0]
        perplexity = perplexity_for(n)
        for col in tqdm.tqdm(range(max_steps), desc=env["env_label"]):
            step_count = col + 1
            step_idx = np.minimum(step_count, compute_time_int) - 1
            latents = step_latents[np.arange(n), step_idx, :]
            # One t-SNE fit per column, reused for both of this env's rows
            # below - the compute-time and shortest-path-length panels plot
            # the exact same points, just recoloured.
            embedding = TSNE(
                n_components=3, perplexity=perplexity, random_state=0, init="pca"
            ).fit_transform(latents)

            ax_compute = axes[compute_row, col]
            # vmin/vmax fixed to [1, max_steps] (not autoscaled to this
            # column's observed range) since compute_time is bounded there by
            # construction - keeps the colour scale identical across every
            # column/row even if, say, 5 never actually occurs in one column.
            row_scatter[compute_row] = ax_compute.scatter(
                embedding[:, 0],
                embedding[:, 1],
                embedding[:, 2],
                c=compute_time_int,
                cmap="viridis",
                vmin=1,
                vmax=max_steps,
                s=0.1,
                alpha=0.9,
            )
            if compute_row == 0:
                ax_compute.set_title(f"Step {col + 1}", fontsize=8, y=1.0)

            ax_hardness = axes[hardness_row, col]
            row_scatter[hardness_row] = ax_hardness.scatter(
                embedding[:, 0],
                embedding[:, 1],
                embedding[:, 2],
                c=hardness,
                cmap="viridis",
                s=0.1,
                alpha=0.9,
            )

            for ax in (ax_compute, ax_hardness):
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])

    # Reserve a right margin for the colourbars added below (fig.colorbar(ax=
    # axes[row, :]) would shrink those axes *after* the row-label loop below
    # already read their positions, so colourbar axes are placed manually via
    # fig.add_axes instead) and tighten the column spacing - the 3D panels'
    # own drawn content doesn't fill their bounding box, so the default
    # wspace left a lot of dead space between columns.
    fig.subplots_adjust(right=0.88, wspace=0.05)

    # One colourbar per row, positioned to match that row's own axes span -
    # every row gets the same treatment now (compute time and hardness both
    # use the continuous viridis spectrum), which is simpler than singling
    # out a discrete legend for the compute-time rows and also avoids that
    # legend's width/collision issues from an earlier version of this script.
    for row in range(nrows):
        row_axes = axes[row, :]
        y0 = min(ax.get_position().y0 for ax in row_axes)
        y1 = max(ax.get_position().y1 for ax in row_axes)
        pad = 0.2 * (y1 - y0)
        cax = fig.add_axes([0.885, y0 + pad, 0.015, (y1 - y0) - 2 * pad])
        cbar = fig.colorbar(row_scatter[row], cax=cax)
        if row % 2 == 0:
            cbar.set_label("Compute time", fontsize=8)
            cbar.set_ticks(range(1, max_steps + 1))
        else:
            cbar.set_label("Hardness", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    # Row labels via fig.text at each row's leftmost axis position, not ax.set_ylabel: on a 3D axes,
    # set_ylabel places a rotated label *inside* the 3D box along its own y-axis (oriented by the
    # current view angle), not a conventional row label on the left margin like it would for a 2D
    # axes - it comes out diagonal and clipped against the next axes.
    for env_idx, env in enumerate(envs):
        for sub_row, coloured_by in [(0, "compute time"), (1, "hardness")]:
            row = 2 * env_idx + sub_row
            pos = axes[row, 0].get_position()
            fig.text(
                pos.x0 - 0.02,
                (pos.y0 + pos.y1) / 2,
                f"{ENV_LABELS[env['env_label']]}\n({coloured_by})",
                fontsize=8,
                rotation=90,
                ha="right",
                va="center",
            )

    # fig.suptitle("t-SNE of CoT step latent states", y=0.98)
    out_path = HERE / "analysis-tsne_step_grid.png"
    fig.savefig(out_path, dpi=600, format="png")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
