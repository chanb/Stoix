#!/usr/bin/env python
"""Adaptive-computation-budget sweep for RAMDP systems on the Lights Out
puzzle (env=lightsout/lightsout_3x3, see
stoix/envs/lightsout/lightsout_env.py).

Companion to lightsout_fixed_budget_sweep.py: that script pins
`min_steps == max_steps == budget`, forbidding halting before `budget` steps
and forcing a halt at exactly `budget` steps - a fixed, non-adaptive
baseline. This script instead sweeps `min_steps` and `max_steps`
*independently* (see the min_steps mechanism added to the compute torsos,
stoix/networks/torso_compute*.py), so the actor's compute torso can
genuinely halt adaptively (sampled/learned halting) anywhere in
`[min_steps, max_steps]` per example, rather than always taking a fixed
number of steps. `min_steps == max_steps` is still possible (as one point in
this more general grid) but isn't the default.

This answers "how much does adaptive halting help, and at what compute
ceiling?" - sweeping both how early halting is allowed (`min_steps`) and how
much compute is available (`max_steps`) - complementing
lightsout_fixed_budget_sweep.py's "does more (fixed) computation help?"
question.

LightsOutEnv's observation is `(m, n, 2)` - the current grid and the goal
grid, stacked as two channels (see the env's module docstring) - so, exactly
like minatar_fixed_budget_sweep.py's MinAtar grids, CNN architectures
(cnn+mlp, cnn+transformer, cnn+gru, cnn+iru) consume it directly via a
CNNTorso input_layer, while non-CNN architectures flatten it to a
`2 * grid_size`-length vector via `stoa.FlattenObservationWrapper`.

Grid axes:
  - grid_size:   Lights Out puzzle grid, e.g. "3x3" - sets env.scenario.name=
                 lightsout-<grid_size> and env.kwargs.episode_length=m*n
                 (overridable via --episode-length)
  - system:      ff_reinforce (PonderNet-style REINFORCE) | ff_qac_fac | ff_qac_naive
  - architecture: mlp (AdaptiveComputationTimeTorso) | transformer
                 (TransformerChainOfThoughtTorso, latent CoT) |
                 gru (GRUAdaptiveComputationTimeTorso, GRU-based recurrent block
                 that re-feeds the encoded observation every pondering step) |
                 iru (IRUAdaptiveComputationTimeTorso, interpolation-recurrent-
                 unit-based recurrent block, same re-feeding) |
                 iru_unshared (UnsharedIRUAdaptiveComputationTimeTorso - like
                 iru, but each pondering step is its own independently-
                 parameterized IRU layer instead of one shared step reused at
                 every iteration, so max_steps grows the parameter count) |
                 transformer_explicit_cot (TransformerExplicitCoTTorso, explicit
                 token CoT - only implemented for system=ff_reinforce, via
                 stoix/systems/ramdp_vpg/ff_reinforce_explicit_cot.py; requested
                 (system, architecture) combos outside that are skipped) |
                 cnn+mlp (CNNTorso input_layer feeding AdaptiveComputationTimeTorso) |
                 cnn+transformer (CNNTorso input_layer feeding
                 TransformerChainOfThoughtTorso) |
                 cnn+gru (CNNTorso input_layer feeding GRUAdaptiveComputationTimeTorso) |
                 cnn+iru (CNNTorso input_layer feeding IRUAdaptiveComputationTimeTorso)
  - min_steps:   forbids halting (voluntarily, in replay, or greedily) before
                 this many pondering steps - see the compute torsos'
                 min_steps mechanism, stoix/networks/torso_compute*.py.
  - max_steps:   forces a halt at this many pondering steps if the example
                 hasn't halted voluntarily already. Combos where
                 min_steps > max_steps are invalid (the torsos assert
                 `1 <= min_steps <= max_steps`) and are skipped, not errored.
  - hidden_dim:  actor torso width (network.actor_network.pre_torso.hidden_dim)
  - lr:          system.actor_lr - has its own value list, independent of critic_lr's
                 (the full lr x critic_lr cross product is still swept)
  - critic_lr:   system.critic_lr - has its own value list, independent of lr's
  - delightful:  whether to gate the REINFORCE weight by the "delightful" surprisal
                 sigmoid (system.delightful); off by default. When on, also sweeps
                 delightful_eta (system.delightful_eta).
  - use_layer_norm: LayerNorm inside the shared ACTStep of
                 AdaptiveComputationTimeTorso; mlp/cnn+mlp only (transformer,
                 cnn+transformer, gru, iru, cnn+gru, cnn+iru, and
                 transformer_explicit_cot have no such param, so this is forced
                 off for them regardless of what's requested).
  - use_input_layer_norm: LayerNorm on the encoded observation before the
                 initial token/state/recurrent-input projection; supported by
                 mlp/cnn+mlp/transformer/cnn+transformer/gru/iru/cnn+gru/cnn+iru,
                 not yet by transformer_explicit_cot (forced off).
  - num_layers:  how many sub-layers are stacked inside each shared pondering
                 step (network.actor_network.pre_torso.num_layers) - Dense
                 layers for mlp/cnn+mlp, GRU cells for gru/cnn+gru, IRU cells
                 for iru/cnn+iru, transformer layers for
                 transformer/cnn+transformer (see
                 stoix/networks/torso_compute*.py); not swept for
                 transformer_explicit_cot, which keeps its own yaml default
                 (2) instead. Default 1.
  - num_heads:   attention head count (network.actor_network.pre_torso.num_heads);
                 only applies to transformer/cnn+transformer
                 (TransformerChainOfThoughtTorso) and transformer_explicit_cot
                 (TransformerExplicitCoTTorso) - every other architecture has
                 no such param, so this is forced to a single value for them.
                 Default 4.
  - seed:        5 seeds per config by default

`difficulty_threshold` (env.kwargs.difficulty_threshold) and `gamma`
(system.gamma) are fixed CLI-level values applied to every job, not swept -
see --difficulty-threshold/--gamma. Lights Out episodes are short
(episode_length defaults to grid_size), so gamma defaults to 0.99 rather than
minatar_fixed_budget_sweep.py's 0.9999 (chosen there for MinAtar's much
longer horizons). total_timesteps defaults to 2e7 (a reduced sweep budget)
rather than the 1e8 used for full runs - re-run the winning config(s) at full
budget afterwards.

Jobs are scheduled across GPUs with a fixed number of concurrent runs per
GPU (a GPU "slot" queue + thread pool), each run pinned via
CUDA_VISIBLE_DEVICES and logged to its own file under <output-dir>/logs/.

Usage:
  python ramdp_experiments/lightsout_sweep.py --dry-run                # preview the grid
  python ramdp_experiments/lightsout_sweep.py --limit 6 --dry-run       # preview a slice
  python ramdp_experiments/lightsout_sweep.py                          # run the full sweep (all grid sizes)
  python ramdp_experiments/lightsout_sweep.py --grid-sizes 3x3,5x5     # only these grid sizes
  python ramdp_experiments/lightsout_sweep.py --systems ff_reinforce --architectures mlp \\
      --grid-sizes 3x3 --min-steps 1 --max-steps 8 --hidden-dim 16 --lr 3e-4 --seeds 1  # small pilot / debug run
  python ramdp_experiments/lightsout_sweep.py --min-steps 1,4 --max-steps 4,8,16 \\
      # sweeps min_steps x max_steps (min_steps > max_steps combos skipped)
  python ramdp_experiments/lightsout_sweep.py --min-steps 8 --max-steps 8  # min_steps == max_steps, the fixed-budget special case
  python ramdp_experiments/lightsout_sweep.py --delightful true,false \\
      --delightful-eta 1.0,3.0                                    # sweep delightful PG on/off
  python ramdp_experiments/lightsout_sweep.py --architectures mlp \\
      --use-layer-norm true,false --use-input-layer-norm true,false  # sweep LayerNorm options
  python ramdp_experiments/lightsout_sweep.py --systems ff_reinforce \\
      --architectures transformer_explicit_cot                       # explicit-CoT sweep
  python ramdp_experiments/lightsout_sweep.py --lr 1e-4,3e-4 --critic-lr 1e-3  # decoupled lr sweeps
  python ramdp_experiments/lightsout_sweep.py --architectures cnn+mlp,cnn+transformer  # CNN-input sweep
  python ramdp_experiments/lightsout_sweep.py --architectures gru,iru,cnn+gru,cnn+iru  # recurrent-block sweep
  python ramdp_experiments/lightsout_sweep.py --difficulty-threshold 0.3  # easier training goals
"""

from __future__ import annotations

import argparse
import itertools
import json
import queue
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parent.parent

SYSTEM_TO_SCRIPT = {
    "ff_reinforce": "stoix/systems/ramdp_vpg/ff_reinforce.py",
    "ff_qac_fac": "stoix/systems/ramdp_vpg/ff_qac.py",
    "ff_qac_naive": "stoix/systems/ramdp_vpg/ff_qac.py",
}
SYSTEM_TO_QAC_VARIANT = {"ff_qac_fac": "fac", "ff_qac_naive": "naive"}
ARCH_TO_NETWORK = {
    "ff_reinforce": {
        "mlp": "mlp_compute",
        "iru_unshared": "iru_unshared_compute",
        "transformer": "transformer_compute",
        "gru": "gru_compute",
        "iru": "iru_compute",
        "cnn+mlp": "cnn_mlp_compute",
        "cnn+transformer": "cnn_transformer_compute",
        "cnn+gru": "cnn_gru_compute",
        "cnn+iru": "cnn_iru_compute",
    },
    "ff_qac_fac": {
        "mlp": "mlp_compute_qac",
        "iru_unshared": "iru_unshared_compute_qac",
        "transformer": "transformer_compute_qac",
        "gru": "gru_compute_qac",
        "iru": "iru_compute_qac",
        "cnn+mlp": "cnn_mlp_compute_qac",
        "cnn+transformer": "cnn_transformer_compute_qac",
        "cnn+gru": "cnn_gru_compute_qac",
        "cnn+iru": "cnn_iru_compute_qac",
    },
    "ff_qac_naive": {
        "mlp": "mlp_compute_qac",
        "iru_unshared": "iru_unshared_compute_qac",
        "transformer": "transformer_compute_qac",
        "gru": "gru_compute_qac",
        "iru": "iru_compute_qac",
        "cnn+mlp": "cnn_mlp_compute_qac",
        "cnn+transformer": "cnn_transformer_compute_qac",
        "cnn+gru": "cnn_gru_compute_qac",
        "cnn+iru": "cnn_iru_compute_qac",
    },
}
# Architectures whose pre_torso has no `use_layer_norm` param - only
# `use_input_layer_norm` - unlike AdaptiveComputationTimeTorso (which has
# both): TransformerChainOfThoughtTorso, GRUAdaptiveComputationTimeTorso,
# IRUAdaptiveComputationTimeTorso, and UnsharedIRUAdaptiveComputationTimeTorso.
# Used to pick the right LayerNorm overrides in Job.command().
NO_LAYER_NORM_ARCHES = (
    "transformer",
    "cnn+transformer",
    "gru",
    "cnn+gru",
    "iru",
    "cnn+iru",
    "iru_unshared",
)
# Architectures whose input_layer is a CNNTorso (need the CNN-specific
# overrides below instead of the flatten-observation wrapper).
CNN_ARCHES = ("cnn+mlp", "cnn+transformer", "cnn+gru", "cnn+iru")
# Architectures whose pre_torso has a `num_heads` param (attention heads) -
# TransformerChainOfThoughtTorso and TransformerExplicitCoTTorso; every other
# torso has no such concept. Used to pick whether --num-heads is swept for a
# given architecture in build_grid() and applied in Job.command().
TRANSFORMER_ARCHES = ("transformer", "cnn+transformer")
DEFAULT_GRID_SIZES = ("3x3", "4x4", "5x5")
GRID_SIZE_RE = re.compile(r"^(\d+)x(\d+)$")

# TransformerExplicitCoTTorso (see stoix/networks/torso_compute_explicit_cot.py)
# doesn't fit ARCH_TO_NETWORK/SYSTEM_TO_SCRIPT's (system, arch) -> network lookup:
# it only exists for ff_reinforce, and via a different training script
# (ff_reinforce_explicit_cot.py, not ff_reinforce.py), so it's handled separately.
EXPLICIT_COT_ARCH = "transformer_explicit_cot"
EXPLICIT_COT_SCRIPT = "stoix/systems/ramdp_vpg/ff_reinforce_explicit_cot.py"
EXPLICIT_COT_NETWORK = "transformer_explicit_cot"
EXPLICIT_COT_SYSTEMS = ("ff_reinforce",)


@dataclass
class Job:
    grid_size: str  # e.g. "3x3" - env.scenario.name becomes "lightsout-3x3"
    system: str
    arch: str
    min_steps: int
    max_steps: int
    hidden_dim: int
    lr: float
    critic_lr: float
    delightful: bool
    delightful_eta: float
    use_layer_norm: bool
    use_input_layer_norm: bool
    num_layers: int
    num_heads: int
    seed: int
    total_timesteps: float
    difficulty_threshold: float
    episode_length: int
    gamma: float
    output_dir: Path
    wandb: bool
    wandb_project: str

    @property
    def env(self) -> str:
        """Lightsout scenario identifier (e.g. "lightsout-3x3") - recorded in
        the manifest as the "env" field, matching every other
        *_fixed_budget_sweep.py script, for plot_fixed_budget_sweep.py's
        per-env grouping."""
        return f"lightsout-{self.grid_size}"

    @property
    def run_name(self) -> str:
        name = (
            f"{self.env}-{self.system}-{self.arch}"
            f"-min_steps_{self.min_steps}-max_steps_{self.max_steps}"
            f"-hidden_dim_{self.hidden_dim}-lr_{self.lr:g}-critic_lr_{self.critic_lr:g}"
        )
        # Not shown for transformer_explicit_cot - num_layers isn't swept for
        # that arch (it keeps its own yaml default), see build_grid.
        if self.arch != EXPLICIT_COT_ARCH:
            name += f"-num_layers_{self.num_layers}"
        # Only shown for architectures with a num_heads param - see
        # TRANSFORMER_ARCHES/build_grid.
        if self.arch in TRANSFORMER_ARCHES or self.arch == EXPLICIT_COT_ARCH:
            name += f"-num_heads_{self.num_heads}"
        if self.delightful:
            name += f"-delightful_eta_{self.delightful_eta:g}"
        if self.use_layer_norm:
            name += "-ln"
        if self.use_input_layer_norm:
            name += "-input_ln"
        return name + f"-seed_{self.seed}"

    def command(self, python_bin: str) -> List[str]:
        if self.arch == EXPLICIT_COT_ARCH:
            script = EXPLICIT_COT_SCRIPT
            network = EXPLICIT_COT_NETWORK
        else:
            script = SYSTEM_TO_SCRIPT[self.system]
            network = ARCH_TO_NETWORK[self.system][self.arch]
        cmd = [
            python_bin,
            script,
            # Base config is a fixed 3x3 example (only one exists), with the
            # actual grid size, task name, and episode length overridden below.
            "env=lightsout/lightsout_3x3",
            f"env.scenario.name={self.env}",
            f"env.scenario.task_name=lightsout_{self.grid_size}",
            f"env.kwargs.episode_length={self.episode_length}",
            f"env.kwargs.difficulty_threshold={self.difficulty_threshold:g}",
            f"network={network}",
            f"system.gamma={self.gamma:g}",
            f"arch.total_timesteps={self.total_timesteps:g}",
            f"arch.seed={self.seed}",
            "arch.num_evaluation=50",
            f"network.actor_network.pre_torso.hidden_dim={self.hidden_dim}",
            f"++network.actor_network.pre_torso.num_layers={self.num_layers}",
            # Independently swept - see the compute torsos' min_steps mechanism
            # (stoix/networks/torso_compute*.py): min_steps forbids halting
            # before that many steps; max_steps forces a halt at that many.
            # min_steps == max_steps (no adaptivity) is one point in this grid,
            # not the default - contrast lightsout_fixed_budget_sweep.py.
            f"network.actor_network.pre_torso.max_steps={self.max_steps}",
            f"network.actor_network.pre_torso.min_steps={self.min_steps}",
            f"system.actor_lr={self.lr:g}",
            f"system.critic_lr={self.critic_lr:g}",
            f"system.ent_coef=0.01",
            f"system.delightful={self.delightful}",
            f"logger.base_exp_path={self.output_dir / self.run_name}",
        ]
        if self.arch in TRANSFORMER_ARCHES or self.arch == EXPLICIT_COT_ARCH:
            # Attention head count - TransformerChainOfThoughtTorso/
            # TransformerExplicitCoTTorso only (see TRANSFORMER_ARCHES).
            cmd.append(f"++network.actor_network.pre_torso.num_heads={self.num_heads}")
        if self.wandb:
            # Fixed project name (not derived per-job) so every job in the
            # sweep lands in the same W&B project. run_id is pinned to
            # run_name (rather than left to WandBLogger's timestamp-based
            # unique_token) so two jobs launched in the same second can't
            # collide on the same W&B run.
            cmd.append("logger.loggers.wandb.enabled=True")
            cmd.append(f"logger.loggers.wandb.project={self.wandb_project}")
        if self.delightful:
            cmd.append(f"system.delightful_eta={self.delightful_eta:g}")
        if self.arch == EXPLICIT_COT_ARCH:
            pass  # TransformerExplicitCoTTorso has no LayerNorm params yet.
        elif self.arch in NO_LAYER_NORM_ARCHES:
            # TransformerChainOfThoughtTorso, GRUAdaptiveComputationTimeTorso, and
            # IRUAdaptiveComputationTimeTorso only have use_input_layer_norm, not
            # use_layer_norm (see stoix/networks/torso_compute_transformer.py and
            # stoix/networks/torso_compute.py). `++` (override-or-add), not `=`:
            # not every yaml config declares this key explicitly (e.g.
            # transformer_compute_qac.yaml was missing it), so a plain `=`
            # override can fail with "Key not in struct" - see num_layers above
            # for the same issue.
            cmd.append(
                f"++network.actor_network.pre_torso.use_input_layer_norm={self.use_input_layer_norm}"
            )
        else:
            cmd.append(f"++network.actor_network.pre_torso.use_layer_norm={self.use_layer_norm}")
            cmd.append(
                f"++network.actor_network.pre_torso.use_input_layer_norm={self.use_input_layer_norm}"
            )
        if self.system in SYSTEM_TO_QAC_VARIANT:
            cmd.append(f"system.qac_variant={SYSTEM_TO_QAC_VARIANT[self.system]}")
        if self.arch not in CNN_ARCHES:
            # LightsOutEnv's native observation is (m, n, 2) (see
            # stoix/envs/lightsout/lightsout_env.py) - flatten it to a
            # 2 * grid_size vector for non-CNN torsos.
            cmd.append(f"+env.wrapper._target_=stoa.FlattenObservationWrapper")
        else:
            cmd.append(f"network.actor_network.input_layer.channel_sizes=[16]")
            cmd.append(f"network.actor_network.input_layer.kernel_sizes=[3]")
            cmd.append(f"network.actor_network.input_layer.strides=[1]")
            cmd.append(f"network.actor_network.input_layer.hidden_sizes=[{self.hidden_dim}]")
            cmd.append(f"network.critic_network.input_layer.channel_sizes=[16]")
            cmd.append(f"network.critic_network.input_layer.kernel_sizes=[3]")
            cmd.append(f"network.critic_network.input_layer.strides=[1]")
            cmd.append(f"network.critic_network.input_layer.hidden_sizes=[256]")
            cmd.append(f"network.critic_network.pre_torso.layer_sizes=[256]")
        return cmd

    def run_dir(self) -> Path:
        return self.output_dir / self.run_name


def build_grid(args: argparse.Namespace) -> List[Job]:
    # (delightful, delightful_eta) combos: eta only matters (and is only swept)
    # when delightful=True, so a delightful=False entry doesn't get needlessly
    # duplicated once per requested eta value.
    delightful_combos = []
    for d in args.delightful:
        if d:
            for eta in args.delightful_eta:
                delightful_combos.append((True, eta))
        else:
            delightful_combos.append((False, args.delightful_eta[0]))
    delightful_combos = list(dict.fromkeys(delightful_combos))

    # (system, arch, use_layer_norm, use_input_layer_norm, num_layers, num_heads) combos:
    #  - transformer_explicit_cot only exists for ff_reinforce (see
    #    EXPLICIT_COT_SYSTEMS) - any other requested (system, architecture)
    #    pair is skipped rather than erroring, so e.g. the default
    #    `--systems ff_reinforce,ff_qac_fac,ff_qac_naive` still works if the
    #    user adds `--architectures ...,transformer_explicit_cot`.
    #  - use_layer_norm only exists on AdaptiveComputationTimeTorso (mlp/cnn+mlp).
    #  - use_input_layer_norm exists on mlp/cnn+mlp/transformer/cnn+transformer/
    #    gru/cnn+gru/iru/cnn+iru, not yet on transformer_explicit_cot.
    #  - num_layers (sub-layers stacked inside each shared pondering step,
    #    see stoix/networks/torso_compute*.py) is swept for every arch except
    #    transformer_explicit_cot, which keeps its own yaml default instead
    #    (see --num-layers help).
    #  - num_heads (attention heads) only exists on transformer/cnn+transformer/
    #    transformer_explicit_cot (see TRANSFORMER_ARCHES) - swept only for
    #    those, everything else forced to a single value.
    # Unsupported axes are forced to a single default value rather than
    # needlessly duplicated per requested setting.
    system_arch_ln_combos = []
    n_skipped_incompatible = 0
    for system in args.systems:
        for arch in args.architectures:
            is_transformer_arch = arch in TRANSFORMER_ARCHES or arch == EXPLICIT_COT_ARCH
            num_heads_options = args.num_heads if is_transformer_arch else [args.num_heads[0]]
            if arch == EXPLICIT_COT_ARCH:
                if system not in EXPLICIT_COT_SYSTEMS:
                    n_skipped_incompatible += 1
                    continue
                ln_options = [(False, False)]
                num_layers_options = [args.num_layers[0]]
            elif arch in NO_LAYER_NORM_ARCHES:
                ln_options = [(False, uiln) for uiln in args.use_input_layer_norm]
                num_layers_options = args.num_layers
            else:
                ln_options = [
                    (uln, uiln) for uln in args.use_layer_norm for uiln in args.use_input_layer_norm
                ]
                num_layers_options = args.num_layers
            for use_layer_norm, use_input_layer_norm in ln_options:
                for num_layers in num_layers_options:
                    for num_heads in num_heads_options:
                        system_arch_ln_combos.append(
                            (
                                system,
                                arch,
                                use_layer_norm,
                                use_input_layer_norm,
                                num_layers,
                                num_heads,
                            )
                        )
    system_arch_ln_combos = list(dict.fromkeys(system_arch_ln_combos))
    if n_skipped_incompatible:
        print(
            f"Skipping {n_skipped_incompatible} (system, architecture) combo(s) requesting "
            f"{EXPLICIT_COT_ARCH}, which is only implemented for {EXPLICIT_COT_SYSTEMS}."
        )

    # (min_steps, max_steps) combos: independently swept (see --min-steps/
    # --max-steps), but the compute torsos assert `1 <= min_steps <= max_steps`
    # (stoix/networks/torso_compute*.py) - a requested min_steps > max_steps
    # pairing is invalid, so it's skipped here rather than erroring at launch.
    step_combos = [
        (min_steps, max_steps)
        for min_steps in args.min_steps
        for max_steps in args.max_steps
        if min_steps <= max_steps
    ]
    n_skipped_step_combos = len(args.min_steps) * len(args.max_steps) - len(step_combos)
    if n_skipped_step_combos:
        print(
            f"Skipping {n_skipped_step_combos} (min_steps, max_steps) combo(s) with "
            "min_steps > max_steps."
        )

    jobs = []
    for (
        grid_size,
        (system, arch, use_layer_norm, use_input_layer_norm, num_layers, num_heads),
        (min_steps, max_steps),
        hidden_dim,
        lr,
        critic_lr,
        (delightful, delightful_eta),
        seed,
    ) in itertools.product(
        args.grid_sizes,
        system_arch_ln_combos,
        step_combos,
        args.hidden_dim,
        args.lr,
        args.critic_lr,
        delightful_combos,
        range(args.seeds),
    ):
        m, n = (int(x) for x in GRID_SIZE_RE.match(grid_size).groups())
        episode_length = args.episode_length if args.episode_length is not None else m * n
        jobs.append(
            Job(
                grid_size=grid_size,
                system=system,
                arch=arch,
                min_steps=min_steps,
                max_steps=max_steps,
                hidden_dim=hidden_dim,
                lr=lr,
                critic_lr=critic_lr,
                delightful=delightful,
                delightful_eta=delightful_eta,
                use_layer_norm=use_layer_norm,
                use_input_layer_norm=use_input_layer_norm,
                num_layers=num_layers,
                num_heads=num_heads,
                seed=seed,
                total_timesteps=args.total_timesteps,
                difficulty_threshold=args.difficulty_threshold,
                episode_length=episode_length,
                gamma=args.gamma,
                output_dir=args.output_dir,
                wandb=args.wandb,
                wandb_project=args.wandb_project,
            )
        )
    return jobs


def run_job(
    job: Job,
    gpu: int,
    python_bin: str,
    log_dir: Path,
    manifest_lock,
    manifest_path: Path,
    mem_fraction: float,
) -> dict:
    log_path = log_dir / f"{job.run_name}.log"
    cmd = job.command(python_bin)
    env = {
        "CUDA_VISIBLE_DEVICES": str(gpu),
        # JAX preallocates this fraction of the *visible* GPU's memory per process.
        # With `runs_per_gpu` processes sharing one physical GPU, each must be capped
        # to roughly 1/runs_per_gpu of the GPU or the later processes to allocate OOM.
        "XLA_PYTHON_CLIENT_MEM_FRACTION": str(mem_fraction),
        "XLA_FLAGS": "--xla_gpu_autotune_level=0",
    }
    import os

    full_env = os.environ.copy()
    full_env.update(env)

    start = time.time()
    with open(log_path, "w") as log_file:
        log_file.write(
            f"# GPU={gpu} XLA_PYTHON_CLIENT_MEM_FRACTION={mem_fraction:g}\n# CMD={' '.join(cmd)}\n\n"
        )
        log_file.flush()
        proc = subprocess.run(
            cmd, cwd=REPO_ROOT, env=full_env, stdout=log_file, stderr=subprocess.STDOUT
        )
    elapsed = time.time() - start

    result = {
        **asdict(job),
        "env": job.env,
        "run_name": job.run_name,
        "output_dir": str(job.output_dir),
        "gpu": gpu,
        "returncode": proc.returncode,
        "elapsed_sec": round(elapsed, 1),
        "log": str(log_path),
    }
    with manifest_lock:
        with open(manifest_path, "a") as f:
            f.write(json.dumps(result) + "\n")
    status = "OK" if proc.returncode == 0 else f"FAILED (rc={proc.returncode})"
    print(f"[{status}] {job.run_name} on GPU {gpu} ({elapsed:.0f}s) -> {log_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--grid-sizes",
        default=",".join(DEFAULT_GRID_SIZES),
        help="Comma-separated Lights Out grid sizes, each 'MxN' (e.g. '3x3'). Sets "
        "env.scenario.name=lightsout-<grid_size>.",
    )
    parser.add_argument(
        "--systems",
        default="ff_reinforce,ff_qac_fac,ff_qac_naive",
        help="Comma-separated subset of {ff_reinforce, ff_qac_fac, ff_qac_naive}.",
    )
    parser.add_argument(
        "--architectures",
        default="mlp,transformer",
        help="Comma-separated subset of {mlp, iru_unshared, transformer, gru, iru, "
        "transformer_explicit_cot, cnn+mlp, cnn+transformer, cnn+gru, cnn+iru}. iru_unshared "
        "(UnsharedIRUAdaptiveComputationTimeTorso) is like iru but with no weight sharing across "
        "pondering steps - each step is its own independently-parameterized IRU layer. "
        "transformer_explicit_cot (TransformerExplicitCoTTorso) is only "
        f"implemented for system in {EXPLICIT_COT_SYSTEMS} - other (system, architecture) combos "
        "requesting it are skipped, not errored.",
    )
    parser.add_argument(
        "--min-steps",
        default="1",
        help="Comma-separated network.actor_network.pre_torso.min_steps values, swept "
        "independently of --max-steps (full cross product) - forbids halting before this many "
        "pondering steps. Default 1 (no forced minimum - the usual adaptive-halting setting).",
    )
    parser.add_argument(
        "--max-steps",
        default="1,2,4,8,16",
        help="Comma-separated network.actor_network.pre_torso.max_steps values, swept "
        "independently of --min-steps (full cross product) - forces a halt at this many "
        "pondering steps if not halted already. Combos where min_steps > max_steps are invalid "
        "(skipped, not errored) - see build_grid.",
    )
    parser.add_argument("--hidden-dim", default="16,32", help="Comma-separated actor torso widths.")
    parser.add_argument(
        "--lr",
        default="1e-4,3e-4,1e-3",
        help="Comma-separated system.actor_lr values, swept independently of --critic-lr "
        "(full cross product).",
    )
    parser.add_argument(
        "--critic-lr",
        default="1e-4,3e-4,1e-3",
        help="Comma-separated system.critic_lr values, swept independently of --lr "
        "(full cross product).",
    )
    parser.add_argument(
        "--delightful",
        default="false",
        help="Comma-separated bools (true/false) - whether to gate the REINFORCE weight by the "
        "'delightful' surprisal sigmoid (system.delightful). Default off, matching prior behavior.",
    )
    parser.add_argument(
        "--delightful-eta",
        default="1.0",
        help="Comma-separated system.delightful_eta values, swept only for delightful=true jobs.",
    )
    parser.add_argument(
        "--use-layer-norm",
        default="false",
        help="Comma-separated bools (true/false) - LayerNorm inside AdaptiveComputationTimeTorso's "
        "shared ACTStep (network.actor_network.pre_torso.use_layer_norm). Only applies to "
        "architecture in {mlp, cnn+mlp}; ignored (forced off) otherwise, since those torsos have "
        "no such param.",
    )
    parser.add_argument(
        "--use-input-layer-norm",
        default="false",
        help="Comma-separated bools (true/false) - LayerNorm on the encoded observation before the "
        "initial token/state/recurrent-input projection "
        "(network.actor_network.pre_torso.use_input_layer_norm). Supported by architecture in "
        "{mlp, transformer, gru, iru, cnn+mlp, cnn+transformer, cnn+gru, cnn+iru}; ignored "
        "(forced off) for transformer_explicit_cot.",
    )
    parser.add_argument(
        "--num-layers",
        default="1",
        help="Comma-separated ints - how many sub-layers are stacked inside each shared pondering "
        "step (network.actor_network.pre_torso.num_layers): Dense layers for mlp/cnn+mlp "
        "(AdaptiveComputationTimeTorso), GRU cells for gru/cnn+gru (GRUAdaptiveComputationTimeTorso), "
        "IRU cells for iru/cnn+iru (IRUAdaptiveComputationTimeTorso), or transformer layers for "
        "transformer/cnn+transformer (TransformerChainOfThoughtTorso) - see "
        "stoix/networks/torso_compute*.py. Default 1 sub-layer per step. Not swept for "
        "transformer_explicit_cot (TransformerExplicitCoTTorso keeps its own yaml default of 2).",
    )
    parser.add_argument(
        "--num-heads",
        default="4",
        help="Comma-separated ints - attention head count "
        "(network.actor_network.pre_torso.num_heads). Only applies to architecture in "
        "{transformer, cnn+transformer, transformer_explicit_cot} (TransformerChainOfThoughtTorso/"
        "TransformerExplicitCoTTorso); ignored (forced to the first value) for every other "
        "architecture, since those torsos have no such param. Default 4.",
    )
    parser.add_argument(
        "--difficulty-threshold",
        type=float,
        default=0.5,
        help="env.kwargs.difficulty_threshold, applied to every job (not swept): goals reachable "
        "in fewer than difficulty_threshold * grid_size presses are sampled during training "
        "('easy'); the eval environment samples harder goals instead - see "
        "stoix/envs/lightsout/lightsout_env.py.",
    )
    parser.add_argument(
        "--episode-length",
        type=int,
        default=None,
        help="env.kwargs.episode_length, applied to every job (not swept). Defaults to grid_size "
        "(m * n) per grid size if omitted.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="system.gamma, applied to every job (not swept). Lower than "
        "minatar_fixed_budget_sweep.py's 0.9999 default, since Lights Out episodes are much "
        "shorter (episode_length defaults to grid_size).",
    )
    parser.add_argument(
        "--wandb",
        type=lambda x: x.strip().lower() in ("1", "true", "yes"),
        default=False,
        help="Enable W&B logging (logger.loggers.wandb.enabled) for every job. Default False.",
    )
    parser.add_argument(
        "--wandb-project",
        default="lightsout_sweep",
        help="W&B project name (logger.loggers.wandb.project), applied to every job so the whole "
        "sweep lands in the same project. Only used when --wandb is set.",
    )
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds per config, seeded 0..seeds-1.")
    parser.add_argument("--total-timesteps", type=float, default=2e7, help="arch.total_timesteps per run.")
    parser.add_argument("--gpus", default="auto", help="Comma-separated GPU ids, or 'auto' to detect via nvidia-smi.")
    parser.add_argument("--runs-per-gpu", type=int, default=2, help="Concurrent runs per GPU.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results_lightsout_sweep",
        help="Where per-run logger.base_exp_path and logs/ + manifest.jsonl are written.",
    )
    parser.add_argument("--python", default=str(REPO_ROOT / ".venv" / "bin" / "python"), help="Python interpreter.")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N jobs (for a pilot / sanity check).")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip jobs whose run directory already exists (default: on; use --no-skip-existing to force rerun).",
    )
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned jobs and exit without running anything.")
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation prompt before launching.")
    args = parser.parse_args()

    args.grid_sizes = args.grid_sizes.split(",")
    args.systems = args.systems.split(",")
    args.architectures = args.architectures.split(",")
    args.min_steps = [int(x) for x in args.min_steps.split(",")]
    args.max_steps = [int(x) for x in args.max_steps.split(",")]
    args.hidden_dim = [int(x) for x in args.hidden_dim.split(",")]
    args.lr = [float(x) for x in args.lr.split(",")]
    args.critic_lr = [float(x) for x in args.critic_lr.split(",")]
    args.delightful = [x.strip().lower() in ("1", "true", "yes") for x in args.delightful.split(",")]
    args.delightful_eta = [float(x) for x in args.delightful_eta.split(",")]
    args.use_layer_norm = [x.strip().lower() in ("1", "true", "yes") for x in args.use_layer_norm.split(",")]
    args.use_input_layer_norm = [
        x.strip().lower() in ("1", "true", "yes") for x in args.use_input_layer_norm.split(",")
    ]
    args.num_layers = [int(x) for x in args.num_layers.split(",")]
    args.num_heads = [int(x) for x in args.num_heads.split(",")]

    for g in args.grid_sizes:
        assert GRID_SIZE_RE.match(g), f"invalid grid size {g!r}, expected 'MxN' (e.g. '3x3')"
    for s in args.systems:
        assert s in SYSTEM_TO_SCRIPT, f"unknown system {s!r}, expected one of {list(SYSTEM_TO_SCRIPT)}"
    valid_architectures = (
        "mlp",
        "iru_unshared",
        "transformer",
        "gru",
        "iru",
        EXPLICIT_COT_ARCH,
    ) + CNN_ARCHES
    for a in args.architectures:
        assert a in valid_architectures, f"unknown architecture {a!r}, expected one of {valid_architectures}"
    for s in args.min_steps:
        assert s >= 1, f"min_steps must be >= 1, got {s}"
    for s in args.max_steps:
        assert s >= 1, f"max_steps must be >= 1, got {s}"
    for n in args.num_layers:
        assert n >= 1, f"num_layers must be >= 1, got {n}"
    for n in args.num_heads:
        assert n >= 1, f"num_heads must be >= 1, got {n}"

    if args.gpus == "auto":
        try:
            out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout
            gpu_ids = list(range(len(out.strip().splitlines())))
        except Exception:
            gpu_ids = [0]
    else:
        gpu_ids = [int(x) for x in args.gpus.split(",")]

    jobs = build_grid(args)
    if args.limit is not None:
        jobs = jobs[: args.limit]

    if args.skip_existing:
        remaining = [j for j in jobs if not j.run_dir().exists()]
        n_skipped = len(jobs) - len(remaining)
        jobs = remaining
    else:
        n_skipped = 0

    concurrency = len(gpu_ids) * args.runs_per_gpu
    mem_fraction = 0.95 / args.runs_per_gpu
    print(
        f"GPUs: {gpu_ids} x {args.runs_per_gpu} runs/GPU = {concurrency} concurrent "
        f"(XLA_PYTHON_CLIENT_MEM_FRACTION={mem_fraction:g} per process)"
    )
    print(f"Grid: {len(jobs)} jobs to run" + (f" ({n_skipped} skipped as already-existing)" if n_skipped else ""))
    print(f"  grid_sizes={args.grid_sizes}")
    print(f"  systems={args.systems} architectures={args.architectures}")
    print(
        f"  min_steps={args.min_steps} max_steps={args.max_steps} hidden_dim={args.hidden_dim} "
        f"seeds=0..{args.seeds - 1}"
    )
    print(f"  lr={args.lr} critic_lr={args.critic_lr}")
    print(f"  delightful={args.delightful} delightful_eta={args.delightful_eta}")
    print(
        f"  use_layer_norm={args.use_layer_norm} (mlp/cnn+mlp only) "
        f"use_input_layer_norm={args.use_input_layer_norm} (not transformer_explicit_cot)"
    )
    print(f"  num_layers={args.num_layers} (not swept for transformer_explicit_cot)")
    print(f"  num_heads={args.num_heads} (transformer/cnn+transformer/transformer_explicit_cot only)")
    print(
        f"  difficulty_threshold={args.difficulty_threshold} "
        f"episode_length={args.episode_length if args.episode_length is not None else 'grid_size (default)'} "
        f"gamma={args.gamma}"
    )
    print(f"  total_timesteps={args.total_timesteps:g} output_dir={args.output_dir}")
    if args.wandb:
        print(f"  wandb=True project={args.wandb_project}")

    if args.dry_run:
        for j in jobs:
            print(" ".join(j.command(args.python)))
        return

    if not jobs:
        print("Nothing to run.")
        return

    if not args.yes:
        resp = input(f"Launch {len(jobs)} jobs with concurrency {concurrency}? [y/N] ")
        if resp.strip().lower() != "y":
            print("Aborted.")
            return

    log_dir = args.output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.jsonl"

    import threading

    manifest_lock = threading.Lock()

    gpu_slots: "queue.Queue[int]" = queue.Queue()
    for gpu in gpu_ids:
        for _ in range(args.runs_per_gpu):
            gpu_slots.put(gpu)

    def worker(job: Job) -> dict:
        gpu = gpu_slots.get()
        try:
            return run_job(job, gpu, args.python, log_dir, manifest_lock, manifest_path, mem_fraction)
        finally:
            gpu_slots.put(gpu)

    failures = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(worker, job): job for job in jobs}
        for future in as_completed(futures):
            result = future.result()
            if result["returncode"] != 0:
                failures.append(result)

    print(f"\nDone. {len(jobs) - len(failures)}/{len(jobs)} succeeded.")
    if failures:
        print(f"{len(failures)} failed, see manifest.jsonl and logs/ for details:")
        for f in failures:
            print(f"  {f['run_name']} (rc={f['returncode']}) -> {f['log']}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    sys.exit(main())
