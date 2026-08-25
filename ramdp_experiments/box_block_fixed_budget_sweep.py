#!/usr/bin/env python
"""Fixed-computation-budget sweep for RAMDP systems on the box-moving
(Sokoban-like) puzzle (env=block_moving/block_moving_variable, see
stoix/envs/block_moving/block_moving_env.py).

Companion to lightsout_fixed_budget_sweep.py/minatar_fixed_budget_sweep.py:
instead of letting the actor's compute torso adaptively halt (min_steps=1,
sampled/learned halting), every job here pins `min_steps == max_steps ==
budget`, which - per the min_steps mechanism added to the compute torsos
(see stoix/networks/torso_compute*.py) - forbids halting before `budget`
steps and forces a halt at exactly `budget` steps. So every example always
takes exactly `budget` steps of computation, with no adaptivity at all.

This answers "does more computation help on this task?" directly (sweeping
`budget` at a fixed, non-adaptive cost) and is also a simpler system to debug
against, since compute_time is deterministic (== budget) rather than a
learned/sampled quantity.

Both box-count settings use `VariableQuarterGenerator` (level_generator=
variable, quarter_size=3 on a 6x6 grid - i.e. four 3x3 quadrants) with
`filtering=quarter` (stoix.envs.block_moving.wrappers.QuarterFilter), which
truncates the episode if the agent strays outside the box/target quadrants -
see stoix/envs/block_moving/generators.py's VariableQuarterGenerator and
wrappers.py's QuarterFilter.

BoxMovingToStoa's observation is the grid's factored-channel encoding,
`(grid_size, grid_size, 4)` (see stoix/envs/block_moving/stoa_adapter.py) -
so, exactly like MinAtar/Lights Out, CNN architectures (cnn+mlp,
cnn+transformer, cnn+gru, cnn+iru) consume it directly via a CNNTorso
input_layer, while non-CNN architectures flatten it to a `4 * grid_size^2`
-length vector via `stoa.FlattenObservationWrapper`.

Grid axes:
  - box_setting: fixed box-count/grid presets (see BOX_SETTINGS below) -
                 "exact-3" (number_of_boxes_min=max=number_of_moving_boxes_max=3)
                 | "exact-4" (same but 4 boxes), both grid_size=6,
                 episode_length=100, quarter_size=3.
  - system:      ff_reinforce (PonderNet-style REINFORCE) | ff_qac_fac | ff_qac_naive
  - architecture: mlp (AdaptiveComputationTimeTorso) | transformer
                 (TransformerChainOfThoughtTorso, latent CoT) |
                 gru (GRUAdaptiveComputationTimeTorso, GRU-based recurrent block
                 that re-feeds the encoded observation every pondering step) |
                 iru (IRUAdaptiveComputationTimeTorso, interpolation-recurrent-
                 unit-based recurrent block, same re-feeding) |
                 transformer_explicit_cot (TransformerExplicitCoTTorso, explicit
                 token CoT - only implemented for system=ff_reinforce, via
                 stoix/systems/ramdp_vpg/ff_reinforce_explicit_cot.py; requested
                 (system, architecture) combos outside that are skipped) |
                 cnn+mlp (CNNTorso input_layer feeding AdaptiveComputationTimeTorso) |
                 cnn+transformer (CNNTorso input_layer feeding
                 TransformerChainOfThoughtTorso) |
                 cnn+gru (CNNTorso input_layer feeding GRUAdaptiveComputationTimeTorso) |
                 cnn+iru (CNNTorso input_layer feeding IRUAdaptiveComputationTimeTorso)
  - budget:      fixed number of steps every example takes (max_steps == min_steps)
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
  - seed:        5 seeds per config by default

`generator_special`/`eval_generator_special` (env.kwargs) control which pool
of quadrant corner-pairs VariableQuarterGenerator samples from - see that
class and make_block_moving_env's docstring. Both default to False (i.e.
train and eval see the same corner-pair pool) via --generator-special/
--eval-generator-special; set --eval-generator-special true to instead
evaluate generalization to the held-out corner-pair pool.

gamma defaults to 0.99 - box-moving episodes are longer (episode_length=100)
than Lights Out's, but still short enough that 0.99 is a reasonable default;
override via --gamma if needed. total_timesteps defaults to 2e7 (a reduced
sweep budget) rather than the 1e8 used for full runs - re-run the winning
config(s) at full budget afterwards.

Jobs are scheduled across GPUs with a fixed number of concurrent runs per
GPU (a GPU "slot" queue + thread pool), each run pinned via
CUDA_VISIBLE_DEVICES and logged to its own file under <output-dir>/logs/.

Usage:
  python ramdp_experiments/box_block_fixed_budget_sweep.py --dry-run                # preview the grid
  python ramdp_experiments/box_block_fixed_budget_sweep.py --limit 6 --dry-run       # preview a slice
  python ramdp_experiments/box_block_fixed_budget_sweep.py                          # run the full sweep (both box settings)
  python ramdp_experiments/box_block_fixed_budget_sweep.py --box-settings exact-3   # only this preset
  python ramdp_experiments/box_block_fixed_budget_sweep.py --systems ff_reinforce --architectures mlp \\
      --box-settings exact-3 --budget 1,8 --hidden-dim 16 --lr 3e-4 --seeds 1  # small pilot / debug run
  python ramdp_experiments/box_block_fixed_budget_sweep.py --delightful true,false \\
      --delightful-eta 1.0,3.0                                    # sweep delightful PG on/off
  python ramdp_experiments/box_block_fixed_budget_sweep.py --architectures mlp \\
      --use-layer-norm true,false --use-input-layer-norm true,false  # sweep LayerNorm options
  python ramdp_experiments/box_block_fixed_budget_sweep.py --systems ff_reinforce \\
      --architectures transformer_explicit_cot                       # explicit-CoT sweep
  python ramdp_experiments/box_block_fixed_budget_sweep.py --lr 1e-4,3e-4 --critic-lr 1e-3  # decoupled lr sweeps
  python ramdp_experiments/box_block_fixed_budget_sweep.py --architectures cnn+mlp,cnn+transformer  # CNN-input sweep
  python ramdp_experiments/box_block_fixed_budget_sweep.py --architectures gru,iru,cnn+gru,cnn+iru  # recurrent-block sweep
  python ramdp_experiments/box_block_fixed_budget_sweep.py --eval-generator-special true  # generalization eval
"""

from __future__ import annotations

import argparse
import itertools
import json
import queue
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
# both): TransformerChainOfThoughtTorso, GRUAdaptiveComputationTimeTorso, and
# IRUAdaptiveComputationTimeTorso. Used to pick the right LayerNorm overrides
# in Job.command().
NO_LAYER_NORM_ARCHES = (
    "transformer",
    "cnn+transformer",
    "gru",
    "cnn+gru",
    "iru",
    "cnn+iru",
)
# Architectures whose input_layer is a CNNTorso (need the CNN-specific
# overrides below instead of the flatten-observation wrapper).
CNN_ARCHES = ("cnn+mlp", "cnn+transformer", "cnn+gru", "cnn+iru")

# Fixed box-count/grid presets - see module docstring's "Grid axes" section.
# All use VariableQuarterGenerator (level_generator=variable) with
# quarter_size=3 on a 6x6 grid (four 3x3 quadrants), satisfying
# VariableQuarterGenerator's number_of_boxes_max <= quarter_size^2 and
# number_of_boxes_min == number_of_boxes_max == number_of_moving_boxes_max
# assertions for both presets.
BOX_SETTINGS = {
    "exact-3": dict(
        number_of_boxes_min=3,
        number_of_boxes_max=3,
        number_of_moving_boxes_max=3,
        grid_size=6,
        quarter_size=3,
        episode_length=100,
    ),
    "exact-4": dict(
        number_of_boxes_min=4,
        number_of_boxes_max=4,
        number_of_moving_boxes_max=4,
        grid_size=6,
        quarter_size=3,
        episode_length=100,
    ),
}

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
    box_setting: str  # "exact-3" | "exact-4" - see BOX_SETTINGS
    system: str
    arch: str
    budget: int
    hidden_dim: int
    lr: float
    critic_lr: float
    delightful: bool
    delightful_eta: float
    use_layer_norm: bool
    use_input_layer_norm: bool
    seed: int
    total_timesteps: float
    generator_special: bool
    eval_generator_special: bool
    gamma: float
    output_dir: Path

    @property
    def env(self) -> str:
        """Box-moving preset identifier (e.g. "box_block-exact-3") - recorded
        in the manifest as the "env" field, matching every other
        *_fixed_budget_sweep.py script, for plot_fixed_budget_sweep.py's
        per-env grouping."""
        return f"box_block-{self.box_setting}"

    @property
    def run_name(self) -> str:
        name = (
            f"{self.env}-{self.system}-{self.arch}-budget_{self.budget}"
            f"-hidden_dim_{self.hidden_dim}-lr_{self.lr:g}-critic_lr_{self.critic_lr:g}"
        )
        if self.delightful:
            name += f"-delightful_eta_{self.delightful_eta:g}"
        if self.use_layer_norm:
            name += "-ln"
        if self.use_input_layer_norm:
            name += "-input_ln"
        if self.eval_generator_special != self.generator_special:
            name += "-eval_special"
        return name + f"-seed_{self.seed}"

    def command(self, python_bin: str) -> List[str]:
        if self.arch == EXPLICIT_COT_ARCH:
            script = EXPLICIT_COT_SCRIPT
            network = EXPLICIT_COT_NETWORK
        else:
            script = SYSTEM_TO_SCRIPT[self.system]
            network = ARCH_TO_NETWORK[self.system][self.arch]
        settings = BOX_SETTINGS[self.box_setting]
        cmd = [
            python_bin,
            script,
            # Base config is a fixed example (only one exists for the
            # "variable" generator), with the actual box/grid settings and
            # task name overridden below. env.scenario.name must stay
            # "block_moving-variable" - make_block_moving_env parses the
            # level generator *type* from it (not a free-form preset name);
            # only task_name carries the "exact-3"/"exact-4" label.
            "env=block_moving/block_moving_variable",
            "env.scenario.name=block_moving-variable",
            f"env.scenario.task_name=box_block_{self.box_setting.replace('-', '_')}",
            f"env.kwargs.grid_size={settings['grid_size']}",
            f"env.kwargs.quarter_size={settings['quarter_size']}",
            f"env.kwargs.number_of_boxes_min={settings['number_of_boxes_min']}",
            f"env.kwargs.number_of_boxes_max={settings['number_of_boxes_max']}",
            f"env.kwargs.number_of_moving_boxes_max={settings['number_of_moving_boxes_max']}",
            f"env.kwargs.episode_length={settings['episode_length']}",
            "env.kwargs.terminate_when_success=True",
            "env.kwargs.filtering=quarter",
            f"env.kwargs.generator_special={self.generator_special}",
            # `+` since eval_generator_special isn't a key in the base yaml
            # (block_moving_variable.yaml) - see make_block_moving_env.
            f"+env.kwargs.eval_generator_special={self.eval_generator_special}",
            f"network={network}",
            "logger.loggers.tensorboard.enabled=True",
            "logger.loggers.json.enabled=True",
            f"system.gamma={self.gamma:g}",
            f"arch.total_timesteps={self.total_timesteps:g}",
            f"arch.seed={self.seed}",
            "arch.num_evaluation=50",
            f"network.actor_network.pre_torso.hidden_dim={self.hidden_dim}",
            # min_steps == max_steps == budget: no adaptivity, every example always
            # takes exactly `budget` steps (see stoix/networks/torso_compute*.py).
            f"network.actor_network.pre_torso.max_steps={self.budget}",
            f"network.actor_network.pre_torso.min_steps={self.budget}",
            f"system.actor_lr={self.lr:g}",
            f"system.critic_lr={self.critic_lr:g}",
            f"system.ent_coef=0.01",
            f"system.delightful={self.delightful}",
            f"logger.base_exp_path={self.output_dir / self.run_name}",
        ]
        if self.delightful:
            cmd.append(f"system.delightful_eta={self.delightful_eta:g}")
        if self.arch == EXPLICIT_COT_ARCH:
            pass  # TransformerExplicitCoTTorso has no LayerNorm params yet.
        elif self.arch in NO_LAYER_NORM_ARCHES:
            # TransformerChainOfThoughtTorso, GRUAdaptiveComputationTimeTorso, and
            # IRUAdaptiveComputationTimeTorso only have use_input_layer_norm, not
            # use_layer_norm (see stoix/networks/torso_compute_transformer.py and
            # stoix/networks/torso_compute.py).
            cmd.append(
                f"network.actor_network.pre_torso.use_input_layer_norm={self.use_input_layer_norm}"
            )
        else:
            cmd.append(f"network.actor_network.pre_torso.use_layer_norm={self.use_layer_norm}")
            cmd.append(
                f"network.actor_network.pre_torso.use_input_layer_norm={self.use_input_layer_norm}"
            )
        if self.system in SYSTEM_TO_QAC_VARIANT:
            cmd.append(f"system.qac_variant={SYSTEM_TO_QAC_VARIANT[self.system]}")
        if self.arch not in CNN_ARCHES:
            # BoxMovingToStoa's native observation is (grid_size, grid_size, 4)
            # (see stoix/envs/block_moving/stoa_adapter.py) - flatten it to a
            # 4 * grid_size^2 vector for non-CNN torsos.
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

    # (system, arch, use_layer_norm, use_input_layer_norm) combos:
    #  - transformer_explicit_cot only exists for ff_reinforce (see
    #    EXPLICIT_COT_SYSTEMS) - any other requested (system, architecture)
    #    pair is skipped rather than erroring, so e.g. the default
    #    `--systems ff_reinforce,ff_qac_fac,ff_qac_naive` still works if the
    #    user adds `--architectures ...,transformer_explicit_cot`.
    #  - use_layer_norm only exists on AdaptiveComputationTimeTorso (mlp/cnn+mlp).
    #  - use_input_layer_norm exists on mlp/cnn+mlp/transformer/cnn+transformer/
    #    gru/cnn+gru/iru/cnn+iru, not yet on transformer_explicit_cot.
    # Unsupported axes are forced to a single False rather than needlessly
    # duplicated per requested setting.
    system_arch_ln_combos = []
    n_skipped_incompatible = 0
    for system in args.systems:
        for arch in args.architectures:
            if arch == EXPLICIT_COT_ARCH:
                if system not in EXPLICIT_COT_SYSTEMS:
                    n_skipped_incompatible += 1
                    continue
                ln_options = [(False, False)]
            elif arch in NO_LAYER_NORM_ARCHES:
                ln_options = [(False, uiln) for uiln in args.use_input_layer_norm]
            else:
                ln_options = [
                    (uln, uiln) for uln in args.use_layer_norm for uiln in args.use_input_layer_norm
                ]
            for use_layer_norm, use_input_layer_norm in ln_options:
                system_arch_ln_combos.append((system, arch, use_layer_norm, use_input_layer_norm))
    system_arch_ln_combos = list(dict.fromkeys(system_arch_ln_combos))
    if n_skipped_incompatible:
        print(
            f"Skipping {n_skipped_incompatible} (system, architecture) combo(s) requesting "
            f"{EXPLICIT_COT_ARCH}, which is only implemented for {EXPLICIT_COT_SYSTEMS}."
        )

    jobs = []
    for (
        box_setting,
        (system, arch, use_layer_norm, use_input_layer_norm),
        budget,
        hidden_dim,
        lr,
        critic_lr,
        (delightful, delightful_eta),
        seed,
    ) in itertools.product(
        args.box_settings,
        system_arch_ln_combos,
        args.budget,
        args.hidden_dim,
        args.lr,
        args.critic_lr,
        delightful_combos,
        range(args.seeds),
    ):
        jobs.append(
            Job(
                box_setting=box_setting,
                system=system,
                arch=arch,
                budget=budget,
                hidden_dim=hidden_dim,
                lr=lr,
                critic_lr=critic_lr,
                delightful=delightful,
                delightful_eta=delightful_eta,
                use_layer_norm=use_layer_norm,
                use_input_layer_norm=use_input_layer_norm,
                seed=seed,
                total_timesteps=args.total_timesteps,
                generator_special=args.generator_special,
                eval_generator_special=args.eval_generator_special,
                gamma=args.gamma,
                output_dir=args.output_dir,
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
        "--box-settings",
        default=",".join(BOX_SETTINGS),
        help=f"Comma-separated subset of {{{','.join(BOX_SETTINGS)}}} - see BOX_SETTINGS.",
    )
    parser.add_argument(
        "--systems",
        default="ff_reinforce,ff_qac_fac,ff_qac_naive",
        help="Comma-separated subset of {ff_reinforce, ff_qac_fac, ff_qac_naive}.",
    )
    parser.add_argument(
        "--architectures",
        default="mlp,transformer",
        help="Comma-separated subset of {mlp, transformer, gru, iru, transformer_explicit_cot, "
        "cnn+mlp, cnn+transformer, cnn+gru, cnn+iru}. transformer_explicit_cot "
        "(TransformerExplicitCoTTorso) is only "
        f"implemented for system in {EXPLICIT_COT_SYSTEMS} - other (system, architecture) combos "
        "requesting it are skipped, not errored.",
    )
    parser.add_argument(
        "--budget",
        default="1,2,4,8,16",
        help="Comma-separated fixed compute budgets - each sets max_steps == min_steps to this value.",
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
        "--generator-special",
        type=lambda x: x.strip().lower() in ("1", "true", "yes"),
        default=False,
        help="env.kwargs.generator_special for the training environment, applied to every job "
        "(not swept) - which pool of VariableQuarterGenerator corner-pairs to sample from. "
        "Default False.",
    )
    parser.add_argument(
        "--eval-generator-special",
        type=lambda x: x.strip().lower() in ("1", "true", "yes"),
        default=False,
        help="env.kwargs.eval_generator_special for the eval environment, applied to every job "
        "(not swept). Default False (matches --generator-special, i.e. train/eval stay "
        "symmetric); set true to evaluate generalization to the held-out corner-pair pool "
        "instead - see stoix/utils/make_env.py's make_block_moving_env.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="system.gamma, applied to every job (not swept).",
    )
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds per config, seeded 0..seeds-1.")
    parser.add_argument("--total-timesteps", type=float, default=2e7, help="arch.total_timesteps per run.")
    parser.add_argument("--gpus", default="auto", help="Comma-separated GPU ids, or 'auto' to detect via nvidia-smi.")
    parser.add_argument("--runs-per-gpu", type=int, default=2, help="Concurrent runs per GPU.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results_box_block_fixed_budget_sweep",
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

    args.box_settings = args.box_settings.split(",")
    args.systems = args.systems.split(",")
    args.architectures = args.architectures.split(",")
    args.budget = [int(x) for x in args.budget.split(",")]
    args.hidden_dim = [int(x) for x in args.hidden_dim.split(",")]
    args.lr = [float(x) for x in args.lr.split(",")]
    args.critic_lr = [float(x) for x in args.critic_lr.split(",")]
    args.delightful = [x.strip().lower() in ("1", "true", "yes") for x in args.delightful.split(",")]
    args.delightful_eta = [float(x) for x in args.delightful_eta.split(",")]
    args.use_layer_norm = [x.strip().lower() in ("1", "true", "yes") for x in args.use_layer_norm.split(",")]
    args.use_input_layer_norm = [
        x.strip().lower() in ("1", "true", "yes") for x in args.use_input_layer_norm.split(",")
    ]

    for b_setting in args.box_settings:
        assert b_setting in BOX_SETTINGS, f"unknown box setting {b_setting!r}, expected one of {list(BOX_SETTINGS)}"
    for s in args.systems:
        assert s in SYSTEM_TO_SCRIPT, f"unknown system {s!r}, expected one of {list(SYSTEM_TO_SCRIPT)}"
    valid_architectures = ("mlp", "transformer", "gru", "iru", EXPLICIT_COT_ARCH) + CNN_ARCHES
    for a in args.architectures:
        assert a in valid_architectures, f"unknown architecture {a!r}, expected one of {valid_architectures}"
    for b in args.budget:
        assert b >= 1, f"budget must be >= 1, got {b}"

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
    print(f"  box_settings={args.box_settings}")
    print(f"  systems={args.systems} architectures={args.architectures}")
    print(f"  budget (min_steps=max_steps)={args.budget} hidden_dim={args.hidden_dim} seeds=0..{args.seeds - 1}")
    print(f"  lr={args.lr} critic_lr={args.critic_lr}")
    print(f"  delightful={args.delightful} delightful_eta={args.delightful_eta}")
    print(
        f"  use_layer_norm={args.use_layer_norm} (mlp/cnn+mlp only) "
        f"use_input_layer_norm={args.use_input_layer_norm} (not transformer_explicit_cot)"
    )
    print(
        f"  generator_special={args.generator_special} "
        f"eval_generator_special={args.eval_generator_special} gamma={args.gamma}"
    )
    print(f"  total_timesteps={args.total_timesteps:g} output_dir={args.output_dir}")

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