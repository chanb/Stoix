#!/usr/bin/env python
"""Adaptive-computation-budget sweep for RAMDP systems on Jumanji environments
(env=jumanji/<env>): sokoban, slidingtile (SlidingTilePuzzle), knapsack
(Knapsack), maze (Maze) - see ramdp_experiments/experiments.md.

Companion to jumanji_fixed_budget_sweep.py: that script pins `min_steps ==
max_steps == budget`, forbidding halting before `budget` steps and forcing a
halt at exactly `budget` steps - a fixed, non-adaptive baseline. This script
instead sweeps `min_steps` and `max_steps` *independently* (see the
min_steps mechanism added to the compute torsos,
stoix/networks/torso_compute*.py), so the actor's compute torso can
genuinely halt adaptively (sampled/learned halting) anywhere in
`[min_steps, max_steps]` per example, rather than always taking a fixed
number of steps. `min_steps == max_steps` is still possible (as one point in
this more general grid) but isn't the default - exactly mirroring how
lightsout_sweep.py complements lightsout_fixed_budget_sweep.py.

This answers "how much does adaptive halting help, and at what compute
ceiling?" - sweeping both how early halting is allowed (`min_steps`) and how
much compute is available (`max_steps`) - complementing
jumanji_fixed_budget_sweep.py's "does more (fixed) computation help?"
question.

See jumanji_fixed_budget_sweep.py's module docstring for the per-env
observation/action/difficulty-knob details (sokoban, slidingtile, knapsack,
maze) - identical here, only the compute-budget axis differs (min_steps/
max_steps instead of budget).

Grid axes (identical to jumanji_fixed_budget_sweep.py except min_steps/
max_steps replacing budget - see that script's docstring for the full
system/architecture/PPO-knob/env-difficulty-knob descriptions):
  - env, <env>-specific difficulty knobs, system, architecture, hidden_dim,
    lr, critic_lr, delightful, delightful_eta, epochs, num_minibatches,
    clip_eps, clip_value_loss, use_layer_norm, use_input_layer_norm,
    num_layers, num_heads, mlp_dim, vocab_size, use_latent_feedback, seed: identical semantics to
    jumanji_fixed_budget_sweep.py, including its five ff_ppo_explicit_*
    systems and its transformer_explicit_cot/cnn+transformer_explicit_cot
    architectures (see EXPLICIT_COT_ARCHES/EXPLICIT_COT_SYSTEMS).
  - min_steps:   forbids halting (voluntarily, in replay, or greedily) before
                 this many pondering steps - see the compute torsos'
                 min_steps mechanism, stoix/networks/torso_compute*.py.
  - max_steps:   forces a halt at this many pondering steps if the example
                 hasn't halted voluntarily already. Combos where
                 min_steps > max_steps are invalid (the torsos assert
                 `1 <= min_steps <= max_steps`) and are skipped, not errored.

gamma defaults to 0.99 (system.gamma, applied to every job, not swept) -
same reasoning as jumanji_fixed_budget_sweep.py (all 4 envs have short
episode horizons). total_timesteps defaults to 2e7.

Jobs are scheduled across GPUs with a fixed number of concurrent runs per
GPU (a GPU "slot" queue + thread pool), each run pinned via
CUDA_VISIBLE_DEVICES and logged to its own file under <output-dir>/logs/ -
identical mechanism to jumanji_fixed_budget_sweep.py/lightsout_sweep.py.

Usage:
  python ramdp_experiments/jumanji_sweep.py --dry-run                # preview the grid
  python ramdp_experiments/jumanji_sweep.py --limit 6 --dry-run       # preview a slice
  python ramdp_experiments/jumanji_sweep.py                          # run the full sweep (all 4 envs)
  python ramdp_experiments/jumanji_sweep.py --envs sokoban,maze       # only these envs
  python ramdp_experiments/jumanji_sweep.py --systems ff_ppo_reinforce --architectures mlp \\
      --envs sokoban --sokoban-generator toy --min-steps 1 --max-steps 8 --hidden-dim 16 \\
      --lr 3e-4 --seeds 1 --total-timesteps 2e5 --limit 2  # small pilot / debug run
  python ramdp_experiments/jumanji_sweep.py --min-steps 1,4 --max-steps 4,8,16 \\
      # sweeps min_steps x max_steps (min_steps > max_steps combos skipped)
  python ramdp_experiments/jumanji_sweep.py --min-steps 8 --max-steps 8  # min_steps == max_steps, the fixed-budget special case
  python ramdp_experiments/jumanji_sweep.py --envs slidingtile \\
      --slidingtile-grid-size 3,4 --slidingtile-num-random-moves 5,20,100       # scramble-depth sweep
  python ramdp_experiments/jumanji_sweep.py --envs knapsack \\
      --knapsack-num-items 5,10,20,50 --knapsack-total-budget 2.5,12.5          # problem-size sweep
  python ramdp_experiments/jumanji_sweep.py --envs maze --maze-size 5,10,15  # maze-size sweep
  python ramdp_experiments/jumanji_sweep.py --architectures cnn+mlp,cnn+transformer \\
      --envs sokoban,slidingtile,maze  # CNN-input sweep (sokoban/slidingtile/maze, via jumanji/*_grid)
  python ramdp_experiments/jumanji_sweep.py --systems ff_ppo_fac,ff_ppo_naive,ff_ppo_reinforce \\
      --epochs 4 --num-minibatches 8,16 --clip-eps 0.1,0.2                 # PPO sweep
  python ramdp_experiments/jumanji_sweep.py --envs sokoban,slidingtile,maze \\
      --systems ff_ppo_explicit_fac --architectures cnn+transformer_explicit_cot  # CNN-input explicit-CoT sweep
  python ramdp_experiments/jumanji_sweep.py \\
      --systems ff_ppo_explicit_fac,ff_ppo_explicit_reinforce   # explicit-CoT PPO sweep (flattened obs)
"""

from __future__ import annotations

import argparse
import itertools
import json
import queue
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent

SYSTEM_TO_SCRIPT = {
    "ff_reinforce": "stoix/systems/ramdp_vpg/ff_reinforce.py",
    "ff_qac_fac": "stoix/systems/ramdp_vpg/ff_qac.py",
    "ff_qac_naive": "stoix/systems/ramdp_vpg/ff_qac.py",
    "ff_ppo_fac": "stoix/systems/ramdp_vpg/ff_ppo.py",
    "ff_ppo_naive": "stoix/systems/ramdp_vpg/ff_ppo.py",
    "ff_ppo_cond_naive": "stoix/systems/ramdp_vpg/ff_ppo.py",
    "ff_ppo_cond_fac": "stoix/systems/ramdp_vpg/ff_ppo.py",
    "ff_ppo_reinforce": "stoix/systems/ramdp_vpg/ff_ppo.py",
    # Explicit-CoT PPO (TransformerExplicitCoTTorso instead of a latent-CoT
    # torso) - one system per qac_variant, mirroring ff_ppo_fac/ff_ppo_naive/
    # ff_ppo_cond_naive/ff_ppo_cond_fac/ff_ppo_reinforce above. Architecture
    # defaults to transformer_explicit_cot (flattened observation) unless
    # --architectures requests transformer_explicit_cot/
    # cnn+transformer_explicit_cot explicitly - see EXPLICIT_COT_ARCHES/build_grid.
    "ff_ppo_explicit_fac": "stoix/systems/ramdp_vpg/ff_ppo_explicit_cot.py",
    "ff_ppo_explicit_naive": "stoix/systems/ramdp_vpg/ff_ppo_explicit_cot.py",
    "ff_ppo_explicit_cond_naive": "stoix/systems/ramdp_vpg/ff_ppo_explicit_cot.py",
    "ff_ppo_explicit_cond_fac": "stoix/systems/ramdp_vpg/ff_ppo_explicit_cot.py",
    "ff_ppo_explicit_reinforce": "stoix/systems/ramdp_vpg/ff_ppo_explicit_cot.py",
}
SYSTEM_TO_QAC_VARIANT = {
    "ff_qac_fac": "fac",
    "ff_qac_naive": "naive",
    "ff_ppo_fac": "fac",
    "ff_ppo_naive": "naive",
    "ff_ppo_cond_naive": "cond_naive",
    "ff_ppo_cond_fac": "cond_fac",
    "ff_ppo_reinforce": "reinforce",
    "ff_ppo_explicit_fac": "fac",
    "ff_ppo_explicit_naive": "naive",
    "ff_ppo_explicit_cond_naive": "cond_naive",
    "ff_ppo_explicit_cond_fac": "cond_fac",
    "ff_ppo_explicit_reinforce": "reinforce",
}
PPO_SYSTEMS = (
    "ff_ppo_fac",
    "ff_ppo_naive",
    "ff_ppo_cond_naive",
    "ff_ppo_cond_fac",
    "ff_ppo_reinforce",
    "ff_ppo_explicit_fac",
    "ff_ppo_explicit_naive",
    "ff_ppo_explicit_cond_naive",
    "ff_ppo_explicit_cond_fac",
    "ff_ppo_explicit_reinforce",
)
# ff_ppo_explicit_* systems whose architecture defaults to
# transformer_explicit_cot (flattened observation) rather than being picked
# via --architectures, unless --architectures requests one of
# EXPLICIT_COT_ARCHES explicitly (e.g. cnn+transformer_explicit_cot) - see
# build_grid.
EXPLICIT_COT_PPO_SYSTEMS = (
    "ff_ppo_explicit_fac",
    "ff_ppo_explicit_naive",
    "ff_ppo_explicit_cond_naive",
    "ff_ppo_explicit_cond_fac",
    "ff_ppo_explicit_reinforce",
)
# The four ff_ppo_explicit_* systems above whose critic is a genuine Q-V
# critic (separate V and Q torsos, see EXPLICIT_COT_NETWORK_BY_SYSTEM) -
# ff_ppo_explicit_reinforce uses a plain V-only critic instead. Used in
# Job.command() to target the right critic network override keys for CNN
# architectures. Unlike lightsout/minatar, jumanji's *non*-explicit-CoT Q-V
# systems (ff_qac_fac/ff_qac_naive/ff_ppo_fac/...) still use a shared-torso
# critic (see ARCH_TO_NETWORK) - only the newly-added explicit-CoT ones use
# separate torsos here.
EXPLICIT_COT_QAC_SYSTEMS = (
    "ff_ppo_explicit_fac",
    "ff_ppo_explicit_naive",
    "ff_ppo_explicit_cond_naive",
    "ff_ppo_explicit_cond_fac",
)
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
ARCH_TO_NETWORK["ff_ppo_fac"] = ARCH_TO_NETWORK["ff_qac_fac"]
ARCH_TO_NETWORK["ff_ppo_naive"] = ARCH_TO_NETWORK["ff_qac_naive"]
ARCH_TO_NETWORK["ff_ppo_cond_naive"] = ARCH_TO_NETWORK["ff_ppo_fac"]
ARCH_TO_NETWORK["ff_ppo_cond_fac"] = ARCH_TO_NETWORK["ff_ppo_fac"]
ARCH_TO_NETWORK["ff_ppo_reinforce"] = ARCH_TO_NETWORK["ff_reinforce"]
NO_LAYER_NORM_ARCHES = ("transformer", "cnn+transformer", "gru", "cnn+gru", "iru", "cnn+iru")
TRANSFORMER_ARCHES = ("transformer", "cnn+transformer")

JUMANJI_ENVS = ("sokoban", "slidingtile", "knapsack", "maze")

# TransformerExplicitCoTTorso (see stoix/networks/torso_compute_explicit_cot.py)
# doesn't fit ARCH_TO_NETWORK/SYSTEM_TO_SCRIPT's (system, arch) -> network lookup:
# it's only trained by a dedicated script per system (ff_reinforce_explicit_cot.py
# for ff_reinforce, ff_ppo_explicit_cot.py for ff_ppo_explicit_*), not the plain
# ff_reinforce.py/ff_ppo.py, so it's handled separately. Two architectures use
# it: transformer_explicit_cot (flattened observation) and
# cnn+transformer_explicit_cot (CNNTorso input_layer feeding the same torso,
# see cnn_transformer_explicit_cot*.yaml) - both listed in CNN_ARCHES/below so
# the CNN-vs-flatten observation handling in Job.command() applies uniformly.
# Unlike lightsout/minatar, ff_reinforce is env-agnostic here too (via
# ff_reinforce_explicit_cot.py) so it's included alongside the ff_ppo_explicit_*
# systems.
EXPLICIT_COT_ARCH = "transformer_explicit_cot"
CNN_EXPLICIT_COT_ARCH = "cnn+transformer_explicit_cot"
EXPLICIT_COT_ARCHES = (EXPLICIT_COT_ARCH, CNN_EXPLICIT_COT_ARCH)
EXPLICIT_COT_SCRIPT_BY_SYSTEM = {"ff_reinforce": "stoix/systems/ramdp_vpg/ff_reinforce_explicit_cot.py"}
EXPLICIT_COT_SCRIPT_BY_SYSTEM.update(
    (system, "stoix/systems/ramdp_vpg/ff_ppo_explicit_cot.py") for system in EXPLICIT_COT_PPO_SYSTEMS
)
# Network name depends on both system (which qac_variant, or plain V-only for
# ff_reinforce/ff_ppo_explicit_reinforce) and arch (flattened observation vs
# CNN input) - nested the same way ARCH_TO_NETWORK is. ff_ppo_explicit_fac/
# naive/cond_naive/cond_fac (EXPLICIT_COT_QAC_SYSTEMS) use the separate-torso
# Q-V critic network; ff_ppo_explicit_reinforce/ff_reinforce use the plain
# V-only network - see ff_ppo_explicit_cot.py's/ff_reinforce_explicit_cot.py's
# learner_setup.
EXPLICIT_COT_NETWORK_BY_SYSTEM = {
    "ff_reinforce": {
        EXPLICIT_COT_ARCH: "transformer_explicit_cot",
        CNN_EXPLICIT_COT_ARCH: "cnn_transformer_explicit_cot",
    },
    "ff_ppo_explicit_fac": {
        EXPLICIT_COT_ARCH: "transformer_explicit_cot_qac_separate_qv",
        CNN_EXPLICIT_COT_ARCH: "cnn_transformer_explicit_cot_qac_separate_qv",
    },
    "ff_ppo_explicit_naive": {
        EXPLICIT_COT_ARCH: "transformer_explicit_cot_qac_separate_qv",
        CNN_EXPLICIT_COT_ARCH: "cnn_transformer_explicit_cot_qac_separate_qv",
    },
    "ff_ppo_explicit_cond_naive": {
        EXPLICIT_COT_ARCH: "transformer_explicit_cot_qac_separate_qv",
        CNN_EXPLICIT_COT_ARCH: "cnn_transformer_explicit_cot_qac_separate_qv",
    },
    "ff_ppo_explicit_cond_fac": {
        EXPLICIT_COT_ARCH: "transformer_explicit_cot_qac_separate_qv",
        CNN_EXPLICIT_COT_ARCH: "cnn_transformer_explicit_cot_qac_separate_qv",
    },
    "ff_ppo_explicit_reinforce": {
        EXPLICIT_COT_ARCH: "transformer_explicit_cot",
        CNN_EXPLICIT_COT_ARCH: "cnn_transformer_explicit_cot",
    },
}
EXPLICIT_COT_SYSTEMS = tuple(EXPLICIT_COT_SCRIPT_BY_SYSTEM)
CNN_ARCHES = ("cnn+mlp", "cnn+transformer", "cnn+gru", "cnn+iru", CNN_EXPLICIT_COT_ARCH)
VALID_ARCHITECTURES = ("mlp", "transformer", "gru", "iru", EXPLICIT_COT_ARCH) + CNN_ARCHES
# Short forms for group_tag/run_name (wandb group names get long fast):
# "transformer" -> implicit-CoT transformer ("TF-iCoT"), "transformer_explicit_cot"
# -> explicit-CoT transformer ("TF-eCoT"); mlp/gru/iru are already short.
ARCH_SHORT_TAG = {
    "transformer": "TF-iCoT",
    "cnn+transformer": "cnn+TF-iCoT",
    EXPLICIT_COT_ARCH: "TF-eCoT",
    CNN_EXPLICIT_COT_ARCH: "cnn+TF-eCoT",
}

# env -> (non-CNN scenario, CNN/grid scenario or None if unsupported).
ENV_SCENARIOS = {
    "sokoban": ("jumanji/sokoban", "jumanji/sokoban_grid"),
    "slidingtile": ("jumanji/slidingtile", "jumanji/slidingtile_grid"),
    "knapsack": ("jumanji/knapsack", None),
    "maze": ("jumanji/maze", "jumanji/maze_grid"),
}
ENV_SUPPORTS_CNN = {env: grid is not None for env, (_, grid) in ENV_SCENARIOS.items()}
# knapsack/maze already set env.wrapper in their yaml (ConcatObservationWrapper,
# since their observation is several equally-necessary fields with no single
# attribute to extract - see those yamls) - unlike sokoban/slidingtile (whose
# native observation is already one array), so non-CNN jobs for those two
# must NOT also append the +env.wrapper._target_=stoa.FlattenObservationWrapper
# override lightsout/minatar-style jobs use, which would conflict.
ENV_HAS_BUILTIN_WRAPPER = {"sokoban": False, "slidingtile": False, "knapsack": True, "maze": True}

SOKOBAN_GENERATOR_CHOICES = (
    "default",
    "toy",
    "simple",
    "unfiltered-train",
    "medium-train",
    "hard",
)
# Shortened dataset_name -> group_tag suffix for the longer Boxoban tiers (the
# override still passes the full HuggingFace dataset_name, only the tag shrinks).
SOKOBAN_TAG_SHORT = {"unfiltered-train": "unfilt-train", "medium-train": "med-train"}

SERVER_MODULES = {
    "vulcan": ["StdEnv/2023", "cuda/12.2"],
}


@dataclass(frozen=True)
class EnvDifficulty:
    """One difficulty-knob setting for a given env: `tag` is the short
    group_tag/run_name suffix, `overrides` are the literal Hydra CLI override
    strings to append (see ENV_DIFFICULTY_AXES)."""

    tag: str
    overrides: Tuple[str, ...] = ()


def _sokoban_difficulty_axis(args: argparse.Namespace) -> List[EnvDifficulty]:
    combos = []
    for choice in args.sokoban_generator:
        if choice == "default":
            combos.append(EnvDifficulty(tag="default"))
        elif choice == "toy":
            combos.append(
                EnvDifficulty(
                    tag="toy",
                    overrides=(
                        "+env.kwargs.generator._target_="
                        "jumanji.environments.routing.sokoban.generator.ToyGenerator",
                    ),
                )
            )
        elif choice == "simple":
            combos.append(
                EnvDifficulty(
                    tag="simple",
                    overrides=(
                        "+env.kwargs.generator._target_="
                        "jumanji.environments.routing.sokoban.generator.SimpleSolveGenerator",
                    ),
                )
            )
        else:
            # unfiltered-train | medium-train | hard: Boxoban difficulty tiers,
            # downloaded from HuggingFace Hub on first use.
            combos.append(
                EnvDifficulty(
                    tag=SOKOBAN_TAG_SHORT.get(choice, choice),
                    overrides=(
                        "+env.kwargs.generator._target_="
                        "jumanji.environments.routing.sokoban.generator.HuggingFaceDeepMindGenerator",
                        f"+env.kwargs.generator.dataset_name={choice}",
                        "+env.kwargs.generator.proportion_of_files=1.0",
                    ),
                )
            )
    return combos


def _slidingtile_difficulty_axis(args: argparse.Namespace) -> List[EnvDifficulty]:
    return [
        EnvDifficulty(
            tag=f"gs{gs}-nrm{nrm}",
            overrides=(
                f"env.kwargs.generator.grid_size={gs}",
                f"env.kwargs.generator.num_random_moves={nrm}",
            ),
        )
        for gs in args.slidingtile_grid_size
        for nrm in args.slidingtile_num_random_moves
    ]


def _knapsack_difficulty_axis(args: argparse.Namespace) -> List[EnvDifficulty]:
    # Item weights are drawn in [0, 1], so total weight is at most num_items -
    # if num_items < total_budget, the budget can never bind (every item
    # always fits), making the task trivial. Skip those combos.
    combos = [
        EnvDifficulty(
            tag=f"ni{ni}-tb{tb:g}",
            overrides=(
                f"env.kwargs.generator.num_items={ni}",
                f"env.kwargs.generator.total_budget={tb:g}",
            ),
        )
        for ni in args.knapsack_num_items
        for tb in args.knapsack_total_budget
        if ni >= tb
    ]
    n_skipped = len(args.knapsack_num_items) * len(args.knapsack_total_budget) - len(combos)
    if n_skipped:
        print(
            f"Skipping {n_skipped} knapsack (num_items, total_budget) combo(s) where "
            "num_items < total_budget (item weights are in [0, 1], so the budget can never bind)."
        )
    return combos


def _maze_difficulty_axis(args: argparse.Namespace) -> List[EnvDifficulty]:
    return [
        EnvDifficulty(
            tag=f"sz{size}",
            overrides=(
                f"env.kwargs.generator.num_rows={size}",
                f"env.kwargs.generator.num_cols={size}",
            ),
        )
        for size in args.maze_size
    ]


ENV_DIFFICULTY_AXES = {
    "sokoban": _sokoban_difficulty_axis,
    "slidingtile": _slidingtile_difficulty_axis,
    "knapsack": _knapsack_difficulty_axis,
    "maze": _maze_difficulty_axis,
}


@dataclass
class Job:
    env: str
    difficulty: EnvDifficulty
    system: str
    arch: str
    min_steps: int
    max_steps: int
    hidden_dim: int
    lr: float
    critic_lr: float
    ent_coef: float
    delightful: bool
    delightful_eta: float
    epochs: int
    num_minibatches: int
    clip_eps: float
    clip_value_loss: bool
    use_layer_norm: bool
    use_input_layer_norm: bool
    num_layers: int
    num_heads: int
    mlp_dim: int
    vocab_size: int
    use_latent_feedback: bool
    seed: int
    total_timesteps: float
    total_num_envs: int
    rollout_length: int
    gamma: float
    output_dir: Path
    wandb: bool
    wandb_project: str

    @property
    def group_tag_parts(self) -> List[str]:
        """The group tag broken into semantic chunks - env/difficulty, algo,
        step budget, network hparams, PPO hparams, misc flags - instead of one
        flat dash-joined string. `group_tag` still joins these with "-" for
        run_name/filenames/manifest (unchanged, filesystem-safe); the parts
        list is for `logger.loggers.wandb.group_tag`, which Neptune stores as
        a real list of tags (see stoix/utils/logger.py) so each axis stays
        independently filterable instead of buried in one long string.
        """
        system_short = (
            self.system.removeprefix("ff_").replace("explicit", "expl").replace("reinforce", "reinf")
        )
        arch_short = ARCH_SHORT_TAG.get(self.arch, self.arch)
        parts = [
            f"{self.env}-{self.difficulty.tag}",
            f"{system_short}-{arch_short}",
            f"mn{self.min_steps}-mx{self.max_steps}",
        ]

        net = (
            f"hd{self.hidden_dim}-lr{self.lr:g}-clr{self.critic_lr:g}-ec{self.ent_coef:g}"
            f"-nl{self.num_layers}"
        )
        if self.arch in TRANSFORMER_ARCHES or self.arch in EXPLICIT_COT_ARCHES:
            net += f"-nh{self.num_heads}-md{self.mlp_dim}"
        # Only shown for the explicit-CoT arches - vocab_size doesn't exist on
        # any other architecture, see EXPLICIT_COT_ARCHES/build_grid.
        if self.arch in EXPLICIT_COT_ARCHES:
            net += f"-vs{self.vocab_size}"
        parts.append(net)

        if self.system in PPO_SYSTEMS:
            ppo = f"ep{self.epochs}-mb{self.num_minibatches}-clip{self.clip_eps:g}"
            if not self.clip_value_loss:
                ppo += "-l2c"
            parts.append(ppo)

        extra = []
        if self.delightful:
            extra.append(f"deta{self.delightful_eta:g}")
        if self.use_layer_norm:
            extra.append("ln")
        if self.use_input_layer_norm:
            extra.append("iln")
        if self.use_latent_feedback:
            extra.append("lf")
        if extra:
            parts.append("-".join(extra))

        return parts

    @property
    def group_tag(self) -> str:
        return "-".join(self.group_tag_parts)

    @property
    def run_name(self) -> str:
        return f"{self.group_tag}-seed_{self.seed}"

    def command(self, python_bin: str) -> List[str]:
        if self.arch in EXPLICIT_COT_ARCHES:
            script = EXPLICIT_COT_SCRIPT_BY_SYSTEM[self.system]
            network = EXPLICIT_COT_NETWORK_BY_SYSTEM[self.system][self.arch]
        else:
            script = SYSTEM_TO_SCRIPT[self.system]
            network = ARCH_TO_NETWORK[self.system][self.arch]
        is_cnn = self.arch in CNN_ARCHES
        flat_scenario, grid_scenario = ENV_SCENARIOS[self.env]
        if is_cnn:
            assert grid_scenario is not None, f"{self.env} has no CNN/grid scenario"
            scenario = grid_scenario
        else:
            scenario = flat_scenario

        cmd = [
            python_bin,
            script,
            f"env={scenario}",
            f"network={network}",
            f"system.gamma={self.gamma:g}",
            f"arch.total_timesteps={self.total_timesteps:g}",
            f"arch.total_num_envs={self.total_num_envs}",
            f"arch.seed={self.seed}",
            "arch.num_evaluation=50",
            f"network.actor_network.pre_torso.hidden_dim={self.hidden_dim}",
            f"++network.actor_network.pre_torso.num_layers={self.num_layers}",
            # Independently swept - see the compute torsos' min_steps mechanism
            # (stoix/networks/torso_compute*.py): min_steps forbids halting
            # before that many steps; max_steps forces a halt at that many.
            # min_steps == max_steps (no adaptivity) is one point in this grid,
            # not the default - contrast jumanji_fixed_budget_sweep.py.
            f"network.actor_network.pre_torso.max_steps={self.max_steps}",
            f"network.actor_network.pre_torso.min_steps={self.min_steps}",
            f"system.actor_lr={self.lr:g}",
            f"system.critic_lr={self.critic_lr:g}",
            f"system.ent_coef={self.ent_coef:g}",
            f"system.rollout_length={self.rollout_length}",
            f"logger.base_exp_path={self.output_dir / self.run_name}",
        ]
        cmd.extend(self.difficulty.overrides)

        if self.system in PPO_SYSTEMS:
            cmd.append(f"system.epochs={self.epochs}")
            cmd.append(f"system.num_minibatches={self.num_minibatches}")
            cmd.append(f"system.clip_eps={self.clip_eps:g}")
            cmd.append(f"system.clip_value_loss={self.clip_value_loss}")
        else:
            cmd.append(f"system.delightful={self.delightful}")
        if self.arch in TRANSFORMER_ARCHES or self.arch in EXPLICIT_COT_ARCHES:
            cmd.append(f"++network.actor_network.pre_torso.num_heads={self.num_heads}")
            cmd.append(f"++network.actor_network.pre_torso.mlp_dim={self.mlp_dim}")
        if self.arch in EXPLICIT_COT_ARCHES:
            # Thought-token vocabulary size - TransformerExplicitCoTTorso only,
            # no other architecture has this param.
            cmd.append(f"++network.actor_network.pre_torso.vocab_size={self.vocab_size}")
            # Latent feedback decoding (Full-Bandwidth Transformer, arXiv:2608.08888) -
            # TransformerExplicitCoTTorso only, see stoix/networks/torso_compute_explicit_cot.py.
            cmd.append(
                f"++network.actor_network.pre_torso.use_latent_feedback={self.use_latent_feedback}"
            )
        if self.wandb:
            cmd.append("logger.loggers.wandb.enabled=True")
            cmd.append(f"logger.loggers.wandb.project={self.wandb_project}")
            cmd.append(f"logger.loggers.wandb.group_tag=[{','.join(self.group_tag_parts)}]")
        if self.delightful:
            cmd.append(f"system.delightful_eta={self.delightful_eta:g}")
        # `++` (override-or-add), not `=`: not every network yaml declares
        # use_layer_norm/use_input_layer_norm explicitly, so a plain `=`
        # override can fail with "Key not in struct" for some (system, arch)
        # combos - matches lightsout_fixed_budget_sweep.py/lightsout_sweep.py.
        if self.arch in EXPLICIT_COT_ARCHES or self.arch in NO_LAYER_NORM_ARCHES:
            # TransformerExplicitCoTTorso only has use_input_layer_norm, not
            # use_layer_norm, same as TransformerChainOfThoughtTorso/
            # GRUAdaptiveComputationTimeTorso/IRUAdaptiveComputationTimeTorso
            # (see stoix/networks/torso_compute_explicit_cot.py and
            # stoix/networks/torso_compute_transformer.py).
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

        if is_cnn:
            cmd.append(f"network.actor_network.input_layer.channel_sizes=[8]")
            cmd.append(f"network.actor_network.input_layer.kernel_sizes=[3]")
            cmd.append(f"network.actor_network.input_layer.strides=[1]")
            cmd.append(f"network.actor_network.input_layer.hidden_sizes=[{self.hidden_dim}]")
            if self.system in EXPLICIT_COT_QAC_SYSTEMS:
                # Separate-torso Q-V critic (value_input_layer/q_input_layer +
                # value_pre_torso/q_pre_torso), not a single input_layer/pre_torso.
                for head in ("value", "q"):
                    cmd.append(f"network.critic_network.{head}_input_layer.channel_sizes=[16]")
                    cmd.append(f"network.critic_network.{head}_input_layer.kernel_sizes=[3]")
                    cmd.append(f"network.critic_network.{head}_input_layer.strides=[1]")
                    cmd.append(f"network.critic_network.{head}_input_layer.hidden_sizes=[256]")
                    cmd.append(f"network.critic_network.{head}_pre_torso.layer_sizes=[256]")
            else:
                cmd.append(f"network.critic_network.input_layer.channel_sizes=[16]")
                cmd.append(f"network.critic_network.input_layer.kernel_sizes=[3]")
                cmd.append(f"network.critic_network.input_layer.strides=[1]")
                cmd.append(f"network.critic_network.input_layer.hidden_sizes=[256]")
                cmd.append(f"network.critic_network.pre_torso.layer_sizes=[256]")
        elif not ENV_HAS_BUILTIN_WRAPPER[self.env]:
            # sokoban (non-CNN)/slidingtile: native observation is a single
            # multi-dim array (grid/puzzle) that needs flattening for
            # non-CNN architectures - knapsack/maze already declare
            # ConcatObservationWrapper in their yaml (see ENV_HAS_BUILTIN_WRAPPER).
            cmd.append("+env.wrapper._target_=stoa.FlattenObservationWrapper")
        return cmd

    def run_dir(self) -> Path:
        return self.output_dir / self.run_name


def build_grid(args: argparse.Namespace) -> List[Job]:
    delightful_combos = []
    for d in args.delightful:
        if d:
            for eta in args.delightful_eta:
                delightful_combos.append((True, eta))
        else:
            delightful_combos.append((False, args.delightful_eta[0]))
    delightful_combos = list(dict.fromkeys(delightful_combos))

    ppo_combos = list(
        itertools.product(args.epochs, args.num_minibatches, args.clip_eps, args.clip_value_loss)
    )

    # (system, arch, use_layer_norm, use_input_layer_norm, num_layers, num_heads,
    # mlp_dim) combos:
    #  - transformer_explicit_cot/cnn+transformer_explicit_cot only exist for
    #    system in EXPLICIT_COT_SYSTEMS - any other requested (system,
    #    architecture) pair is skipped rather than erroring.
    #  - ff_ppo_explicit_*'s architecture defaults to transformer_explicit_cot
    #    (flattened observation, see EXPLICIT_COT_PPO_SYSTEMS) when
    #    --architectures doesn't request either EXPLICIT_COT_ARCHES value;
    #    requesting cnn+transformer_explicit_cot (optionally alongside
    #    transformer_explicit_cot) opts into the CNN-input variant instead -
    #    see EXPLICIT_COT_NETWORK_BY_SYSTEM.
    #  - num_layers is swept for every arch, including the explicit-CoT arches;
    #    num_heads/mlp_dim are also swept for them (like TRANSFORMER_ARCHES),
    #    everything else forced to a single value.
    #  - vocab_size (thought-token vocabulary size) only exists on the
    #    explicit-CoT arches (see EXPLICIT_COT_ARCHES) - swept only for them,
    #    everything else (including plain transformer) forced to a single
    #    value.
    #  - use_latent_feedback (latent feedback decoding) likewise only exists on
    #    the explicit-CoT arches - swept only for them, everything else forced
    #    to a single value.
    system_arch_ln_combos = []
    n_skipped_incompatible = 0
    for system in args.systems:
        if system in EXPLICIT_COT_PPO_SYSTEMS:
            requested_explicit_cot_archs = [a for a in args.architectures if a in EXPLICIT_COT_ARCHES]
            archs = requested_explicit_cot_archs or (EXPLICIT_COT_ARCH,)
        else:
            archs = args.architectures
        for arch in archs:
            is_transformer_arch = arch in TRANSFORMER_ARCHES or arch in EXPLICIT_COT_ARCHES
            num_heads_options = args.num_heads if is_transformer_arch else [args.num_heads[0]]
            mlp_dim_options = args.mlp_dim if is_transformer_arch else [args.mlp_dim[0]]
            vocab_size_options = (
                args.vocab_size if arch in EXPLICIT_COT_ARCHES else [args.vocab_size[0]]
            )
            use_latent_feedback_options = (
                args.use_latent_feedback
                if arch in EXPLICIT_COT_ARCHES
                else [args.use_latent_feedback[0]]
            )
            if arch in EXPLICIT_COT_ARCHES:
                if system not in EXPLICIT_COT_SYSTEMS:
                    n_skipped_incompatible += 1
                    continue
                ln_options = [(False, uiln) for uiln in args.use_input_layer_norm]
                num_layers_options = args.num_layers
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
                        for mlp_dim in mlp_dim_options:
                            for vocab_size, use_latent_feedback in itertools.product(
                                vocab_size_options, use_latent_feedback_options
                            ):
                                system_arch_ln_combos.append(
                                    (
                                        system,
                                        arch,
                                        use_layer_norm,
                                        use_input_layer_norm,
                                        num_layers,
                                        num_heads,
                                        mlp_dim,
                                        vocab_size,
                                        use_latent_feedback,
                                    )
                                )
    system_arch_ln_combos = list(dict.fromkeys(system_arch_ln_combos))
    if n_skipped_incompatible:
        print(
            f"Skipping {n_skipped_incompatible} (system, architecture) combo(s) requesting "
            f"one of {EXPLICIT_COT_ARCHES}, which is only implemented for {EXPLICIT_COT_SYSTEMS}."
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
    n_skipped_cnn = 0
    for env in args.envs:
        difficulty_combos = ENV_DIFFICULTY_AXES[env](args)
        for (
            difficulty,
            (
                system,
                arch,
                use_layer_norm,
                use_input_layer_norm,
                num_layers,
                num_heads,
                mlp_dim,
                vocab_size,
                use_latent_feedback,
            ),
            (min_steps, max_steps),
            hidden_dim,
            lr,
            critic_lr,
            ent_coef,
            (delightful, delightful_eta),
            (epochs, num_minibatches, clip_eps, clip_value_loss),
            seed,
        ) in itertools.product(
            difficulty_combos,
            system_arch_ln_combos,
            step_combos,
            args.hidden_dim,
            args.lr,
            args.critic_lr,
            args.ent_coef,
            delightful_combos,
            ppo_combos,
            range(args.seeds),
        ):
            if arch in CNN_ARCHES and not ENV_SUPPORTS_CNN[env]:
                n_skipped_cnn += 1
                continue
            if system in PPO_SYSTEMS:
                delightful, delightful_eta = False, args.delightful_eta[0]
            else:
                epochs, num_minibatches, clip_eps, clip_value_loss = ppo_combos[0]
            jobs.append(
                Job(
                    env=env,
                    difficulty=difficulty,
                    system=system,
                    arch=arch,
                    min_steps=min_steps,
                    max_steps=max_steps,
                    hidden_dim=hidden_dim,
                    lr=lr,
                    critic_lr=critic_lr,
                    ent_coef=ent_coef,
                    delightful=delightful,
                    delightful_eta=delightful_eta,
                    epochs=epochs,
                    num_minibatches=num_minibatches,
                    clip_eps=clip_eps,
                    clip_value_loss=clip_value_loss,
                    use_layer_norm=use_layer_norm,
                    use_input_layer_norm=use_input_layer_norm,
                    num_layers=num_layers,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    vocab_size=vocab_size,
                    use_latent_feedback=use_latent_feedback,
                    seed=seed,
                    total_timesteps=args.total_timesteps,
                    total_num_envs=args.total_num_envs,
                    rollout_length=args.rollout_length,
                    gamma=args.gamma,
                    output_dir=args.output_dir,
                    wandb=args.wandb,
                    wandb_project=args.wandb_project,
                )
            )

    if n_skipped_cnn:
        print(
            f"Skipping {n_skipped_cnn} job(s) requesting a CNN architecture for an env with no "
            f"grid/CNN scenario (knapsack has no spatial structure - see ENV_SUPPORTS_CNN)."
        )

    seen_run_names = set()
    deduped_jobs = []
    for job in jobs:
        if job.run_name in seen_run_names:
            continue
        seen_run_names.add(job.run_name)
        deduped_jobs.append(job)
    n_deduped = len(jobs) - len(deduped_jobs)
    if n_deduped:
        print(
            f"Deduplicated {n_deduped} job(s) with identical run_name (an axis not applicable "
            "to that job's system, e.g. delightful for a PPO system)."
        )
    return deduped_jobs


def run_job(
    job: Job,
    gpu: int,
    python_bin: str,
    log_dir: Path,
    manifest_lock,
    manifest_path: Path,
    mem_fraction: float,
    server: str = None,
) -> dict:
    log_path = log_dir / f"{job.run_name}.log"
    cmd = job.command(python_bin)
    env = {
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "XLA_PYTHON_CLIENT_MEM_FRACTION": str(mem_fraction),
        "XLA_FLAGS": "--xla_gpu_autotune_level=0",
    }
    import os

    full_env = os.environ.copy()
    full_env.update(env)

    if server is not None:
        module_preamble = " && ".join(f"module load {m}" for m in SERVER_MODULES[server])
        run_cmd = ["bash", "-lc", f"{module_preamble} && exec {shlex.join(cmd)}"]
    else:
        run_cmd = cmd

    start = time.time()
    with open(log_path, "w") as log_file:
        log_file.write(
            f"# GPU={gpu} XLA_PYTHON_CLIENT_MEM_FRACTION={mem_fraction:g} server={server}\n"
            f"# CMD={' '.join(cmd)}\n\n"
        )
        log_file.flush()
        proc = subprocess.run(
            run_cmd, cwd=REPO_ROOT, env=full_env, stdout=log_file, stderr=subprocess.STDOUT
        )
    elapsed = time.time() - start

    result_job = asdict(job)
    result_job["difficulty"] = asdict(job.difficulty)
    result = {
        **result_job,
        "run_name": job.run_name,
        "group_tag": job.group_tag,
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
        "--envs",
        default=",".join(JUMANJI_ENVS),
        help=f"Comma-separated subset of {{{','.join(JUMANJI_ENVS)}}} (env=jumanji/<env>).",
    )
    parser.add_argument(
        "--systems",
        default="ff_reinforce,ff_qac_fac,ff_qac_naive",
        help="Comma-separated subset of {ff_reinforce, ff_qac_fac, ff_qac_naive, ff_ppo_fac, "
        "ff_ppo_naive, ff_ppo_cond_naive, ff_ppo_cond_fac, ff_ppo_reinforce, ff_ppo_explicit_fac, "
        "ff_ppo_explicit_naive, ff_ppo_explicit_cond_naive, ff_ppo_explicit_cond_fac, "
        "ff_ppo_explicit_reinforce}. See jumanji_fixed_budget_sweep.py's module docstring for what "
        "each means. The five ff_ppo_explicit_* systems train "
        "stoix/systems/ramdp_vpg/ff_ppo_explicit_cot.py (explicit chain-of-thought tokens) and "
        "default to a flattened-observation architecture regardless of --architectures unless it "
        "requests transformer_explicit_cot/cnn+transformer_explicit_cot explicitly.",
    )
    parser.add_argument(
        "--architectures",
        default="mlp,transformer",
        help=f"Comma-separated subset of {{{','.join(VALID_ARCHITECTURES)}}}. CNN architectures "
        "(including cnn+transformer_explicit_cot) are only valid for env in "
        "{sokoban, slidingtile, maze} (see ENV_SUPPORTS_CNN) - requested for knapsack, they're "
        "skipped, not errored. transformer_explicit_cot/cnn+transformer_explicit_cot "
        f"(TransformerExplicitCoTTorso) are only implemented for system in {EXPLICIT_COT_SYSTEMS} "
        "- other (system, architecture) combos requesting them are skipped too.",
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
        "--lr", default="1e-4,3e-4,1e-3",
        help="Comma-separated system.actor_lr values, swept independently of --critic-lr.",
    )
    parser.add_argument(
        "--critic-lr", default="1e-4,3e-4,1e-3",
        help="Comma-separated system.critic_lr values, swept independently of --lr.",
    )
    parser.add_argument(
        "--ent-coef", default="0.01",
        help="Comma-separated system.ent_coef values (entropy bonus coefficient).",
    )
    parser.add_argument("--delightful", default="false", help="Comma-separated bools - system.delightful (not PPO).")
    parser.add_argument("--delightful-eta", default="1.0", help="Comma-separated system.delightful_eta values.")
    parser.add_argument("--epochs", default="4", help="Comma-separated system.epochs values (PPO only).")
    parser.add_argument("--num-minibatches", default="16", help="Comma-separated system.num_minibatches values (PPO only).")
    parser.add_argument("--clip-eps", default="0.2", help="Comma-separated system.clip_eps values (PPO only).")
    parser.add_argument("--clip-value-loss", default="true", help="Comma-separated bools (PPO only).")
    parser.add_argument("--use-layer-norm", default="false", help="Comma-separated bools (mlp/cnn+mlp only).")
    parser.add_argument("--use-input-layer-norm", default="false", help="Comma-separated bools.")
    parser.add_argument("--num-layers", default="1", help="Comma-separated ints - sub-layers per pondering step.")
    parser.add_argument("--num-heads", default="4", help="Comma-separated ints (transformer archs only).")
    parser.add_argument("--mlp-dim", default="256", help="Comma-separated ints (transformer archs only).")
    parser.add_argument(
        "--vocab-size",
        default="32",
        help="Comma-separated ints - thought-token vocabulary size "
        "(network.actor_network.pre_torso.vocab_size). Only applies to the explicit-CoT "
        "arches (EXPLICIT_COT_ARCHES); ignored (forced to the first value) for every other "
        "architecture, including plain transformer. Default 32 (the network yaml default).",
    )
    parser.add_argument(
        "--use-latent-feedback",
        default="false",
        help="Comma-separated bools (true/false) - latent feedback decoding "
        "(network.actor_network.pre_torso.use_latent_feedback), the Full-Bandwidth "
        "Transformer's gated hidden-state feedback (arXiv:2608.08888) - see "
        "stoix/networks/torso_compute_explicit_cot.py. Only applies to the explicit-CoT "
        "arches (EXPLICIT_COT_ARCHES); ignored (forced to the first value) for every other "
        "architecture. Default false.",
    )

    parser.add_argument(
        "--sokoban-generator",
        default="default",
        help=f"Comma-separated subset of {{{','.join(SOKOBAN_GENERATOR_CHOICES)}}} - env.kwargs.generator "
        "for sokoban jobs (ignored for other envs). 'default' leaves it unset (whatever "
        "sokoban.yaml/the Sokoban class defaults to). 'toy'/'simple' are tiny fixed, "
        "network-free levels - good for a pilot/debug run. 'unfiltered-train'/'medium-train'/"
        "'hard' are increasingly hard Boxoban dataset tiers, downloaded from HuggingFace Hub "
        "on first use (needs network access).",
    )
    parser.add_argument(
        "--slidingtile-grid-size", default="3",
        help="Comma-separated ints - NxN puzzle size (env.kwargs.generator.grid_size), ignored for other envs.",
    )
    parser.add_argument(
        "--slidingtile-num-random-moves", default="10,50,100",
        help="Comma-separated ints - scramble depth from the solved state "
        "(env.kwargs.generator.num_random_moves), ignored for other envs.",
    )
    parser.add_argument(
        "--knapsack-num-items", default="10,20,50",
        help="Comma-separated ints - env.kwargs.generator.num_items, ignored for other envs.",
    )
    parser.add_argument(
        "--knapsack-total-budget", default="2.5",
        help="Comma-separated floats - env.kwargs.generator.total_budget, ignored for other envs.",
    )
    parser.add_argument(
        "--maze-size", default="5,10,15",
        help="Comma-separated ints - square maze side length (env.kwargs.generator.num_rows == "
        "num_cols), ignored for other envs.",
    )

    parser.add_argument(
        "--gamma", type=float, default=0.99,
        help="system.gamma, applied to every job (not swept). All 4 envs here have short "
        "(<=a few hundred step) episodes, unlike MinAtar's gamma=0.9999 default.",
    )
    parser.add_argument("--wandb", type=lambda x: x.strip().lower() in ("1", "true", "yes"), default=False)
    parser.add_argument("--wandb-project", default="jumanji_sweep")
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds per config, seeded 0..seeds-1.")
    parser.add_argument("--total-timesteps", type=float, default=2e7, help="arch.total_timesteps per run.")
    parser.add_argument("--total-num-envs", type=int, default=1024, help="arch.total_num_envs, applied to every job.")
    parser.add_argument("--rollout-length", type=int, default=32, help="system.rollout_length, applied to every job.")
    parser.add_argument("--gpus", default="auto", help="Comma-separated GPU ids, or 'auto' to detect via nvidia-smi.")
    parser.add_argument("--runs-per-gpu", type=int, default=2, help="Concurrent runs per GPU.")
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "results_jumanji_sweep",
        help="Where per-run logger.base_exp_path and logs/ + manifest.jsonl are written.",
    )
    parser.add_argument("--python", default=str(REPO_ROOT / ".venv" / "bin" / "python"), help="Python interpreter.")
    parser.add_argument(
        "--server", default=None, choices=sorted(SERVER_MODULES),
        help="If set, `module load` this server's required environment modules before each job.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N jobs (for a pilot / sanity check).")
    parser.add_argument(
        "--skip-existing", action="store_true", default=True,
        help="Skip jobs already OK in manifest.jsonl (default: on; --no-skip-existing to force rerun).",
    )
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned jobs and exit without running anything.")
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation prompt before launching.")
    args = parser.parse_args()

    args.envs = args.envs.split(",")
    args.systems = args.systems.split(",")
    args.architectures = args.architectures.split(",")
    args.min_steps = [int(x) for x in args.min_steps.split(",")]
    args.max_steps = [int(x) for x in args.max_steps.split(",")]
    args.hidden_dim = [int(x) for x in args.hidden_dim.split(",")]
    args.lr = [float(x) for x in args.lr.split(",")]
    args.critic_lr = [float(x) for x in args.critic_lr.split(",")]
    args.ent_coef = [float(x) for x in args.ent_coef.split(",")]
    args.delightful = [x.strip().lower() in ("1", "true", "yes") for x in args.delightful.split(",")]
    args.delightful_eta = [float(x) for x in args.delightful_eta.split(",")]
    args.epochs = [int(x) for x in args.epochs.split(",")]
    args.num_minibatches = [int(x) for x in args.num_minibatches.split(",")]
    args.clip_eps = [float(x) for x in args.clip_eps.split(",")]
    args.clip_value_loss = [x.strip().lower() in ("1", "true", "yes") for x in args.clip_value_loss.split(",")]
    args.use_layer_norm = [x.strip().lower() in ("1", "true", "yes") for x in args.use_layer_norm.split(",")]
    args.use_input_layer_norm = [x.strip().lower() in ("1", "true", "yes") for x in args.use_input_layer_norm.split(",")]
    args.num_layers = [int(x) for x in args.num_layers.split(",")]
    args.num_heads = [int(x) for x in args.num_heads.split(",")]
    args.mlp_dim = [int(x) for x in args.mlp_dim.split(",")]
    args.vocab_size = [int(x) for x in args.vocab_size.split(",")]
    args.use_latent_feedback = [
        x.strip().lower() in ("1", "true", "yes") for x in args.use_latent_feedback.split(",")
    ]
    args.sokoban_generator = args.sokoban_generator.split(",")
    args.slidingtile_grid_size = [int(x) for x in args.slidingtile_grid_size.split(",")]
    args.slidingtile_num_random_moves = [int(x) for x in args.slidingtile_num_random_moves.split(",")]
    args.knapsack_num_items = [int(x) for x in args.knapsack_num_items.split(",")]
    args.knapsack_total_budget = [float(x) for x in args.knapsack_total_budget.split(",")]
    args.maze_size = [int(x) for x in args.maze_size.split(",")]

    for e in args.envs:
        assert e in JUMANJI_ENVS, f"unknown env {e!r}, expected one of {list(JUMANJI_ENVS)}"
    for s in args.systems:
        assert s in SYSTEM_TO_SCRIPT, f"unknown system {s!r}, expected one of {list(SYSTEM_TO_SCRIPT)}"
    for a in args.architectures:
        assert a in VALID_ARCHITECTURES, f"unknown architecture {a!r}, expected one of {VALID_ARCHITECTURES}"
    for s in args.min_steps:
        assert s >= 1, f"min_steps must be >= 1, got {s}"
    for s in args.max_steps:
        assert s >= 1, f"max_steps must be >= 1, got {s}"
    for c in args.sokoban_generator:
        assert c in SOKOBAN_GENERATOR_CHOICES, f"unknown sokoban generator {c!r}, expected one of {SOKOBAN_GENERATOR_CHOICES}"

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
        completed_run_names = set()
        manifest_path = args.output_dir / "manifest.jsonl"
        if manifest_path.exists():
            with open(manifest_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    if entry.get("returncode") == 0:
                        completed_run_names.add(entry["run_name"])
        remaining = [j for j in jobs if j.run_name not in completed_run_names]
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
    print(f"  envs={args.envs}")
    print(f"  systems={args.systems} architectures={args.architectures}")
    print(
        f"  min_steps={args.min_steps} max_steps={args.max_steps} hidden_dim={args.hidden_dim} "
        f"seeds=0..{args.seeds - 1}"
    )
    print(f"  lr={args.lr} critic_lr={args.critic_lr} ent_coef={args.ent_coef}")
    print(f"  sokoban_generator={args.sokoban_generator}")
    print(f"  slidingtile_grid_size={args.slidingtile_grid_size} slidingtile_num_random_moves={args.slidingtile_num_random_moves}")
    print(f"  knapsack_num_items={args.knapsack_num_items} knapsack_total_budget={args.knapsack_total_budget}")
    print(f"  maze_size={args.maze_size}")
    print(f"  gamma={args.gamma}")
    print(
        f"  total_timesteps={args.total_timesteps:g} total_num_envs={args.total_num_envs} "
        f"rollout_length={args.rollout_length} output_dir={args.output_dir}"
    )
    if args.server is not None:
        print(f"  server={args.server} -> module load {SERVER_MODULES[args.server]}")
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
            return run_job(job, gpu, args.python, log_dir, manifest_lock, manifest_path, mem_fraction, args.server)
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
