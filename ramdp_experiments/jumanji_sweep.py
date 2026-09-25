#!/usr/bin/env python
"""Adaptive-computation-budget sweep for RAMDP systems on Jumanji environments
(env=jumanji/<env>): sokoban, slidingtile (SlidingTilePuzzle), knapsack
(Knapsack), maze (Maze), pacman (PacMan) - see ramdp_experiments/experiments.md.

Usage:
  python ramdp_experiments/jumanji_sweep.py --dry-run                # preview the grid
  python ramdp_experiments/jumanji_sweep.py --limit 6 --dry-run       # preview a slice
  python ramdp_experiments/jumanji_sweep.py                          # run the full sweep (all 5 envs)
  python ramdp_experiments/jumanji_sweep.py --envs sokoban,maze       # only these envs
  python ramdp_experiments/jumanji_sweep.py --systems ff_ppo_reinforce --architectures mlp \\
      --envs sokoban --sokoban-generator toy --min-steps 1 --max-steps 8 --hidden-dim 16 \\
      --lr 3e-4 --seeds 1 --total-timesteps 2e5 --limit 2  # small pilot / debug run
  python ramdp_experiments/jumanji_sweep.py --min-steps 1,4 --max-steps 4,8,16 \\
      # sweeps min_steps x max_steps (min_steps > max_steps combos skipped)
  python ramdp_experiments/jumanji_sweep.py --min-steps 8 --max-steps 8  # min_steps == max_steps, the fixed-budget special case
  python ramdp_experiments/jumanji_sweep.py --seeds 5 --base-seed 5    # seeds 5..9 (extend an earlier seeds 0..4 sweep)
  python ramdp_experiments/jumanji_sweep.py --envs slidingtile \\
      --slidingtile-grid-size 3,4 --slidingtile-num-random-moves 5,20,100       # scramble-depth sweep
  python ramdp_experiments/jumanji_sweep.py --envs knapsack \\
      --knapsack-num-items 5,10,20,50 --knapsack-max-budget 10,50              # problem-size sweep
  python ramdp_experiments/jumanji_sweep.py --envs maze --maze-size 5,10,15  # maze-size sweep
  python ramdp_experiments/jumanji_sweep.py --envs pacman  # pacman has no difficulty knob (fixed maze)
  python ramdp_experiments/jumanji_sweep.py --architectures cnn+mlp,cnn+transformer \\
      --envs sokoban,slidingtile,maze,pacman  # CNN-input sweep (via jumanji/*_grid)
  python ramdp_experiments/jumanji_sweep.py --systems ff_ppo_reinforce \\
      --epochs 4 --num-minibatches 8,16 --clip-eps 0.1,0.2                 # PPO sweep
  python ramdp_experiments/jumanji_sweep.py --systems ff_ppo_reinforce \\
      --standardize-advantages true,false                 # sweep PPO advantage standardization
  python ramdp_experiments/jumanji_sweep.py --systems ff_ppo_reinforce \\
      --recompute-advantages true,false           # sweep per-epoch advantage/target recompute
  python ramdp_experiments/jumanji_sweep.py --systems ff_ppo_reinforce \\
      --gae-lambda 0.9,0.95,1.0                    # sweep GAE(lambda)
  python ramdp_experiments/jumanji_sweep.py --envs sokoban,slidingtile,maze \\
      --systems ff_ppo_explicit_reinforce --architectures cnn+transformer_explicit_cot_merged  # CNN-input explicit-CoT sweep
  python ramdp_experiments/jumanji_sweep.py \\
      --systems ff_ppo_explicit_reinforce   # explicit-CoT PPO sweep (flattened obs)
"""

from __future__ import annotations

import argparse
import hashlib
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
    "ff_ppo_reinforce": "stoix/systems/ramdp_vpg/ff_ppo.py",
    # Explicit-CoT PPO (TransformerMergedActionCoTTorso instead of a latent-CoT torso): the halting
    # decision and the environment action are the same draw from one vocabulary - see
    # stoix/networks/torso_compute_explicit_cot_merged.py.
    "ff_ppo_explicit_reinforce": "stoix/systems/ramdp_vpg/ff_ppo_explicit_cot.py",
}
PPO_SYSTEMS = (
    "ff_ppo_reinforce",
    "ff_ppo_explicit_reinforce",
)
# Explicit-CoT systems whose architecture defaults to
# transformer_explicit_cot_merged (flattened observation) rather than being
# picked via --architectures, unless --architectures requests one of
# EXPLICIT_COT_ARCHES explicitly (e.g. cnn+transformer_explicit_cot_merged) -
# see build_grid.
EXPLICIT_COT_PPO_SYSTEMS = ("ff_ppo_explicit_reinforce",)
# PPO_SYSTEMS minus EXPLICIT_COT_PPO_SYSTEMS: the systems trained by ff_ppo.py (implicit/latent CoT)
# rather than ff_ppo_explicit_cot.py.
LATENT_KL_PPO_SYSTEMS = tuple(s for s in PPO_SYSTEMS if s not in EXPLICIT_COT_PPO_SYSTEMS)
ARCH_TO_NETWORK = {
    "ff_ppo_reinforce": {
        "mlp": "mlp_compute",
        "transformer": "transformer_compute",
        "gru": "gru_compute",
        "iru": "iru_compute",
        "cnn+mlp": "cnn_mlp_compute",
        "cnn+transformer": "cnn_transformer_compute",
        "cnn+gru": "cnn_gru_compute",
        "cnn+iru": "cnn_iru_compute",
    },
}
NO_LAYER_NORM_ARCHES = ("transformer", "cnn+transformer", "gru", "cnn+gru", "iru", "cnn+iru")
TRANSFORMER_ARCHES = ("transformer", "cnn+transformer")
# Architectures whose pre_torso is IRUStep-based (IRUAdaptiveComputationTimeTorso,
# see stoix/networks/torso_compute.py).
IRU_ARCHES = ("iru", "cnn+iru")
# Architectures whose pre_torso has a stop_gradient_halting_input param - IRUStep-based torsos
# (IRU_ARCHES) and TransformerChainOfThoughtTorso (TRANSFORMER_ARCHES; *not*
# TransformerMergedActionCoTTorso/EXPLICIT_COT_ARCHES, which has no such param) - see
# stoix/networks/torso_compute.py's IRUStep and stoix/networks/torso_compute_transformer.py's
# _CoTStep docstrings.
STOP_GRADIENT_HALTING_ARCHES = IRU_ARCHES + TRANSFORMER_ARCHES
# Architectures whose pre_torso has a halting_temperature param - every
# ACTStep/RecurrentACTStep/IRUStep-based torso (mlp/gru/iru and their cnn+ variants), see
# stoix/networks/torso_compute.py, plus TransformerChainOfThoughtTorso (TRANSFORMER_ARCHES), see
# stoix/networks/torso_compute_transformer.py.
HALTING_TEMPERATURE_ARCHES = (
    "mlp",
    "gru",
    "iru",
    "cnn+mlp",
    "cnn+gru",
    "cnn+iru",
) + TRANSFORMER_ARCHES
# Architectures whose pre_torso has a halting_hidden_dims param - only
# TransformerChainOfThoughtTorso (TRANSFORMER_ARCHES), via its HaltingHead - see
# stoix/networks/torso_compute_transformer.py.
HALTING_HIDDEN_DIMS_ARCHES = TRANSFORMER_ARCHES

JUMANJI_ENVS = ("sokoban", "slidingtile", "knapsack", "maze", "pacman")

# TransformerMergedActionCoTTorso (see stoix/networks/torso_compute_explicit_cot_merged.py) doesn't
# fit ARCH_TO_NETWORK/SYSTEM_TO_SCRIPT's (system, arch) -> network lookup: it's only trained by
# ff_ppo_explicit_cot.py, not the plain ff_ppo.py, so it's handled separately.
EXPLICIT_COT_ARCH = "transformer_explicit_cot_merged"
CNN_EXPLICIT_COT_ARCH = "cnn+transformer_explicit_cot_merged"
EXPLICIT_COT_ARCHES = (EXPLICIT_COT_ARCH, CNN_EXPLICIT_COT_ARCH)
EXPLICIT_COT_SCRIPT_BY_SYSTEM = {
    system: "stoix/systems/ramdp_vpg/ff_ppo_explicit_cot.py" for system in EXPLICIT_COT_PPO_SYSTEMS
}
# Network name by arch (flattened observation vs CNN input) - nested the same
# way ARCH_TO_NETWORK is. Plain V-only critic.
EXPLICIT_COT_NETWORK_BY_SYSTEM = {
    "ff_ppo_explicit_reinforce": {
        EXPLICIT_COT_ARCH: "transformer_explicit_cot_merged",
        CNN_EXPLICIT_COT_ARCH: "cnn_transformer_explicit_cot_merged",
    },
}
EXPLICIT_COT_SYSTEMS = tuple(EXPLICIT_COT_SCRIPT_BY_SYSTEM)
CNN_ARCHES = ("cnn+mlp", "cnn+transformer", "cnn+gru", "cnn+iru", CNN_EXPLICIT_COT_ARCH)
VALID_ARCHITECTURES = ("mlp", "transformer", "gru", "iru", EXPLICIT_COT_ARCH) + CNN_ARCHES
# Short forms for group_tag/run_name (wandb group names get long fast):
# "transformer" -> implicit-CoT transformer ("TF-iCoT"), "transformer_explicit_cot_merged"
# -> explicit-CoT transformer ("TF-eCoT"); mlp/gru/iru are already short.
ARCH_SHORT_TAG = {
    "transformer": "TF-iCoT",
    "cnn+transformer": "cnn+TF-iCoT",
    EXPLICIT_COT_ARCH: "TF-eCoT",
    CNN_EXPLICIT_COT_ARCH: "cnn+TF-eCoT",
}

# W&B's "group" field (and, in practice, run IDs) are capped at 128
# characters - Job.group_tag_parts enforces this (with room held back for
# run_name's "-seed_N" suffix, see _SEED_SUFFIX_RESERVE), rather than
# leaving it to come out under the limit by luck: a handful of extra flags
# (weight decay, latent_kl_coef, ...) stacked on top of a
# transformer/explicit-CoT run's already-long net/ppo segments can push the
# un-capped tag past 128.
MAX_GROUP_TAG_LEN = 128
_SEED_SUFFIX_RESERVE = len("-seed_9999")  # generous - --seeds never gets near 4 digits


def _cap_tag_length(parts: List[str], max_len: int) -> List[str]:
    """Cap the "-"-joined (equivalently "_"-joined, same length) length of
    `parts` at `max_len` chars, keeping as many whole leading parts (env,
    system/arch, budget, network/PPO hparams - the parts most useful for
    identifying a run at a glance) as fit, then replacing everything after
    that with a single short hash of the *full*, untruncated tag - so two
    configs that would otherwise collide after truncation (e.g. differing
    only in a dropped trailing flag) still get distinct group tags/run names
    (needed for the run_name dedup in build_grid(), and to avoid two jobs
    silently writing to the same output directory)."""
    full = "-".join(parts)
    if len(full) <= max_len:
        return parts
    digest = hashlib.sha1(full.encode()).hexdigest()[:8]
    kept: List[str] = []
    length = 0
    for part in parts:
        added = len(part) + (1 if kept else 0)  # +1 for the joining separator
        if length + added + 1 + len(digest) > max_len:  # +1 for the separator before digest
            break
        kept.append(part)
        length += added
    kept.append(digest)
    return kept


# env -> (non-CNN scenario, CNN/grid scenario or None if unsupported).
ENV_SCENARIOS = {
    "sokoban": ("jumanji/sokoban", "jumanji/sokoban_grid"),
    "slidingtile": ("jumanji/slidingtile", "jumanji/slidingtile_grid"),
    "knapsack": ("jumanji/knapsack", None),
    "maze": ("jumanji/maze", "jumanji/maze_grid"),
    "pacman": ("jumanji/pacman", "jumanji/pacman_grid"),
}
ENV_SUPPORTS_CNN = {env: grid is not None for env, (_, grid) in ENV_SCENARIOS.items()}
# knapsack/maze/pacman already set env.wrapper in their yaml
# (ConcatObservationWrapper, since their observation is several
# equally-necessary fields with no single attribute to extract - see those
# yamls) - unlike sokoban/slidingtile (whose native observation is already
# one array), so non-CNN jobs for those three must NOT also append the
# +env.wrapper._target_=stoa.FlattenObservationWrapper override
# lightsout/minatar-style jobs use, which would conflict.
ENV_HAS_BUILTIN_WRAPPER = {
    "sokoban": False,
    "slidingtile": False,
    "knapsack": True,
    "maze": True,
    "pacman": True,
}

# Per-env CNN architecture - env-specific rather than one shared CNN, since the four CNN-capable
# envs (see ENV_SUPPORTS_CNN) differ substantially in per-cell channel semantics: - sokoban: fixed
# 10x10 grid, 2 channels bundling walls/boxes/targets/ player (see
# stoix/configs/env/jumanji/sokoban_grid.yaml) - gets a deep conv stack and wide MLPs (matches
# cnn_mlp_compute.yaml's own defaults throughout).
ENV_CNN_ARCH = {
    # "sokoban": {
    #     "channel_sizes": (128, 128, 128),
    #     "kernel_sizes": (3, 3, 3),
    #     "strides": (2, 1, 1),
    #     "hidden_sizes": (128,),
    #     "critic_hidden_sizes": (128,),
    #     "critic_layer_sizes": (128, 128),
    # },
    # "sokoban": { # sep11
    #     "channel_sizes": (128, 128),
    #     "kernel_sizes": (3, 3),
    #     "strides": (2, 1),
    #     "hidden_sizes": (128,),
    #     "critic_hidden_sizes": (128,),
    #     "critic_layer_sizes": (128, 128),
    # },
    # "sokoban": { # sep12
    #     "channel_sizes": (64, 64),
    #     "kernel_sizes": (3, 3),
    #     "strides": (2, 1),
    #     "hidden_sizes": (128,),
    #     "critic_hidden_sizes": (128,),
    #     "critic_layer_sizes": (128, 128),
    # },
    "sokoban": { # sokoban-final
        "channel_sizes": (64,),
        "kernel_sizes": (3,),
        "strides": (2,),
        "hidden_sizes": (128,),
        "critic_hidden_sizes": (128,),
        "critic_layer_sizes": (128, 128),
    },
    "slidingtile": {
        "channel_sizes": (8, 8),
        "kernel_sizes": (2, 1),
        "strides": (1,),
        "hidden_sizes": (64,),
        "critic_hidden_sizes": (64,),
        "critic_layer_sizes": (64, 64),
    },
    "maze": {
        "channel_sizes": (8,),
        "kernel_sizes": (3,),
        "strides": (1,),
        "hidden_sizes": (64,),
        "critic_hidden_sizes": (128,),
        "critic_layer_sizes": (128, 128),
    },
    "pacman": {
        "channel_sizes": (64,),
        "kernel_sizes": (3,),
        "strides": (2,),
        "hidden_sizes": (128,),
        "critic_hidden_sizes": (128,),
        "critic_layer_sizes": (128, 128),
    },
}

SOKOBAN_GENERATOR_CHOICES = (
    "default",
    "toy",
    "simple",
    "unfiltered-train",
    "medium-train",
    "hard",
)
# --sokoban-eval-generator: 'same' keeps eval on the train generator; otherwise
# any non-default generator above, or a held-out Boxoban split.
SOKOBAN_EVAL_GENERATOR_CHOICES = (
    "same",
    "toy",
    "simple",
    "unfiltered-train",
    "unfiltered-valid",
    "unfiltered-test",
    "medium-train",
    "medium-valid",
    "hard",
)
# Shortened dataset_name -> group_tag suffix for the longer Boxoban tiers (the
# override still passes the full HuggingFace dataset_name, only the tag shrinks).
SOKOBAN_TAG_SHORT = {"unfiltered-train": "unfilt-train", "medium-train": "med-train"}

SERVER_MODULES = {
    "slurm": ["StdEnv/2023", "cuda/12.2"],
}


@dataclass(frozen=True)
class EnvDifficulty:
    """One difficulty-knob setting for a given env: `tag` is the short
    group_tag/run_name suffix, `overrides` are the literal Hydra CLI override
    strings to append (see ENV_DIFFICULTY_AXES)."""

    tag: str
    overrides: Tuple[str, ...] = ()


def _sokoban_generator_overrides(choice: str, prefix: str) -> Tuple[str, ...]:
    """Hydra overrides selecting sokoban generator `choice` under `prefix`:
    `env.kwargs` (train + eval) or `env.eval_kwargs` (eval env only, see
    stoix/utils/make_env.py's make_jumanji_env)."""
    if choice == "default":
        return ()
    if choice == "toy":
        return (
            f"+{prefix}.generator._target_="
            "jumanji.environments.routing.sokoban.generator.ToyGenerator",
        )
    if choice == "simple":
        return (
            f"+{prefix}.generator._target_="
            "jumanji.environments.routing.sokoban.generator.SimpleSolveGenerator",
        )
    # Boxoban dataset splits (train/valid/test/hard), downloaded from
    # HuggingFace Hub on first use.
    return (
        f"+{prefix}.generator._target_="
        "jumanji.environments.routing.sokoban.generator.HuggingFaceDeepMindGenerator",
        f"+{prefix}.generator.dataset_name={choice}",
        f"+{prefix}.generator.proportion_of_files=1.0",
    )


def _sokoban_difficulty_axis(args: argparse.Namespace) -> List[EnvDifficulty]:
    combos = []
    eval_choice = args.sokoban_eval_generator
    for choice in args.sokoban_generator:
        tag = "default" if choice == "default" else SOKOBAN_TAG_SHORT.get(choice, choice)
        overrides = _sokoban_generator_overrides(choice, "env.kwargs")
        if eval_choice != "same":
            tag += f"-eval-{SOKOBAN_TAG_SHORT.get(eval_choice, eval_choice)}"
            overrides += _sokoban_generator_overrides(eval_choice, "env.eval_kwargs")
        combos.append(EnvDifficulty(tag=tag, overrides=overrides))
    return combos


def _slidingtile_difficulty_axis(args: argparse.Namespace) -> List[EnvDifficulty]:
    # --slidingtile-time-limit: None (default) leaves env.kwargs.time_limit at the yaml value (no
    # tag, so run names match pre-flag sweeps); otherwise each value is swept and overrides the
    # train (and, unless --slidingtile-eval-time-limit is set, eval) env's time_limit.
    eval_time_limit = args.slidingtile_eval_time_limit
    eval_nrm = args.slidingtile_eval_num_random_moves
    eval_overrides = ()
    eval_tag = ""
    if eval_time_limit != "same":
        eval_overrides = (f"+env.eval_kwargs.time_limit={eval_time_limit}",)
        eval_tag = f"-evaltl{eval_time_limit}"
    if eval_nrm != "same":
        eval_tag += f"-evalnrm{eval_nrm}"

    def _eval_generator_overrides(gs: int) -> Tuple[str, ...]:
        if eval_nrm == "same":
            return ()
        return (
            "+env.eval_kwargs.generator._target_="
            "jumanji.environments.logic.sliding_tile_puzzle.generator.RandomWalkGenerator",
            f"+env.eval_kwargs.generator.grid_size={gs}",
            f"+env.eval_kwargs.generator.num_random_moves={eval_nrm}",
        )

    return [
        EnvDifficulty(
            tag=f"gs{gs}-nrm{nrm}{'' if tl is None else f'-tl{tl}'}{eval_tag}",
            overrides=(
                f"env.kwargs.generator.grid_size={gs}",
                f"env.kwargs.generator.num_random_moves={nrm}",
            )
            + (() if tl is None else (f"env.kwargs.time_limit={tl}",))
            + eval_overrides
            + _eval_generator_overrides(gs),
        )
        for gs in args.slidingtile_grid_size
        for nrm in args.slidingtile_num_random_moves
        for tl in args.slidingtile_time_limit
    ]


def _knapsack_difficulty_axis(args: argparse.Namespace) -> List[EnvDifficulty]:
    # Item weights are integers in {1, ..., max_weight} (see
    # stoix.envs.knapsack.generator.IntegerRandomGenerator), so the maximum possible total weight is
    # num_items * max_weight - if max_budget is at least that, the top of the per-episode budget
    # draw (Uniform{1, ..., max_budget}) can always fit every item, making the hardest episodes in
    # that combo trivial.
    combos = [
        EnvDifficulty(
            tag=f"ni{ni}-mw{mw}-mv{mv}-mb{mb}",
            overrides=(
                f"env.kwargs.generator.num_items={ni}",
                f"env.kwargs.generator.max_weight={mw}",
                f"env.kwargs.generator.max_value={mv}",
                f"env.kwargs.generator.max_budget={mb}",
            ),
        )
        for ni in args.knapsack_num_items
        for mw in args.knapsack_max_weight
        for mv in args.knapsack_max_value
        for mb in args.knapsack_max_budget
        if mb < ni * mw
    ]
    n_total = (
        len(args.knapsack_num_items)
        * len(args.knapsack_max_weight)
        * len(args.knapsack_max_value)
        * len(args.knapsack_max_budget)
    )
    n_skipped = n_total - len(combos)
    if n_skipped:
        print(
            f"Skipping {n_skipped} knapsack (num_items, max_weight, max_value, max_budget) "
            "combo(s) where max_budget >= num_items * max_weight (the budget could always fit "
            "every item)."
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


def _pacman_difficulty_axis(args: argparse.Namespace) -> List[EnvDifficulty]:
    # PacMan ships a single fixed maze (AsciiGenerator(DEFAULT_MAZE), see
    # jumanji.environments.routing.pac_man.env.PacMan.__init__) with no
    # generator params to sweep, unlike sokoban/slidingtile/knapsack/maze -
    # one difficulty point, no overrides.
    return [EnvDifficulty(tag="default")]


ENV_DIFFICULTY_AXES = {
    "sokoban": _sokoban_difficulty_axis,
    "slidingtile": _slidingtile_difficulty_axis,
    "knapsack": _knapsack_difficulty_axis,
    "maze": _maze_difficulty_axis,
    "pacman": _pacman_difficulty_axis,
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
    actor_weight_decay: float
    critic_weight_decay: float
    ent_coef: float
    max_grad_norm: float
    epochs: int
    num_minibatches: int
    clip_eps: float
    clip_value_loss: bool
    gae_lambda: float
    latent_kl_coef: float
    clip_halting_head: bool
    halting_lr: float
    halting_weight_decay: float
    halting_ent_coef: float
    halting_temperature: float
    halting_hidden_dims: Tuple[int, ...]
    use_dpo_loss: bool
    dpo_alpha: float
    dpo_beta: float
    use_expectile_value_loss: bool
    expectile: float
    standardize_advantages: bool
    recompute_advantages: bool
    critic_before_actor: bool
    use_layer_norm: bool
    use_input_layer_norm: bool
    stop_gradient_halting_input: bool
    num_layers: int
    num_heads: int
    mlp_dim: int
    qkv_dim: int
    vocab_size: int
    use_latent_feedback: bool
    use_sandwich_norm: bool
    use_rmsnorm: bool
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
        """The group tag broken into semantic chunks - env/difficulty, algo, step budget, gamma,
        network hparams, PPO hparams, misc flags - instead of one flat dash-joined string."""
        system_short = (
            self.system.removeprefix("ff_").replace("explicit", "expl").replace("reinforce", "reinf")
        )
        arch_short = ARCH_SHORT_TAG.get(self.arch, self.arch)
        parts = [
            f"{self.env}-{self.difficulty.tag}",
            f"{system_short}-{arch_short}",
            f"mn{self.min_steps}-mx{self.max_steps}",
            f"g{self.gamma:g}",
        ]

        net = (
            f"hd{self.hidden_dim}-lr{self.lr:g}-clr{self.critic_lr:g}-ec{self.ent_coef:g}"
            f"-mgn{self.max_grad_norm:g}-nl{self.num_layers}"
        )
        if self.arch in TRANSFORMER_ARCHES or self.arch in EXPLICIT_COT_ARCHES:
            net += f"-nh{self.num_heads}-md{self.mlp_dim}"
            if self.qkv_dim:
                net += f"-qkv{self.qkv_dim}"
        # Only shown for the explicit-CoT arches - vocab_size doesn't exist on
        # any other architecture, see EXPLICIT_COT_ARCHES/build_grid.
        if self.arch in EXPLICIT_COT_ARCHES:
            net += f"-vs{self.vocab_size}"
        parts.append(net)

        ppo = f"ep{self.epochs}-mb{self.num_minibatches}-clip{self.clip_eps:g}"
        if self.gae_lambda != 1.0:
            ppo += f"-gae{self.gae_lambda:g}"
        if not self.clip_value_loss:
            ppo += "-l2c"
        if self.standardize_advantages:
            ppo += "-stdadv"
        if self.recompute_advantages:
            ppo += "-radv"
        if self.critic_before_actor:
            ppo += "-cba"
        if self.use_dpo_loss:
            ppo += f"-dpoa{self.dpo_alpha:g}b{self.dpo_beta:g}"
        if self.use_expectile_value_loss:
            ppo += f"-iql{self.expectile:g}"
        parts.append(ppo)

        extra = []
        if self.latent_kl_coef:
            extra.append(f"lkl{self.latent_kl_coef:g}")
        if not self.clip_halting_head:
            extra.append("nch")
        if self.halting_lr != self.lr:
            extra.append(f"hlr{self.halting_lr:g}")
        if self.halting_weight_decay != self.actor_weight_decay:
            extra.append(f"hwd{self.halting_weight_decay:g}")
        if self.halting_ent_coef:
            extra.append(f"hec{self.halting_ent_coef:g}")
        if self.halting_temperature != 1.0:
            extra.append(f"ht{self.halting_temperature:g}")
        if self.halting_hidden_dims:
            extra.append(f"hh{'x'.join(str(d) for d in self.halting_hidden_dims)}")
        if self.actor_weight_decay:
            extra.append(f"wd{self.actor_weight_decay:g}")
        if self.critic_weight_decay:
            extra.append(f"cwd{self.critic_weight_decay:g}")
        if self.use_layer_norm:
            extra.append("ln")
        if self.use_input_layer_norm:
            extra.append("iln")
        if self.stop_gradient_halting_input:
            extra.append("sgh")
        if self.use_latent_feedback:
            extra.append("lf")
        if self.use_sandwich_norm:
            extra.append("sn")
        if self.use_rmsnorm:
            extra.append("rms")
        if extra:
            parts.append("-".join(extra))

        return _cap_tag_length(parts, MAX_GROUP_TAG_LEN - _SEED_SUFFIX_RESERVE)

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
            "arch.num_eval_episodes=10",
            f"network.actor_network.pre_torso.hidden_dim={self.hidden_dim}",
            f"++network.actor_network.pre_torso.num_layers={self.num_layers}",
            # Independently swept - see the compute torsos' min_steps mechanism
            # (stoix/networks/torso_compute*.py): min_steps forbids halting before that many steps;
            # max_steps forces a halt at that many.
            f"network.actor_network.pre_torso.max_steps={self.max_steps}",
            f"network.actor_network.pre_torso.min_steps={self.min_steps}",
            f"system.actor_lr={self.lr:g}",
            f"system.critic_lr={self.critic_lr:g}",
            f"system.actor_weight_decay={self.actor_weight_decay:g}",
            f"system.critic_weight_decay={self.critic_weight_decay:g}",
            f"system.ent_coef={self.ent_coef:g}",
            f"system.max_grad_norm={self.max_grad_norm:g}",
            f"system.rollout_length={self.rollout_length}",
            f"logger.base_exp_path={self.output_dir / self.run_name}",
        ]
        cmd.extend(self.difficulty.overrides)

        cmd.append(f"system.epochs={self.epochs}")
        cmd.append(f"system.num_minibatches={self.num_minibatches}")
        cmd.append(f"system.clip_eps={self.clip_eps:g}")
        cmd.append(f"system.clip_value_loss={self.clip_value_loss}")
        cmd.append(f"system.gae_lambda={self.gae_lambda:g}")
        cmd.append(f"system.standardize_advantages={self.standardize_advantages}")
        cmd.append(f"system.recompute_advantages={self.recompute_advantages}")
        cmd.append(f"system.critic_before_actor={self.critic_before_actor}")
        if self.system in LATENT_KL_PPO_SYSTEMS:
            # Latent trust-region penalty - ff_ppo.py (implicit CoT)
            # only, see LATENT_KL_PPO_SYSTEMS.
            cmd.append(f"system.latent_kl_coef={self.latent_kl_coef:g}")
        if self.system in LATENT_KL_PPO_SYSTEMS and self.arch in TRANSFORMER_ARCHES:
            # Whether the halting head's own params are exempt from actor
            # gradient clipping - only meaningful for the transformer
            # implicit-CoT torso's separately-named HaltingHead submodule
            # (see ff_ppo.py's _label_actor_params_by_halting_head), so
            # gated the same as latent_kl_coef above plus TRANSFORMER_ARCHES.
            cmd.append(f"system.clip_halting_head={self.clip_halting_head}")
            # Per-parameter-group learning rate/weight decay for the
            # halting head's own params, overriding actor_lr/
            # actor_weight_decay for just that submodule - same
            # applicability as clip_halting_head above (only the
            # transformer implicit-CoT torso's separately-named
            # HaltingHead submodule, see ff_ppo.py's
            # _label_actor_params_by_halting_head).
            cmd.append(f"system.halting_lr={self.halting_lr:g}")
            cmd.append(f"system.halting_weight_decay={self.halting_weight_decay:g}")
        # Halting-decision entropy bonus - ff_ppo.py's own systems only
        # (LATENT_KL_PPO_SYSTEMS): ff_ppo_explicit_cot.py has no separate
        # halting decision, its single per-step categorical's entropy is
        # already covered by ent_coef - see ff_ppo.py's/
        # ff_ppo_explicit_cot.py's module docstrings.
        if self.system in LATENT_KL_PPO_SYSTEMS:
            cmd.append(f"system.halting_ent_coef={self.halting_ent_coef:g}")
        # Discovered Policy Optimisation actor surrogate - both ff_ppo.py's
        # own systems and ff_ppo_explicit_* (same applicability as
        # halting_ent_coef above) - see stoix/utils/loss.py's
        # dpo_loss/dpo_surrogate and ff_ppo.py's/ff_ppo_explicit_cot.py's
        # module docstrings.
        cmd.append(f"system.use_dpo_loss={self.use_dpo_loss}")
        cmd.append(f"system.dpo_alpha={self.dpo_alpha:g}")
        cmd.append(f"system.dpo_beta={self.dpo_beta:g}")
        # Expectile regression for V's loss - both
        # ff_ppo.py's own systems and ff_ppo_explicit_* (same
        # applicability as halting_ent_coef/use_dpo_loss above) - see
        # stoix/utils/loss.py's expectile_loss and ff_ppo.py's/
        # ff_ppo_explicit_cot.py's module docstrings.
        cmd.append(f"system.use_expectile_value_loss={self.use_expectile_value_loss}")
        cmd.append(f"system.expectile={self.expectile:g}")
        if self.arch in TRANSFORMER_ARCHES or self.arch in EXPLICIT_COT_ARCHES:
            cmd.append(f"++network.actor_network.pre_torso.num_heads={self.num_heads}")
            cmd.append(f"++network.actor_network.pre_torso.mlp_dim={self.mlp_dim}")
            if self.qkv_dim:
                # Decouples the Q/K/V projection width from hidden_dim - see
                # stoix/networks/torso_compute_transformer.py's TransformerBlock docstring.
                cmd.append(f"++network.actor_network.pre_torso.qkv_dim={self.qkv_dim}")
            # Sandwich LayerNorm placement / RMSNorm instead of LayerNorm - every
            # TransformerBlock-based torso only, same applicability as num_heads/
            # mlp_dim above - see stoix/networks/torso_compute_transformer.py's
            # TransformerBlock/_norm_cls.
            cmd.append(
                f"++network.actor_network.pre_torso.use_sandwich_norm={self.use_sandwich_norm}"
            )
            cmd.append(f"++network.actor_network.pre_torso.use_rmsnorm={self.use_rmsnorm}")
        if self.arch in EXPLICIT_COT_ARCHES:
            # Thought-token vocabulary size - TransformerMergedActionCoTTorso only,
            # no other architecture has this param.
            cmd.append(f"++network.actor_network.pre_torso.vocab_size={self.vocab_size}")
            # Latent feedback decoding (Full-Bandwidth Transformer, arXiv:2608.08888) -
            # TransformerMergedActionCoTTorso only, see stoix/networks/torso_compute_explicit_cot_merged.py.
            cmd.append(
                f"++network.actor_network.pre_torso.use_latent_feedback={self.use_latent_feedback}"
            )
        if self.wandb:
            cmd.append("logger.loggers.wandb.enabled=True")
            cmd.append(f"logger.loggers.wandb.project={self.wandb_project}")
            # Each element is single-quoted so OmegaConf parses it as a str
            # even when it's all-digits (e.g. a _cap_tag_length hash suffix) -
            # unquoted, OmegaConf's list grammar infers such elements as int,
            # which breaks WandBLogger's "_".join(group_tag).
            quoted_parts = ",".join(f"'{part}'" for part in self.group_tag_parts)
            cmd.append(f"logger.loggers.wandb.group_tag=[{quoted_parts}]")
        # `++` (override-or-add), not `=`: not every network yaml declares
        # use_layer_norm/use_input_layer_norm explicitly, so a plain `=`
        # override can fail with "Key not in struct" for some (system, arch)
        # combos - matches lightsout_fixed_budget_sweep.py/lightsout_sweep.py.
        if self.arch in EXPLICIT_COT_ARCHES or self.arch in NO_LAYER_NORM_ARCHES:
            # TransformerMergedActionCoTTorso only has use_input_layer_norm, not
            # use_layer_norm, same as TransformerChainOfThoughtTorso/
            # GRUAdaptiveComputationTimeTorso/IRUAdaptiveComputationTimeTorso
            # (see stoix/networks/torso_compute_explicit_cot_merged.py and
            # stoix/networks/torso_compute_transformer.py).
            cmd.append(
                f"++network.actor_network.pre_torso.use_input_layer_norm={self.use_input_layer_norm}"
            )
        else:
            cmd.append(f"++network.actor_network.pre_torso.use_layer_norm={self.use_layer_norm}")
            cmd.append(
                f"++network.actor_network.pre_torso.use_input_layer_norm={self.use_input_layer_norm}"
            )
        if self.arch in STOP_GRADIENT_HALTING_ARCHES:
            # Detaches the state fed into the halting head (IRUStep's or _CoTStep's) so the halting
            # REINFORCE loss can't backprop into the shared recurrent/transformer weights that also
            # produce the action head's representation - see stoix.networks.torso_compute.IRUStep's
            # and stoix.networks.torso_compute_transformer._CoTStep's docstrings.
            cmd.append(
                "++network.actor_network.pre_torso.stop_gradient_halting_input="
                f"{self.stop_gradient_halting_input}"
            )
        if self.arch in HALTING_TEMPERATURE_ARCHES:
            # Divides the halting head's logit before the sigmoid - see
            # stoix.networks.torso_compute's ACTStep/RecurrentACTStep/IRUStep and
            # stoix.networks.torso_compute_transformer's TransformerChainOfThoughtTorso docstrings.
            cmd.append(f"network.actor_network.pre_torso.halting_temperature={self.halting_temperature:g}")
        if self.arch in HALTING_HIDDEN_DIMS_ARCHES:
            # Hidden layer widths of the halting head's MLP - () (default) is a bare linear readout
            # (the original nn.Dense(1) design), a non-empty tuple gives it that many
            # Dense+activation hidden layers first - see
            # stoix.networks.torso_compute_transformer.HaltingHead.
            dims = ",".join(str(d) for d in self.halting_hidden_dims)
            cmd.append(f"++network.actor_network.pre_torso.halting_hidden_dims=[{dims}]")

        if is_cnn:
            cnn_arch = ENV_CNN_ARCH[self.env]
            critic_channel_sizes = channel_sizes = ",".join(str(c) for c in cnn_arch["channel_sizes"])
            critic_kernel_sizes = kernel_sizes = ",".join(str(k) for k in cnn_arch["kernel_sizes"])
            critic_strides = strides = ",".join(str(s) for s in cnn_arch["strides"])
            if "critic_channel_sizes" in cnn_arch:
                critic_channel_sizes = cnn_arch["critic_channel_sizes"]
            if "critic_kernel_sizes" in cnn_arch:
                critic_kernel_sizes = cnn_arch["critic_kernel_sizes"]
            if "critic_strides" in cnn_arch:
                critic_strides = cnn_arch["critic_strides"]
            hidden_sizes = ",".join(str(h) for h in cnn_arch["hidden_sizes"])
            critic_hidden_sizes = ",".join(str(h) for h in cnn_arch["critic_hidden_sizes"])
            critic_layer_sizes = ",".join(str(h) for h in cnn_arch["critic_layer_sizes"])
            cmd.append(f"network.actor_network.input_layer.channel_sizes=[{channel_sizes}]")
            cmd.append(f"network.actor_network.input_layer.kernel_sizes=[{kernel_sizes}]")
            cmd.append(f"network.actor_network.input_layer.strides=[{strides}]")
            cmd.append(f"network.actor_network.input_layer.hidden_sizes=[{hidden_sizes}]")
            cmd.append(f"network.critic_network.input_layer.channel_sizes=[{critic_channel_sizes}]")
            cmd.append(f"network.critic_network.input_layer.kernel_sizes=[{critic_kernel_sizes}]")
            cmd.append(f"network.critic_network.input_layer.strides=[{critic_strides}]")
            cmd.append(f"network.critic_network.input_layer.hidden_sizes=[{critic_hidden_sizes}]")
            cmd.append(f"network.critic_network.pre_torso.layer_sizes=[{critic_layer_sizes}]")
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
    # (use_dpo_loss, dpo_alpha, dpo_beta) combos: alpha/beta only matter (and
    # are only swept) when use_dpo_loss=True, for the same reason
    # as ppo_combos above.
    dpo_combos = []
    for use_dpo in args.use_dpo_loss:
        if use_dpo:
            for alpha in args.dpo_alpha:
                for beta in args.dpo_beta:
                    dpo_combos.append((True, alpha, beta))
        else:
            dpo_combos.append((False, args.dpo_alpha[0], args.dpo_beta[0]))
    dpo_combos = list(dict.fromkeys(dpo_combos))

    # (use_expectile_value_loss, expectile) combos: expectile only matters
    # (and is only swept) when use_expectile_value_loss=True, mirroring
    # dpo_combos above.
    expectile_combos = []
    for use_expectile in args.use_expectile_value_loss:
        if use_expectile:
            for expectile in args.expectile:
                expectile_combos.append((True, expectile))
        else:
            expectile_combos.append((False, args.expectile[0]))
    expectile_combos = list(dict.fromkeys(expectile_combos))

    ppo_combos = list(
        itertools.product(
            args.epochs,
            args.num_minibatches,
            args.clip_eps,
            args.clip_value_loss,
            args.gae_lambda,
            args.standardize_advantages,
            args.recompute_advantages,
            args.critic_before_actor,
        )
    )

    # (system, arch, use_layer_norm, use_input_layer_norm, num_layers, num_heads,
    # mlp_dim) combos:
    #  - transformer_explicit_cot_merged/cnn+transformer_explicit_cot_merged only exist for
    #    system in EXPLICIT_COT_SYSTEMS - any other requested (system,
    #    architecture) pair is skipped rather than erroring.
    #  - ff_ppo_explicit_*'s architecture defaults to transformer_explicit_cot_merged
    #    (flattened observation, see EXPLICIT_COT_PPO_SYSTEMS) when
    #    --architectures doesn't request either EXPLICIT_COT_ARCHES value;
    #    requesting cnn+transformer_explicit_cot_merged (optionally alongside
    #    transformer_explicit_cot_merged) opts into the CNN-input variant instead -
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
            qkv_dim_options = args.qkv_dim if is_transformer_arch else [args.qkv_dim[0]]
            vocab_size_options = (
                args.vocab_size if arch in EXPLICIT_COT_ARCHES else [args.vocab_size[0]]
            )
            use_latent_feedback_options = (
                args.use_latent_feedback
                if arch in EXPLICIT_COT_ARCHES
                else [args.use_latent_feedback[0]]
            )
            stop_gradient_halting_input_options = (
                args.stop_gradient_halting_input
                if arch in STOP_GRADIENT_HALTING_ARCHES
                else [args.stop_gradient_halting_input[0]]
            )
            # use_sandwich_norm/use_rmsnorm only exist on TransformerBlock-based
            # torsos - same applicability as num_heads/mlp_dim above (see
            # is_transformer_arch).
            use_sandwich_norm_options = (
                args.use_sandwich_norm if is_transformer_arch else [args.use_sandwich_norm[0]]
            )
            use_rmsnorm_options = args.use_rmsnorm if is_transformer_arch else [args.use_rmsnorm[0]]
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
                            for qkv_dim in qkv_dim_options:
                                for (
                                    vocab_size,
                                    use_latent_feedback,
                                    stop_gradient_halting_input,
                                    use_sandwich_norm,
                                    use_rmsnorm,
                                ) in itertools.product(
                                    vocab_size_options,
                                    use_latent_feedback_options,
                                    stop_gradient_halting_input_options,
                                    use_sandwich_norm_options,
                                    use_rmsnorm_options,
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
                                            qkv_dim,
                                            vocab_size,
                                            use_latent_feedback,
                                            stop_gradient_halting_input,
                                            use_sandwich_norm,
                                            use_rmsnorm,
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
                qkv_dim,
                vocab_size,
                use_latent_feedback,
                stop_gradient_halting_input,
                use_sandwich_norm,
                use_rmsnorm,
            ),
            (min_steps, max_steps),
            hidden_dim,
            lr,
            critic_lr,
            actor_weight_decay,
            critic_weight_decay,
            ent_coef,
            max_grad_norm,
            (
                epochs,
                num_minibatches,
                clip_eps,
                clip_value_loss,
                gae_lambda,
                standardize_advantages,
                recompute_advantages,
                critic_before_actor,
            ),
            latent_kl_coef,
            clip_halting_head,
            halting_lr,
            halting_weight_decay,
            halting_ent_coef,
            halting_temperature,
            halting_hidden_dims,
            (use_dpo_loss, dpo_alpha, dpo_beta),
            (use_expectile_value_loss, expectile),
            seed,
        ) in itertools.product(
            difficulty_combos,
            system_arch_ln_combos,
            step_combos,
            args.hidden_dim,
            args.lr,
            args.critic_lr,
            args.actor_weight_decay,
            args.critic_weight_decay,
            args.ent_coef,
            args.max_grad_norm,
            ppo_combos,
            args.latent_kl_coef,
            args.clip_halting_head,
            args.halting_lr,
            args.halting_weight_decay,
            args.halting_ent_coef,
            args.halting_temperature,
            args.halting_hidden_dims,
            dpo_combos,
            expectile_combos,
            range(args.base_seed, args.base_seed + args.seeds),
        ):
            if arch in CNN_ARCHES and not ENV_SUPPORTS_CNN[env]:
                n_skipped_cnn += 1
                continue
            # latent_kl_coef only exists on ff_ppo.py's own systems
            # (LATENT_KL_PPO_SYSTEMS) - forced to the first requested value
            # for every other system (including explicit-CoT PPO systems).
            if system not in LATENT_KL_PPO_SYSTEMS:
                latent_kl_coef = args.latent_kl_coef[0]
            # clip_halting_head/halting_lr/halting_weight_decay only exist on
            # ff_ppo.py's own systems and only have an actual halting head to
            # single out on TRANSFORMER_ARCHES (see
            # _label_actor_params_by_halting_head in ff_ppo.py) - forced to
            # the first requested value for every other (system, arch).
            if system not in LATENT_KL_PPO_SYSTEMS or arch not in TRANSFORMER_ARCHES:
                clip_halting_head = args.clip_halting_head[0]
                halting_lr = args.halting_lr[0]
                halting_weight_decay = args.halting_weight_decay[0]
            # halting_ent_coef exists on ff_ppo.py's own systems only (same
            # as latent_kl_coef above) - forced to the first requested value
            # for every other system, which have no such config knob.
            if system not in LATENT_KL_PPO_SYSTEMS:
                halting_ent_coef = args.halting_ent_coef[0]
            # halting_temperature only exists on HALTING_TEMPERATURE_ARCHES -
            # forced to the first requested value (default 1.0, a no-op) for
            # every other architecture.
            if arch not in HALTING_TEMPERATURE_ARCHES:
                halting_temperature = args.halting_temperature[0]
            # halting_hidden_dims only exists on HALTING_HIDDEN_DIMS_ARCHES -
            # forced to the first requested value (default (), a no-op) for
            # every other architecture.
            if arch not in HALTING_HIDDEN_DIMS_ARCHES:
                halting_hidden_dims = args.halting_hidden_dims[0]
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
                    actor_weight_decay=actor_weight_decay,
                    critic_weight_decay=critic_weight_decay,
                    ent_coef=ent_coef,
                    max_grad_norm=max_grad_norm,
                    epochs=epochs,
                    num_minibatches=num_minibatches,
                    clip_eps=clip_eps,
                    clip_value_loss=clip_value_loss,
                    gae_lambda=gae_lambda,
                    latent_kl_coef=latent_kl_coef,
                    clip_halting_head=clip_halting_head,
                    halting_lr=halting_lr,
                    halting_weight_decay=halting_weight_decay,
                    halting_ent_coef=halting_ent_coef,
                    halting_temperature=halting_temperature,
                    halting_hidden_dims=halting_hidden_dims,
                    use_dpo_loss=use_dpo_loss,
                    dpo_alpha=dpo_alpha,
                    dpo_beta=dpo_beta,
                    use_expectile_value_loss=use_expectile_value_loss,
                    expectile=expectile,
                    standardize_advantages=standardize_advantages,
                    recompute_advantages=recompute_advantages,
                    critic_before_actor=critic_before_actor,
                    use_layer_norm=use_layer_norm,
                    use_input_layer_norm=use_input_layer_norm,
                    stop_gradient_halting_input=stop_gradient_halting_input,
                    num_layers=num_layers,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    qkv_dim=qkv_dim,
                    vocab_size=vocab_size,
                    use_latent_feedback=use_latent_feedback,
                    use_sandwich_norm=use_sandwich_norm,
                    use_rmsnorm=use_rmsnorm,
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
            "to that job's system)."
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
        default="ff_ppo_reinforce",
        help="Comma-separated subset of {ff_ppo_reinforce, "
        "ff_ppo_explicit_reinforce}. See jumanji_fixed_budget_sweep.py's module docstring for what "
        "each means. ff_ppo_explicit_reinforce trains "
        "stoix/systems/ramdp_vpg/ff_ppo_explicit_cot.py (explicit chain-of-thought tokens, G - V "
        "advantage) and defaults to a flattened-observation architecture regardless of --architectures unless it "
        "requests transformer_explicit_cot_merged/cnn+transformer_explicit_cot_merged explicitly.",
    )
    parser.add_argument(
        "--architectures",
        default="mlp,transformer",
        help=f"Comma-separated subset of {{{','.join(VALID_ARCHITECTURES)}}}. CNN architectures "
        "(including cnn+transformer_explicit_cot_merged) are only valid for env in "
        "{sokoban, slidingtile, maze, pacman} (see ENV_SUPPORTS_CNN) - requested for knapsack, "
        "they're skipped, not errored. transformer_explicit_cot_merged/cnn+transformer_explicit_cot_merged "
        f"(TransformerMergedActionCoTTorso) are only implemented for system in {EXPLICIT_COT_SYSTEMS} "
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
        "--actor-weight-decay", default="0.0",
        help="Comma-separated system.actor_weight_decay values (AdamW weight decay for the actor; "
        "0.0 recovers plain Adam), swept independently of --critic-weight-decay.",
    )
    parser.add_argument(
        "--critic-weight-decay", default="0.0",
        help="Comma-separated system.critic_weight_decay values (AdamW weight decay for the critic; "
        "0.0 recovers plain Adam), swept independently of --actor-weight-decay.",
    )
    parser.add_argument(
        "--ent-coef", default="0.01",
        help="Comma-separated system.ent_coef values (entropy bonus coefficient).",
    )
    parser.add_argument(
        "--max-grad-norm", default="0.5",
        help="Comma-separated system.max_grad_norm values (global gradient-clipping norm, "
        "applied before the optimizer update). Default 0.5, matching the yaml default.",
    )
    parser.add_argument("--epochs", default="4", help="Comma-separated system.epochs values (PPO only).")
    parser.add_argument("--num-minibatches", default="16", help="Comma-separated system.num_minibatches values (PPO only).")
    parser.add_argument("--clip-eps", default="0.2", help="Comma-separated system.clip_eps values (PPO only).")
    parser.add_argument("--clip-value-loss", default="true", help="Comma-separated bools (PPO only).")
    parser.add_argument(
        "--gae-lambda",
        default="1.0",
        help="Comma-separated system.gae_lambda values (GAE(lambda) mixing parameter for the "
        "critic's regression target). 1.0 (default) recovers the original to-the-end "
        "Monte-Carlo return exactly; <1.0 blends in earlier bootstrapped value estimates for "
        "a lower-variance, more-biased target - and, for ff_ppo_reinforce, makes the "
        "advantage itself GAE(lambda). Swept independently of --epochs/--num-minibatches/"
        "--clip-eps/--clip-value-loss. PPO systems only.",
    )
    parser.add_argument(
        "--standardize-advantages",
        default="false",
        help="Comma-separated bools (true/false) - system.standardize_advantages: whether the "
        "advantage is standardized (zero mean, unit variance) across the rollout before being "
        "used in the PPO clipped surrogate. Swept independently of --epochs/--num-minibatches/"
        "--clip-eps/--clip-value-loss. PPO systems only.",
    )
    parser.add_argument(
        "--recompute-advantages",
        default="false",
        help="Comma-separated bools (true/false) - system.recompute_advantages: whether the "
        "advantage's 'what changed' term and the critic's own regression target are recomputed "
        "at the end of every PPO epoch from that epoch's just-updated critic params, rather than "
        "staying pinned at their rollout-time values for the whole update (vanilla-PPO style, "
        "the default). PPO systems only.",
    )
    parser.add_argument(
        "--critic-before-actor",
        default="false",
        help="Comma-separated bools (true/false) - system.critic_before_actor: replaces the "
        "joint per-minibatch actor+critic update with two fully sequential phases, `epochs` "
        "epochs of critic-only updates followed by `epochs` epochs of actor-only updates - see "
        "ff_ppo.py's module docstring. PPO systems only.",
    )
    parser.add_argument(
        "--latent-kl-coef",
        default="0.0",
        help="Comma-separated system.latent_kl_coef values - optional trust-region penalty on "
        "how far the actor torso's per-step latent 'thought' states may drift across a PPO "
        "update (0.0 disables it, see ff_ppo.py's module docstring). ff_ppo.py's own systems "
        "only (LATENT_KL_PPO_SYSTEMS), not ff_ppo_explicit_*.",
    )
    parser.add_argument(
        "--clip-halting-head",
        default="true",
        help="Comma-separated bools (true/false) - system.clip_halting_head: whether the "
        "halting head's own params are subject to the actor's max_grad_norm clipping like "
        "every other actor param (true, the default) or updated with plain unclipped AdamW "
        "instead (false). ff_ppo.py's own systems only (LATENT_KL_PPO_SYSTEMS), and only has "
        "an actual halting head to exempt on TRANSFORMER_ARCHES - forced to true otherwise.",
    )
    parser.add_argument(
        "--halting-lr",
        default="3e-4",
        help="Comma-separated system.halting_lr values - learning rate for the halting head's "
        "own params, overriding --lr for just that submodule (matching --lr, the same value, "
        "recovers the original shared-learning-rate behaviour exactly). Swept independently of "
        "--lr. ff_ppo.py's own systems only (LATENT_KL_PPO_SYSTEMS), and only has an actual "
        "halting head to single out on TRANSFORMER_ARCHES - forced to the first requested "
        "value otherwise.",
    )
    parser.add_argument(
        "--halting-weight-decay",
        default="0.0",
        help="Comma-separated system.halting_weight_decay values - AdamW weight decay for the "
        "halting head's own params, overriding --actor-weight-decay for just that submodule. "
        "Swept independently of --actor-weight-decay. Same applicability as --halting-lr above.",
    )
    parser.add_argument(
        "--halting-ent-coef",
        default="0.0",
        help="Comma-separated system.halting_ent_coef values - entropy regularisation "
        "coefficient for the halting decision itself (a per-step Bernoulli for the IRU/GRU/mlp/"
        "latent-CoT torsos), separate from --ent-coef, which only ever reaches the "
        "environment action's distribution. Without this, nothing keeps the halting policy from "
        "collapsing to a degenerate, non-adaptive compute-time before discovering genuine "
        "per-example structure (0.0 disables it) - see ff_ppo.py's module docstring. Applies to "
        "ff_ppo.py's own systems only (LATENT_KL_PPO_SYSTEMS) - "
        "ff_ppo_explicit_reinforce's single per-step categorical entropy "
        "is already covered by --ent-coef.",
    )
    parser.add_argument(
        "--use-dpo-loss",
        default="false",
        help="Comma-separated bools (true/false) - system.use_dpo_loss: whether the env-action "
        "(and halting/CoT-step) actor surrogate uses Discovered Policy Optimisation "
        "(stoix.utils.loss.dpo_loss/dpo_surrogate, Lu et al. 2022, "
        "https://arxiv.org/abs/2210.05639) instead of PPO's clipped surrogate "
        "(stoix.utils.loss.ppo_clip_loss). False (default) recovers the original ppo_clip_loss "
        "behaviour exactly. Applies to every system in PPO_SYSTEMS (both ff_ppo.py's own "
        "systems and ff_ppo_explicit_*, same applicability as --halting-ent-coef).",
    )
    parser.add_argument(
        "--dpo-alpha",
        default="2.0",
        help="Comma-separated system.dpo_alpha values - DPO's positive-advantage drift "
        "coefficient, only used (and only swept) when --use-dpo-loss includes true - paired "
        "with --use-dpo-loss/--dpo-beta via dpo_combos so a sweep isn't needlessly duplicated "
        "across every use_dpo_loss=false job Default "
        "2.0, the value found by Lu et al. (2022)'s meta-optimisation.",
    )
    parser.add_argument(
        "--dpo-beta",
        default="0.6",
        help="Comma-separated system.dpo_beta values - DPO's negative-advantage drift "
        "coefficient, same applicability/pairing as --dpo-alpha. Default 0.6, the value found "
        "by Lu et al. (2022)'s meta-optimisation.",
    )
    parser.add_argument(
        "--use-expectile-value-loss",
        default="false",
        help="Comma-separated bools (true/false) - system.use_expectile_value_loss: whether V's "
        "loss uses expectile regression "
        "(stoix.utils.loss.expectile_loss, Kostrikov et al. 2021's Implicit Q-Learning, "
        "https://arxiv.org/abs/2110.06169) instead of PPO's clipped value loss / plain L2 (per "
        "--clip-value-loss). False (default) recovers the original clip_value_loss/L2 "
        "behaviour exactly. Applies to every system in PPO_SYSTEMS (both ff_ppo.py's own "
        "systems and ff_ppo_explicit_*, same applicability as --use-dpo-loss).",
    )
    parser.add_argument(
        "--expectile",
        default="0.1",
        help="Comma-separated system.expectile values - V's target expectile, only used (and "
        "only swept) when --use-expectile-value-loss includes true - paired with it via "
        "expectile_combos so a sweep isn't needlessly duplicated across every "
        "use_expectile_value_loss=false job (mirrors --dpo-alpha/--use-dpo-loss). <0.5 makes V "
        "deliberately (and persistently, not uncertainty-dependent) underestimate the return "
        "distribution - with no per-action information in the loss, it just "
        "reinforces whichever action was sampled a bit more; >0.5 would make V overestimate "
        "instead (IQL's usual direction); 0.5 recovers plain squared-error regression. Default "
        "0.1.",
    )
    parser.add_argument(
        "--halting-temperature",
        default="1.0",
        help="Comma-separated network.actor_network.pre_torso.halting_temperature values - "
        "divides the halting head's logit before the sigmoid (see "
        "stoix.networks.torso_compute's ACTStep/RecurrentACTStep/IRUStep and "
        "stoix.networks.torso_compute_transformer's TransformerChainOfThoughtTorso "
        "docstrings): below 1.0 "
        "sharpens the halting probability towards 0/1, above 1.0 softens it towards 0.5, 1.0 is "
        "a no-op. Only applies to architecture in {mlp, gru, iru, cnn+mlp, cnn+gru, cnn+iru, "
        "transformer, cnn+transformer} "
        "(HALTING_TEMPERATURE_ARCHES); ignored (forced to the first value) for every other "
        "architecture. A no-op whenever min_steps == max_steps (halting is always forced, so "
        "the halting head is never actually queried for a decision).",
    )
    parser.add_argument(
        "--halting-hidden-dims",
        default="none",
        help="Comma-separated network.actor_network.pre_torso.halting_hidden_dims values - "
        "hidden layer widths of the halting head's MLP (see "
        "stoix.networks.torso_compute_transformer.HaltingHead), each value itself an "
        "'x'-joined tuple of layer widths, e.g. '--halting-hidden-dims none,64,128x128' sweeps "
        "a bare linear readout (the original design), one 64-unit hidden layer, and two "
        "128-unit hidden layers. 'none' (default) or an empty string means a bare linear "
        "readout - the original nn.Dense(1) design, a no-op. Only applies to architecture in "
        "{transformer, cnn+transformer} (HALTING_HIDDEN_DIMS_ARCHES) - unlike "
        "--halting-temperature, does *not* apply to mlp/gru/iru (their halting heads don't "
        "have this option yet); ignored (forced to the first value) for every other "
        "architecture.",
    )
    parser.add_argument("--use-layer-norm", default="false", help="Comma-separated bools (mlp/cnn+mlp only).")
    parser.add_argument("--use-input-layer-norm", default="false", help="Comma-separated bools.")
    parser.add_argument(
        "--stop-gradient-halting-input",
        default="false",
        help="Comma-separated bools - network.actor_network.pre_torso.stop_gradient_halting_input: "
        "detaches the state fed into the halting head (IRUStep's or _CoTStep's Dense(1)) before "
        "it's read, so the halting REINFORCE loss can't backprop into the shared "
        "recurrent/transformer weights that also produce the action head's representation - see "
        "stoix.networks.torso_compute.IRUStep's and "
        "stoix.networks.torso_compute_transformer._CoTStep's docstrings. Only applies to "
        "architecture in {iru, cnn+iru, transformer, cnn+transformer} "
        "(STOP_GRADIENT_HALTING_ARCHES); ignored (forced to the first value) for every other "
        "architecture. A no-op when min_steps == max_steps.",
    )
    parser.add_argument("--num-layers", default="1", help="Comma-separated ints - sub-layers per pondering step.")
    parser.add_argument("--num-heads", default="4", help="Comma-separated ints (transformer archs only).")
    parser.add_argument("--mlp-dim", default="256", help="Comma-separated ints (transformer archs only).")
    parser.add_argument(
        "--qkv-dim",
        default="0",
        help="Comma-separated ints - network.actor_network.pre_torso.qkv_dim: the total Q/K/V "
        "projection width inside every TransformerBlock, decoupled from hidden_dim (the "
        "residual-stream/input-projection width) - see "
        "stoix/networks/torso_compute_transformer.py's TransformerBlock docstring. 0 (default) "
        "means unset - the override is omitted and the torso falls back to qkv_dim == hidden_dim, "
        "its original behavior. Only applies to TRANSFORMER_ARCHES/EXPLICIT_COT_ARCHES (same "
        "applicability as --num-heads/--mlp-dim); ignored (forced to the first value) otherwise.",
    )
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
        "stoix/networks/torso_compute_explicit_cot_merged.py. Only applies to the explicit-CoT "
        "arches (EXPLICIT_COT_ARCHES); ignored (forced to the first value) for every other "
        "architecture. Default false.",
    )
    parser.add_argument(
        "--use-sandwich-norm",
        default="false",
        help="Comma-separated bools (true/false) - sandwich LayerNorm placement "
        "(network.actor_network.pre_torso.use_sandwich_norm): normalizes each TransformerBlock "
        "sub-layer's residual sum, not just its input as in plain pre-norm - see "
        "stoix/networks/torso_compute_transformer.py's TransformerBlock docstring. Only applies "
        "to TRANSFORMER_ARCHES/EXPLICIT_COT_ARCHES (every other architecture has no "
        "TransformerBlock); ignored (forced to the first value) otherwise. Default false.",
    )
    parser.add_argument(
        "--use-rmsnorm",
        default="false",
        help="Comma-separated bools (true/false) - RMSNorm instead of LayerNorm "
        "(network.actor_network.pre_torso.use_rmsnorm): switches every norm inside the "
        "pre_torso (use_input_layer_norm's and every TransformerBlock norm) from nn.LayerNorm "
        "to nn.RMSNorm - see stoix/networks/torso_compute_transformer.py's _norm_cls. Same "
        "applicability as --use-sandwich-norm. Default false.",
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
        "--sokoban-eval-generator",
        default="same",
        choices=SOKOBAN_EVAL_GENERATOR_CHOICES,
        help="env.eval_kwargs.generator for sokoban jobs: the generator the *eval* environment "
        "draws levels from, while training keeps using --sokoban-generator. Single value (not "
        "swept). 'same' (default) evaluates on the train generator. E.g. train on "
        "'unfiltered-train' and evaluate on the held-out 'unfiltered-test'. Adds an "
        "'-eval-<name>' suffix to the group_tag/run_name.",
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
        "--slidingtile-time-limit",
        default="default",
        help="Comma-separated ints - env.kwargs.time_limit for slidingtile jobs: the train "
        "environment's max steps per episode before truncation (also the eval env's, unless "
        "--slidingtile-eval-time-limit is set), ignored for other envs. 'default' (default) keeps "
        "the time_limit set in jumanji/slidingtile.yaml/slidingtile_grid.yaml (40) and adds no "
        "tag; otherwise adds a '-tl<N>' suffix to the group_tag/run_name.",
    )
    parser.add_argument(
        "--slidingtile-eval-time-limit",
        default="same",
        help="env.eval_kwargs.time_limit for slidingtile jobs: the eval environment's max steps "
        "per episode before truncation, while training keeps using --slidingtile-time-limit. "
        "Single value (not swept), "
        "ignored for other envs. 'same' (default) evaluates with the train time_limit. Adds an "
        "'-evaltl<N>' suffix to the group_tag/run_name.",
    )
    parser.add_argument(
        "--slidingtile-eval-num-random-moves",
        default="same",
        help="env.eval_kwargs.generator.num_random_moves for slidingtile jobs: the eval "
        "environment's scramble depth, while training keeps using --slidingtile-num-random-moves. "
        "Single value (not swept), ignored for other envs. 'same' (default) evaluates with the "
        "train scramble depth. Adds an '-evalnrm<N>' suffix to the group_tag/run_name.",
    )
    parser.add_argument(
        "--knapsack-num-items", default="10,20,50",
        help="Comma-separated ints - env.kwargs.generator.num_items, ignored for other envs.",
    )
    parser.add_argument(
        "--knapsack-max-weight", default="10",
        help="Comma-separated ints - env.kwargs.generator.max_weight: item weights are drawn "
        "from {1, ..., max_weight} (stoix.envs.knapsack.generator.IntegerRandomGenerator, the "
        "classic 0-1 knapsack formulation), ignored for other envs.",
    )
    parser.add_argument(
        "--knapsack-max-value", default="10",
        help="Comma-separated ints - env.kwargs.generator.max_value: item values are drawn from "
        "{1, ..., max_value}, ignored for other envs.",
    )
    parser.add_argument(
        "--knapsack-max-budget", default="20",
        help="Comma-separated ints - env.kwargs.generator.max_budget: the bag's capacity is "
        "redrawn every episode from Uniform{1, ..., max_budget} (not a fixed constant), ignored "
        "for other envs.",
    )
    parser.add_argument(
        "--maze-size", default="5,10,15",
        help="Comma-separated ints - square maze side length (env.kwargs.generator.num_rows == "
        "num_cols), ignored for other envs.",
    )

    parser.add_argument(
        "--gamma", type=float, default=0.99,
        help="system.gamma, applied to every job (not swept). sokoban/slidingtile/knapsack/maze "
        "have short (<=a few hundred step) episodes; pacman's run up to 1000 steps - 0.99 is "
        "still used as the shared default rather than special-cased per env, unlike MinAtar's "
        "gamma=0.9999 default.",
    )
    parser.add_argument("--wandb", type=lambda x: x.strip().lower() in ("1", "true", "yes"), default=False)
    parser.add_argument("--wandb-project", default="jumanji_sweep")
    parser.add_argument(
        "--seeds",
        type=int,
        default=5,
        help="Number of seeds per config, seeded base_seed..base_seed+seeds-1.",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=0,
        help="First seed (default 0); e.g. --base-seed 5 --seeds 5 runs seeds 5..9, "
        "extending an earlier --seeds 5 sweep with new seeds.",
    )
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
    args.actor_weight_decay = [float(x) for x in args.actor_weight_decay.split(",")]
    args.critic_weight_decay = [float(x) for x in args.critic_weight_decay.split(",")]
    args.ent_coef = [float(x) for x in args.ent_coef.split(",")]
    args.max_grad_norm = [float(x) for x in args.max_grad_norm.split(",")]
    args.epochs = [int(x) for x in args.epochs.split(",")]
    args.num_minibatches = [int(x) for x in args.num_minibatches.split(",")]
    args.clip_eps = [float(x) for x in args.clip_eps.split(",")]
    args.clip_value_loss = [x.strip().lower() in ("1", "true", "yes") for x in args.clip_value_loss.split(",")]
    args.gae_lambda = [float(x) for x in args.gae_lambda.split(",")]
    args.standardize_advantages = [
        x.strip().lower() in ("1", "true", "yes") for x in args.standardize_advantages.split(",")
    ]
    args.recompute_advantages = [
        x.strip().lower() in ("1", "true", "yes") for x in args.recompute_advantages.split(",")
    ]
    args.critic_before_actor = [
        x.strip().lower() in ("1", "true", "yes") for x in args.critic_before_actor.split(",")
    ]
    args.latent_kl_coef = [float(x) for x in args.latent_kl_coef.split(",")]
    args.clip_halting_head = [
        x.strip().lower() in ("1", "true", "yes") for x in args.clip_halting_head.split(",")
    ]
    args.halting_lr = [float(x) for x in args.halting_lr.split(",")]
    args.halting_weight_decay = [float(x) for x in args.halting_weight_decay.split(",")]
    args.halting_ent_coef = [float(x) for x in args.halting_ent_coef.split(",")]
    args.use_dpo_loss = [
        x.strip().lower() in ("1", "true", "yes") for x in args.use_dpo_loss.split(",")
    ]
    args.dpo_alpha = [float(x) for x in args.dpo_alpha.split(",")]
    args.dpo_beta = [float(x) for x in args.dpo_beta.split(",")]
    args.use_expectile_value_loss = [
        x.strip().lower() in ("1", "true", "yes") for x in args.use_expectile_value_loss.split(",")
    ]
    args.expectile = [float(x) for x in args.expectile.split(",")]
    args.halting_temperature = [float(x) for x in args.halting_temperature.split(",")]
    args.halting_hidden_dims = [
        () if part.strip().lower() in ("", "none") else tuple(int(d) for d in part.split("x"))
        for part in args.halting_hidden_dims.split(",")
    ]
    args.use_layer_norm = [x.strip().lower() in ("1", "true", "yes") for x in args.use_layer_norm.split(",")]
    args.use_input_layer_norm = [x.strip().lower() in ("1", "true", "yes") for x in args.use_input_layer_norm.split(",")]
    args.stop_gradient_halting_input = [
        x.strip().lower() in ("1", "true", "yes")
        for x in args.stop_gradient_halting_input.split(",")
    ]
    args.num_layers = [int(x) for x in args.num_layers.split(",")]
    args.num_heads = [int(x) for x in args.num_heads.split(",")]
    args.mlp_dim = [int(x) for x in args.mlp_dim.split(",")]
    args.qkv_dim = [int(x) for x in args.qkv_dim.split(",")]
    args.vocab_size = [int(x) for x in args.vocab_size.split(",")]
    args.use_latent_feedback = [
        x.strip().lower() in ("1", "true", "yes") for x in args.use_latent_feedback.split(",")
    ]
    args.use_sandwich_norm = [
        x.strip().lower() in ("1", "true", "yes") for x in args.use_sandwich_norm.split(",")
    ]
    args.use_rmsnorm = [
        x.strip().lower() in ("1", "true", "yes") for x in args.use_rmsnorm.split(",")
    ]
    args.sokoban_generator = args.sokoban_generator.split(",")
    args.slidingtile_grid_size = [int(x) for x in args.slidingtile_grid_size.split(",")]
    args.slidingtile_num_random_moves = [int(x) for x in args.slidingtile_num_random_moves.split(",")]
    args.slidingtile_time_limit = (
        [None]
        if args.slidingtile_time_limit == "default"
        else [int(x) for x in args.slidingtile_time_limit.split(",")]
    )
    if args.slidingtile_eval_time_limit != "same":
        int(args.slidingtile_eval_time_limit)  # validate - kept as str, embedded directly below
    if args.slidingtile_eval_num_random_moves != "same":
        int(args.slidingtile_eval_num_random_moves)  # validate - kept as str, embedded directly below
    args.knapsack_num_items = [int(x) for x in args.knapsack_num_items.split(",")]
    args.knapsack_max_weight = [int(x) for x in args.knapsack_max_weight.split(",")]
    args.knapsack_max_value = [int(x) for x in args.knapsack_max_value.split(",")]
    args.knapsack_max_budget = [int(x) for x in args.knapsack_max_budget.split(",")]
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
        f"seeds={args.base_seed}..{args.base_seed + args.seeds - 1}"
    )
    print(
        f"  lr={args.lr} critic_lr={args.critic_lr} ent_coef={args.ent_coef} "
        f"max_grad_norm={args.max_grad_norm}"
    )
    print(f"  actor_weight_decay={args.actor_weight_decay} critic_weight_decay={args.critic_weight_decay}")
    print(
        f"  latent_kl_coef={args.latent_kl_coef} "
        f"(ff_ppo.py's own systems only: {LATENT_KL_PPO_SYSTEMS})"
    )
    print(
        f"  clip_halting_head={args.clip_halting_head} "
        f"(ff_ppo.py's own systems x TRANSFORMER_ARCHES only: "
        f"{LATENT_KL_PPO_SYSTEMS}, {TRANSFORMER_ARCHES})"
    )
    print(
        f"  halting_lr={args.halting_lr} halting_weight_decay={args.halting_weight_decay} "
        f"(ff_ppo.py's own systems x TRANSFORMER_ARCHES only: "
        f"{LATENT_KL_PPO_SYSTEMS}, {TRANSFORMER_ARCHES})"
    )
    print(f"  halting_ent_coef={args.halting_ent_coef} (ff_ppo.py's own systems only: {LATENT_KL_PPO_SYSTEMS})")
    print(
        f"  use_dpo_loss={args.use_dpo_loss} dpo_alpha={args.dpo_alpha} dpo_beta={args.dpo_beta} "
        f"(PPO systems only: {PPO_SYSTEMS})"
    )
    print(
        f"  use_expectile_value_loss={args.use_expectile_value_loss} expectile={args.expectile} "
        f"(PPO systems only: {PPO_SYSTEMS})"
    )
    print(
        f"  halting_temperature={args.halting_temperature} "
        f"(HALTING_TEMPERATURE_ARCHES only: {HALTING_TEMPERATURE_ARCHES})"
    )
    print(
        f"  halting_hidden_dims={args.halting_hidden_dims} "
        f"(HALTING_HIDDEN_DIMS_ARCHES only: {HALTING_HIDDEN_DIMS_ARCHES})"
    )
    print(f"  sokoban_generator={args.sokoban_generator} sokoban_eval_generator={args.sokoban_eval_generator}")
    print(
        f"  slidingtile_grid_size={args.slidingtile_grid_size} "
        f"slidingtile_num_random_moves={args.slidingtile_num_random_moves} "
        f"slidingtile_time_limit={args.slidingtile_time_limit} "
        f"slidingtile_eval_time_limit={args.slidingtile_eval_time_limit} "
        f"slidingtile_eval_num_random_moves={args.slidingtile_eval_num_random_moves}"
    )
    print(
        f"  knapsack_num_items={args.knapsack_num_items} knapsack_max_weight={args.knapsack_max_weight} "
        f"knapsack_max_value={args.knapsack_max_value} knapsack_max_budget={args.knapsack_max_budget}"
    )
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
