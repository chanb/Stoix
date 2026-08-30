#!/usr/bin/env python
"""Fixed-computation-budget sweep for RAMDP systems on MinAtar environments
(gymnax/<env>, e.g. gymnax/seaquest).

Companion to minatar_sweep.py: instead of letting the actor's compute torso
adaptively halt (min_steps=1, sampled/learned halting), every job here pins
`min_steps == max_steps == budget`, which - per the min_steps mechanism added
to the compute torsos (see stoix/networks/torso_compute*.py) - forbids
halting before `budget` steps and forces a halt at exactly `budget` steps.
So every example always takes exactly `budget` steps of computation, with no
adaptivity at all.

This answers "does more computation help on this task?" directly (sweeping
`budget` at a fixed, non-adaptive cost, comparable to minatar_sweep.py's
adaptive min_steps/max_steps axes) and is also a simpler system to debug
against, since compute_time is deterministic (== budget) rather than a
learned/sampled quantity.

Grid axes:
  - env:         MinAtar game (env=gymnax/<env>) - subset of MINATAR_GAMES
  - system:      ff_reinforce (PonderNet-style REINFORCE, G - V) | ff_qac_fac (Q - V,
                 runtime-factorized) | ff_qac_naive (Q - V, full table) | ff_ppo_fac
                 (PPO, Q - V runtime-factorized) | ff_ppo_naive (PPO, Q - V full table) |
                 ff_ppo_cond_naive (PPO, Q - V with compute_time fed into the critic as
                 an input instead of the output shape) | ff_ppo_cond_fac (PPO, same
                 c-conditioned architecture/parameter count as ff_ppo_cond_naive, but
                 the conditioned output is additionally scaled by gamma^(c-1) - compare
                 the two to isolate whether that analytic prior helps, holding capacity
                 fixed) | ff_ppo_reinforce (PPO, G - V REINFORCE-with-baseline) |
                 ff_ppo_explicit_fac/ff_ppo_explicit_naive/ff_ppo_explicit_cond_naive/
                 ff_ppo_explicit_cond_fac/ff_ppo_explicit_reinforce (PPO with an
                 *explicit* chain of thought - TransformerExplicitCoTTorso instead of a
                 latent-CoT torso, one system per qac_variant mirroring the five ff_ppo_*
                 systems above; trains stoix/systems/ramdp_vpg/ff_ppo_explicit_cot.py.
                 Their architecture is implied (always cnn+transformer_explicit_cot,
                 regardless of --architectures) - see EXPLICIT_COT_PPO_SYSTEMS). The
                 ff_ppo_* systems train stoix/systems/ramdp_vpg/ff_ppo.py - PPO's clipped
                 surrogate over several epochs of minibatch updates per rollout, vs. one
                 REINFORCE/QAC gradient step per rollout for the others - see --epochs/
                 --num-minibatches/--clip-eps/--clip-value-loss. ff_ppo_cond_naive/
                 ff_ppo_cond_fac (and their ff_ppo_explicit_* counterparts) have no
                 ff_qac.py equivalent (see stoix/systems/ramdp_vpg/ff_ppo.py's module
                 docstring) - they only exist as ff_ppo_*/ff_ppo_explicit_* systems.
  - architecture: every architecture consumes the raw MinAtar grid observation
                 through a CNNTorso input_layer (see CNN_ARCHES): cnn+mlp (CNNTorso
                 feeding AdaptiveComputationTimeTorso) | cnn+transformer (CNNTorso
                 feeding TransformerChainOfThoughtTorso, latent CoT) | cnn+gru
                 (CNNTorso feeding GRUAdaptiveComputationTimeTorso, GRU-based
                 recurrent block that re-feeds the encoded observation every
                 pondering step) | cnn+iru (CNNTorso feeding
                 IRUAdaptiveComputationTimeTorso, interpolation-recurrent-unit-based
                 recurrent block, same re-feeding) | cnn+transformer_explicit_cot
                 (CNNTorso feeding TransformerExplicitCoTTorso, explicit token CoT -
                 only implemented for system in EXPLICIT_COT_SYSTEMS; requested
                 (system, architecture) combos outside that are skipped)
  - budget:      fixed number of steps every example takes (max_steps == min_steps)
  - hidden_dim:  actor torso width (network.actor_network.pre_torso.hidden_dim)
  - lr:          system.actor_lr - has its own value list, independent of critic_lr's
                 (the full lr x critic_lr cross product is still swept)
  - critic_lr:   system.critic_lr - has its own value list, independent of lr's
  - delightful:  whether to gate the REINFORCE weight by the "delightful" surprisal
                 sigmoid (system.delightful); off by default. When on, also sweeps
                 delightful_eta (system.delightful_eta). Not supported by the
                 ff_ppo_* systems (forced off - see PPO_SYSTEMS).
  - epochs:      system.epochs - PPO epochs per rollout; swept independently of
                 num_minibatches/clip_eps (full cross product). ff_ppo_* systems
                 only, forced to the first requested value otherwise.
  - num_minibatches: system.num_minibatches - PPO minibatches per epoch; ff_ppo_*
                 systems only, see epochs.
  - clip_eps:    system.clip_eps - PPO ratio/value clipping; ff_ppo_* systems only,
                 see epochs.
  - clip_value_loss: system.clip_value_loss - whether the critic's value/Q loss uses
                 PPO-style clipping (True, default) against the old value/Q estimate,
                 or plain L2 regression instead (False, as in ff_reinforce.py/
                 ff_qac.py) - see stoix/systems/ramdp_vpg/ff_ppo.py's module
                 docstring. ff_ppo_* systems only, see epochs.
  - use_layer_norm: LayerNorm inside the shared ACTStep of
                 AdaptiveComputationTimeTorso; cnn+mlp only (cnn+transformer,
                 cnn+gru, cnn+iru, and cnn+transformer_explicit_cot have no such
                 param, so this is forced off for them regardless of what's
                 requested).
  - use_input_layer_norm: LayerNorm on the encoded observation before the
                 initial token/state/recurrent-input projection; supported by
                 cnn+mlp/cnn+transformer/cnn+gru/cnn+iru, not yet by
                 cnn+transformer_explicit_cot (forced off).
  - num_layers:  how many sub-layers are stacked inside each shared pondering
                 step (network.actor_network.pre_torso.num_layers) - Dense
                 layers for cnn+mlp, GRU cells for cnn+gru, IRU cells
                 for cnn+iru, transformer layers for
                 cnn+transformer (see
                 stoix/networks/torso_compute*.py); not swept for
                 cnn+transformer_explicit_cot, which keeps its own yaml default
                 (2) instead. Default 1.
  - num_heads:   attention head count (network.actor_network.pre_torso.num_heads);
                 only applies to cnn+transformer
                 (TransformerChainOfThoughtTorso) and cnn+transformer_explicit_cot
                 (TransformerExplicitCoTTorso) - every other architecture has
                 no such param, so this is forced to a single value for them.
                 Default 4.
  - mlp_dim:     transformer feedforward width (network.actor_network.pre_torso.mlp_dim);
                 same applicability as num_heads (cnn+transformer/
                 cnn+transformer_explicit_cot only). Default 256.
  - vocab_size:  thought-token vocabulary size (network.actor_network.pre_torso.vocab_size),
                 i.e. how many discrete "thought" classes TransformerExplicitCoTTorso can
                 emit before the extra "act now" class - see
                 stoix/networks/torso_compute_explicit_cot.py. cnn+transformer_explicit_cot
                 only (every other architecture, including cnn+transformer, has no such
                 param, so this is forced to a single value for them). Default 32 (the
                 network yaml default).
  - seed:        5 seeds per config by default

gamma defaults to 0.9999 (system.gamma), applied to every job (not swept), for
MinAtar's long horizons (same default as minatar_sweep.py) - override via
--gamma if needed. total_timesteps defaults to 2e7 (a reduced sweep budget)
rather than the 1e8 used for full runs -- re-run the winning config(s) at
full budget afterwards.

Jobs are scheduled across GPUs with a fixed number of concurrent runs per
GPU (a GPU "slot" queue + thread pool), each run pinned via
CUDA_VISIBLE_DEVICES and logged to its own file under <output-dir>/logs/.

Usage:
  python ramdp_experiments/minatar_fixed_budget_sweep.py --dry-run                # preview the grid
  python ramdp_experiments/minatar_fixed_budget_sweep.py --limit 6 --dry-run       # preview a slice
  python ramdp_experiments/minatar_fixed_budget_sweep.py                          # run the full sweep (all MinAtar games)
  python ramdp_experiments/minatar_fixed_budget_sweep.py --envs seaquest,breakout  # only these games
  python ramdp_experiments/minatar_fixed_budget_sweep.py --systems ff_reinforce --architectures cnn+mlp \\
      --envs seaquest --budget 1,8 --hidden-dim 16 --lr 3e-4 --seeds 1  # small pilot / debug run
  python ramdp_experiments/minatar_fixed_budget_sweep.py --delightful true,false \\
      --delightful-eta 1.0,3.0                                    # sweep delightful PG on/off
  python ramdp_experiments/minatar_fixed_budget_sweep.py --architectures cnn+mlp \\
      --use-layer-norm true,false --use-input-layer-norm true,false  # sweep LayerNorm options
  python ramdp_experiments/minatar_fixed_budget_sweep.py --systems ff_reinforce \\
      --architectures cnn+transformer_explicit_cot                    # explicit-CoT sweep
  python ramdp_experiments/minatar_fixed_budget_sweep.py --lr 1e-4,3e-4 --critic-lr 1e-3  # decoupled lr sweeps
  python ramdp_experiments/minatar_fixed_budget_sweep.py --architectures cnn+gru,cnn+iru  # recurrent-block sweep
  python ramdp_experiments/minatar_fixed_budget_sweep.py --systems ff_ppo_fac,ff_ppo_naive,ff_ppo_reinforce \\
      --epochs 4 --num-minibatches 8,16 --clip-eps 0.1,0.2                 # PPO sweep
  python ramdp_experiments/minatar_fixed_budget_sweep.py --systems ff_ppo_fac \\
      --clip-value-loss true,false                       # PPO clipped vs. L2 critic loss
  python ramdp_experiments/minatar_fixed_budget_sweep.py --systems ff_ppo_cond_naive,ff_ppo_cond_fac \\
      --architectures cnn+mlp                     # compare conditioned Q-V variants, same capacity
  python ramdp_experiments/minatar_fixed_budget_sweep.py \\
      --systems ff_ppo_explicit_fac,ff_ppo_explicit_reinforce   # explicit-CoT PPO sweep
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
from typing import List

REPO_ROOT = Path(__file__).resolve().parent.parent

SYSTEM_TO_SCRIPT = {
    "ff_reinforce": "stoix/systems/ramdp_vpg/ff_reinforce.py",
    "ff_qac_fac": "stoix/systems/ramdp_vpg/ff_qac.py",
    "ff_qac_naive": "stoix/systems/ramdp_vpg/ff_qac.py",
    # PPO (ff_ppo.py) reuses the same "qac_variant" config knob as ff_qac.py to
    # select its advantage estimator, extended with a "reinforce" (G - V)
    # value - see SYSTEM_TO_QAC_VARIANT and stoix/systems/ramdp_vpg/ff_ppo.py's
    # module docstring.
    "ff_ppo_fac": "stoix/systems/ramdp_vpg/ff_ppo.py",
    "ff_ppo_naive": "stoix/systems/ramdp_vpg/ff_ppo.py",
    # cond_naive/cond_fac have no ff_qac.py equivalent - they condition the
    # critic's q_value on compute_time as an extra input (a learned linear
    # feature, see stoix.networks.base_qac.ValueAndQCritic._q_input) instead
    # of "naive"'s output-shaped table or "fac"'s analytic scaling, and only
    # exist for ff_ppo.py (see its module docstring).
    "ff_ppo_cond_naive": "stoix/systems/ramdp_vpg/ff_ppo.py",
    "ff_ppo_cond_fac": "stoix/systems/ramdp_vpg/ff_ppo.py",
    "ff_ppo_reinforce": "stoix/systems/ramdp_vpg/ff_ppo.py",
    # Explicit-CoT PPO (TransformerExplicitCoTTorso instead of a latent-CoT
    # torso) - one system per qac_variant, mirroring ff_ppo_fac/ff_ppo_naive/
    # ff_ppo_cond_naive/ff_ppo_cond_fac/ff_ppo_reinforce above. Architecture is
    # implied (always cnn+transformer_explicit_cot) rather than selected via
    # --architectures - see EXPLICIT_COT_PPO_SYSTEMS/build_grid.
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
# Systems whose critic is a Q-V critic (a value head and a Q head, using
# separate torsos - see ARCH_TO_NETWORK/EXPLICIT_COT_NETWORK_BY_SYSTEM) rather
# than the plain V-only critic ff_reinforce.py/ff_ppo_reinforce/
# ff_ppo_explicit_reinforce use - every SYSTEM_TO_QAC_VARIANT entry except the
# "reinforce" qac_variant. Used in Job.command() to target the right critic
# network override keys for CNN architectures.
QAC_CRITIC_SYSTEMS = tuple(s for s, v in SYSTEM_TO_QAC_VARIANT.items() if v != "reinforce")
# Systems trained by ff_ppo.py/ff_ppo_explicit_cot.py (PPO's clipped
# surrogate, several epochs of minibatch updates per rollout) rather than
# ff_reinforce.py/ff_qac.py (a single REINFORCE/QAC gradient step per
# rollout) - these get the epochs/num_minibatches/clip_eps axes (see
# Job.command()) and don't support system.delightful (neither script has
# that knob - see ff_ppo.py's module docstring for why).
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
# ff_ppo_explicit_* systems whose architecture is implied (always
# cnn+transformer_explicit_cot) rather than picked via --architectures - see
# build_grid.
EXPLICIT_COT_PPO_SYSTEMS = (
    "ff_ppo_explicit_fac",
    "ff_ppo_explicit_naive",
    "ff_ppo_explicit_cond_naive",
    "ff_ppo_explicit_cond_fac",
    "ff_ppo_explicit_reinforce",
)
ARCH_TO_NETWORK = {
    "ff_reinforce": {
        "cnn+mlp": "cnn_mlp_compute",
        "cnn+transformer": "cnn_transformer_compute",
        "cnn+gru": "cnn_gru_compute",
        "cnn+iru": "cnn_iru_compute",
    },
    # Every Q-V system uses SeparateValueAndQCritic (the "_separate_qv" network
    # variant): V and Q each get their own torso, independently initialised,
    # so neither head's gradient shares parameters with the other - see
    # base_qac.py's module docstring for the shared- vs separate-torso tradeoff.
    "ff_qac_fac": {
        "cnn+mlp": "cnn_mlp_compute_qac_separate_qv",
        "cnn+transformer": "cnn_transformer_compute_qac_separate_qv",
        "cnn+gru": "cnn_gru_compute_qac_separate_qv",
        "cnn+iru": "cnn_iru_compute_qac_separate_qv",
    },
    "ff_qac_naive": {
        "cnn+mlp": "cnn_mlp_compute_qac_separate_qv",
        "cnn+transformer": "cnn_transformer_compute_qac_separate_qv",
        "cnn+gru": "cnn_gru_compute_qac_separate_qv",
        "cnn+iru": "cnn_iru_compute_qac_separate_qv",
    },
}
# ff_ppo_fac/ff_ppo_naive use the same separate-torso Q-V critic
# (SeparateValueAndQCritic) as ff_qac_fac/ff_qac_naive; ff_ppo_cond_naive/
# ff_ppo_cond_fac condition that same critic on compute_time in code (see
# ValueAndQCritic._q_input) rather than via a different network yaml, so
# they reuse ff_ppo_fac's network too; ff_ppo_reinforce uses the same plain
# V-only critic as ff_reinforce - so they all reuse those networks'
# (arch -> network) mappings rather than duplicating them.
ARCH_TO_NETWORK["ff_ppo_fac"] = ARCH_TO_NETWORK["ff_qac_fac"]
ARCH_TO_NETWORK["ff_ppo_naive"] = ARCH_TO_NETWORK["ff_qac_naive"]
ARCH_TO_NETWORK["ff_ppo_cond_naive"] = ARCH_TO_NETWORK["ff_ppo_fac"]
ARCH_TO_NETWORK["ff_ppo_cond_fac"] = ARCH_TO_NETWORK["ff_ppo_fac"]
ARCH_TO_NETWORK["ff_ppo_reinforce"] = ARCH_TO_NETWORK["ff_reinforce"]
# Architectures whose pre_torso has no `use_layer_norm` param - only
# `use_input_layer_norm` - unlike AdaptiveComputationTimeTorso (which has
# both): TransformerChainOfThoughtTorso, GRUAdaptiveComputationTimeTorso, and
# IRUAdaptiveComputationTimeTorso. Used to pick the right LayerNorm overrides
# in Job.command().
NO_LAYER_NORM_ARCHES = (
    "cnn+transformer",
    "cnn+gru",
    "cnn+iru",
)
# TransformerExplicitCoTTorso (see stoix/networks/torso_compute_explicit_cot.py)
# doesn't fit ARCH_TO_NETWORK/SYSTEM_TO_SCRIPT's (system, arch) -> network lookup:
# it's only trained by a dedicated script per system (ff_reinforce_explicit_cot.py
# for ff_reinforce, ff_ppo_explicit_cot.py for ff_ppo_explicit_*), not the plain
# ff_reinforce.py/ff_ppo.py, so it's handled separately. Its architecture is
# always cnn+transformer_explicit_cot (CNNTorso input_layer feeding the same
# torso, see cnn_transformer_explicit_cot*.yaml) - listed in CNN_ARCHES below
# so the CNN input-layer handling in Job.command() applies uniformly.
EXPLICIT_COT_ARCH = "cnn+transformer_explicit_cot"
EXPLICIT_COT_SCRIPT_BY_SYSTEM = {"ff_reinforce": "stoix/systems/ramdp_vpg/ff_reinforce_explicit_cot.py"}
EXPLICIT_COT_SCRIPT_BY_SYSTEM.update(
    (system, "stoix/systems/ramdp_vpg/ff_ppo_explicit_cot.py") for system in EXPLICIT_COT_PPO_SYSTEMS
)
# ff_ppo_explicit_fac/naive/cond_naive/cond_fac use the separate-torso Q-V
# critic network; ff_ppo_explicit_reinforce/ff_reinforce use the plain V-only
# network - see ff_ppo_explicit_cot.py's/ff_reinforce_explicit_cot.py's
# learner_setup.
EXPLICIT_COT_NETWORK_BY_SYSTEM = {
    "ff_reinforce": "cnn_transformer_explicit_cot",
    "ff_ppo_explicit_fac": "cnn_transformer_explicit_cot_qac_separate_qv",
    "ff_ppo_explicit_naive": "cnn_transformer_explicit_cot_qac_separate_qv",
    "ff_ppo_explicit_cond_naive": "cnn_transformer_explicit_cot_qac_separate_qv",
    "ff_ppo_explicit_cond_fac": "cnn_transformer_explicit_cot_qac_separate_qv",
    "ff_ppo_explicit_reinforce": "cnn_transformer_explicit_cot",
}
EXPLICIT_COT_SYSTEMS = tuple(EXPLICIT_COT_SCRIPT_BY_SYSTEM)
# Every supported architecture consumes the raw grid observation through a
# CNNTorso input_layer (see Job.command()) - this is also the full list of
# valid --architectures values.
CNN_ARCHES = ("cnn+mlp", "cnn+transformer", "cnn+gru", "cnn+iru", EXPLICIT_COT_ARCH)
# Architectures whose pre_torso has `num_heads`/`mlp_dim` params (attention
# heads / transformer feedforward width) - TransformerChainOfThoughtTorso and
# TransformerExplicitCoTTorso; every other torso has no such concept. Used to
# pick whether --num-heads/--mlp-dim are swept for a given architecture in
# build_grid() and applied in Job.command().
TRANSFORMER_ARCHES = ("cnn+transformer",)
MINATAR_GAMES = (
    "asterix",
    "breakout",
    "freeway",
    "seaquest",
    "space_invaders",
)

# --server <key> -> modules to `module load` (in order) before each job's python
# process. `module` is a shell function, not a binary, so jobs run under this
# option are launched via `bash -lc "module load ... && exec <cmd>"` instead of
# the normal argv-list subprocess.run (see run_job()).
SERVER_MODULES = {
    "vulcan": ["StdEnv/2023", "cuda/12.2"],
}

@dataclass
class Job:
    env: str
    system: str
    arch: str
    budget: int
    hidden_dim: int
    lr: float
    critic_lr: float
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
    seed: int
    total_timesteps: float
    total_num_envs: int
    rollout_length: int
    gamma: float
    output_dir: Path
    wandb: bool
    wandb_project: str

    @property
    def group_tag(self) -> str:
        """Abbreviated form of run_name with the seed suffix omitted -
        identifies the hyperparameter setting shared by every seed of a
        config, for W&B grouping (logger.loggers.wandb.group_tag) and
        manifest.jsonl. Uses short axis prefixes (b/hd/lr/clr/nl/nh/md/ep/
        mb/clip/deta/ln/iln) rather than run_name's full field names, since
        W&B's group field gets unwieldy at run_name's length."""
        system_short = self.system.removeprefix("ff_")
        name = (
            f"{self.env}-{system_short}-{self.arch}-b{self.budget}"
            f"-hd{self.hidden_dim}-lr{self.lr:g}-clr{self.critic_lr:g}"
        )
        # Not shown for transformer_explicit_cot - num_layers isn't swept for
        # that arch (it keeps its own yaml default), see build_grid.
        if self.arch != EXPLICIT_COT_ARCH:
            name += f"-nl{self.num_layers}"
        # Only shown for architectures with num_heads/mlp_dim params - see
        # TRANSFORMER_ARCHES/build_grid.
        if self.arch in TRANSFORMER_ARCHES or self.arch == EXPLICIT_COT_ARCH:
            name += f"-nh{self.num_heads}-md{self.mlp_dim}"
        # Only shown for cnn+transformer_explicit_cot - vocab_size doesn't
        # exist on any other architecture, see EXPLICIT_COT_ARCH/build_grid.
        if self.arch == EXPLICIT_COT_ARCH:
            name += f"-vs{self.vocab_size}"
        # Only shown for PPO systems - epochs/num_minibatches/clip_eps don't
        # exist for ff_reinforce.py/ff_qac.py (single gradient step per
        # rollout, no clipped ratio) - see PPO_SYSTEMS/Job.command().
        if self.system in PPO_SYSTEMS:
            name += f"-ep{self.epochs}-mb{self.num_minibatches}-clip{self.clip_eps:g}"
            if not self.clip_value_loss:
                name += "-l2c"
        if self.delightful:
            name += f"-deta{self.delightful_eta:g}"
        if self.use_layer_norm:
            name += "-ln"
        if self.use_input_layer_norm:
            name += "-iln"
        return name

    @property
    def run_name(self) -> str:
        return f"{self.group_tag}-seed_{self.seed}"

    def command(self, python_bin: str) -> List[str]:
        if self.arch == EXPLICIT_COT_ARCH:
            script = EXPLICIT_COT_SCRIPT_BY_SYSTEM[self.system]
            network = EXPLICIT_COT_NETWORK_BY_SYSTEM[self.system]
        else:
            script = SYSTEM_TO_SCRIPT[self.system]
            network = ARCH_TO_NETWORK[self.system][self.arch]
        cmd = [
            python_bin,
            script,
            f"env=gymnax/{self.env}",
            f"network={network}",
            f"system.gamma={self.gamma:g}",
            f"arch.total_timesteps={self.total_timesteps:g}",
            f"arch.total_num_envs={self.total_num_envs}",
            f"arch.seed={self.seed}",
            "arch.num_evaluation=50",
            f"network.actor_network.pre_torso.hidden_dim={self.hidden_dim}",
            f"++network.actor_network.pre_torso.num_layers={self.num_layers}",
            # min_steps == max_steps == budget: no adaptivity, every example always
            # takes exactly `budget` steps (see stoix/networks/torso_compute*.py).
            f"network.actor_network.pre_torso.max_steps={self.budget}",
            f"network.actor_network.pre_torso.min_steps={self.budget}",
            f"system.actor_lr={self.lr:g}",
            f"system.critic_lr={self.critic_lr:g}",
            f"system.ent_coef=0.01",
            f"system.rollout_length={self.rollout_length}",
            f"logger.base_exp_path={self.output_dir / self.run_name}",
        ]
        if self.system in PPO_SYSTEMS:
            # epochs/num_minibatches/clip_eps: PPO-specific hyperparameters,
            # not present in ff_reinforce.py/ff_qac.py's system config - see
            # ff_ppo.yaml. ff_ppo.py also has no system.delightful knob (see
            # its module docstring), so that override is skipped here.
            cmd.append(f"system.epochs={self.epochs}")
            cmd.append(f"system.num_minibatches={self.num_minibatches}")
            cmd.append(f"system.clip_eps={self.clip_eps:g}")
            cmd.append(f"system.clip_value_loss={self.clip_value_loss}")
        else:
            cmd.append(f"system.delightful={self.delightful}")
        if self.arch in TRANSFORMER_ARCHES or self.arch == EXPLICIT_COT_ARCH:
            # Attention head count / feedforward width - TransformerChainOfThoughtTorso/
            # TransformerExplicitCoTTorso only (see TRANSFORMER_ARCHES).
            cmd.append(f"++network.actor_network.pre_torso.num_heads={self.num_heads}")
            cmd.append(f"++network.actor_network.pre_torso.mlp_dim={self.mlp_dim}")
        if self.arch == EXPLICIT_COT_ARCH:
            # Thought-token vocabulary size - TransformerExplicitCoTTorso only,
            # no other architecture has this param.
            cmd.append(f"++network.actor_network.pre_torso.vocab_size={self.vocab_size}")
        if self.wandb:
            # Fixed project name (not derived per-job) so every job in the
            # sweep lands in the same W&B project. run_id is pinned to
            # run_name (rather than left to WandBLogger's timestamp-based
            # unique_token) so two jobs launched in the same second can't
            # collide on the same W&B run.
            cmd.append("logger.loggers.wandb.enabled=True")
            cmd.append(f"logger.loggers.wandb.project={self.wandb_project}")
            # Single-element list: WandBLogger joins group_tag with "_" into
            # W&B's group field (see stoix/utils/logger.py), so every seed of
            # this hyperparameter setting groups together in the W&B UI.
            cmd.append(f"logger.loggers.wandb.group_tag=[{self.group_tag}]")
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
        # Every supported architecture consumes the raw grid observation through
        # a CNNTorso input_layer (see CNN_ARCHES).
        cmd.append(f"network.actor_network.input_layer.channel_sizes=[4]")
        cmd.append(f"network.actor_network.input_layer.kernel_sizes=[3]")
        cmd.append(f"network.actor_network.input_layer.strides=[1]")
        cmd.append(f"network.actor_network.input_layer.hidden_sizes=[{self.hidden_dim}]")
        if self.system in QAC_CRITIC_SYSTEMS:
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

    # (epochs, num_minibatches, clip_eps, clip_value_loss) combos: only meaningful
    # for PPO_SYSTEMS (ff_ppo.py) - forced to the first requested value for every
    # other system in the main product loop below, then deduplicated by run_name,
    # mirroring how delightful_combos/num_heads_options collapse axes that don't
    # apply.
    ppo_combos = list(
        itertools.product(args.epochs, args.num_minibatches, args.clip_eps, args.clip_value_loss)
    )

    # (system, arch, use_layer_norm, use_input_layer_norm, num_layers, num_heads,
    # mlp_dim) combos:
    #  - cnn+transformer_explicit_cot only exists for system in
    #    EXPLICIT_COT_SYSTEMS - any other requested (system, architecture) pair
    #    is skipped rather than erroring, so e.g. the default
    #    `--systems ff_reinforce,ff_qac_fac,ff_qac_naive` still works if the
    #    user adds `--architectures ...,cnn+transformer_explicit_cot`.
    #  - ff_ppo_explicit_*'s architecture is implied (always
    #    cnn+transformer_explicit_cot, see EXPLICIT_COT_PPO_SYSTEMS) rather than
    #    selected via --architectures, so it's forced here regardless of what
    #    --architectures requests, mirroring how the other unsupported axes
    #    below are forced to a single value.
    #  - use_layer_norm only exists on AdaptiveComputationTimeTorso (cnn+mlp).
    #  - use_input_layer_norm exists on cnn+mlp/cnn+transformer/cnn+gru/cnn+iru,
    #    not yet on cnn+transformer_explicit_cot.
    #  - num_layers (sub-layers stacked inside each shared pondering step,
    #    see stoix/networks/torso_compute*.py) is swept for every arch except
    #    cnn+transformer_explicit_cot, which keeps its own yaml default instead
    #    (see --num-layers help).
    #  - num_heads (attention heads) and mlp_dim (transformer feedforward width)
    #    only exist on cnn+transformer/cnn+transformer_explicit_cot (see
    #    TRANSFORMER_ARCHES) - swept only for those, everything else forced to a
    #    single value.
    #  - vocab_size (thought-token vocabulary size) only exists on
    #    cnn+transformer_explicit_cot (see EXPLICIT_COT_ARCH) - swept only for
    #    that architecture, everything else (including cnn+transformer) forced
    #    to a single value.
    # Unsupported axes are forced to a single default value rather than
    # needlessly duplicated per requested setting.
    system_arch_ln_combos = []
    n_skipped_incompatible = 0
    for system in args.systems:
        archs = (EXPLICIT_COT_ARCH,) if system in EXPLICIT_COT_PPO_SYSTEMS else args.architectures
        for arch in archs:
            is_transformer_arch = arch in TRANSFORMER_ARCHES or arch == EXPLICIT_COT_ARCH
            num_heads_options = args.num_heads if is_transformer_arch else [args.num_heads[0]]
            mlp_dim_options = args.mlp_dim if is_transformer_arch else [args.mlp_dim[0]]
            vocab_size_options = (
                args.vocab_size if arch == EXPLICIT_COT_ARCH else [args.vocab_size[0]]
            )
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
                        for mlp_dim in mlp_dim_options:
                            for vocab_size in vocab_size_options:
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
                                    )
                                )
    system_arch_ln_combos = list(dict.fromkeys(system_arch_ln_combos))
    if n_skipped_incompatible:
        print(
            f"Skipping {n_skipped_incompatible} (system, architecture) combo(s) requesting "
            f"{EXPLICIT_COT_ARCH}, which is only implemented for {EXPLICIT_COT_SYSTEMS}."
        )

    jobs = []
    for (
        env,
        (
            system,
            arch,
            use_layer_norm,
            use_input_layer_norm,
            num_layers,
            num_heads,
            mlp_dim,
            vocab_size,
        ),
        budget,
        hidden_dim,
        lr,
        critic_lr,
        (delightful, delightful_eta),
        (epochs, num_minibatches, clip_eps, clip_value_loss),
        seed,
    ) in itertools.product(
        args.envs,
        system_arch_ln_combos,
        args.budget,
        args.hidden_dim,
        args.lr,
        args.critic_lr,
        delightful_combos,
        ppo_combos,
        range(args.seeds),
    ):
        # Neither axis applies to both kinds of system at once (see
        # PPO_SYSTEMS) - force the inapplicable one to its default so a sweep
        # over both doesn't multiply out into identical duplicate jobs.
        if system in PPO_SYSTEMS:
            delightful, delightful_eta = False, args.delightful_eta[0]
        else:
            epochs, num_minibatches, clip_eps, clip_value_loss = ppo_combos[0]
        jobs.append(
            Job(
                env=env,
                system=system,
                arch=arch,
                budget=budget,
                hidden_dim=hidden_dim,
                lr=lr,
                critic_lr=critic_lr,
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

    # Collapsing the inapplicable axis above (delightful for PPO_SYSTEMS,
    # epochs/num_minibatches/clip_eps for everything else) can produce
    # duplicate run_names (e.g. sweeping --delightful true,false together with
    # a PPO system) - dedupe rather than schedule two jobs that would write to
    # the same output directory.
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
        # JAX preallocates this fraction of the *visible* GPU's memory per process.
        # With `runs_per_gpu` processes sharing one physical GPU, each must be capped
        # to roughly 1/runs_per_gpu of the GPU or the later processes to allocate OOM.
        "XLA_PYTHON_CLIENT_MEM_FRACTION": str(mem_fraction),
        "XLA_FLAGS": "--xla_gpu_autotune_level=0",
    }
    import os

    full_env = os.environ.copy()
    full_env.update(env)

    if server is not None:
        # `module` is a shell function (from Lmod's profile.d init scripts), not
        # a binary, so it can't be exec'd directly via an argv list - run it
        # through a login shell instead, which sources the init scripts that
        # define it. `exec` replaces the shell with the python process so
        # `proc.returncode` still reflects the actual job, not bash's.
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

    result = {
        **asdict(job),
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
        default=",".join(MINATAR_GAMES),
        help=f"Comma-separated subset of {{{','.join(MINATAR_GAMES)}}} (env=gymnax/<env>).",
    )
    parser.add_argument(
        "--systems",
        default="ff_reinforce,ff_qac_fac,ff_qac_naive",
        help="Comma-separated subset of {ff_reinforce, ff_qac_fac, ff_qac_naive, ff_ppo_fac, "
        "ff_ppo_naive, ff_ppo_cond_naive, ff_ppo_cond_fac, ff_ppo_reinforce, ff_ppo_explicit_fac, "
        "ff_ppo_explicit_naive, ff_ppo_explicit_cond_naive, ff_ppo_explicit_cond_fac, "
        "ff_ppo_explicit_reinforce}. The ff_ppo_* systems train stoix/systems/ramdp_vpg/ff_ppo.py "
        "(PPO's clipped surrogate, several epochs of minibatch updates per rollout) instead of a "
        "single REINFORCE/QAC gradient step per rollout - ff_ppo_fac/ff_ppo_naive use the same "
        "Q-V advantage as ff_qac_fac/ff_qac_naive; ff_ppo_cond_naive/ff_ppo_cond_fac condition "
        "the critic's Q on compute_time as an input instead (same architecture/parameter count "
        "for both, differing only in whether the conditioned output is also scaled by "
        "gamma^(c-1) - no ff_qac.py equivalent); ff_ppo_reinforce uses the same G-V "
        "(REINFORCE-with-baseline) advantage as ff_reinforce. See --epochs/--num-minibatches/"
        "--clip-eps (PPO-only; system.delightful is not supported by ff_ppo.py/"
        "ff_ppo_explicit_cot.py). The five ff_ppo_explicit_* systems mirror the five ff_ppo_* "
        "qac_variants exactly, but train stoix/systems/ramdp_vpg/ff_ppo_explicit_cot.py instead "
        "(explicit chain-of-thought tokens, TransformerExplicitCoTTorso) - their architecture is "
        "implied (cnn+transformer_explicit_cot) rather than picked via --architectures.",
    )
    parser.add_argument(
        "--architectures",
        default="cnn+mlp,cnn+transformer",
        help="Comma-separated subset of {cnn+mlp, cnn+transformer, cnn+gru, cnn+iru, "
        "cnn+transformer_explicit_cot}. Every architecture consumes the raw MinAtar grid "
        "observation through a CNNTorso input_layer (see CNN_ARCHES). "
        "cnn+transformer_explicit_cot (TransformerExplicitCoTTorso) is only "
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
        "--epochs",
        default="4",
        help="Comma-separated system.epochs values (PPO epochs per rollout), swept independently "
        "of --num-minibatches/--clip-eps (full cross product). Only applies to PPO systems "
        "(ff_ppo_fac, ff_ppo_naive, ff_ppo_cond_naive, ff_ppo_cond_fac, ff_ppo_reinforce) - "
        "ignored (forced to the first value) for ff_reinforce/ff_qac_*, which take one "
        "gradient step per rollout.",
    )
    parser.add_argument(
        "--num-minibatches",
        default="16",
        help="Comma-separated system.num_minibatches values (PPO minibatches per epoch), swept "
        "independently of --epochs/--clip-eps. PPO systems only, see --epochs.",
    )
    parser.add_argument(
        "--clip-eps",
        default="0.2",
        help="Comma-separated system.clip_eps values (PPO ratio/value clipping), swept "
        "independently of --epochs/--num-minibatches. PPO systems only, see --epochs.",
    )
    parser.add_argument(
        "--clip-value-loss",
        default="true",
        help="Comma-separated bools (true/false) - system.clip_value_loss: whether the critic's "
        "value/Q loss uses PPO-style clipping (True, default) against the old value/Q estimate, "
        "or plain L2 regression instead (False, as in ff_reinforce.py/ff_qac.py). Swept "
        "independently of --epochs/--num-minibatches/--clip-eps. PPO systems only, see --epochs.",
    )
    parser.add_argument(
        "--use-layer-norm",
        default="false",
        help="Comma-separated bools (true/false) - LayerNorm inside AdaptiveComputationTimeTorso's "
        "shared ACTStep (network.actor_network.pre_torso.use_layer_norm). Only applies to "
        "architecture in {cnn+mlp}; ignored (forced off) otherwise, since those torsos have "
        "no such param.",
    )
    parser.add_argument(
        "--use-input-layer-norm",
        default="false",
        help="Comma-separated bools (true/false) - LayerNorm on the raw observation before the "
        "initial token/state projection (network.actor_network.pre_torso.use_input_layer_norm). "
        "Supported by architecture in {cnn+mlp, cnn+transformer, cnn+gru, cnn+iru}; ignored "
        "(forced off) for cnn+transformer_explicit_cot.",
    )
    parser.add_argument(
        "--num-layers",
        default="1",
        help="Comma-separated ints - how many sub-layers are stacked inside each shared pondering "
        "step (network.actor_network.pre_torso.num_layers): Dense layers for cnn+mlp "
        "(AdaptiveComputationTimeTorso), GRU cells for cnn+gru (GRUAdaptiveComputationTimeTorso), "
        "IRU cells for cnn+iru (IRUAdaptiveComputationTimeTorso), or transformer layers for "
        "cnn+transformer (TransformerChainOfThoughtTorso) - see "
        "stoix/networks/torso_compute*.py. Default 1 sub-layer per step. Not swept for "
        "cnn+transformer_explicit_cot (TransformerExplicitCoTTorso keeps its own yaml default of 2).",
    )
    parser.add_argument(
        "--num-heads",
        default="4",
        help="Comma-separated ints - attention head count "
        "(network.actor_network.pre_torso.num_heads). Only applies to architecture in "
        "{cnn+transformer, cnn+transformer_explicit_cot} (TransformerChainOfThoughtTorso/"
        "TransformerExplicitCoTTorso); ignored (forced to the first value) for every other "
        "architecture, since those torsos have no such param. Default 4.",
    )
    parser.add_argument(
        "--mlp-dim",
        default="256",
        help="Comma-separated ints - transformer feedforward width "
        "(network.actor_network.pre_torso.mlp_dim). Same applicability as --num-heads: only "
        "architecture in {cnn+transformer, cnn+transformer_explicit_cot}; ignored "
        "(forced to the first value) for every other architecture. Default 256.",
    )
    parser.add_argument(
        "--vocab-size",
        default="32",
        help="Comma-separated ints - thought-token vocabulary size "
        "(network.actor_network.pre_torso.vocab_size): how many discrete 'thought' classes "
        "TransformerExplicitCoTTorso can emit before the extra 'act now' class - see "
        "stoix/networks/torso_compute_explicit_cot.py. Only applies to architecture "
        "cnn+transformer_explicit_cot (unlike --num-heads/--mlp-dim, NOT swept for "
        "cnn+transformer, which has no such param); ignored (forced to the first value) for "
        "every other architecture. Default 32 (the network yaml default).",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.9999,
        help="system.gamma, applied to every job (not swept). Matches minatar_sweep.py's default "
        "for MinAtar's long horizons.",
    )
    parser.add_argument(
        "--wandb",
        type=lambda x: x.strip().lower() in ("1", "true", "yes"),
        default=False,
        help="Enable W&B logging (logger.loggers.wandb.enabled) for every job. Default False.",
    )
    parser.add_argument(
        "--wandb-project",
        default="minatar_fixed_budget_sweep",
        help="W&B project name (logger.loggers.wandb.project), applied to every job so the whole "
        "sweep lands in the same project. Only used when --wandb is set.",
    )
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds per config, seeded 0..seeds-1.")
    parser.add_argument("--total-timesteps", type=float, default=2e7, help="arch.total_timesteps per run.")
    parser.add_argument(
        "--total-num-envs",
        type=int,
        default=1024,
        help="arch.total_num_envs, applied to every job (not swept). Total number of vectorised "
        "environments across all devices and batched updates.",
    )
    parser.add_argument(
        "--rollout-length",
        type=int,
        default=32,
        help="system.rollout_length, applied to every job (not swept). Number of environment "
        "steps per vectorised environment collected per rollout.",
    )
    parser.add_argument("--gpus", default="auto", help="Comma-separated GPU ids, or 'auto' to detect via nvidia-smi.")
    parser.add_argument("--runs-per-gpu", type=int, default=2, help="Concurrent runs per GPU.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results_minatar_fixed_budget_sweep",
        help="Where per-run logger.base_exp_path and logs/ + manifest.jsonl are written.",
    )
    parser.add_argument("--python", default=str(REPO_ROOT / ".venv" / "bin" / "python"), help="Python interpreter.")
    parser.add_argument(
        "--server",
        default=None,
        choices=sorted(SERVER_MODULES),
        help="If set, `module load` this server's required environment modules (see "
        "SERVER_MODULES) before each job's python process, e.g. --server vulcan loads "
        f"{SERVER_MODULES['vulcan']}. Jobs are then launched via `bash -lc` instead of "
        "directly, since `module` is a shell function, not a binary.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N jobs (for a pilot / sanity check).")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help=(
            "Skip jobs that already have a successful (returncode 0) entry in "
            "manifest.jsonl (default: on; use --no-skip-existing to force rerun)."
        ),
    )
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned jobs and exit without running anything.")
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation prompt before launching.")
    args = parser.parse_args()

    args.envs = args.envs.split(",")
    args.systems = args.systems.split(",")
    args.architectures = args.architectures.split(",")
    args.budget = [int(x) for x in args.budget.split(",")]
    args.hidden_dim = [int(x) for x in args.hidden_dim.split(",")]
    args.lr = [float(x) for x in args.lr.split(",")]
    args.critic_lr = [float(x) for x in args.critic_lr.split(",")]
    args.delightful = [x.strip().lower() in ("1", "true", "yes") for x in args.delightful.split(",")]
    args.delightful_eta = [float(x) for x in args.delightful_eta.split(",")]
    args.epochs = [int(x) for x in args.epochs.split(",")]
    args.num_minibatches = [int(x) for x in args.num_minibatches.split(",")]
    args.clip_eps = [float(x) for x in args.clip_eps.split(",")]
    args.clip_value_loss = [
        x.strip().lower() in ("1", "true", "yes") for x in args.clip_value_loss.split(",")
    ]
    args.use_layer_norm = [x.strip().lower() in ("1", "true", "yes") for x in args.use_layer_norm.split(",")]
    args.use_input_layer_norm = [
        x.strip().lower() in ("1", "true", "yes") for x in args.use_input_layer_norm.split(",")
    ]
    args.num_layers = [int(x) for x in args.num_layers.split(",")]
    args.num_heads = [int(x) for x in args.num_heads.split(",")]
    args.mlp_dim = [int(x) for x in args.mlp_dim.split(",")]

    for e in args.envs:
        assert e in MINATAR_GAMES, f"unknown env {e!r}, expected one of {list(MINATAR_GAMES)}"
    for s in args.systems:
        assert s in SYSTEM_TO_SCRIPT, f"unknown system {s!r}, expected one of {list(SYSTEM_TO_SCRIPT)}"
    valid_architectures = CNN_ARCHES
    for a in args.architectures:
        assert a in valid_architectures, f"unknown architecture {a!r}, expected one of {valid_architectures}"
    for b in args.budget:
        assert b >= 1, f"budget must be >= 1, got {b}"
    for n in args.num_layers:
        assert n >= 1, f"num_layers must be >= 1, got {n}"
    for n in args.num_heads:
        assert n >= 1, f"num_heads must be >= 1, got {n}"
    for d in args.mlp_dim:
        assert d >= 1, f"mlp_dim must be >= 1, got {d}"
    for e in args.epochs:
        assert e >= 1, f"epochs must be >= 1, got {e}"
    for m in args.num_minibatches:
        assert m >= 1, f"num_minibatches must be >= 1, got {m}"
    for c in args.clip_eps:
        assert c > 0, f"clip_eps must be > 0, got {c}"

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
    print(f"  budget (min_steps=max_steps)={args.budget} hidden_dim={args.hidden_dim} seeds=0..{args.seeds - 1}")
    print(f"  lr={args.lr} critic_lr={args.critic_lr}")
    print(f"  delightful={args.delightful} delightful_eta={args.delightful_eta} (not PPO_SYSTEMS)")
    print(
        f"  epochs={args.epochs} num_minibatches={args.num_minibatches} clip_eps={args.clip_eps} "
        f"clip_value_loss={args.clip_value_loss} (PPO systems only: {PPO_SYSTEMS})"
    )
    print(
        f"  use_layer_norm={args.use_layer_norm} (cnn+mlp only) "
        f"use_input_layer_norm={args.use_input_layer_norm} (not cnn+transformer_explicit_cot)"
    )
    print(f"  num_layers={args.num_layers} (not swept for cnn+transformer_explicit_cot)")
    print(
        f"  num_heads={args.num_heads} mlp_dim={args.mlp_dim} "
        f"(cnn+transformer/cnn+transformer_explicit_cot only)"
    )
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
            return run_job(
                job, gpu, args.python, log_dir, manifest_lock, manifest_path, mem_fraction, args.server
            )
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
