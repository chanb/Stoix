#!/bin/bash
set -euo pipefail

# Submits one slurm job per (seed, line) combination for the sokoban-icot
# "EXPECTILE LOSS" sweep, since each job can only run a single seed.
# 5 seeds x 4 lines = 20 jobs total.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="${SCRIPT_DIR}/sokoban-icot-run.sh"

project_name=no_cnn
seeds=(0 1 2 3 4)

COMMON="--seeds 1 --runs-per-gpu 1 --architectures cnn+transformer --hidden-dim 32 --qkv-dim 128 --mlp-dim 512 --num-layers 4 --num-heads 8 --total-timesteps 3e9 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --envs sokoban --sokoban-generator unfiltered-train --sokoban-eval-generator unfiltered-valid --gpus 0 --rollout-length 20 --total-num-envs 128 --epochs 2 --num-minibatches 4 --ent-coef 0.01 --gamma 0.999 --actor-weight-decay 0.01 --critic-before-actor false --standardize-advantages true --use-input-layer-norm true --use-rmsnorm true --gae-lambda 0.95 --max-grad-norm 5.0 --wandb true --wandb-project sokoban-${project_name} --output-dir /home/chanb/scratch/logs/ramdp/sokoban-icot-${project_name} --yes --no-skip-existing --server vulcan --use-expectile-value-loss true --expectile 0.9"

# sbatch --export is comma-delimited, so a CMD containing literal commas
# (e.g. "--budget 1,2,3") gets silently truncated at the first comma.
# Base64-encode CMD before export and decode it in the run script to avoid this.
for seed in "${seeds[@]}"; do
    CMD="python ramdp_experiments/jumanji_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 1,2,3 --base-seed ${seed} ${COMMON}"
    CMD_B64="$(base64 -w0 <<< "${CMD}")"
    sbatch --export=ALL,CMD_B64="${CMD_B64}" --job-name="sokoban-icot-budget123-seed${seed}" "${RUN_SCRIPT}"

    CMD="python ramdp_experiments/jumanji_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 4 --base-seed ${seed} ${COMMON}"
    CMD_B64="$(base64 -w0 <<< "${CMD}")"
    sbatch --export=ALL,CMD_B64="${CMD_B64}" --job-name="sokoban-icot-budget4-seed${seed}" "${RUN_SCRIPT}"

    CMD="python ramdp_experiments/jumanji_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 5 --base-seed ${seed} ${COMMON}"
    CMD_B64="$(base64 -w0 <<< "${CMD}")"
    sbatch --export=ALL,CMD_B64="${CMD_B64}" --job-name="sokoban-icot-budget5-seed${seed}" "${RUN_SCRIPT}"

    CMD="python ramdp_experiments/jumanji_sweep.py --systems ff_ppo_reinforce --max-steps 5 --base-seed ${seed} ${COMMON} --halting-ent-coef 0.1 --halting-temperature 1.0 --clip-halting-head true  --stop-gradient-halting-input false --halting-lr 1e-3 --halting-weight-decay 0"
    CMD_B64="$(base64 -w0 <<< "${CMD}")"
    sbatch --export=ALL,CMD_B64="${CMD_B64}" --job-name="sokoban-icot-maxsteps5-halt_lr_0.001-seed${seed}" "${RUN_SCRIPT}"

    CMD="python ramdp_experiments/jumanji_sweep.py --systems ff_ppo_reinforce --max-steps 5 --base-seed ${seed} ${COMMON} --halting-ent-coef 0.1 --halting-temperature 1.0 --clip-halting-head true  --stop-gradient-halting-input false --halting-lr 1e-4 --halting-weight-decay 0"
    CMD_B64="$(base64 -w0 <<< "${CMD}")"
    sbatch --export=ALL,CMD_B64="${CMD_B64}" --job-name="sokoban-icot-maxsteps5-halt_lr_0.0001-seed${seed}" "${RUN_SCRIPT}"
done
