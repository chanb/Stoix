#!/bin/bash
set -euo pipefail

# Submits one slurm job per (gamma, line) combination for the lightsout-icot

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="${SCRIPT_DIR}/lightsout-icot-run.sh"

project_name=shallow_cnn
gammas=( 0.99 0.995 0.999 0.9995 )

COMMON="--seeds 10 --architectures transformer --hidden-dim 32 --mlp-dim 512 --qkv-dim 128 --num-layers 4 --num-heads 8 --total-timesteps 3e8 --grid-sizes 5x4 --episode-length 10 --eval-episode-length 20 --difficulty-threshold 0.5 --lr 3e-4 --critic-lr 3e-4 --epochs 4 --num-minibatches 8 --use-input-layer-norm true --clip-value-loss false --critic-before-actor false --ent-coef 0.01 --clip-eps 0.2 --actor-weight-decay 0.1 --critic-weight-decay 0.0 --gae-lambda 0.95 --standardize-advantages true --use-rmsnorm true --use-sandwich-norm false --wandb true --wandb-project lightsout-icot-${project_name} --output-dir /home/chanb/scratch/logs/ramdp/lightsout-icot-${project_name} --runs-per-gpu 6 --gpus 0 --yes --server vulcan --no-skip-existing --use-expectile-value-loss true"

# sbatch --export is comma-delimited, so a CMD containing literal commas
# (e.g. "--budget 1,2,3") gets silently truncated at the first comma.
# Base64-encode CMD before export and decode it in the run script to avoid this.
for gamma in "${gammas[@]}"; do
    CMD="python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_reinforce --gamma ${gamma} --budget 1,2 ${COMMON}"
    CMD_B64="$(base64 -w0 <<< "${CMD}")"
    sbatch --export=ALL,CMD_B64="${CMD_B64}" --job-name="lightsout-icot-budget12-gamma${gamma}" "${RUN_SCRIPT}"

    CMD="python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_reinforce --gamma ${gamma} --budget 3 ${COMMON}"
    CMD_B64="$(base64 -w0 <<< "${CMD}")"
    sbatch --export=ALL,CMD_B64="${CMD_B64}" --job-name="lightsout-icot-budget3-gamma${gamma}" "${RUN_SCRIPT}"

    CMD="python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_reinforce --gamma ${gamma} --budget 4 ${COMMON}"
    CMD_B64="$(base64 -w0 <<< "${CMD}")"
    sbatch --export=ALL,CMD_B64="${CMD_B64}" --job-name="lightsout-icot-budget4-gamma${gamma}" "${RUN_SCRIPT}"

    CMD="python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_reinforce --gamma ${gamma} --budget 5 ${COMMON}"
    CMD_B64="$(base64 -w0 <<< "${CMD}")"
    sbatch --export=ALL,CMD_B64="${CMD_B64}" --job-name="lightsout-icot-budget5-gamma${gamma}" "${RUN_SCRIPT}"

    CMD="python ramdp_experiments/lightsout_sweep.py --systems ff_ppo_reinforce --gamma ${gamma} --max-steps 5 ${COMMON} --halting-ent-coef 0.1 --halting-temperature 1.0 --clip-halting-head true  --stop-gradient-halting-input false --halting-lr 1e-3 --halting-weight-decay 0"
    CMD_B64="$(base64 -w0 <<< "${CMD}")"
    sbatch --export=ALL,CMD_B64="${CMD_B64}" --job-name="lightsout-icot-maxsteps5-halt_lr_0.001-gamma${gamma}" "${RUN_SCRIPT}"

    CMD="python ramdp_experiments/lightsout_sweep.py --systems ff_ppo_reinforce --gamma ${gamma} --max-steps 5 ${COMMON} --halting-ent-coef 0.1 --halting-temperature 1.0 --clip-halting-head true  --stop-gradient-halting-input false --halting-lr 1e-4 --halting-weight-decay 0"
    CMD_B64="$(base64 -w0 <<< "${CMD}")"
    sbatch --export=ALL,CMD_B64="${CMD_B64}" --job-name="lightsout-icot-maxsteps5-halt_lr_0.0001-gamma${gamma}" "${RUN_SCRIPT}"
done
