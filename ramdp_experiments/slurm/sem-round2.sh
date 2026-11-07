#!/bin/bash
#SBATCH --account=aip-schuurma
#SBATCH --time=12:00:00
#SBATCH --mem=24GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=/home/chanb/scratch/logs/ramdp/Stoix/%x_%A_%a.out

# SEM round 2 (hyperparameters: ramdp_experiments/sem_round2_hparams.md), 2 runs per GPU.
# Usage: sbatch [--time=...] [--array=...] sem-round2.sh <env> <budgets|adaptive> [seeds]
#   env: lightsout | slidingtile_bank | sokoban | minatar (one array task per game, --array=0-4)
#   budgets: comma-separated fixed budgets (e.g. 1,2,3), or "adaptive" (learned halting, 1-5 steps)
# SMOKE=1 runs 5 of the 50 iterations (1/10 of total_timesteps), no wandb, to a separate dir.
# Optional env: OUT (output dir), SEM_PROJECT (wandb project), EXTRA_ARGS (sweep flags, override the above).

env=${1:?usage: $0 <env> <budgets|adaptive> [seeds]}
budget=${2:?usage: $0 <env> <budgets|adaptive> [seeds]}
seeds=${3:-5}

module load StdEnv/2023
module load cuda/12.2

mkdir -p $SLURM_TMPDIR/tmp
export CUDA_MPS_LOG_DIRECTORY=$SLURM_TMPDIR/tmp
nvidia-cuda-mps-control -d

cd /home/chanb/research/iclr_2027/Stoix

if [ "${env}" = "minatar" ]; then
    games=(asterix breakout freeway seaquest space_invaders)
    env=${games[$SLURM_ARRAY_TASK_ID]}
fi

timesteps=1e8; gamma=0.999; mgn=1.0
case ${env} in
    sokoban) timesteps=3e8; mgn=5.0 ;;
    asterix|breakout|freeway|seaquest|space_invaders) gamma=0.99999 ;;
esac

out=${OUT:-/home/chanb/scratch/logs/ramdp/sem-round2}
wandb_args="--wandb true --wandb-project ${SEM_PROJECT:-stoix-sem}"
if [ "${SMOKE:-0}" = "1" ]; then
    export NUM_EVALUATION=5
    timesteps=$(awk "BEGIN{print ${timesteps} / 10}")
    out=${out}-smoke
    wandb_args="--wandb false"
fi

echo "hostname: $(hostname)"
echo "env: ${env} budget: ${budget} seeds: ${seeds} timesteps: ${timesteps} smoke: ${SMOKE:-0}"
echo "starting at: $(date)"

if [ "${env}" = "lightsout" ]; then
    prefix=lightsout
    env_args="--architectures transformer --grid-sizes 5x4 --episode-length 10 --eval-episode-length 20 --difficulty-threshold 0.5 --num-state-tokens 8 --critic-layer-sizes 256,256,256 --critic-activation gelu"
else
    prefix=jumanji
    env_args="--envs ${env} --architectures cnn+transformer --cnn-tokens true --slidingtile-grid-size 4 --slidingtile-time-limit 60 --slidingtile-eval-time-limit 75 --sokoban-generator unfiltered-train --sokoban-eval-generator unfiltered-valid"
fi

if [ "${budget}" = "adaptive" ]; then
    sweep_args="ramdp_experiments/${prefix}_sweep.py --min-steps 1 --max-steps 5 --halting-lr 1e-4 --halting-weight-decay 0 --halting-ent-coef 0.01 --halting-temperature 1.0 --halting-hidden-dims none --stop-gradient-halting-input false"
else
    sweep_args="ramdp_experiments/${prefix}_fixed_budget_sweep.py --budget ${budget}"
fi

python ${sweep_args} ${env_args} --systems ff_ppo_reinforce --seeds ${seeds} --runs-per-gpu 2 --gpus 0 \
    --total-timesteps ${timesteps} --total-num-envs 1024 --rollout-length 64 --gamma ${gamma} --max-grad-norm ${mgn} \
    --lr 1e-4 --critic-lr 1e-4 --actor-weight-decay 0 --critic-weight-decay 0 \
    --epochs 2 --num-minibatches 64 --clip-eps 0.3 --clip-value-loss false --use-expectile-value-loss false \
    --vf-coef 1.0 --ent-coef 0.01 --gae-lambda 0.95 --standardize-advantages true --recompute-advantages false \
    --critic-before-actor false --use-dpo-loss false --latent-kl-coef 0 --clip-halting-head true \
    --hidden-dim 128 --mlp-dim 512 --num-layers 2 --num-heads 4 --use-input-layer-norm true --use-rmsnorm true \
    --use-sandwich-norm false --sem-group-size 8 --latent-rms-norm true --input-injection true --residual-scaling true \
    --block-init trunc_normal ${wandb_args} --output-dir ${out} --server slurm --yes ${EXTRA_ARGS}

echo "finished with exit code $? at: $(date)"
