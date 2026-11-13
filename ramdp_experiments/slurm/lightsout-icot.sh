#!/bin/bash
#SBATCH --account=<ACCOUNT>
#SBATCH --time=23:59:00
#SBATCH --mem=24GB
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:2
#SBATCH --array=1-1
#SBATCH --output=<LOG_DIR>/logs/ramdp/Stoix/%x_%A_%a.out

module load StdEnv/2023
module load cuda/12.2

mkdir -p $SLURM_TMPDIR/tmp
export CUDA_MPS_LOG_DIRECTORY=$SLURM_TMPDIR/tmp
nvidia-cuda-mps-control -d

LOG_DIR=""
REPO_PATH=""
cd ${REPO_PATH}/Stoix

echo "hostname: $(hostname)"
echo "starting at: $(date)"

project_name=final

python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 1,2,3,4,5 --seeds 10 --architectures transformer --hidden-dim 128 --mlp-dim 256 --qkv-dim 256 --num-layers 2 --num-heads 8 --total-timesteps 3e8 --grid-sizes 5x4 --episode-length 10 --eval-episode-length 20 --difficulty-threshold 0.5 --lr 3e-4 --critic-lr 3e-4 --epochs 4 --num-minibatches 8 --use-input-layer-norm true --clip-value-loss false --critic-before-actor false --ent-coef 0.01 --clip-eps 0.2 --gamma 0.9995 --actor-weight-decay 0.1 --critic-weight-decay 0.0 --gae-lambda 0.95 --standardize-advantages true --use-rmsnorm true --use-sandwich-norm false --wandb true --wandb-project lightsout-icot-${project_name} --output-dir ${LOG_DIR}/logs/ramdp/lightsout-icot-${project_name} --runs-per-gpu 3 --gpus 0,1 --server slurm --yes --no-skip-existing --use-expectile-value-loss true

python ramdp_experiments/lightsout_sweep.py --systems ff_ppo_reinforce --max-steps 5 --seeds 10 --architectures transformer --hidden-dim 128 --mlp-dim 256 --qkv-dim 256 --num-layers 2 --num-heads 8 --total-timesteps 3e8 --grid-sizes 5x4 --episode-length 10 --eval-episode-length 20 --difficulty-threshold 0.5 --lr 3e-4 --critic-lr 3e-4 --epochs 4 --num-minibatches 8 --use-input-layer-norm true --clip-value-loss false --critic-before-actor false --ent-coef 0.01 --clip-eps 0.2 --gamma 0.99 --actor-weight-decay 0.1 --critic-weight-decay 0.0 --gae-lambda 0.95 --standardize-advantages true --use-rmsnorm true --use-sandwich-norm false --halting-ent-coef 0.01 --halting-temperature 5.0 --clip-halting-head true --stop-gradient-halting-input false --wandb true --wandb-project lightsout-icot-${project_name} --output-dir ${LOG_DIR}/logs/ramdp/lightsout-icot-${project_name} --runs-per-gpu 3 --gpus 0 --server slurm --yes --no-skip-existing --use-expectile-value-loss true

echo "finished with exit code $? at: $(date)"
