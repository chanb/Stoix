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
vocab_size=1,2,4,8,16

python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_explicit_reinforce --budget 1,2,3,4,5 --seeds 10 --vocab-size=${vocab_size} --architectures transformer_explicit_cot --hidden-dim 128 --mlp-dim 256 --qkv-dim 256 --num-layers 2 --num-heads 8 --total-timesteps 3e8 --grid-sizes 5x4 --episode-length 10 --eval-episode-length 20 --difficulty-threshold 0.5 --lr 3e-4 --critic-lr 3e-4 --epochs 4 --num-minibatches 8 --use-input-layer-norm true --use-rmsnorm true --ent-coef 0.05 --clip-eps 0.2 --gamma 0.999 --actor-weight-decay 0.1 --gae-lambda 0.95 --standardize-advantages true --wandb true --wandb-project lightsout-ecot-${project_name} --output-dir ${LOG_DIR}/logs/ramdp/lightsout-ecot-${project_name} --runs-per-gpu 3 --gpus 0,1 --no-skip-existing --server slurm --yes --use-expectile-value-loss true

python ramdp_experiments/lightsout_sweep.py --systems ff_ppo_explicit_reinforce --max-steps 5 --seeds 10 --vocab-size=${vocab_size} --architectures transformer_explicit_cot --hidden-dim 128 --mlp-dim 256 --qkv-dim 256 --num-layers 2 --num-heads 8 --total-timesteps 3e8 --grid-sizes 5x4 --episode-length 10 --eval-episode-length 20 --difficulty-threshold 0.5 --lr 3e-4 --critic-lr 3e-4 --epochs 4 --num-minibatches 8 --use-input-layer-norm true --use-rmsnorm true --ent-coef 0.01 --clip-eps 0.2 --gamma 0.999 --actor-weight-decay 0.1 --gae-lambda 0.95 --standardize-advantages true --wandb true --wandb-project lightsout-ecot-${project_name} --output-dir ${LOG_DIR}/logs/ramdp/lightsout-ecot-${project_name} --runs-per-gpu 3 --gpus 0,1 --no-skip-existing --server slurm --yes --use-expectile-value-loss true

echo "finished with exit code $? at: $(date)"
