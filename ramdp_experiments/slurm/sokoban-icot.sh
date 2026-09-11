#!/bin/bash
#SBATCH --account=aip-schuurma
#SBATCH --time=11:59:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --array=1-1
#SBATCH --output=/home/chanb/scratch/logs/ramdp/Stoix/%x_%A_%a.out

module load StdEnv/2023
module load cuda/12.2

mkdir -p $SLURM_TMPDIR/tmp
export CUDA_MPS_LOG_DIRECTORY=$SLURM_TMPDIR/tmp
nvidia-cuda-mps-control -d

cd /home/chanb/research/iclr_2027/Stoix

echo "hostname: $(hostname)"
echo "starting at: $(date)"

project_name=sep11

python ramdp_experiments/jumanji_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 1,2,3,4,5 --seeds 1 --runs-per-gpu 1 --architectures cnn+transformer --hidden-dim 64 --mlp-dim 128 --num-layers 2 --num-heads 4 --total-timesteps 2e9 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --envs sokoban --sokoban-generator unfiltered-train --gpus 0,1 --rollout-length 20 --total-num-envs 128 --epochs 2 --num-minibatches 4 --ent-coef 0.01 --gamma 0.99 --actor-weight-decay 0.005 --critic-before-actor false --standardize-advantages true --use-input-layer-norm true --gae-lambda 0.95 --wandb true --wandb-project sokoban-${project_name} --output-dir /home/chanb/scratch/logs/ramdp/sokoban-icot-${project_name} --yes --no-skip-existing --server vulcan

python ramdp_experiments/jumanji_sweep.py --systems ff_ppo_reinforce --max-steps 5 --seeds 1 --runs-per-gpu 1 --architectures cnn+transformer --hidden-dim 64 --mlp-dim 128 --num-layers 2 --num-heads 4 --total-timesteps 2e9 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --envs sokoban --sokoban-generator unfiltered-train --gpus 2 --rollout-length 20 --total-num-envs 128 --epochs 2 --num-minibatches 4 --ent-coef 0.01 --gamma 0.99 --actor-weight-decay 0.005 --critic-before-actor false --standardize-advantages true --use-input-layer-norm true --gae-lambda 0.95 --halting-ent-coef 0.01 --wandb true --wandb-project sokoban-${project_name} --output-dir /home/chanb/scratch/logs/ramdp/sokoban-icot-${project_name} --yes --no-skip-existing --server vulcan


wait

echo "finished with exit code $? at: $(date)"