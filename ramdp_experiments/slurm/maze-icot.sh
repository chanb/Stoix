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

python ramdp_experiments/jumanji_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 1,2,3,4,5 --seeds 3 --runs-per-gpu 3 --architectures cnn+transformer --hidden-dim 64 --mlp-dim 128 --num-layers 2 --num-heads 4 --total-timesteps 1e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --envs maze --maze-size 10 --gpus 0,1 --rollout-length 10 --total-num-envs 128 --epochs 2 --num-minibatches 2 --ent-coef 0.01 --gamma 0.99 --standardize-advantages true --gae-lambda 0.95 --wandb true --wandb-project maze-${project_name} --output-dir /home/chanb/scratch/logs/ramdp/maze-icot-${project_name} --yes --no-skip-existing --server vulcan

python ramdp_experiments/jumanji_sweep.py --systems ff_ppo_reinforce --max-steps 5 --seeds 3 --runs-per-gpu 3 --architectures cnn+transformer --hidden-dim 64 --mlp-dim 128 --num-layers 2 --num-heads 4 --total-timesteps 1e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --envs maze --maze-size 10 --gpus 0,1 --rollout-length 10 --total-num-envs 128 --epochs 2 --num-minibatches 2 --ent-coef 0.01 --gamma 0.99 --standardize-advantages true --gae-lambda 0.95 --halting-ent-coef 0.01 --wandb true --wandb-project maze-${project_name} --output-dir /home/chanb/scratch/logs/ramdp/maze-icot-${project_name} --yes --no-skip-existing --server vulcan

wait

echo "finished with exit code $? at: $(date)"