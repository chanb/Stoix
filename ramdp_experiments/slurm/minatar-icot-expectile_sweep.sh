#!/bin/bash
#SBATCH --account=aip-schuurma
#SBATCH --time=23:59:00
#SBATCH --mem=24GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=0-14
#SBATCH --output=/home/chanb/scratch/logs/ramdp/Stoix/%x_%A_%a.out

# Expectile sweep for the fixed-budget (budget=1) iCoT baseline on MinAtar.
# One array task per (game, expectile): 5 games x 3 expectiles = 15 tasks,
# each running 5 seeds, 2 concurrently on one GPU.

module load StdEnv/2023
module load cuda/12.2

mkdir -p $SLURM_TMPDIR/tmp
export CUDA_MPS_LOG_DIRECTORY=$SLURM_TMPDIR/tmp
nvidia-cuda-mps-control -d

cd /home/chanb/research/iclr_2027/Stoix

games=(asterix breakout freeway seaquest space_invaders)
expectiles=(0.2 0.5 0.8)
game=${games[$((SLURM_ARRAY_TASK_ID / ${#expectiles[@]}))]}
expectile=${expectiles[$((SLURM_ARRAY_TASK_ID % ${#expectiles[@]}))]}

echo "hostname: $(hostname)"
echo "game: ${game} expectile: ${expectile}"
echo "starting at: $(date)"

project_name=nov06-expectile_sweep

python ramdp_experiments/jumanji_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 1 --seeds 5 --runs-per-gpu 2 --architectures cnn+transformer --hidden-dim 64 --mlp-dim 128 --num-layers 2 --num-heads 4 --total-timesteps 1e8 --total-num-envs 128 --rollout-length 64 --envs ${game} --gamma 0.99 --lr 3e-4 --critic-lr 3e-4 --epochs 4 --num-minibatches 8 --clip-eps 0.2 --clip-value-loss false --ent-coef 0.01 --gae-lambda 0.95 --standardize-advantages true --critic-before-actor false --use-input-layer-norm true --actor-weight-decay 0.0 --max-grad-norm 0.5 --use-expectile-value-loss true --expectile ${expectile} --wandb true --wandb-project minatar-icot-${project_name} --output-dir /home/chanb/scratch/logs/ramdp/minatar-icot-${project_name} --gpus 0 --server vulcan --yes

echo "finished with exit code $? at: $(date)"
