#!/bin/bash
#SBATCH --account=aip-schuurma
#SBATCH --time=23:59:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l40s:2
#SBATCH --array=0-4
#SBATCH --output=/home/chanb/scratch/logs/ramdp/Stoix/%x_%A_%a.out

# One array task per MinAtar game (env=gymnax/<game>), each running the
# fixed-budget iCoT sweep (budget 1-5) and the adaptive-halting iCoT sweep
# (max_steps 5) concurrently on the same two GPUs.

module load StdEnv/2023
module load cuda/12.2

mkdir -p $SLURM_TMPDIR/tmp
export CUDA_MPS_LOG_DIRECTORY=$SLURM_TMPDIR/tmp
nvidia-cuda-mps-control -d

cd /home/chanb/research/iclr_2027/Stoix

games=(asterix breakout freeway seaquest space_invaders)
game=${games[$SLURM_ARRAY_TASK_ID]}

echo "hostname: $(hostname)"
echo "game: ${game}"
echo "starting at: $(date)"

project_name=nov06

common_args="--systems ff_ppo_reinforce --seeds 5 --runs-per-gpu 3 --architectures cnn+transformer --hidden-dim 64 --mlp-dim 128 --num-layers 2 --num-heads 4 --total-timesteps 1e8 --total-num-envs 128 --rollout-length 64 --envs ${game} --gamma 0.99 --lr 3e-4 --critic-lr 3e-4 --epochs 4 --num-minibatches 8 --clip-eps 0.2 --clip-value-loss false --ent-coef 0.01 --gae-lambda 0.95 --standardize-advantages true --critic-before-actor false --use-input-layer-norm true --actor-weight-decay 0.0 --max-grad-norm 0.5 --wandb true --wandb-project minatar-icot-${project_name} --output-dir /home/chanb/scratch/logs/ramdp/minatar-icot-${project_name} --gpus 0,1 --server vulcan --yes"

python ramdp_experiments/jumanji_fixed_budget_sweep.py ${common_args} --budget 1,2,3,4,5 &

python ramdp_experiments/jumanji_sweep.py ${common_args} --max-steps 5 --halting-ent-coef 0.01 --halting-temperature 5.0 &

wait

echo "finished with exit code $? at: $(date)"
