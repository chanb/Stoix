#!/bin/bash
#SBATCH --account=aip-schuurma
#SBATCH --time=11:59:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
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

project_name=icot_sweep_2

# python ramdp_experiments/jumanji_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 1,2,3,4 --seeds 10 --architectures cnn+transformer --hidden-dim 16 --qkv-dim 256 --mlp-dim 256 --num-layers 2 --num-heads 16 --total-timesteps 1e8 --total-num-envs 256 --envs slidingtile --slidingtile-grid-size 3 --slidingtile-num-random-moves 200 --lr 2e-4 --critic-lr 3e-4 --epochs 4 --num-minibatches 8 --use-input-layer-norm true --use-rmsnorm true --ent-coef 0.01 --clip-eps 0.2 --gamma 0.999 --actor-weight-decay 0.01 --gae-lambda 0.95 --standardize-advantages true --wandb true --wandb-project slidingpuzzle-${project_name} --output-dir /home/chanb/scratch/logs/ramdp/slidingpuzzle-icot-${project_name} --runs-per-gpu 4 --gpus 0 --yes --no-skip-existing --server vulcan --use-expectile-value-loss true --expectile 0.2 &

# python ramdp_experiments/jumanji_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 5 --seeds 10 --architectures cnn+transformer --hidden-dim 16 --qkv-dim 256 --mlp-dim 256 --num-layers 2 --num-heads 16 --total-timesteps 1e8 --total-num-envs 256 --envs slidingtile --slidingtile-grid-size 3 --slidingtile-num-random-moves 200 --lr 2e-4 --critic-lr 3e-4 --epochs 4 --num-minibatches 8 --use-input-layer-norm true --use-rmsnorm true --ent-coef 0.01 --clip-eps 0.2 --gamma 0.999 --actor-weight-decay 0.01 --gae-lambda 0.95 --standardize-advantages true --wandb true --wandb-project slidingpuzzle-${project_name} --output-dir /home/chanb/scratch/logs/ramdp/slidingpuzzle-icot-${project_name} --runs-per-gpu 4 --gpus 0 --yes --no-skip-existing --server vulcan --use-expectile-value-loss true --expectile 0.2 &

python ramdp_experiments/jumanji_sweep.py --systems ff_ppo_reinforce --max-steps 5 --seeds 10 --architectures cnn+transformer --hidden-dim 16 --qkv-dim 256 --mlp-dim 256 --num-layers 2 --num-heads 16 --total-timesteps 1e8 --total-num-envs 256 --envs slidingtile --slidingtile-grid-size 3 --slidingtile-num-random-moves 200 --lr 3e-4 --critic-lr 3e-4 --epochs 4 --num-minibatches 8 --use-input-layer-norm true --use-rmsnorm true --ent-coef 0.01 --clip-eps 0.2 --gamma 0.999 --actor-weight-decay 0.001 --gae-lambda 0.95 --standardize-advantages true --wandb true --wandb-project slidingpuzzle-${project_name} --output-dir /home/chanb/scratch/logs/ramdp/slidingpuzzle-icot-${project_name} --runs-per-gpu 4 --gpus 0 --yes --no-skip-existing --server vulcan --use-expectile-value-loss true --expectile 0.2 --halting-ent-coef 0.01 --halting-temperature 1.0 --clip-halting-head false &

wait

echo "finished with exit code $? at: $(date)"
