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

# python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 1,2,3 --seeds 10 --architectures transformer --hidden-dim 64 --mlp-dim 128 --num-layers 4 --num-heads 8 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 4 --num-minibatches 16 --grid-sizes 5x4 --use-input-layer-norm true --episode-length 10 --ent-coef 0.001 --gamma 0.99 --actor-weight-decay 0.01 --gae-lambda 0.95 --standardize-advantages true --wandb true --wandb-project lightsout-${project_name} --output-dir /home/chanb/scratch/logs/ramdp/lightsout-icot-${project_name} --runs-per-gpu 4 --gpus 0,1 --no-skip-existing --server vulcan --yes

# python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 4,5 --seeds 10 --architectures transformer --hidden-dim 64 --mlp-dim 128 --num-layers 4 --num-heads 8 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 4 --num-minibatches 16 --grid-sizes 5x4 --use-input-layer-norm true --episode-length 10 --ent-coef 0.001 --gamma 0.99 --actor-weight-decay 0.01 --gae-lambda 0.95 --standardize-advantages true --wandb true --wandb-project lightsout-${project_name} --output-dir /home/chanb/scratch/logs/ramdp/lightsout-icot-${project_name} --runs-per-gpu 4 --gpus 0,1 --no-skip-existing --server vulcan --yes

# python ramdp_experiments/lightsout_sweep.py --systems ff_ppo_reinforce --max-steps 5 --seeds 10 --architectures transformer --hidden-dim 64 --mlp-dim 128 --num-layers 4 --num-heads 8 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 4 --num-minibatches 16 --grid-sizes 5x4 --use-input-layer-norm true --episode-length 10 --ent-coef 0.001 --gamma 0.99 --actor-weight-decay 0.01 --gae-lambda 0.95 --standardize-advantages true --halting-ent-coef 0.01 --wandb true --wandb-project lightsout-${project_name} --output-dir /home/chanb/scratch/logs/ramdp/lightsout-icot-${project_name} --runs-per-gpu 4 --gpus 0,1 --no-skip-existing --server vulcan --yes


##### HYPERPARAM SWEEP
project_name=sep12

# python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 1,2,3 --seeds 3 --architectures transformer --hidden-dim 128 --mlp-dim 512 --num-layers 2,4 --num-heads 8 --total-timesteps 3e8 --grid-sizes 5x4 --episode-length 10 --lr 3e-4 --critic-lr 3e-4 --epochs 4 --num-minibatches 8 --use-input-layer-norm true --ent-coef 0.01 --clip-eps 0.2 --gamma 0.99 --actor-weight-decay 0.1,0.2 --gae-lambda 0.95 --standardize-advantages true --wandb true --wandb-project lightsout-icot-${project_name}-hyperparam --output-dir /home/chanb/scratch/logs/ramdp/lightsout-icot-${project_name}-hyperparam --runs-per-gpu 3 --gpus 0,1 --no-skip-existing --server vulcan --yes

# python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 4,5 --seeds 3 --architectures transformer --hidden-dim 128 --mlp-dim 512 --num-layers 2,4 --num-heads 8 --total-timesteps 3e8 --grid-sizes 5x4 --episode-length 10 --lr 3e-4 --critic-lr 3e-4 --epochs 4 --num-minibatches 8 --use-input-layer-norm true --ent-coef 0.01 --clip-eps 0.2 --gamma 0.99 --actor-weight-decay 0.1,0.2 --gae-lambda 0.95 --standardize-advantages true --wandb true --wandb-project lightsout-icot-${project_name}-hyperparam --output-dir /home/chanb/scratch/logs/ramdp/lightsout-icot-${project_name}-hyperparam --runs-per-gpu 3 --gpus 0,1 --no-skip-existing --server vulcan --yes

# python ramdp_experiments/lightsout_sweep.py --systems ff_ppo_reinforce --max-steps 5 --seeds 3 --architectures transformer --hidden-dim 128 --mlp-dim 512 --num-layers 2,4 --num-heads 8 --total-timesteps 3e8 --grid-sizes 5x4 --episode-length 10 --lr 3e-4 --critic-lr 3e-4 --epochs 4 --num-minibatches 8 --use-input-layer-norm true --ent-coef 0.01 --clip-eps 0.2 --gamma 0.99 --actor-weight-decay 0.1,0.2 --gae-lambda 0.95 --standardize-advantages true --halting-ent-coef 0.0,0.01,0.001 --halting-temperature 1.0,5.0 --wandb true --wandb-project lightsout-icot-${project_name}-hyperparam --output-dir /home/chanb/scratch/logs/ramdp/lightsout-icot-${project_name}-hyperparam --runs-per-gpu 3 --gpus 0,1 --no-skip-existing --server vulcan --yes


echo "finished with exit code $? at: $(date)"
