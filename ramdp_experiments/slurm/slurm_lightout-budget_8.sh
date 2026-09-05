#!/bin/bash
#SBATCH --account=aip-schuurma
#SBATCH --time=11:59:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --array=1-1
#SBATCH --output=/home/chanb/scratch/logs/ramdp/Stoix/%x_%A_%a.out

module load StdEnv/2023
module load cuda/12.2

cd /home/chanb/research/iclr_2027/Stoix

echo "hostname: $(hostname)"
echo "starting at: $(date)"

python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 8 --seeds 3 --architectures transformer --hidden-dim 32 --mlp-dim 64 --num-layers 2 --num-heads 4 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 8 --num-minibatches 16 --grid-sizes 4x4 --use-input-layer-norm false --episode-length 10 --wandb true --wandb-project lightsout_sweep-ppo_only-tf_test-sep_3-4x4_weight_decay_search --runs-per-gpu 4 --gpus 0 --no-skip-existing --ent-coef 0.0001 --server vulcan --actor-weight-decay 0.0,0.1,0.01 --yes

echo "finished with exit code $? at: $(date)"