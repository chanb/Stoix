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

cd /home/chanb/research/iclr_2027/Stoix

echo "hostname: $(hostname)"
echo "starting at: $(date)"


python ramdp_experiments/jumanji_fixed_budget_sweep.py --systems ff_ppo_explicit_reinforce --budget 1 --seeds 5 --runs-per-gpu 2 --architectures iru --hidden-dim 64,128,256 --num-layers 2 --total-timesteps 1e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --envs knapsack --knapsack-num-items 100 --knapsack-total-budget 12.5 --gpus 0,1 --rollout-length 64 --epochs 4 --num-minibatches 16 --ent-coef 0.001 --gamma 0.995 --actor-weight-decay 0.005 --critic-before-actor true --wandb true --wandb-project knapsack-tf-sep6 --yes --server vulcan --no-skip-existing

python ramdp_experiments/jumanji_fixed_budget_sweep.py --systems ff_ppo_explicit_reinforce --budget 2,4,8,16 --seeds 5 --runs-per-gpu 2 --architectures iru --hidden-dim 64,128,256 --mlp-dim 64 --num-layers 2 --num-heads 4 --total-timesteps 1e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --envs knapsack --knapsack-num-items 100 --knapsack-total-budget 12.5 --gpus 0,1 --rollout-length 64 --epochs 4 --num-minibatches 16 --ent-coef 0.001 --gamma 0.995 --actor-weight-decay 0.005 --critic-before-actor true --wandb true --wandb-project knapsack-tf-sep6 --yes --server vulcan --no-skip-existing

python ramdp_experiments/jumanji_sweep.py --systems ff_ppo_explicit_cond_fac,ff_ppo_explicit_cond_naive,ff_ppo_explicit_reinforce --max-steps 16 --seeds 5 --runs-per-gpu 2 --architectures iru --hidden-dim 64,128,256 --mlp-dim 64 --num-layers 2 --num-heads 4 --total-timesteps 1e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --envs knapsack --knapsack-num-items 100 --knapsack-total-budget 12.5 --gpus 0,1 --rollout-length 64 --epochs 4 --num-minibatches 16 --ent-coef 0.001 --gamma 0.995 --actor-weight-decay 0.005 --critic-before-actor true --wandb true --wandb-project knapsack-tf-sep6 --yes --server vulcan --no-skip-existing

wait

echo "finished with exit code $? at: $(date)"