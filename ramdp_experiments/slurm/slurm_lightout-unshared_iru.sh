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

python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 1,2,4,8,16 --seeds 5 --architectures iru_unshared --hidden-dim 32 --num-layers 1 --total-timesteps 1e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 4 --num-minibatches 16 --grid-sizes 4x5 --use-input-layer-norm true --episode-length 10 --wandb true --wandb-project lightsout-unshared_iru-sep5-recompute_adv --runs-per-gpu 4 --gpus 0 --no-skip-existing --ent-coef 0.001 --gamma 0.999 --server vulcan --actor-weight-decay 0.01 --yes &

python ramdp_experiments/lightsout_sweep.py --systems ff_ppo_cond_fac,ff_ppo_cond_naive,ff_ppo_reinforce --max-steps 16 --seeds 5 --architectures iru_unshared --hidden-dim 32 --num-layers 1 --total-timesteps 1e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 4 --num-minibatches 16 --grid-sizes 4x5 --use-input-layer-norm true --episode-length 10 --wandb true --wandb-project lightsout-unshared_iru-sep5-recompute_adv --runs-per-gpu 4 --gpus 1 --no-skip-existing --ent-coef 0.001 --gamma 0.999 --server vulcan --actor-weight-decay 0.01 --yes &

wait

echo "finished with exit code $? at: $(date)"