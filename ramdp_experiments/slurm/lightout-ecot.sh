#!/bin/bash
#SBATCH --account=aip-schuurma
#SBATCH --time=23:59:00
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


# python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_explicit_reinforce --budget 1 --seeds 5 --architectures transformer_explicit_cot --hidden-dim 64 --mlp-dim 128 --num-layers 2 --num-heads 4 --vocab-size=1 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 4 --num-minibatches 16 --grid-sizes 5x5 --use-input-layer-norm false --episode-length 10 --wandb true --wandb-project lightsout-tf_ecot-sep5-recompute_adv --runs-per-gpu 3 --gpus 0,1 --no-skip-existing --ent-coef 0.001 --gamma 0.999 --server vulcan --actor-weight-decay 0.01 --recompute-advantages true --yes

# python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_explicit_reinforce --budget 2,4,8,16 --seeds 5 --architectures transformer_explicit_cot --hidden-dim 64 --mlp-dim 128 --num-layers 2 --num-heads 4 --vocab-size=2,4,8 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 4 --num-minibatches 16 --grid-sizes 5x5 --use-input-layer-norm false --episode-length 10 --wandb true --wandb-project lightsout-tf_ecot-sep5-recompute_adv --runs-per-gpu 3 --gpus 0,1 --no-skip-existing --ent-coef 0.001 --gamma 0.999 --server vulcan --actor-weight-decay 0.01 --recompute-advantages true --yes


# 2 GPUs can only complete one variant at a time.
# python ramdp_experiments/lightsout_sweep.py --systems ff_ppo_explicit_cond_fac --max-steps 16 --seeds 5 --architectures transformer_explicit_cot --hidden-dim 64 --mlp-dim 128 --num-layers 2 --num-heads 4 --vocab-size=2,4,8 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 4 --num-minibatches 16 --grid-sizes 5x5 --use-input-layer-norm false --episode-length 10 --wandb true --wandb-project lightsout-tf_ecot-sep5-recompute_adv --runs-per-gpu 3 --gpus 0,1 --no-skip-existing --ent-coef 0.001 --gamma 0.999 --server vulcan --actor-weight-decay 0.01 --recompute-advantages true --yes


# python ramdp_experiments/lightsout_sweep.py --systems ff_ppo_explicit_cond_naive --max-steps 16 --seeds 5 --architectures transformer_explicit_cot --hidden-dim 64 --mlp-dim 128 --num-layers 2 --num-heads 4 --vocab-size=2,4,8 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 4 --num-minibatches 16 --grid-sizes 5x5 --use-input-layer-norm false --episode-length 10 --wandb true --wandb-project lightsout-tf_ecot-sep5-recompute_adv --runs-per-gpu 3 --gpus 0,1 --no-skip-existing --ent-coef 0.001 --gamma 0.999 --server vulcan --actor-weight-decay 0.01 --recompute-advantages true --yes

python ramdp_experiments/lightsout_sweep.py --systems ff_ppo_explicit_reinforce --max-steps 16 --seeds 5 --architectures transformer_explicit_cot --hidden-dim 64 --mlp-dim 128 --num-layers 2 --num-heads 4 --vocab-size=2,4,8 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 4 --num-minibatches 16 --grid-sizes 5x5 --use-input-layer-norm false --episode-length 10 --wandb true --wandb-project lightsout-tf_ecot-sep5-recompute_adv --runs-per-gpu 3 --gpus 0,1 --no-skip-existing --ent-coef 0.001 --gamma 0.999 --server vulcan --actor-weight-decay 0.01 --recompute-advantages true --yes

# wait

echo "finished with exit code $? at: $(date)"