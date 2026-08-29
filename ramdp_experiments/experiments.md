# Frozen Lake
- Purpose:
  - Check with a known environment that with more compute performs better with linear PonderNet
  - Check what happens with tabular Q-function
  - Compare estimators

# Lightsout
- Purpose:
  - Sanity check with a known environment that with more compute performs better
  - Compare four different architectures: IRU, IRU without parameter sharing, TF with implicit CoT, TF with explicit CoT

## IRU without parameter sharing
- Env: grid size 3x3
- Unshared parameters
  - epochs=8, num_minibatches=16, hidden-dim=16
- Commands:
  - Fixed budget: `python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 1,2,4,8,16 --seeds 10 --runs-per-gpu 2 --yes --architectures iru_unshared --hidden-dim 16 --num-layers 1 --total-timesteps 5e7 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 8 --num-minibatches 16 --grid-sizes 3x3 --use-input-layer-norm true --episode-length 6 --gpus 0,1,2,3 --wandb true --wandb-project lightsout_sweep-ppo_only-iru`
  - Adaptive budget: `python ramdp_experiments/lightsout_sweep.py --systems ff_ppo_reinforce,ff_ppo_cond_fac,ff_ppo_cond_naive --max-steps 16 --seeds 10 --runs-per-gpu 2 --yes --architectures iru_unshared --hidden-dim=16 --total-timesteps 5e7 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 8 --num-minibatches 16 --grid-sizes 3x3 --use-input-layer-norm true --episode-length 6 --gpus 4,5,6,7 --wandb true --wandb-project lightsout_sweep-ppo_only-iru`

## IRU with parameter sharing
- Env: grid size 5x5
- Shared parameters
- Commands:
  - Fixed budget: `python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 1,2,4,8,16 --seeds 10 --runs-per-gpu 2 --yes --architectures iru --hidden-dim 16 --num-layers 1 --total-timesteps 5e7 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 8 --num-minibatches 16 --grid-sizes 5x5 --difficulty-threshold 0.25 --use-input-layer-norm true --episode-length 10 --gpus 0,1,2,3 --wandb true --wandb-project lightsout_sweep-ppo_only-iru`


## Transformer with implicit CoT

## Transformer with explicit CoT


# Jumanji
- Purpose:
  - Known to be difficult problem, can do CNN for some envs
  - Envs: Sokoban, sliding puzzle tile, maze, knapsack
