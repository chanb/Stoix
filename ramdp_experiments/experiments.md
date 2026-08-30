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
- PPO hyperparameters
  - epochs=8, num_minibatches=16, hidden-dim=16
- Commands:
  - Fixed budget: `python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 1,2,4,8,16 --seeds 10 --runs-per-gpu 2 --yes --architectures iru_unshared --hidden-dim 16 --num-layers 1 --total-timesteps 5e7 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 8 --num-minibatches 16 --grid-sizes 3x3 --use-input-layer-norm true --episode-length 6 --gpus 0,1,2,3 --wandb true --wandb-project lightsout_sweep-ppo_only-iru`
  - Adaptive budget: `python ramdp_experiments/lightsout_sweep.py --systems ff_ppo_reinforce,ff_ppo_cond_fac,ff_ppo_cond_naive --max-steps 16 --seeds 10 --runs-per-gpu 2 --yes --architectures iru_unshared --hidden-dim=16 --total-timesteps 5e7 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 8 --num-minibatches 16 --grid-sizes 3x3 --use-input-layer-norm true --episode-length 6 --gpus 4,5,6,7 --wandb true --wandb-project lightsout_sweep-ppo_only-unshared_iru`
- Status: Setting seems reasonable

## IRU with parameter sharing
- Env: grid size 5x5
- PPO hyperparameters
- Commands:
  - Fixed budget: `python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 1,2,4,,168 --seeds 10 --runs-per-gpu 2 --yes --architectures iru --hidden-dim 16 --num-layers 1 --total-timesteps 1e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 8 --num-minibatches 16 --grid-sizes 4x4 --difficulty-threshold 1.0 --use-input-layer-norm true --episode-length 10 --gpus 0,1,2,3 --wandb true --wandb-project lightsout_sweep-ppo_only-iru`


## Transformer with implicit CoT
- Env: grid size 5x5
- PPO hyperparameters
- Commands:
  - Fixed budget: `python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 1,2,4,8 --seeds 10 --runs-per-gpu 2 --yes --architectures transformer --hidden-dim 16 --mlp-dim 16 --num-layers 2 --num-heads 2 --total-timesteps 1e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 8 --num-minibatches 16 --grid-sizes 4x4 --difficulty-threshold 1.0 --use-input-layer-norm true --episode-length 10 --wandb true --wandb-project lightsout_sweep-ppo_only-tf_implicit_cot --gpus 4,5,6,7`

## Transformer with explicit CoT
- Env: grid size 5x5
- PPO hyperparameters
- Commands:
  - Fixed budget: `python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 1,2,4,8 --seeds 10 --runs-per-gpu 2 --yes --architectures transformer_explicit_cot --hidden-dim 16 --mlp-dim 16 --num-layers 2 --num-heads 2 --total-timesteps 5e7 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 8 --num-minibatches 16 --grid-sizes 5x5 --difficulty-threshold 0.25 --use-input-layer-norm true --episode-length 10 --wandb true --wandb-project lightsout_sweep-ppo_only-tf_explicit_cot --gpus 0,1,2,3`


# Jumanji
- Purpose:
  - Known to be difficult problem, can do CNN for some envs
  - Envs: Sokoban, sliding puzzle tile, maze, knapsack

## Sokoban
`python ramdp_experiments/jumanji_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 4,1 --seeds 3 --runs-per-gpu 1 --architectures cnn+transformer --hidden-dim 16 --mlp-dim 16 --num-layers 2 --num-heads 4 --total-timesteps 5e7 --clip-value-loss false --lr 1e-4 --critic-lr 3e-4 --envs sokoban --sokoban-generator toy --use-input-layer-norm true --gpus 4,5,6,7 --total-num-envs 128 --rollout-length 64 --epochs 8 --num-minibatches 16 --wandb true --wandb-project jumanji_sweep-ppo_only-cnn_tf_arch`

## Sliding puzzle
`python ramdp_experiments/jumanji_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 4,1 --seeds 3 --runs-per-gpu 1 --architectures cnn+transformer --hidden-dim 16 --mlp-dim 16 --num-layers 2 --num-heads 4 --total-timesteps 5e7 --clip-value-loss false --lr 1e-4 --critic-lr 3e-4 --envs slidingtile --slidingtile-grid-size 3,4 --slidingtile-num-random-moves 5,20 --use-input-layer-norm true --gpus 4,5,6,7 --total-num-envs 128 --rollout-length 64 --epochs 8 --num-minibatches 16 --wandb true --wandb-project jumanji_sweep-ppo_only-cnn_tf_arch`

## Knapsack
`python ramdp_experiments/jumanji_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 4,1 --seeds 3 --runs-per-gpu 1 --architectures cnn+transformer --hidden-dim 16 --mlp-dim 16 --num-layers 2 --num-heads 4 --total-timesteps 5e7 --clip-value-loss false --lr 1e-4 --critic-lr 3e-4 --envs knapsack --knapsack-num-items 5,10,20,50 --knapsack-total-budget 2.5,12.5 --use-input-layer-norm true --gpus 4,5,6,7 --total-num-envs 128 --rollout-length 64 --epochs 8 --num-minibatches 16 --wandb true --wandb-project jumanji_sweep-ppo_only-cnn_tf_arch`

## Maze
`python ramdp_experiments/jumanji_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 4,1 --seeds 3 --runs-per-gpu 1 --architectures cnn+transformer --hidden-dim 16 --mlp-dim 16 --num-layers 2 --num-heads 4 --total-timesteps 5e7 --clip-value-loss false --lr 1e-4 --critic-lr 3e-4 --envs maze --maze-size 5,10,15 --use-input-layer-norm true --gpus 4,5,6,7 --total-num-envs 128 --rollout-length 64 --epochs 8 --num-minibatches 16 --wandb true --wandb-project jumanji_sweep-ppo_only-cnn_tf_arch`





## Comments Aug 30:
- Episode = 10 is a bit better since there's a higher chance of random walking into a good solution.
- Gridsize 5x4 with `difficulty_threshold=0.5` seems good so far.

### IRU
- So far minibatches=16 and epoch=8 is better than other combinations of mb=8,16, epoch=2,8. Need to test mb=32 with epoch=8 (IRU paper uses it for lightsout)---it is not as good as mb=16.
- Also try `hidden_dim=128` (IRU paper uses 64 for IRU model)
- We should still do MinAtar since we should see if the algorithm recovers the case where all actions take `c=1`

python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 1,2,4,8 --seeds 3 --runs-per-gpu 2 --architectures iru --hidden-dim 128 --num-layers 2 --total-timesteps 1e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 8 --num-minibatches 16,32 --grid-sizes 5x4 --difficulty-threshold 0.5 --use-input-layer-norm true --episode-length 6 --gpus 0,1 --wandb true --wandb-project lightsout_sweep-ppo_only-iru --no-skip-existing --yes

### TF with implicit CoT
- When `num_layers=1`, `hidden_dim=64` seems to have better trends.
- `hidden_dim=32` is bad.
- Still figuring out `num_heads=2,4` and `num_layers=1,2`.
- Should be using GeLU rather than ReLU.
- Currently we have LN before feeding state to halt predictor.

python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 1,2,4,8 --seeds 3 --runs-per-gpu 2 --yes
 --architectures transformer --hidden-dim 32,64 --mlp-dim 64 --num-layers 1,2 --num-heads 2,4 --total-timesteps 1e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 8 --num-minibatches 16
 --grid-sizes 5x4 --difficulty-threshold 0.5 --use-input-layer-norm true --episode-length 10 --wandb true --wandb-project lightsout_sweep-ppo_only-tf_implicit_cot