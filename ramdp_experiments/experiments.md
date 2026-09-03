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
- Env: grid size 3x3 with `difficulty-threshold=0.5`
- PPO hyperparameters
  - epochs=8, num_minibatches=16, hidden-dim=16, num_layers=1
- Commands:
  - Fixed budget: `python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 1,2,4,8,16 --seeds 10 --architectures iru_unshared --hidden-dim 16 --num-layers 1 --total-timesteps 5e7 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 8 --num-minibatches 16 --grid-sizes 3x3 --use-input-layer-norm true --episode-length 6 --wandb true --wandb-project lightsout_sweep-ppo_only-unshared_iru --no-skip-existing --yes --gpus 0,1,2,3 --runs-per-gpu 2`
  - Adaptive budget: `python ramdp_experiments/lightsout_sweep.py --systems ff_ppo_reinforce,ff_ppo_cond_fac,ff_ppo_cond_naive --max-steps 16 --seeds 10 --architectures iru_unshared --hidden-dim=16 --num-layers 1 --total-timesteps 5e7 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 8 --num-minibatches 16 --grid-sizes 3x3 --use-input-layer-norm true --episode-length 6 --wandb true --wandb-project lightsout_sweep-ppo_only-unshared_iru --no-skip-existing --yes --gpus 4,5,6,7 --runs-per-gpu 2`
- Status: Setting seems reasonable

## IRU with parameter sharing
- Env: grid size 5x4 with `difficulty-threshold=0.5`
- PPO hyperparameters
  - epochs=8, num_minibatches=16, hidden-dim=64, num_layers=2
- Commands:
  - Fixed budget: `python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 1,2,4,8,16 --seeds 10 --architectures iru --hidden-dim 64 --num-layers 2 --total-timesteps 1e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 8 --num-minibatches 16,32 --grid-sizes 5x4 --use-input-layer-norm true --episode-length 10 --wandb true --wandb-project lightsout_sweep-ppo_only-iru --no-skip-existing --yes --gpus 0,1 --runs-per-gpu 2`
  - Adaptive budget: `python ramdp_experiments/lightsout_sweep.py --systems ff_ppo_reinforce,ff_ppo_cond_fac,ff_ppo_cond_naive --max-steps 16 --seeds 10 --architectures iru --hidden-dim 64 --num-layers 2 --total-timesteps 1e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 8 --num-minibatches 16,32 --grid-sizes 5x4 --use-input-layer-norm true --episode-length 10 --wandb true --wandb-project lightsout_sweep-ppo_only-iru --no-skip-existing --yes --gpus 0,1 --runs-per-gpu 2`


## Transformer with implicit CoT
- Env: grid size 5x4 with `difficulty-threshold=0.5`
- PPO hyperparameters
  - epoch=8, num_minibatches=16, num_layers=1, hidden_dim=64, num_heads=2, mlp_dim=64
- Commands:
  - Fixed budget: `python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 1,2,4,8 --seeds 10 --architectures transformer --hidden-dim 64 --mlp-dim 64 --num-layers 1 --num-heads 2 --total-timesteps 1e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 8 --num-minibatches 16 --grid-sizes 5x4 --use-input-layer-norm true --episode-length 10 --wandb true --wandb-project lightsout_sweep-ppo_only-tf_implicit_cot --yes --gpus 0,1 --runs-per-gpu 2`

## Transformer with explicit CoT
- Env: grid size 5x4 with `difficulty-threshold=0.5`
- PPO hyperparameters
- Commands:
  - Fixed budget: 


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

## CNN + explicit CoT
- Purpose: does the explicit-CoT torso (see Lightsout's "Transformer with explicit CoT") also help on the CNN-input envs, not just Lightsout's flattened grid? `cnn+transformer_explicit_cot` pairs the same CNNTorso input used above with `TransformerExplicitCoTTorso` instead of the implicit-CoT `TransformerChainOfThoughtTorso`. Not supported for knapsack (no spatial structure).
- Also available in minatar_fixed_budget_sweep.py/minatar_sweep.py the same way.
- Commands:
  - Sokoban: `python ramdp_experiments/jumanji_fixed_budget_sweep.py --systems ff_ppo_explicit_reinforce --budget 4,1 --seeds 3 --runs-per-gpu 1 --architectures cnn+transformer_explicit_cot --hidden-dim 16 --mlp-dim 16 --num-heads 4 --total-timesteps 5e7 --clip-value-loss false --lr 1e-4 --critic-lr 3e-4 --envs sokoban --sokoban-generator toy --gpus 4,5,6,7 --total-num-envs 128 --rollout-length 64 --epochs 8 --num-minibatches 16 --wandb true --wandb-project jumanji_sweep-ppo_only-cnn_tf_explicit_cot`
  - Sliding puzzle: same command with `--envs slidingtile --slidingtile-grid-size 3,4 --slidingtile-num-random-moves 5,20` instead of the sokoban flags.
  - Maze: same command with `--envs maze --maze-size 5,10,15` instead of the sokoban flags.
  - Q-V variants (fac/naive/cond_naive/cond_fac): swap `--systems ff_ppo_explicit_reinforce` for `--systems ff_ppo_explicit_fac` etc.


# Comments Aug 30:
- Episode = 10 is a bit better since there's a higher chance of random walking into a good solution.
- Gridsize 5x4 with `difficulty_threshold=0.5` seems good so far.

### IRU
- So far minibatches=16 and epoch=8 is better than other combinations of mb=8,16, epoch=2,8. Need to test mb=32 with epoch=8 (IRU paper uses it for lightsout)---it is not as good as mb=16.
- Also try `hidden_dim=128` (IRU paper uses 64 for IRU model)---doesn't help much
- We should still do MinAtar since we should see if the algorithm recovers the case where all actions take `c=1`

### TF with implicit CoT
- When `num_layers=1`, `hidden_dim=64` seems to have better trends.
- Still figuring out `num_heads=2,4` and `num_layers=1,2`.
- As of now, `num_layers=1`, `hidden_dim=64`, `num_heads=2`, `epoch=8`, `mb=16` has the best compute trend.
- Should be using GeLU rather than ReLU.
- Currently we have LN before feeding state to halt predictor.

python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 1,2,4,8 --seeds 3 --architectures transformer --hidden-dim 32,64 --mlp-dim 64 --num-layers 1,2 --num-heads 2,4 --total-timesteps 1e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 8 --num-minibatches 16 --grid-sizes 5x4 --use-input-layer-norm true --episode-length 10 --wandb true --wandb-project lightsout_sweep-ppo_only-tf_implicit_cot --yes --runs-per-gpu 2 --gpus 4,5,6,7


 ### TF with explicit CoT
python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_explicit_reinforce --budget 1,2,4,8 --seeds 3 --architectures transformer_explicit_cot --hidden-dim 32,64 --mlp-dim 64 --num-layers 1,2 --num-heads 2,4 --total-timesteps 1e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 8 --num-minibatches 16 --grid-sizes 5x4 --use-input-layer-norm true --episode-length 10 --wandb true --wandb-project lightsout_sweep-ppo_only-tf_explicit_cot --yes --runs-per-gpu 2 --gpus 0,1




# Comments on Aug 31:
## Comments about experiments from Aug. 30
### Implicit CoT, `lightsout`
- Group: `lightsout-5x4-ppo_reinforce-transformer-hd32-lr0.0003-clr0.0003-nl1-nh4-md64-ep8-mb16-clip0.2-l2c-iln`
  - Budget = 8 is better than all other variants, perhaps 100M steps is insufficient compute.
- Group: `lightsout-5x4-ppo_reinforce-transformer-hd64-lr0.0003-clr0.0003-nl1-nh2-md64-ep8-mb16-clip0.2-l2c-iln`
  - General trend is that more budget gives better performance---of course we can increase number of training steps
- Group: `lightsout-5x4-ppo_reinforce-transformer-hd64-lr0.0003-clr0.0003-nl1-nh4-md64-ep8-mb16-clip0.2-l2c-iln`
  - General trend is that more budget gives better performance---of course we can increase number of training steps---not as dramatic as `num_heads=2`
- Group: `lightsout-5x4-ppo_reinforce-transformer-hd32-lr0.0003-clr0.0003-nl2-nh4-md64-ep8-mb16-clip0.2-l2c-iln`
  - Performance is similar across budget, but it seems like with more training step there could be separation

We will choose the last group: `lightsout-5x4-ppo_reinforce-transformer-hd32-lr0.0003-clr0.0003-nl2-nh4-md64-ep8-mb16-clip0.2-l2c-iln` since it's the closest to current TF models---rerunning


### Explicit CoT, `lightsout`
- Needed to expose `num_layers`---it was missing.
- Use similar setting as above, it seems more training steps can help. Setting `epoch=8` is generally better than `epoch=16`.
- Use shared embed/umembed---this seems to help!


## Jumanji architecture search
Start with knapsack: `python ramdp_experiments/jumanji_fixed_budget_sweep.py --systems ff_ppo_explicit_reinforce --budget 1,4 --seeds 3 --runs-per-gpu 4 --architectures transformer_explicit_cot --hidden-dim 32 --mlp-dim 64 --num-layers 2 --num-heads 4 --total-timesteps 5e7 --clip-value-loss false --lr 1e-4 --critic-lr 3e-4 --envs knapsack --knapsack-num-items 5,10,20,50 --knapsack-total-budget 2.5,12.5 --use-input-layer-norm true --gpus 0,1 --total-num-envs 128 --rollout-length 64 --epochs 8 --num-minibatches 16 --wandb true --wandb-project jumanji_sweep-ppo_only-test`


# Comments on Sept 1:
### Implicit CoT, `lightsout`
`lightsout-5x4-ppo_reinforce-transformer-hd32-lr0.0003-clr0.0003-nl2-nh4-md64-ep8-mb16-clip0.2-l2c-iln` has a reasonably nice trend, in that with more compute it can do better.
Command ran on salient:
```
python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 1,2,4,8 --seeds 3 --architectures transformer --hidden-dim 32 --mlp-dim 64 --num-layers 2 --num-heads 4 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 8 --num-minibatches 16 --grid-sizes 5x4 --use-input-layer-norm true --episode-length 10 --wandb true --wandb-project lightsout_sweep-ppo_only-tf_test --yes --runs-per-gpu 2 --gpus 4,5,6,7 --no-skip-existing
```

Now we try adaptive budget
```
python ramdp_experiments/lightsout_sweep.py --systems ff_ppo_reinforce,ff_ppo_cond_naive,ff_ppo_cond_fac --max-steps 10 --seeds 3 --architectures transformer --hidden-dim 32 --mlp-dim 64 --num-layers 2 --num-heads 4 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 8 --num-minibatches 16 --grid-sizes 5x4 --use-input-layer-norm true --episode-length 10 --wandb true --wandb-project lightsout_sweep-ppo_only-tf_test --runs-per-gpu 2 --gpus 4,5,6,7 --no-skip-existing
```

### Explicit CoT, `lightsout`
`lightsout-5x4-ppo_explicit_reinforce-transformer_explicit_cot-hd32-lr0.0003-clr0.0003-nl2-nh4-md64-ep8-mb16-clip0.2-l2c` generally has high entropy when `vocab_size` is > 1. This somewhat makes sense because it has to explore all possible paths.

Command:
```
python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_explicit_reinforce --budget 1,2,4,8 --seeds 3 --architectures transformer_explicit_cot --hidden-dim 32 --mlp-dim 64 --num-layers 2 --num-heads 4 --vocab-size=1,2,4,8 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 8 --num-minibatches 16 --grid-sizes 5x4 --use-input-layer-norm true --episode-length 10 --wandb true --wandb-project lightsout_sweep-ppo_only-tf_test-shared_embed_explicit_cot --runs-per-gpu 2 --gpus 0,1 --no-skip-existing --yes
```

One way is to decrease the entropy regularization effect.

Command:
```
python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_explicit_reinforce --budget 1,2,4,8 --seeds 3 --architectures transformer_explicit_cot --hidden-dim 32 --mlp-dim 64 --num-layers 2 --num-heads 4 --vocab-size=1,2,4,8 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 8 --num-minibatches 16 --grid-sizes 5x4 --use-input-layer-norm true --episode-length 10 --wandb true --wandb-project lightsout_sweep-ppo_only-tf_test-shared_embed_explicit_cot --runs-per-gpu 2 --gpus 0,1,2,3 --no-skip-existing --ent-coef 0.001,0.0001
```


### Jumanji architecture search
Start with maze: `python ramdp_experiments/jumanji_fixed_budget_sweep.py --systems ff_ppo_explicit_reinforce --budget 1,4,8,16 --seeds 3 --runs-per-gpu 4 --architectures transformer_explicit_cot --hidden-dim 8,16,32 --mlp-dim 32,64 --num-layers 2 --num-heads 4 --vocab-size=1,2,4,8 --total-timesteps 5e7 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --envs maze --maze-size 10 --use-input-layer-norm true --gpus 0,1 --rollout-length 64 --epochs 8 --num-minibatches 16 --wandb true --wandb-project jumanji_sweep-ppo_only-test-sept_1 --server vulcan`


## Seems to be a bug... Rerun
```
####### implicit, layernorm
python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 1,2 --seeds 3 --architectures transformer --hidden-dim 32 --mlp-dim 64 --num-layers 2 --num-heads 4 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 8 --num-minibatches 16 --grid-sizes 5x4 --use-input-layer-norm true --episode-length 10 --wandb true --wandb-project lightsout_sweep-ppo_only-tf_test-sep_1 --yes --runs-per-gpu 2 --gpus 4,5,6,7 --no-skip-existing --ent-coef 0.001,0.0001

python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 4,8 --seeds 3 --architectures transformer --hidden-dim 32 --mlp-dim 64 --num-layers 2 --num-heads 4 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 8 --num-minibatches 16 --grid-sizes 5x4 --use-input-layer-norm true --episode-length 10 --wandb true --wandb-project lightsout_sweep-ppo_only-tf_test-sep_1 --yes --runs-per-gpu 1 --gpus 4,5 --no-skip-existing --ent-coef 0.001,0.0001

python ramdp_experiments/lightsout_sweep.py --systems ff_ppo_reinforce,ff_ppo_cond_naive,ff_ppo_cond_fac --max-steps 16 --seeds 3 --architectures transformer --hidden-dim 32 --mlp-dim 64 --num-layers 2 --num-heads 4 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 8 --num-minibatches 16 --grid-sizes 5x4 --use-input-layer-norm true --episode-length 10 --wandb true --wandb-project lightsout_sweep-ppo_only-tf_test-sep_1 --yes --runs-per-gpu 1 --gpus 6,7 --no-skip-existing --ent-coef 0.001,0.0001



####### implicit, no layernorm
python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 1,2 --seeds 3 --architectures transformer --hidden-dim 32 --mlp-dim 64 --num-layers 2 --num-heads 4 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 8 --num-minibatches 16 --grid-sizes 5x4 --use-input-layer-norm false --episode-length 10 --wandb true --wandb-project lightsout_sweep-ppo_only-tf_test-sep_1 --yes --runs-per-gpu 2 --gpus 6,7 --no-skip-existing --ent-coef 0.001,0.0001

python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 4,8 --seeds 3 --architectures transformer --hidden-dim 32 --mlp-dim 64 --num-layers 2 --num-heads 4 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 8 --num-minibatches 16 --grid-sizes 5x4 --use-input-layer-norm false --episode-length 10 --wandb true --wandb-project lightsout_sweep-ppo_only-tf_test-sep_1 --yes --runs-per-gpu 2 --gpus 0,1 --no-skip-existing --ent-coef 0.001,0.0001

python ramdp_experiments/lightsout_sweep.py --systems ff_ppo_reinforce,ff_ppo_cond_naive,ff_ppo_cond_fac --max-steps 16 --seeds 3 --architectures transformer --hidden-dim 32 --mlp-dim 64 --num-layers 2 --num-heads 4 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 8 --num-minibatches 16 --grid-sizes 5x4 --use-input-layer-norm false --episode-length 10 --wandb true --wandb-project lightsout_sweep-ppo_only-tf_test-sep_1 --yes --runs-per-gpu 1 --gpus 6,7 --no-skip-existing --ent-coef 0.001,0.0001


####### explicit, layernorm, latent_feedback (Earlier runs use full-bandwidth tf implementation (Pre Sept. 2), later runs use skip connection (Post Sept. 2))
python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_explicit_reinforce --budget 1,2 --seeds 3 --architectures transformer_explicit_cot --hidden-dim 32 --mlp-dim 64 --num-layers 2 --num-heads 4 --vocab-size=2,4,8 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 8 --num-minibatches 16 --grid-sizes 5x4 --use-input-layer-norm true --episode-length 10 --wandb true --wandb-project lightsout_sweep-ppo_only-tf_test-sep_1 --runs-per-gpu 2 --gpus 6,7 --no-skip-existing --use-latent-feedback true --ent-coef 0.001,0.0001

python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_explicit_reinforce --budget 4,8 --seeds 3 --architectures transformer_explicit_cot --hidden-dim 32 --mlp-dim 64 --num-layers 2 --num-heads 4 --vocab-size=2,4,8 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 8 --num-minibatches 16 --grid-sizes 5x4 --use-input-layer-norm true --episode-length 10 --wandb true --wandb-project lightsout_sweep-ppo_only-tf_test-sep_1 --runs-per-gpu 1 --gpus 2,3 --no-skip-existing --use-latent-feedback true --ent-coef 0.001,0.0001

####### explicit, layernorm, no latent_feedback
python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_explicit_reinforce --budget 1,2 --seeds 3 --architectures transformer_explicit_cot --hidden-dim 32 --mlp-dim 64 --num-layers 2 --num-heads 4 --vocab-size=2,4,8 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 8 --num-minibatches 16 --grid-sizes 5x4 --use-input-layer-norm true --episode-length 10 --wandb true --wandb-project lightsout_sweep-ppo_only-tf_test-sep_1 --runs-per-gpu 2 --gpus 2,3 --no-skip-existing --use-latent-feedback false --ent-coef 0.001,0.0001
```


# Comments on Sept 2:
When `input_layer_norm=True`:
- Large `max_steps` should have smaller entropy coef, e.g. `max_steps>=4` should have at `ent_coef=0.0001`.
- Large `max_steps` should have smaller entropy coef, e.g. `max_steps<4` should have at `ent_coef=0.001`.

```
####### implicit, layernorm
python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 1,2 --seeds 3 --architectures transformer --hidden-dim 32 --mlp-dim 64 --num-layers 2 --num-heads 4 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 8 --num-minibatches 16 --grid-sizes 5x4 --use-input-layer-norm true --episode-length 10 --wandb true --wandb-project lightsout_sweep-ppo_only-tf_test-sep_1 --yes --runs-per-gpu 2 --gpus 4,5,6,7 --no-skip-existing --ent-coef 0.001,0.0001

python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 4,8 --seeds 3 --architectures transformer --hidden-dim 32 --mlp-dim 64 --num-layers 2 --num-heads 4 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 8 --num-minibatches 16 --grid-sizes 5x4 --use-input-layer-norm true --episode-length 10 --wandb true --wandb-project lightsout_sweep-ppo_only-tf_test-sep_1 --yes --runs-per-gpu 1 --gpus 4,5 --no-skip-existing --ent-coef 0.001,0.0001

python ramdp_experiments/lightsout_sweep.py --systems ff_ppo_reinforce,ff_ppo_cond_naive,ff_ppo_cond_fac --max-steps 16 --seeds 3 --architectures transformer --hidden-dim 32 --mlp-dim 64 --num-layers 2 --num-heads 4 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 8 --num-minibatches 16 --grid-sizes 5x4 --use-input-layer-norm true --episode-length 10 --wandb true --wandb-project lightsout_sweep-ppo_only-tf_test-sep_1 --yes --runs-per-gpu 1 --gpus 6,7 --no-skip-existing --ent-coef 0.001,0.0001



####### implicit, no layernorm
python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 1,2 --seeds 3 --architectures transformer --hidden-dim 32 --mlp-dim 64 --num-layers 2 --num-heads 4 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 8 --num-minibatches 16 --grid-sizes 5x4 --use-input-layer-norm false,true --episode-length 10 --wandb true --wandb-project lightsout_sweep-ppo_only-tf_test-sep_2 --runs-per-gpu 2 --gpus 4,5,6,7 --no-skip-existing --ent-coef 0.001 --yes
# tmux attach -t1: DONE, gpus 4 5 6 7

python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 4,8 --seeds 3 --architectures transformer --hidden-dim 32 --mlp-dim 64 --num-layers 2 --num-heads 4 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 8 --num-minibatches 16 --grid-sizes 5x4 --use-input-layer-norm false,true --episode-length 10 --wandb true --wandb-project lightsout_sweep-ppo_only-tf_test-sep_2 --runs-per-gpu 2 --gpus 0,1,2,3 --no-skip-existing --ent-coef 0.0001 --yes
# tmux attach -t0: cancelled (this architecture is too strong for budget = 1), gpus 0 1 2 3

## WAIT FOR ABOVE TWO
python ramdp_experiments/lightsout_sweep.py --systems ff_ppo_cond_fac,ff_ppo_cond_naive,ff_ppo_reinforce --max-steps 16 --seeds 3 --architectures transformer --hidden-dim 32 --mlp-dim 64 --num-layers 2 --num-heads 4 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 8 --num-minibatches 16 --grid-sizes 5x4 --use-input-layer-norm false --episode-length 10 --wandb true --wandb-project lightsout_sweep-ppo_only-tf_test-sep_2 --runs-per-gpu 1 --gpus 6,7 --no-skip-existing --ent-coef 0.0001  --yes


####### explicit: for latent_feedback (Earlier runs use full-bandwidth tf implementation (Pre Sept. 2), later runs use skip connection (Post Sept. 2))
python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_explicit_reinforce --budget 1,2 --seeds 3 --architectures transformer_explicit_cot --hidden-dim 32 --mlp-dim 64 --num-layers 2 --num-heads 4 --vocab-size=2,4,8 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 8 --num-minibatches 16 --grid-sizes 5x4 --use-input-layer-norm false,true --use-latent-feedback false --episode-length 10 --wandb true --wandb-project lightsout_sweep-ppo_only-tf_test-sep_2 --runs-per-gpu 2 --gpus 6,7 --no-skip-existing --ent-coef 0.001
# tmux attach -t2: cancelled half way (input layer norm unnecessary), gpus 6 7

python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_explicit_reinforce --budget 1,2 --seeds 3 --architectures transformer_explicit_cot --hidden-dim 32 --mlp-dim 64 --num-layers 2 --num-heads 4 --vocab-size=2,4,8 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 8 --num-minibatches 16 --grid-sizes 5x4 --use-input-layer-norm false,true --use-latent-feedback true --episode-length 10 --wandb true --wandb-project lightsout_sweep-ppo_only-tf_test-sep_2 --runs-per-gpu 2 --gpus 4,5 --no-skip-existing --ent-coef 0.001
# tmux attach -t1: cancelled half way (latent feedback unnecessary), gpus 4 5


python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_explicit_reinforce --budget 4,8 --seeds 3 --architectures transformer_explicit_cot --hidden-dim 32 --mlp-dim 64 --num-layers 2 --num-heads 4 --vocab-size=2,4,8 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 8 --num-minibatches 16 --grid-sizes 5x4 --use-input-layer-norm false,true --use-latent-feedback false,true --episode-length 10 --wandb true --wandb-project lightsout_sweep-ppo_only-tf_test-sep_2 --runs-per-gpu 1 --gpus 2,3 --no-skip-existing --ent-coef 0.0001

## WAIT FOR ABOVE TWO
python ramdp_experiments/lightsout_sweep.py --systems ff_ppo_cond_fac,ff_ppo_cond_naive,ff_ppo_reinforce --max-steps 16 --seeds 3 --architectures transformer_explicit_cot --hidden-dim 32 --mlp-dim 64 --num-layers 2 --num-heads 4 --vocab-size=2,4,8 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 8 --num-minibatches 16 --grid-sizes 5x4 --use-input-layer-norm false,true --use-latent-feedback false,true --episode-length 10 --wandb true --wandb-project lightsout_sweep-ppo_only-tf_test-sep_2 --runs-per-gpu 1 --gpus 6,7 --no-skip-existing --ent-coef 0.0001  --yes
```

- It seems like the architecture is already pretty good with budget = 1. Next step is to do architecture sweep again...
```
# IMPLICIT CoT
# Check architecture with budget of 1
python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 1 --seeds 3 --architectures transformer --hidden-dim 16 --mlp-dim 16,32 --num-layers 2,4 --num-heads 2,4 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 8 --num-minibatches 16 --grid-sizes 5x4 --use-input-layer-norm false --episode-length 10 --wandb true --wandb-project lightsout_sweep-ppo_only-tf_test-sep_2-architecture_search --runs-per-gpu 3 --gpus 2,3 --no-skip-existing --ent-coef 0.001 --yes
# tmux attach -t8: DONE, gpus 2 3

# Check architecture with budget of 8
python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 8 --seeds 3 --architectures transformer --hidden-dim 16 --mlp-dim 16,32 --num-layers 2,4 --num-heads 2,4 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 8 --num-minibatches 16 --grid-sizes 5x4 --use-input-layer-norm false --episode-length 10 --wandb true --wandb-project lightsout_sweep-ppo_only-tf_test-sep_2-architecture_search --runs-per-gpu 2 --gpus 6,7 --no-skip-existing --ent-coef 0.0001 --yes
# tmux attach -t2: CANCELLED, gpus 6 7


# EXPLICIT CoT
python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_explicit_reinforce --budget 1 --seeds 3 --architectures transformer_explicit_cot --hidden-dim 16 --mlp-dim 16,32 --num-layers 2,4 --num-heads 2,4 --vocab-size=1 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 8 --num-minibatches 16 --grid-sizes 5x4 --use-input-layer-norm false --use-latent-feedback false --episode-length 10 --wandb true --wandb-project lightsout_sweep-ppo_only-tf_test-sep_2-architecture_search --runs-per-gpu 3 --gpus 4,5 --no-skip-existing --ent-coef 0.001 --yes
# tmux attach -t1: DONE, gpus 4 5

python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_explicit_reinforce --budget 8 --seeds 3 --architectures transformer_explicit_cot --hidden-dim 16 --mlp-dim 16,32 --num-layers 2,4 --num-heads 2,4 --vocab-size=2 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 8 --num-minibatches 16 --grid-sizes 5x4 --use-input-layer-norm false --use-latent-feedback false --episode-length 10 --wandb true --wandb-project lightsout_sweep-ppo_only-tf_test-sep_2-architecture_search --runs-per-gpu 3 --gpus 0,1 --no-skip-existing --ent-coef 0.0001 --yes
# tmux attach -t0: CANCELLED, gpus 0 1


python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_explicit_reinforce --budget 8 --seeds 3 --architectures transformer_explicit_cot --hidden-dim 16 --mlp-dim 16,32 --num-layers 2,4 --num-heads 2,4 --vocab-size=4 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 8 --num-minibatches 16 --grid-sizes 5x4 --use-input-layer-norm false --use-latent-feedback false --episode-length 10 --wandb true --wandb-project lightsout_sweep-ppo_only-tf_test-sep_2-architecture_search --runs-per-gpu 3 --gpus 2,3 --no-skip-existing --ent-coef 0.0001 --yes
# tmux attach -t8: CANCELLED, gpus 2 3

python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_explicit_reinforce --budget 8 --seeds 3 --architectures transformer_explicit_cot --hidden-dim 16 --mlp-dim 16,32 --num-layers 2,4 --num-heads 2,4 --vocab-size=8 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 8 --num-minibatches 16 --grid-sizes 5x4 --use-input-layer-norm false --use-latent-feedback false --episode-length 10 --wandb true --wandb-project lightsout_sweep-ppo_only-tf_test-sep_2-architecture_search --runs-per-gpu 3 --gpus 4,5 --no-skip-existing --ent-coef 0.0001 --yes
# tmux attach -t1: CANCELLED, gpus 4 5
```


# Comments on Sept 3:
- It seems like when `hidden-dim=16`, the performances with `c=1` is around 0.55 (worse than `hidden-dim=32`)
- Unfortunately it seems like for `c=8` this is significantly worse as well.
- Idea: Use 4x4 grid, add weight decay (this also encourages grokking)
  - Weight decay = 0.1 is too strong, 0.05 has a visible effect already

```
# IMPLICIT CoT
#### Check architecture with budget of 1
python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 1 --seeds 3 --architectures transformer --hidden-dim 64 --mlp-dim 128 --num-layers 2 --num-heads 4 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 8 --num-minibatches 16 --grid-sizes 5x4 --use-input-layer-norm false --episode-length 10 --wandb true --wandb-project lightsout_sweep-ppo_only-tf_test-sep_3-salient-5x4_weight_decay_search --runs-per-gpu 3 --gpus 0,1,2 --no-skip-existing --ent-coef 0.001 --actor-weight-decay 0.0,0.1,0.01
# tmux attach -t0: done, gpus 0 1 2

python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 1 --seeds 3 --architectures transformer --hidden-dim 64 --mlp-dim 128 --num-layers 2 --num-heads 4 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 8 --num-minibatches 16 --grid-sizes 5x4 --use-input-layer-norm false --episode-length 10 --wandb true --wandb-project lightsout_sweep-ppo_only-tf_test-sep_3-salient-5x4_weight_decay_search --runs-per-gpu 3 --gpus 6,7 --no-skip-existing --ent-coef 0.001 --actor-weight-decay 0.02,0.05
# tmux attach -t2: done, gpus 6 7


python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 1 --seeds 3 --architectures transformer --hidden-dim 64 --mlp-dim 128 --num-layers 2 --num-heads 4 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 4 --num-minibatches 8 --grid-sizes 5x4 --use-input-layer-norm false --episode-length 10 --wandb true --wandb-project lightsout_sweep-ppo_only-tf_test-sep_3-salient-5x4_weight_decay_search --runs-per-gpu 3 --gpus 6,7 --no-skip-existing --ent-coef 0.001 --actor-weight-decay 0.0,0.01
# tmux attach -t2: running, gpus 6 7


#### Check architecture with budget of 2
python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 2 --seeds 3 --architectures transformer --hidden-dim 64 --mlp-dim 128 --num-layers 2 --num-heads 4 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 8 --num-minibatches 16 --grid-sizes 5x4 --use-input-layer-norm false --episode-length 10 --wandb true --wandb-project lightsout_sweep-ppo_only-tf_test-sep_3-salient-5x4_weight_decay_search --runs-per-gpu 3 --gpus 0,1,2 --no-skip-existing --ent-coef 0.001 --actor-weight-decay 0.0,0.1,0.01
# tmux attach -t0: running, gpus 0 1 2



#### Check architecture with budget of 8
python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 4 --seeds 3 --architectures transformer --hidden-dim 64 --mlp-dim 128 --num-layers 2 --num-heads 4 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 8 --num-minibatches 16 --grid-sizes 5x4 --use-input-layer-norm false --episode-length 10 --wandb true --wandb-project lightsout_sweep-ppo_only-tf_test-sep_3-salient-5x4_weight_decay_search --runs-per-gpu 3 --gpus 3,4,5 --no-skip-existing --ent-coef 0.0001 --actor-weight-decay 0.0,0.1,0.01
# tmux attach -t1: running, gpus 3 4 5
```


```
python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 1 --seeds 3 --architectures transformer --hidden-dim 64 --mlp-dim 128 --num-layers 2 --num-heads 4 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 4 --num-minibatches 32 --grid-sizes 5x5,4x6,5x6 --use-input-layer-norm false --episode-length 10 --wandb true --wandb-project lightsout_sweep-ppo_only-tf_test-sep_3-salient-maze_search --runs-per-gpu 2 --gpus 0,1,2 --no-skip-existing --ent-coef 0.001 --actor-weight-decay 0.0,0.01

```