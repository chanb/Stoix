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
  - Seems like weight decay = 0.01 is good to have.
- When tuning between epochs and minibatches, more epochs usually take longer than more minibatches even the number of updates are the same (note that the difference here is the batch size)
  - epoch=8, minibatch=8 is fine in performance with 300M steps compared to epoch=8, minibatch=16
  - The best is epoch=4, minibatch=16, in both speed and performance.

```
# IMPLICIT CoT
#### Check architecture with budget of 1
python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 1 --seeds 3 --architectures transformer --hidden-dim 64 --mlp-dim 128 --num-layers 2 --num-heads 4 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 8 --num-minibatches 16 --grid-sizes 5x4 --use-input-layer-norm false --episode-length 10 --wandb true --wandb-project lightsout_sweep-ppo_only-tf_test-sep_3-salient-5x4_weight_decay_search --runs-per-gpu 3 --gpus 0,1,2 --no-skip-existing --ent-coef 0.001 --actor-weight-decay 0.0,0.1,0.01
# tmux attach -t0: done, gpus 0 1 2

python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 1 --seeds 3 --architectures transformer --hidden-dim 64 --mlp-dim 128 --num-layers 2 --num-heads 4 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 8 --num-minibatches 16 --grid-sizes 5x4 --use-input-layer-norm false --episode-length 10 --wandb true --wandb-project lightsout_sweep-ppo_only-tf_test-sep_3-salient-5x4_weight_decay_search --runs-per-gpu 3 --gpus 6,7 --no-skip-existing --ent-coef 0.001 --actor-weight-decay 0.02,0.05
# tmux attach -t2: done, gpus 6 7


python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 1 --seeds 3 --architectures transformer --hidden-dim 64 --mlp-dim 128 --num-layers 2 --num-heads 4 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 4 --num-minibatches 8 --grid-sizes 5x4 --use-input-layer-norm false --episode-length 10 --wandb true --wandb-project lightsout_sweep-ppo_only-tf_test-sep_3-salient-5x4_weight_decay_search --runs-per-gpu 3 --gpus 6,7 --no-skip-existing --ent-coef 0.001 --actor-weight-decay 0.0,0.01
# tmux attach -t2: done, gpus 6 7


python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 1 --seeds 3 --architectures transformer --hidden-dim 64 --mlp-dim 128 --num-layers 2 --num-heads 4 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 8 --num-minibatches 8 --grid-sizes 5x4 --use-input-layer-norm false --episode-length 10 --wandb true --wandb-project lightsout_sweep-ppo_only-tf_test-sep_3-salient-5x4_weight_decay_search --runs-per-gpu 3 --gpus 6,7 --no-skip-existing --ent-coef 0.001 --actor-weight-decay 0.0,0.01 --yes
# tmux attach -t2: done, gpus 6 7

python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 1 --seeds 3 --architectures transformer --hidden-dim 64 --mlp-dim 128 --num-layers 2 --num-heads 4 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 4 --num-minibatches 16 --grid-sizes 5x4 --use-input-layer-norm false --episode-length 10 --wandb true --wandb-project lightsout_sweep-ppo_only-tf_test-sep_3-salient-5x4_weight_decay_search --runs-per-gpu 3 --gpus 6,7 --no-skip-existing --ent-coef 0.001 --actor-weight-decay 0.0,0.01 --yes
# tmux attach -t2: done, gpus 6 7

python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 1 --seeds 3 --architectures transformer --hidden-dim 64 --mlp-dim 128 --num-layers 2 --num-heads 4 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 16 --num-minibatches 4 --grid-sizes 5x4 --use-input-layer-norm false --episode-length 10 --wandb true --wandb-project lightsout_sweep-ppo_only-tf_test-sep_3-salient-5x4_weight_decay_search --runs-per-gpu 3 --gpus 0,1 --no-skip-existing --ent-coef 0.001 --actor-weight-decay 0.0,0.01 --yes
# tmux attach -t0: CANCELLED, gpus 0 1

python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 1 --seeds 3 --architectures transformer --hidden-dim 64 --mlp-dim 128 --num-layers 2 --num-heads 4 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 4 --num-minibatches 32 --grid-sizes 5x4 --use-input-layer-norm false --episode-length 10 --wandb true --wandb-project lightsout_sweep-ppo_only-tf_test-sep_3-salient-5x4_weight_decay_search --runs-per-gpu 3 --gpus 6,7 --no-skip-existing --ent-coef 0.001 --actor-weight-decay 0.0,0.01 --yes
# tmux attach -t2: done, gpus 6 7

python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 1 --seeds 3 --architectures transformer --hidden-dim 64 --mlp-dim 128 --num-layers 2 --num-heads 4 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 2 --num-minibatches 64 --grid-sizes 5x4 --use-input-layer-norm false --episode-length 10 --wandb true --wandb-project lightsout_sweep-ppo_only-tf_test-sep_3-salient-5x4_weight_decay_search --runs-per-gpu 3 --gpus 0,1 --no-skip-existing --ent-coef 0.001 --actor-weight-decay 0.0,0.01 --yes
# tmux attach -t0: done, gpus 0 1


#### Check architecture with budget of 2
python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 2 --seeds 3 --architectures transformer --hidden-dim 64 --mlp-dim 128 --num-layers 2 --num-heads 4 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 8 --num-minibatches 16 --grid-sizes 5x4 --use-input-layer-norm false --episode-length 10 --wandb true --wandb-project lightsout_sweep-ppo_only-tf_test-sep_3-salient-5x4_weight_decay_search --runs-per-gpu 3 --gpus 0,1,2 --no-skip-existing --ent-coef 0.001 --actor-weight-decay 0.0,0.1,0.01
# tmux attach -t0: done, gpus 0 1 2



#### Check architecture with budget of 8
python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 4 --seeds 3 --architectures transformer --hidden-dim 64 --mlp-dim 128 --num-layers 2 --num-heads 4 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 8 --num-minibatches 16 --grid-sizes 5x4 --use-input-layer-norm false --episode-length 10 --wandb true --wandb-project lightsout_sweep-ppo_only-tf_test-sep_3-salient-5x4_weight_decay_search --runs-per-gpu 3 --gpus 3,4,5 --no-skip-existing --ent-coef 0.0001 --actor-weight-decay 0.0,0.1,0.01
# tmux attach -t1: done, gpus 3 4 5
```


Try different grid size:
- Seems like 5x5 is sufficiently large for the problem to be difficult for budget = 1
```
python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 1 --seeds 3 --architectures transformer --hidden-dim 64 --mlp-dim 128 --num-layers 2 --num-heads 4 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 16 --num-minibatches 4 --grid-sizes 4x6,5x5,6x5 --use-input-layer-norm false --episode-length 10 --wandb true --wandb-project lightsout_sweep-ppo_only-tf_test-sep_3-salient-maze_search --runs-per-gpu 3 --gpus 0,1,2 --no-skip-existing --ent-coef 0.001 --actor-weight-decay 0.01 --yes
# tmux attach -t2: done, gpus 0 1 2
```


Explicit CoT
```
python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_explicit_reinforce --budget 1 --seeds 3 --architectures transformer_explicit_cot --hidden-dim 64 --mlp-dim 128 --num-layers 2 --num-heads 4 --vocab-size=1 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 4 --num-minibatches 16 --grid-sizes 5x4 --use-input-layer-norm false --use-latent-feedback false --episode-length 10 --wandb true --wandb-project lightsout_sweep-ppo_only-tf_test-sep_3-salient-5x4_weight_decay_search --runs-per-gpu 3 --gpus 3,4,5 --no-skip-existing --ent-coef 0.001 --actor-weight-decay 0.0,0.01,0.1 --yes
# tmux attach -t1: done, gpus 3 4 5
```

5x5 maze (salient3)
- For implcit CoT, we don't seem to be getting better performance.
  - Can we add a similar technique to the latent states as explicit CoT, where we penalize how far the latent states can drift?
```
python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 1,2,4 --seeds 3 --architectures transformer --hidden-dim 64 --mlp-dim 128 --num-layers 2 --num-heads 4 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 4 --num-minibatches 16 --grid-sizes 5x5 --use-input-layer-norm false --episode-length 10 --wandb true --wandb-project lightsout_sweep-ppo_only-tf_test-sep_3-salient-5x5_weight_decay_search --runs-per-gpu 1 --gpus 0,1,2 --no-skip-existing --ent-coef 0.001 --actor-weight-decay 0.01
# tmux attach -t11: done, gpus 0 1 2, salient3

python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 8,16 --seeds 3 --architectures transformer --hidden-dim 64 --mlp-dim 128 --num-layers 2 --num-heads 4 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 4 --num-minibatches 16 --grid-sizes 5x5 --use-input-layer-norm false --episode-length 10 --wandb true --wandb-project lightsout_sweep-ppo_only-tf_test-sep_3-salient-5x5_weight_decay_search --runs-per-gpu 1 --gpus 0,1,2 --no-skip-existing --ent-coef 0.001 --actor-weight-decay 0.01
# tmux attach -t11: running, gpus 0 1 2, salient3
```

5x5 maze (salient4)
- For explicit CoT, maybe the clipping needs to be less aggressive?
  - It is also the case that we clip based on the (unnormalized) sequence-level probability rather than per-token level, so that might be too aggressive as well.
    - XXX: CHANGING THIS TO BE PER TOKEN LEVEL
      - This seems to be helping so far (for vocab_size=2)
```
# explicit CoT run
python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_explicit_reinforce --budget 1 --seeds 3 --architectures transformer_explicit_cot --hidden-dim 64 --mlp-dim 128 --num-layers 2 --num-heads 4 --vocab-size=1 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 4 --num-minibatches 16 --grid-sizes 5x5 --use-input-layer-norm false --use-latent-feedback false --episode-length 10 --wandb true --wandb-project lightsout_sweep-ppo_only-tf_test-sep_3-salient-5x5_weight_decay_search --runs-per-gpu 1 --gpus 3,4,5 --no-skip-existing --ent-coef 0.001 --actor-weight-decay 0.01 --yes
# tmux attach -t1: done, gpus 3 4 5

python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_explicit_reinforce --budget 2,4,8 --seeds 3 --architectures transformer_explicit_cot --hidden-dim 64 --mlp-dim 128 --num-layers 2 --num-heads 4 --vocab-size=2,4,8 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 4 --num-minibatches 16 --grid-sizes 5x5 --use-input-layer-norm false --use-latent-feedback false --episode-length 10 --wandb true --wandb-project lightsout_sweep-ppo_only-tf_test-sep_3-salient-5x5_weight_decay_search --runs-per-gpu 1 --gpus 3,4,5,6,7 --no-skip-existing --ent-coef 0.001 --actor-weight-decay 0.01 --yes
# tmux attach -t1: done, gpus 3 4 5 6 7 (per-sequence clip)


python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_explicit_reinforce --budget 2,4,8 --seeds 3 --architectures transformer_explicit_cot --hidden-dim 64 --mlp-dim 128 --num-layers 2 --num-heads 4 --vocab-size=2,4,8 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 4 --num-minibatches 16 --grid-sizes 5x5 --use-input-layer-norm false --use-latent-feedback false --episode-length 10 --wandb true --wandb-project lightsout_sweep-ppo_only-tf_test-sep_3-salient-5x5_weight_decay_search --runs-per-gpu 1 --gpus 0,1,2,3,4,5,6,7 --no-skip-existing --ent-coef 0.001 --actor-weight-decay 0.01 --yes
# tmux attach -t1: done, gpus 0 .. 7 (per-token clip)


# implicit CoT run
python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 2,4,8 --seeds 3 --architectures transformer --hidden-dim 64 --mlp-dim 128 --num-layers 2 --num-heads 4 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 4 --num-minibatches 16 --grid-sizes 5x5 --use-input-layer-norm false --episode-length 10 --wandb true --wandb-project lightsout_sweep-ppo_only-tf_test-sep_3-salient-5x5_weight_decay_search --runs-per-gpu 1 --gpus 0,1,2,3,4,5,6,7 --no-skip-existing --ent-coef 0.001 --actor-weight-decay 0.01 --latent-kl-coef 0.1,0.01,0.001 --yes
# tmux attach -t1: done, gpus 0 .. 7 (kl penalty, salient4)
```


# Comments on Sept 5:
- Both shared and unshared IRU runs can be done within 3 hours of vulcan server.
- For shared IRU, 5x4 seems like a good spot. The current model still fails to follow the increasing budget -> increasing performance trend
  - Q - V estimators are worse than G - V. The former never learns to get positive return (critic converges too quickly?)
    - Running 797795_[1] on vulcan with larger model + try smaller critic LR (1e-5)
      - Smaller critic LR just learns too slow
  - Try 2 layers: 797917
    - It learns now, but unfortunately performance is poorer than G - V
  

- For unshared IRU, somewhat a similar story for IRU, 5x4 is a good spot. The increasing budget -> increasing performance trend is there however.
  - Q - V is still worse than G - V. In 5x5 the former never learns
  - Try 2 layers: 797911
    - Same as shared IRU

- For eCoT,
  - with input layer norm
    - FAC seems to be get non-trivial return
    - With 2 GPUs, vulcan can complete each variant within 12 hours (5 seeds, each run is ~4 hours)
      - Running both Q_sep - V (797818_1) and G - V (797820_1) on vulcan
      - Running fixed budget 797859_1
  - without input layer norm
    - Running 797851_1
    - Running 797852_1
    - Running 797853_1
    - Fixed budget: 797858_1

- For iCoT, latent KL penalty seems good with range ~1e-5 to ~1e-4:
  - `python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 2,4,8 --seeds 3 --architectures transformer --hidden-dim 64 --mlp-dim 128 --num-layers 2 --num-heads 4 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 4 --num-minibatches 16 --grid-sizes 5x5 --use-input-layer-norm false --episode-length 10 --wandb true --wandb-project lightsout_sweep-ppo_only-tf_test-sep_3-salient-5x5_weight_decay_search --runs-per-gpu 1 --gpus 0,1,2,3,4,5,6,7 --no-skip-existing --ent-coef 0.001 --actor-weight-decay 0.01 --latent-kl-coef 5e-5 --yes`
  - tmux attach -t1: done, gpus 0 .. 7 (kl penalty, salient4)
  - But the trend is still not monotonically increasing


- NEW IDEA: We should be recomputing the advantage per epoch, especially for Q - V variants because otherwise the bias is too high in the beginning
  - Unshared IRU:
    - ALL: 798656_1
  - IRU:
    - ALL: 798663_1
  - eCoT:
    - Fixed budget: 798681_1
    - `none`: 798747_1
    - `naive`: 798746_1
    - `fac`: 798745_1
  - eCoT (vs=32):
    - Fixed budget: 798678_1
    - `none`: 798754_1
    - `naive`: 798751_1
    - `fac`: 798748_1
  - iCoT:
    - Fixed budget: 803346_1
    - `none`: 803339_1
    - `naive`: 803343_1
    - `fac`: 803345_1

  - It seems like the Q-function is stalling the learning for naive and fac.
    - Update critic before policy test: `python ramdp_experiments/lightsout_sweep.py --systems ff_ppo_cond_fac,ff_ppo_cond_naive,ff_ppo_reinforce --max-steps 16 --seeds 5 --architectures iru --hidden-dim 32 --num-layers 1 --total-timesteps 1e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 4 --num-minibatches 16 --grid-sizes 4x5 --use-input-layer-norm true --episode-length 10 --wandb true --wandb-project lightsout-iru-sep5-recompute_adv --runs-per-gpu 1 --gpus 0,1,2,3,4,5,6,7 --no-skip-existing --ent-coef 0.001 --gamma 0.999 --actor-weight-decay 0.01 --recompute-advantages true --yes --critic-before-actor true`
      - tmux attach -t0, DONE, gpus 0 .. 7 (salient4)
      - Doesn't seem to impact much arguably even worse...
    - Smaller actor weight decay: `python ramdp_experiments/lightsout_sweep.py --systems ff_ppo_cond_fac,ff_ppo_cond_naive,ff_ppo_reinforce --max-steps 16 --seeds 5 --architectures iru --hidden-dim 32 --num-layers 1 --total-timesteps 1e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 4 --num-minibatches 16 --grid-sizes 4x5 --use-input-layer-norm true --episode-length 10 --wandb true --wandb-project lightsout-iru-sep5-recompute_adv --runs-per-gpu 1 --gpus 0,1,2,3,4,5,6,7 --no-skip-existing --ent-coef 0.001 --gamma 0.999 --actor-weight-decay 0.005 --recompute-advantages true --yes --critic-before-actor false`
      - tmux attach -t0, DONE, gpus 0 .. 7 (salient4)
      - All performs better, but the trend is still the same as before
    - Larger critic learning rate: `python ramdp_experiments/lightsout_sweep.py --systems ff_ppo_cond_fac,ff_ppo_cond_naive,ff_ppo_reinforce --max-steps 16 --seeds 5 --architectures iru --hidden-dim 32 --num-layers 1 --total-timesteps 1e8 --clip-value-loss false --lr 3e-4 --critic-lr 5e-4 --epochs 4 --num-minibatches 16 --grid-sizes 4x5 --use-input-layer-norm true --episode-length 10 --wandb true --wandb-project lightsout-iru-sep5-recompute_adv --runs-per-gpu 1 --gpus 0,1,2,3,4,5,6,7 --no-skip-existing --ent-coef 0.001 --gamma 0.999 --actor-weight-decay 0.01 --recompute-advantages true --yes --critic-before-actor false`
      - tmux attach -t0, DONE, gpus 0 .. 7 (salient4)
      - Worse.
    - Larger entropy coef: `python ramdp_experiments/lightsout_sweep.py --systems ff_ppo_cond_fac,ff_ppo_cond_naive,ff_ppo_reinforce --max-steps 16 --seeds 5 --architectures iru --hidden-dim 32 --num-layers 1 --total-timesteps 1e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 4 --num-minibatches 16 --grid-sizes 4x5 --use-input-layer-norm true --episode-length 10 --wandb true --wandb-project lightsout-iru-sep5-recompute_adv --runs-per-gpu 1 --gpus 0,1,2,3,4,5,6,7 --no-skip-existing --ent-coef 0.005 --gamma 0.999 --actor-weight-decay 0.01 --recompute-advantages true --yes --critic-before-actor false`
      - tmux attach -t0, DONE, gpus 0 .. 7 (salient4)
    - Just train for longer: `python ramdp_experiments/lightsout_sweep.py --systems ff_ppo_cond_fac,ff_ppo_cond_naive,ff_ppo_reinforce --max-steps 16 --seeds 5 --architectures iru --hidden-dim 32 --num-layers 1 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 4 --num-minibatches 16 --grid-sizes 4x5 --use-input-layer-norm true --episode-length 10 --wandb true --wandb-project lightsout-iru-sep5-recompute_adv --runs-per-gpu 1 --gpus 0,1,2,3,4,5,6,7 --no-skip-existing --ent-coef 0.001 --gamma 0.999 --actor-weight-decay 0.005 --recompute-advantages true --yes --critic-before-actor false`
      - tmux attach -t0, DONE, gpus 0 .. 7 (salient4)
    - Just train for longer w/ larger ent coef: `python ramdp_experiments/lightsout_sweep.py --systems ff_ppo_cond_fac,ff_ppo_cond_naive,ff_ppo_reinforce --max-steps 16 --seeds 5 --architectures iru --hidden-dim 32 --num-layers 1 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 4 --num-minibatches 16 --grid-sizes 4x5 --use-input-layer-norm true --episode-length 10 --wandb true --wandb-project lightsout-iru-sep5-recompute_adv --runs-per-gpu 1 --gpus 0,1,2,3,4,5,6,7 --no-skip-existing --ent-coef 0.005 --gamma 0.999 --actor-weight-decay 0.005 --recompute-advantages true --yes --critic-before-actor false`
      - tmux attach -t0, DONE, gpus 0 .. 7 (salient4)
    - Just train for longer + standardize advantage: `python ramdp_experiments/lightsout_sweep.py --systems ff_ppo_cond_fac,ff_ppo_cond_naive,ff_ppo_reinforce --max-steps 16 --seeds 5 --architectures iru --hidden-dim 32 --num-layers 1 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 4 --num-minibatches 16 --grid-sizes 4x5 --use-input-layer-norm true --episode-length 10 --wandb true --wandb-project lightsout-iru-sep5-recompute_adv --runs-per-gpu 1 --gpus 0,1,2,3,4,5,6,7 --no-skip-existing --ent-coef 0.001 --gamma 0.999 --actor-weight-decay 0.005 --recompute-advantages true --yes --critic-before-actor false --standardize-advantages true`
      - tmux attach -t0, DONE, gpus 0 .. 7 (salient4)
      - HORRIBLE PERFORMANCE
    - Just train for longer + standardize advantage + more layers: `python ramdp_experiments/lightsout_sweep.py --systems ff_ppo_cond_fac,ff_ppo_cond_naive,ff_ppo_reinforce --max-steps 16 --seeds 5 --architectures iru --hidden-dim 64 --num-layers 1,2 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 4 --num-minibatches 16 --grid-sizes 4x5 --use-input-layer-norm true --episode-length 10 --wandb true --wandb-project lightsout-iru-sep5-recompute_adv --runs-per-gpu 1 --gpus 0,1,2,3,4,5,6,7 --no-skip-existing --ent-coef 0.001 --gamma 0.999 --actor-weight-decay 0.005 --recompute-advantages true --yes --critic-before-actor false --standardize-advantages false,true`
      - tmux attach -t0, DONE, gpus 0 .. 7 (salient4)
    - Just train for longer + more layers for fixed budget: `python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 1,2,4,8,16 --seeds 5 --architectures iru --hidden-dim 64 --num-layers 2 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 4 --num-minibatches 16 --grid-sizes 4x5 --use-input-layer-norm true --episode-length 10 --wandb true --wandb-project lightsout-iru-sep5-recompute_adv --runs-per-gpu 1 --gpus 0,1,2,3,4,5,6,7 --no-skip-existing --ent-coef 0.001 --gamma 0.999 --actor-weight-decay 0.005 --recompute-advantages true --yes --critic-before-actor false`
      - tmux attach -t0, DONE, gpus 0 .. 7 (salient4)
    - Just train for longer + standardize advantage + more layers: `python ramdp_experiments/lightsout_sweep.py --systems ff_ppo_cond_fac,ff_ppo_cond_naive,ff_ppo_reinforce --max-steps 16 --seeds 5 --architectures iru --hidden-dim 32 --num-layers 2,4 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 4 --num-minibatches 16 --grid-sizes 4x5 --use-input-layer-norm true --episode-length 10 --wandb true --wandb-project lightsout-iru-sep5-recompute_adv --runs-per-gpu 1 --gpus 0,1,2,3,4,5,6,7 --no-skip-existing --ent-coef 0.001 --gamma 0.999 --actor-weight-decay 0.005 --recompute-advantages true --yes --critic-before-actor false --standardize-advantages false,true`
      - tmux attach -t0, DONE, gpus 0 .. 7 (salient4)
    - Just train for longer + standardize advantage + larger critic learning rate: `python ramdp_experiments/lightsout_sweep.py --systems ff_ppo_cond_fac,ff_ppo_cond_naive,ff_ppo_reinforce --max-steps 16 --seeds 5 --architectures iru --hidden-dim 32 --num-layers 1 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 5e-4 --epochs 4 --num-minibatches 16 --grid-sizes 4x5 --use-input-layer-norm true --episode-length 10 --wandb true --wandb-project lightsout-iru-sep5-recompute_adv --runs-per-gpu 1 --gpus 0,1,2,3,4,5,6,7 --no-skip-existing --ent-coef 0.001 --gamma 0.999 --actor-weight-decay 0.01 --recompute-advantages true --yes --critic-before-actor false --standardize-advantages false,true`
      - tmux attach -t0, DONE, gpus 0 .. 7 (salient4)

# Comments on Sept 6:
- Don't use standardized advantage (even though it gives a bit of signal to learn even before critic is good)
- It's better to set `ent_coef=0.005` for both `naive` and `fac`.
  - The policy entropy is significantly lower than `reinforce` with same `ent_coef`.
- It's better to set `actor_weight_decay=0.005`
- Increase max grad norm: 811267_1
- Critic before actor, no recompute advantage: 811268_1

## Knapsack
### eCoT
```
python ramdp_experiments/jumanji_fixed_budget_sweep.py --systems ff_ppo_explicit_reinforce --budget 1 --seeds 5 --runs-per-gpu 2 --architectures transformer_explicit_cot --hidden-dim 128 --mlp-dim 512 --num-layers 2 --num-heads 8 --vocab-size 1 --total-timesteps 1e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --envs knapsack --knapsack-num-items 50 --knapsack-total-budget 12.5 --gpus 0,1,2,3 --rollout-length 64 --epochs 4 --num-minibatches 16 --ent-coef 0.001 --gamma 0.995 --actor-weight-decay 0.005 --critic-before-actor true --wandb true --wandb-project knapsack-tf-sep6 --yes

python ramdp_experiments/jumanji_fixed_budget_sweep.py --systems ff_ppo_explicit_reinforce --budget 2,4,8,16 --seeds 5 --runs-per-gpu 2 --architectures transformer_explicit_cot --hidden-dim 128 --mlp-dim 512 --num-layers 2 --num-heads 8 --vocab-size 2,4,8 --total-timesteps 1e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --envs knapsack --knapsack-num-items 50 --knapsack-total-budget 12.5 --gpus 0,1,2,3 --rollout-length 64 --epochs 4 --num-minibatches 16 --ent-coef 0.001 --gamma 0.995 --actor-weight-decay 0.005 --critic-before-actor true --wandb true --wandb-project knapsack-tf-sep6 --yes

python ramdp_experiments/jumanji_sweep.py --systems ff_ppo_explicit_cond_fac,ff_ppo_explicit_cond_naive,ff_ppo_explicit_reinforce --max-steps 16 --seeds 5 --runs-per-gpu 2 --architectures transformer_explicit_cot --hidden-dim 128 --mlp-dim 512 --num-layers 2 --num-heads 8 --vocab-size 2,4,8 --total-timesteps 1e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --envs knapsack --knapsack-num-items 50 --knapsack-total-budget 12.5 --gpus 4,5,6,7 --rollout-length 64 --epochs 4 --num-minibatches 16 --ent-coef 0.001 --gamma 0.995 --actor-weight-decay 0.005 --critic-before-actor true --wandb true --wandb-project knapsack-tf-sep6 --no-skip-existing --yes
```

### iCoT
813612


### IRU
813611


## Maze
(stoix) chanb@vulcan2:~/research/iclr_2027/Stoix$ sbatch ramdp_experiments/slurm/maze-iru.sh
Submitted batch job 814459
(stoix) chanb@vulcan2:~/research/iclr_2027/Stoix$ sbatch ramdp_experiments/slurm/maze-ecot.sh
Submitted batch job 814460
(stoix) chanb@vulcan2:~/research/iclr_2027/Stoix$ sbatch ramdp_experiments/slurm/maze-icot.sh
Submitted batch job 814461


# Comments on Sept 7
- Knapsack has slightly nice trend, but they're all close to optimal.
- I think for lightsout and maze (with lower returns) we can just have larger models and decreasing the budget
- Maybe it's worthwhile to think about whether the halting mechanism learns as quickly as the policy.
  - Stop gradient from flowing to torso from halting predictor: Seems to help in the small parameter space (maze)---but not for IRU lightsout with larger hidden dim
  - Halt entropy?
- For IRU lightsout, 32 hidden dim is too weak, (max - min: ~4% pt)
  - Both 64 and 128 hidden dims have better improvement (max - min: ~7% pt and ~14% pt resp.)
  - Train for 3e8 instead of just 1e8 (like the TF models)
- Maze should use CNN architecture

## Knapsack
TODO

## Maze
### IRU
```
python ramdp_experiments/jumanji_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 1 --seeds 5 --runs-per-gpu 1 --architectures cnn+iru --hidden-dim 32 --num-layers 4 --total-timesteps 1e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --envs maze --maze-size 10 --gpus 0,1,2,3,4,5,6,7 --rollout-length 10 --epochs 2 --num-minibatches 4 --ent-coef 0.01 --gamma 0.999 --actor-weight-decay 0.005 --critic-before-actor true --wandb true --wandb-project maze-sep7-v2 --yes --no-skip-existing

python ramdp_experiments/jumanji_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 2,3,4,5 --seeds 5 --runs-per-gpu 1 --architectures cnn+iru --hidden-dim 32 --num-layers 4 --total-timesteps 1e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --envs maze --maze-size 10 --gpus 0,1,2,3,4,5,6,7 --rollout-length 10 --epochs 2 --num-minibatches 4 --ent-coef 0.01 --gamma 0.999 --actor-weight-decay 0.005 --critic-before-actor true --wandb true --wandb-project maze-sep7-v2 --yes --no-skip-existing

python ramdp_experiments/jumanji_sweep.py --systems ff_ppo_cond_fac,ff_ppo_reinforce,ff_ppo_cond_naive --max-steps 5 --seeds 5 --runs-per-gpu 1 --architectures cnn+iru --hidden-dim 32 --num-layers 4 --total-timesteps 1e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --envs maze --maze-size 10 --gpus 0,1,2,3,4,5,6,7 --rollout-length 10 --epochs 2 --num-minibatches 4 --ent-coef 0.01 --gamma 0.999 --actor-weight-decay 0.005 --critic-before-actor true --stop-gradient-halting-input false --halting-ent-coef 0.0,0.001,0.01 --wandb true --wandb-project maze-sep7-v2 --yes --no-skip-existing
```


## Sokoban
### IRU
```
python ramdp_experiments/jumanji_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 1 --seeds 5 --runs-per-gpu 1 --architectures cnn+iru --hidden-dim 32 --num-layers 4 --total-timesteps 1e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --envs sokoban --sokoban-generator unfiltered-train --gpus 0,1,2,3,4,5,6,7 --rollout-length 10 --epochs 2 --num-minibatches 4 --ent-coef 0.01 --gamma 0.999 --actor-weight-decay 0.005 --critic-before-actor true --wandb true --wandb-project sokoban-sep7 --yes --no-skip-existing

python ramdp_experiments/jumanji_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 2,3,4,5 --seeds 5 --runs-per-gpu 1 --architectures cnn+iru --hidden-dim 32 --num-layers 4 --total-timesteps 1e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --envs sokoban --sokoban-generator unfiltered-train --gpus 0,1,2,3,4,5,6,7 --rollout-length 10 --epochs 2 --num-minibatches 4 --ent-coef 0.01 --gamma 0.999 --actor-weight-decay 0.005 --critic-before-actor true --wandb true --wandb-project sokoban-sep7 --yes --no-skip-existing

python ramdp_experiments/jumanji_sweep.py --systems ff_ppo_cond_fac,ff_ppo_reinforce,ff_ppo_cond_naive --max-steps 5 --seeds 5 --runs-per-gpu 1 --architectures cnn+iru --hidden-dim 32 --num-layers 4 --total-timesteps 1e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --envs sokoban --sokoban-generator unfiltered-train --gpus 0,1,2,3,4,5,6,7 --rollout-length 10 --epochs 2 --num-minibatches 4 --ent-coef 0.01 --gamma 0.999 --actor-weight-decay 0.005 --critic-before-actor true --stop-gradient-halting-input false --wandb true --wandb-project sokoban-sep7 --yes --no-skip-existing
```


# Comments on Sept 8
- For lightsout, 64 dim is the best for all architectures. When using 128 dim, the models are already pretty good without extra compute: See https://wandb.ai/bpychan-university-of-alberta/lightsout-sep7
- There was a bug in `cond_fac`, where the Q-function inference is `Q(s, a, c)` rather than `gamma**(c - 1) Q(s, a, 1)`
  - IRU lightsout: `cond_fac` is somewhat noisy, the performance is fluctuating a lot as the model if decreasing its runtime
- https://wandb.ai/bpychan-university-of-alberta/lightsout-sep8 : no cba nor recompute adv
- https://wandb.ai/bpychan-university-of-alberta/lightsout-sep7 : cba but no recompute adv

## Maze
### IRU
```
python ramdp_experiments/jumanji_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 1,2,3,4,5 --seeds 5 --runs-per-gpu 1 --architectures cnn+iru --hidden-dim 32 --num-layers 4 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --envs maze --maze-size 10 --gpus 0,1,2,3,4,5,6,7 --rollout-length 10 --epochs 2 --num-minibatches 4 --ent-coef 0.01 --gamma 0.999 --actor-weight-decay 0.005 --critic-before-actor true --wandb true --wandb-project maze-sep7-v2 --yes --no-skip-existing

python ramdp_experiments/jumanji_sweep.py --systems ff_ppo_cond_fac,ff_ppo_reinforce,ff_ppo_cond_naive --max-steps 5 --seeds 5 --runs-per-gpu 1 --architectures cnn+iru --hidden-dim 32 --num-layers 4 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --envs maze --maze-size 10 --gpus 0,1,2,3,4,5,6,7 --rollout-length 10 --epochs 2 --num-minibatches 4 --ent-coef 0.01 --gamma 0.999 --actor-weight-decay 0.005 --critic-before-actor true --stop-gradient-halting-input false --wandb true --wandb-project maze-sep7-v2 --yes --no-skip-existing

python ramdp_experiments/jumanji_sweep.py --systems ff_ppo_cond_fac,ff_ppo_cond_naive --max-steps 5 --seeds 5 --runs-per-gpu 1 --architectures cnn+iru --hidden-dim 32 --num-layers 4 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --envs maze --maze-size 10 --gpus 0,1,2,3,4,5,6,7 --rollout-length 10 --epochs 2 --num-minibatches 4 --ent-coef 0.01 --gamma 0.999 --actor-weight-decay 0.005 --critic-before-actor true --stop-gradient-halting-input false --wandb true --wandb-project maze-sep7-v2 --yes --no-skip-existing --qv-critic separate


python ramdp_experiments/jumanji_sweep.py --systems ff_ppo_cond_fac,ff_ppo_reinforce,ff_ppo_cond_naive --max-steps 5 --seeds 5 --runs-per-gpu 1 --architectures cnn+iru --hidden-dim 32 --num-layers 4 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --envs maze --maze-size 10 --gpus 0,1,2,3,4,5,6,7 --rollout-length 10 --epochs 2 --num-minibatches 4 --ent-coef 0.01 --gamma 0.999 --actor-weight-decay 0.005 --critic-before-actor false --stop-gradient-halting-input false --wandb true --wandb-project maze-sep7-v2 --yes --no-skip-existing --qv-critic separate


python ramdp_experiments/jumanji_sweep.py --systems ff_ppo_cond_fac,ff_ppo_reinforce,ff_ppo_cond_naive --max-steps 5 --seeds 5 --runs-per-gpu 1 --architectures cnn+iru --hidden-dim 32 --num-layers 4 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --envs maze --maze-size 10 --gpus 0,1,2,3,4,5,6,7 --rollout-length 10 --epochs 2 --num-minibatches 4 --ent-coef 0.01 --gamma 0.999 --actor-weight-decay 0.005 --critic-before-actor false --stop-gradient-halting-input false --wandb true --wandb-project maze-sep7-v2 --yes --no-skip-existing --qv-critic shared


python ramdp_experiments/jumanji_sweep.py --systems ff_ppo_cond_fac,ff_ppo_reinforce,ff_ppo_cond_naive --max-steps 5 --seeds 5 --runs-per-gpu 1 --architectures cnn+iru --hidden-dim 32 --num-layers 4 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 1e-4 --envs maze --maze-size 10 --gpus 0,1,2,3,4,5,6,7 --rollout-length 10 --epochs 2 --num-minibatches 4 --ent-coef 0.01,0.001 --gamma 0.999 --actor-weight-decay 0.005 --critic-before-actor false --stop-gradient-halting-input false --wandb true --wandb-project maze-sep7-v2 --yes --no-skip-existing --qv-critic shared

python ramdp_experiments/jumanji_sweep.py --systems ff_ppo_cond_fac,ff_ppo_reinforce,ff_ppo_cond_naive --max-steps 5 --seeds 5 --runs-per-gpu 1 --architectures cnn+iru --hidden-dim 32 --num-layers 4 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --envs maze --maze-size 10 --gpus 0,1,2,3,4,5,6,7 --rollout-length 10 --epochs 2 --num-minibatches 4 --ent-coef 0.001 --gamma 0.999 --actor-weight-decay 0.005 --critic-before-actor false --stop-gradient-halting-input false --wandb true --wandb-project maze-sep7-v2 --yes --no-skip-existing --qv-critic shared

python ramdp_experiments/jumanji_sweep.py --systems ff_ppo_cond_fac,ff_ppo_cond_naive --max-steps 5 --seeds 5 --runs-per-gpu 1 --architectures cnn+iru --hidden-dim 32 --num-layers 4 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --envs maze --maze-size 10 --gpus 0,1,2,3,4,5,6,7 --rollout-length 10 --epochs 2 --num-minibatches 4 --ent-coef 0.0001 --gamma 0.999 --actor-weight-decay 0.005 --critic-before-actor false --stop-gradient-halting-input false --wandb true --wandb-project maze-sep7-v2 --yes --no-skip-existing --qv-critic shared

python ramdp_experiments/jumanji_sweep.py --systems ff_ppo_cond_fac,ff_ppo_cond_naive --max-steps 5 --seeds 3 --runs-per-gpu 1 --architectures cnn+iru --hidden-dim 32 --num-layers 4 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --envs maze --maze-size 10 --gpus 0,1,2,3,4,5,6,7 --rollout-length 10 --epochs 2 --num-minibatches 4 --ent-coef 0.001 --gamma 0.999 --actor-weight-decay 0.005 --critic-before-actor false --stop-gradient-halting-input false --wandb true --wandb-project maze-sep7-v2 --yes --no-skip-existing --qv-critic separate
```

## Sokoban
### IRU
```
python ramdp_experiments/jumanji_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 1,2,3,4,5 --seeds 5 --runs-per-gpu 1 --architectures cnn+iru --hidden-dim 128 --num-layers 4 --total-timesteps 1e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --envs sokoban --sokoban-generator unfiltered-train --gpus 0,1,2,3,4,5,6,7 --rollout-length 32 --epochs 4 --num-minibatches 8 --ent-coef 0.01 --gamma 0.999 --actor-weight-decay 0.005 --critic-before-actor true --wandb true --wandb-project sokoban-sep8 --yes --no-skip-existing

python ramdp_experiments/jumanji_sweep.py --systems ff_ppo_cond_fac,ff_ppo_reinforce,ff_ppo_cond_naive --max-steps 5 --seeds 5 --runs-per-gpu 1 --architectures cnn+iru --hidden-dim 128 --num-layers 4 --total-timesteps 1e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --envs sokoban --sokoban-generator unfiltered-train --gpus 0,1,2,3,4,5,6,7 --rollout-length 32 --epochs 2 --num-minibatches 4 --ent-coef 0.01 --gamma 0.999 --actor-weight-decay 0.005 --critic-before-actor true --stop-gradient-halting-input false --wandb true --wandb-project sokoban-sep8 --yes --no-skip-existing
```


# Comments on Sept 9
tmux attach -t0
```
python ramdp_experiments/jumanji_sweep.py --systems ff_ppo_cond_fac,ff_ppo_cond_naive --max-steps 5 --seeds 1 --runs-per-gpu 1 --architectures cnn+iru --hidden-dim 16,32,64 --num-layers 2,4 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4,1e-3 --envs maze --maze-size 10 --gpus 0,1,2,3,4,5 --rollout-length 10 --epochs 2 --num-minibatches 4 --ent-coef 0.001 --gamma 0.999 --actor-weight-decay 0.0 --critic-before-actor false --stop-gradient-halting-input false --wandb true --wandb-project maze-sep9 --yes --no-skip-existing --qv-critic separate,shared --critic-weight-decay 0.0
```
- Bad runs
  - `['maze-sz10', 'ppo_cond_fac-cnn+iru', 'mn1-mx5', 'hd32-lr0.0003-clr0.0003-ec0.001-mgn0.5-nl2-sharedqv', 'ep2-mb4-clip0.2-l2c']`
  - `['maze-sz10', 'ppo_cond_fac-cnn+iru', 'mn1-mx5', 'hd16-lr0.0003-clr0.001-ec0.001-mgn0.5-nl2-sharedqv', 'ep2-mb4-clip0.2-l2c']`
  - `['maze-sz10', 'ppo_cond_fac-cnn+iru', 'mn1-mx5', 'hd32-lr0.0003-clr0.001-ec0.001-mgn0.5-nl2-sharedqv', 'ep2-mb4-clip0.2-l2c']`
  - `['maze-sz10', 'ppo_cond_fac-cnn+iru', 'mn1-mx5', 'hd64-lr0.0003-clr0.001-ec0.001-mgn0.5-nl2-sharedqv', 'ep2-mb4-clip0.2-l2c']`
  - `['maze-sz10', 'ppo_cond_fac-cnn+iru', 'mn1-mx5', 'hd16-lr0.0003-clr0.001-ec0.001-mgn0.5-nl4-sharedqv', 'ep2-mb4-clip0.2-l2c']`
  - `['maze-sz10', 'ppo_cond_fac-cnn+iru', 'mn1-mx5', 'hd64-lr0.0003-clr0.0003-ec0.001-mgn0.5-nl2-sharedqv', 'ep2-mb4-clip0.2-l2c']`
- Bad but better than above: <0.5 return
  - `['maze-sz10', 'ppo_cond_fac-cnn+iru', 'mn1-mx5', 'hd32-lr0.0003-clr0.001-ec0.001-mgn0.5-nl2-sepqv', 'ep2-mb4-clip0.2-l2c']`
  - `['maze-sz10', 'ppo_cond_fac-cnn+iru', 'mn1-mx5', 'hd16-lr0.0003-clr0.001-ec0.001-mgn0.5-nl2-sepqv', 'ep2-mb4-clip0.2-l2c']`
  - `['maze-sz10', 'ppo_cond_fac-cnn+iru', 'mn1-mx5', 'hd16-lr0.0003-clr0.001-ec0.001-mgn0.5-nl4-sepqv', 'ep2-mb4-clip0.2-l2c']`
  - `['maze-sz10', 'ppo_cond_fac-cnn+iru', 'mn1-mx5', 'hd64-lr0.0003-clr0.001-ec0.001-mgn0.5-nl2-sepqv', 'ep2-mb4-clip0.2-l2c']`
  - `['maze-sz10', 'ppo_cond_fac-cnn+iru', 'mn1-mx5', 'hd16-lr0.0003-clr0.0003-ec0.001-mgn0.5-nl2-sharedqv', 'ep2-mb4-clip0.2-l2c']`
- Similar: ~0.5 return
  - `['maze-sz10', 'ppo_cond_fac-cnn+iru', 'mn1-mx5', 'hd32-lr0.0003-clr0.0003-ec0.001-mgn0.5-nl4-sharedqv', 'ep2-mb4-clip0.2-l2c']`
  - `['maze-sz10', 'ppo_cond_fac-cnn+iru', 'mn1-mx5', 'hd32-lr0.0003-clr0.0003-ec0.001-mgn0.5-nl4-sepqv', 'ep2-mb4-clip0.2-l2c']`
  - `['maze-sz10', 'ppo_cond_fac-cnn+iru', 'mn1-mx5', 'hd16-lr0.0003-clr0.0003-ec0.001-mgn0.5-nl4-sharedqv', 'ep2-mb4-clip0.2-l2c']`
  - `['maze-sz10', 'ppo_cond_fac-cnn+iru', 'mn1-mx5', 'hd32-lr0.0003-clr0.0003-ec0.001-mgn0.5-nl2-sepqv', 'ep2-mb4-clip0.2-l2c']`
  - `['maze-sz10', 'ppo_cond_fac-cnn+iru', 'mn1-mx5', 'hd16-lr0.0003-clr0.0003-ec0.001-mgn0.5-nl2-sepqv', 'ep2-mb4-clip0.2-l2c']`


Based on above, use separate q-v, try standardized advantage
```
python ramdp_experiments/jumanji_sweep.py --systems ff_ppo_cond_fac --max-steps 5 --seeds 1 --runs-per-gpu 1 --architectures cnn+iru --hidden-dim 64 --num-layers 1,2,4 --total-timesteps 1e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4,1e-3 --envs maze --total-num-envs 128 --maze-size 10 --gpus 0,1,2,3,4,5 --rollout-length 10 --epochs 2 --num-minibatches 2 --ent-coef 0.01 --gamma 0.99 --actor-weight-decay 0.0 --critic-before-actor false,true --wandb true --wandb-project maze-sep9 --yes --no-skip-existing --qv-critic separate --critic-weight-decay 0.0,0.001 --use-input-layer-norm false,true --standardize-advantages false,true
```



tmux attach -t1, -t2
```
python ramdp_experiments/jumanji_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 1 --seeds 1 --runs-per-gpu 1 --architectures cnn+iru --hidden-dim 256 --num-layers 4 --total-timesteps 1e9 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --envs sokoban --sokoban-generator unfiltered-train --gpus 7 --rollout-length 128 --total-num-envs 256 --epochs 2 --num-minibatches 16 --ent-coef 0.01 --gamma 0.995 --actor-weight-decay 0.005 --critic-before-actor false --wandb true --wandb-project sokoban-sep9 --yes --no-skip-existing
```
- Maybe need standardized advantage

```
python ramdp_experiments/jumanji_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 1 --seeds 1 --runs-per-gpu 1 --architectures cnn+iru --hidden-dim 64 --num-layers 2 --total-timesteps 2e9 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --envs sokoban --sokoban-generator unfiltered-train --gpus 7 --rollout-length 20 --total-num-envs 128 --epochs 2 --num-minibatches 4 --ent-coef 0.01 --gamma 0.99 --actor-weight-decay 0.005 --critic-before-actor false --wandb true --wandb-project sokoban-sep9 --yes --no-skip-existing --standardize-advantages true --use-input-layer-norm true
```


# Comments on Sept 10
## Maze
### IRU
Search:
```
python ramdp_experiments/jumanji_sweep.py --systems ff_ppo_cond_fac --max-steps 5 --seeds 1 --runs-per-gpu 1 --architectures cnn+iru --hidden-dim 64 --num-layers 1,2,4 --total-timesteps 1e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4,1e-3 --envs maze --total-num-envs 128 --maze-size 10 --gpus 0,1,2,3,4,5 --rollout-length 10 --epochs 2 --num-minibatches 2 --ent-coef 0.01 --gamma 0.99 --actor-weight-decay 0.0 --critic-before-actor false,true --wandb true --wandb-project maze-sep9 --yes --no-skip-existing --qv-critic separate --critic-weight-decay 0.0,0.001 --use-input-layer-norm false,true --standardize-advantages false,true
```

New run:
```
python ramdp_experiments/jumanji_sweep.py --systems ff_ppo_cond_fac --max-steps 5 --seeds 1 --runs-per-gpu 1 --architectures cnn+iru --hidden-dim 64 --num-layers 1 --total-timesteps 1e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --envs maze --total-num-envs 128 --maze-size 10 --gpus 0,1,2,3 --rollout-length 10 --epochs 2 --num-minibatches 2 --ent-coef 0.01 --gamma 0.99 --critic-before-actor false,true --wandb true --wandb-project maze-sep9 --yes --no-skip-existing --qv-critic separate --use-input-layer-norm true --standardize-advantages false,true
```

- Seems to be reasonably good `['maze-sz10', 'ppo_cond_fac-cnn+iru', 'mn1-mx5', 'hd64-lr0.0003-clr0.0003-ec0.01-mgn0.5-nl1-sepqv', '145f8f33']`---takes <2hrs on salient4:
```
/home/bryanpu1/projects/iclr_2027/Stoix/stoix/systems/ramdp_vpg/ff_ppo.py
    env=jumanji/maze_grid
    network=cnn_iru_compute_qac_separate_qv
    system.gamma=0.99
    arch.total_timesteps=1e+08
    arch.total_num_envs=128
    arch.seed=0
    arch.num_evaluation=50
    arch.num_eval_episodes=10
    network.actor_network.pre_torso.hidden_dim=64
    ++network.actor_network.pre_torso.num_layers=1
    network.actor_network.pre_torso.max_steps=5
    network.actor_network.pre_torso.min_steps=1
    system.actor_lr=0.0003
    system.critic_lr=0.0003
    system.actor_weight_decay=0
    system.critic_weight_decay=0.001
    system.ent_coef=0.01
    system.max_grad_norm=0.5
    system.rollout_length=10
    logger.base_exp_path=/home/bryanpu1/projects/iclr_2027/Stoix/results_jumanji_sweep/maze-sz10-ppo_cond_fac-cnn+iru-mn1-mx5-hd64-lr0.0003-clr0.0003-ec0.01-mgn0.5-nl1-sepqv-145f8f33-seed_0
    env.kwargs.generator.num_rows=10
    env.kwargs.generator.num_cols=10
    system.epochs=2
    system.num_minibatches=2
    system.clip_eps=0.2
    system.clip_value_loss=False
    system.standardize_advantages=False
    system.recompute_advantages=False
    system.critic_before_actor=True
    system.latent_kl_coef=0
    system.halting_ent_coef=0
    logger.loggers.wandb.enabled=True
    logger.loggers.wandb.project=maze-sep9
    logger.loggers.wandb.group_tag=['maze-sz10','ppo_cond_fac-cnn+iru','mn1-mx5','hd64-lr0.0003-clr0.0003-ec0.01-mgn0.5-nl1-sepqv','145f8f33']
    ++network.actor_network.pre_torso.use_input_layer_norm=False
    ++network.actor_network.pre_torso.stop_gradient_halting_input=False
    system.qac_variant=cond_fac
    network.actor_network.input_layer.channel_sizes=[16,16]
    network.actor_network.input_layer.kernel_sizes=[3,3]
    network.actor_network.input_layer.strides=[2,1]
    network.actor_network.input_layer.hidden_sizes=[64]
    network.critic_network.value_input_layer.channel_sizes=[16,16]
    network.critic_network.value_input_layer.kernel_sizes=[3,3]
    network.critic_network.value_input_layer.strides=[2,1]
    network.critic_network.value_input_layer.hidden_sizes=[128]
    network.critic_network.value_pre_torso.layer_sizes=[128,128]
    network.critic_network.q_input_layer.channel_sizes=[16,16]
    network.critic_network.q_input_layer.kernel_sizes=[3,3]
    network.critic_network.q_input_layer.strides=[2,1]
    network.critic_network.q_input_layer.hidden_sizes=[128]
    network.critic_network.q_pre_torso.layer_sizes=[128,128]
```
- Generally, it seems like critic_lr=1e-3 is too high, standardize-advantages=True gives flat line in some cases
- critic_before_actor and critic_weight_decay don't seem to matter too much (yet), when num_layers=1 and hidden_dim=64 and input_layer_norm=False
  - Here, standardize-advantage=True seems to actually give lower episode return but higher discounted return (it takes less steps)
    e.g.
```
/home/bryanpu1/projects/iclr_2027/Stoix/stoix/systems/ramdp_vpg/ff_ppo.py
    env=jumanji/maze_grid
    network=cnn_iru_compute_qac_separate_qv
    system.gamma=0.99
    arch.total_timesteps=1e+08
    arch.total_num_envs=128
    arch.seed=0
    arch.num_evaluation=50
    arch.num_eval_episodes=10
    network.actor_network.pre_torso.hidden_dim=64
    ++network.actor_network.pre_torso.num_layers=2
    network.actor_network.pre_torso.max_steps=5
    network.actor_network.pre_torso.min_steps=1
    system.actor_lr=0.0003
    system.critic_lr=0.0003
    system.actor_weight_decay=0
    system.critic_weight_decay=0.001
    system.ent_coef=0.01
    system.max_grad_norm=0.5
    system.rollout_length=10
    logger.base_exp_path=/home/bryanpu1/projects/iclr_2027/Stoix/results_jumanji_sweep/maze-sz10-ppo_cond_fac-cnn+iru-mn1-mx5-hd64-lr0.0003-clr0.0003-ec0.01-mgn0.5-nl2-sepqv-2eb84e8f-seed_0
    env.kwargs.generator.num_rows=10
    env.kwargs.generator.num_cols=10
    system.epochs=2
    system.num_minibatches=2
    system.clip_eps=0.2
    system.clip_value_loss=False
    system.standardize_advantages=True
    system.recompute_advantages=False
    system.critic_before_actor=False
    system.latent_kl_coef=0
    system.halting_ent_coef=0
    logger.loggers.wandb.enabled=True
    logger.loggers.wandb.project=maze-sep9
    logger.loggers.wandb.group_tag=['maze-sz10','ppo_cond_fac-cnn+iru','mn1-mx5','hd64-lr0.0003-clr0.0003-ec0.01-mgn0.5-nl2-sepqv','2eb84e8f']
    ++network.actor_network.pre_torso.use_input_layer_norm=False
    ++network.actor_network.pre_torso.stop_gradient_halting_input=False
    system.qac_variant=cond_fac
    network.actor_network.input_layer.channel_sizes=[16,16]
    network.actor_network.input_layer.kernel_sizes=[3,3]
    network.actor_network.input_layer.strides=[2,1]
    network.actor_network.input_layer.hidden_sizes=[64]
    network.critic_network.value_input_layer.channel_sizes=[16,16]
    network.critic_network.value_input_layer.kernel_sizes=[3,3]
    network.critic_network.value_input_layer.strides=[2,1]
    network.critic_network.value_input_layer.hidden_sizes=[128]
    network.critic_network.value_pre_torso.layer_sizes=[128,128]
    network.critic_network.q_input_layer.channel_sizes=[16,16]
    network.critic_network.q_input_layer.kernel_sizes=[3,3]
    network.critic_network.q_input_layer.strides=[2,1]
    network.critic_network.q_input_layer.hidden_sizes=[128]
    network.critic_network.q_pre_torso.layer_sizes=[128,128]
```


### eCoT
```
# tmux attach -t6, gpus 4,5
python ramdp_experiments/jumanji_sweep.py --systems ff_ppo_explicit_cond_fac --max-steps 5 --seeds 1 --runs-per-gpu 1 --architectures cnn+transformer_explicit_cot --hidden-dim 64 --mlp-dim 128 --num-layers 2 --num-heads 4 --vocab-size 4 --total-timesteps 1e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --envs maze --maze-size 10 --gpus 4,5 --rollout-length 10 --total-num-envs 128 --epochs 2 --num-minibatches 2 --ent-coef 0.01 --gamma 0.99 --critic-before-actor false,true --qv-critic separate --standardize-advantages false,true --wandb true --wandb-project maze-sep10-ecot_sweep --yes --no-skip-existing


python ramdp_experiments/jumanji_fixed_budget_sweep.py --systems ff_ppo_explicit_reinforce --budget 1 --seeds 1 --runs-per-gpu 1 --architectures cnn+transformer_explicit_cot --hidden-dim 64 --mlp-dim 128 --num-layers 2 --num-heads 4 --vocab-size 1 --total-timesteps 1e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --envs maze --maze-size 10 --gpus 0,1 --rollout-length 10 --total-num-envs 128 --epochs 2 --num-minibatches 2 --ent-coef 0.01 --gamma 0.99 --critic-before-actor false,true --qv-critic separate --standardize-advantages false,true --wandb true --wandb-project maze-sep10-ecot_sweep --yes --no-skip-existing
```



## Sokoban
This run is learning:
```
python ramdp_experiments/jumanji_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 1 --seeds 1 --runs-per-gpu 1 --architectures cnn+iru --hidden-dim 64 --num-layers 2 --total-timesteps 2e9 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --envs sokoban --sokoban-generator unfiltered-train --gpus 0 --rollout-length 20 --total-num-envs 128 --epochs 2 --num-minibatches 4 --ent-coef 0.01 --gamma 0.99 --actor-weight-decay 0.005 --critic-before-actor false --wandb true --wandb-project sokoban-sep9 --yes --no-skip-existing --standardize-advantages true --use-input-layer-norm true
```

Try adaptive:
```
python ramdp_experiments/jumanji_sweep.py --systems ff_ppo_cond_fac --max-steps 5 --seeds 1 --runs-per-gpu 1 --architectures cnn+iru --hidden-dim 64 --num-layers 2 --total-timesteps 2e9 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --envs sokoban --sokoban-generator unfiltered-train --gpus 1 --rollout-length 20 --total-num-envs 128 --epochs 2 --num-minibatches 4 --ent-coef 0.01 --gamma 0.99 --actor-weight-decay 0.005 --critic-before-actor false --wandb true --wandb-project sokoban-sep9 --yes --no-skip-existing --standardize-advantages true --use-input-layer-norm true


python ramdp_experiments/jumanji_sweep.py --systems ff_ppo_reinforce --max-steps 5 --seeds 1 --runs-per-gpu 1 --architectures cnn+iru --hidden-dim 64 --num-layers 2 --total-timesteps 2e9 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --envs sokoban --sokoban-generator unfiltered-train --gpus 2 --rollout-length 20 --total-num-envs 128 --epochs 2 --num-minibatches 4 --ent-coef 0.01 --gamma 0.99 --actor-weight-decay 0.005 --critic-before-actor false --wandb true --wandb-project sokoban-sep9 --yes --no-skip-existing --standardize-advantages true --use-input-layer-norm true --gae-lambda 0.95
```


```
python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 1,5 --seeds 3 --architectures sps --hidden-dim 64 --mlp-dim 128 --num-layers 2,4 --num-heads 8 --total-timesteps 3e8 --grid-sizes 5x4 --episode-length 10 --lr 3e-4 --critic-lr 3e-4 --epochs 4 --num-minibatches 8 --use-input-layer-norm true --ent-coef 0.01 --clip-eps 0.2,0.3 --gamma 0.99 --actor-weight-decay 0.0 --gae-lambda 0.95 --standardize-advantages true --wandb true --wandb-project lightsout-icot-${project_name}-sps --output-dir /home/bryanpu1/scratch/logs/ramdp/lightsout-icot-${project_name}-sps --runs-per-gpu 1 --gpus 0,1,4,5 --no-skip-existing --yes &

python ramdp_experiments/lightsout_sweep.py --systems ff_ppo_reinforce --max-steps 5 --seeds 3 --architectures sps --hidden-dim 64 --mlp-dim 128 --num-layers 2,4 --num-heads 8 --total-timesteps 3e8 --grid-sizes 5x4 --episode-length 10 --lr 3e-4 --critic-lr 3e-4 --epochs 4 --num-minibatches 8 --use-input-layer-norm true --ent-coef 0.01 --clip-eps 0.2,0.3 --gamma 0.99 --actor-weight-decay 0.0 --gae-lambda 0.95 --standardize-advantages true --halting-ent-coef 0.0,0.01,0.001 --wandb true --wandb-project lightsout-icot-${project_name}-sps --output-dir /home/bryanpu1/scratch/logs/ramdp/lightsout-icot-${project_name}-sps --runs-per-gpu 1 --gpus 6,7 --no-skip-existing --yes &

wait
```