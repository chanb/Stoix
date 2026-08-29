##
Gymnax 1.0.0 supports seaquest
```
uv pip install --no-deps "gymnax==1.0.0"
```

## Vulcan
```
module load StdEnv/2023
module load cuda/12.2
```

## Transformer with implicit CoT
```
c_max=8
CUDA_VISIBLE_DEVICES="0" python stoix/systems/ramdp_vpg/ff_reinforce.py \
  network=transformer_compute \
  env=gymnax/breakout \
  logger.loggers.tensorboard.enabled=True \
  system.gamma=0.9999 \
  arch.total_timesteps=1e8 \
  network.actor_network.pre_torso.hidden_dim=32 \
  network.actor_network.pre_torso.mlp_dim=128 \
  network.actor_network.pre_torso.max_steps=${c_max} \
  logger.base_exp_path=results/transformers-c_max_${c_max}-breakout
```

## PonderNet
```
c_max=8

CUDA_VISIBLE_DEVICES="0" python stoix/systems/ramdp_vpg/ff_reinforce.py \
  env=gymnax/breakout \
  logger.loggers.tensorboard.enabled=True \
  system.gamma=0.9999 \
  arch.total_timesteps=1e8 \
  network.actor_network.pre_torso.hidden_dim=32 \
  network.actor_network.pre_torso.max_steps=${c_max} \
  logger.base_exp_path=results/pondernet-c_max_${c_max}-breakout

CUDA_VISIBLE_DEVICES="0" python stoix/systems/ramdp_vpg/ff_reinforce.py \
  env=gymnax/breakout \
  logger.loggers.tensorboard.enabled=True \
  system.gamma=0.9999 \
  arch.total_timesteps=1e8 \
  network.actor_network.pre_torso.hidden_dim=16 \
  network.actor_network.pre_torso.max_steps=${c_max} \
  logger.base_exp_path=results/pondernet-c_max_${c_max}-breakout-hidden_dim_16
```

## Q - V
```
c_max=8
CUDA_VISIBLE_DEVICES="0" python stoix/systems/ramdp_vpg/ff_qac.py \
  env=gymnax/breakout \
  logger.loggers.tensorboard.enabled=True \
  system.gamma=0.9999 \
  arch.total_timesteps=1e8 \
  network.actor_network.pre_torso.hidden_dim=16 \
  network.actor_network.pre_torso.max_steps=${c_max} \
  system.qac_variant=fac \
  logger.base_exp_path=results/pondernet-fac-c_max_${c_max}-breakout-hidden_dim_16

CUDA_VISIBLE_DEVICES="1" python stoix/systems/ramdp_vpg/ff_qac.py \
  env=gymnax/breakout \
  logger.loggers.tensorboard.enabled=True \
  system.gamma=0.9999 \
  arch.total_timesteps=1e8 \
  network.actor_network.pre_torso.hidden_dim=16 \
  network.actor_network.pre_torso.max_steps=${c_max} \
  system.qac_variant=naive \
  logger.base_exp_path=results/pondernet-naive-c_max_${c_max}-breakout-hidden_dim_16
```

## ARC
```
CUDA_VISIBLE_DEVICES="0" python stoix/systems/ramdp_vpg/ff_qac.py \
    env=jaxarc/default \
    logger.loggers.tensorboard.enabled=True \
    system.gamma=0.9999 \
    arch.total_timesteps=1e8 \
    network.actor_network.pre_torso.hidden_dim=16 \
    network.actor_network.pre_torso.max_steps=${c_max} \
    system.qac_variant=fac \
    logger.base_exp_path=results_arc/pondernet-fac-c_max_${c_max}-hidden_dim_16 \
    +env.reward.step_penalty=0.0

CUDA_VISIBLE_DEVICES="0" python stoix/systems/ramdp_vpg/ff_qac.py \
    env=jaxarc/default \
    logger.loggers.tensorboard.enabled=True \
    system.gamma=0.9999 \
    arch.total_timesteps=1e8 \
    network.actor_network.pre_torso.hidden_dim=16 \
    network.actor_network.pre_torso.max_steps=${c_max} \
    system.qac_variant=naive \
    logger.base_exp_path=results_arc/pondernet-naive-c_max_${c_max}-hidden_dim_16 \
    +env.reward.step_penalty=0.0

CUDA_VISIBLE_DEVICES="0" python stoix/systems/ramdp_vpg/ff_reinforce.py \
    env=jaxarc/default \
    logger.loggers.tensorboard.enabled=True \
    system.gamma=0.9999 \
    arch.total_timesteps=1e8 \
    network.actor_network.pre_torso.hidden_dim=16 \
    network.actor_network.pre_torso.max_steps=${c_max} \
    logger.base_exp_path=results_arc/pondernet-c_max_${c_max}-hidden_dim_16 \
    +env.reward.step_penalty=0.0
```

```
python ramdp_experiments/sokoban_sweep.py --hidden-dim=8,16,32,64 --lr=1e-5,1e-4,3e-4,1e-3 --runs-per-gpu=1 --seeds=3

CUDA_VISIBLE_DEVICES="0" XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

c_max=8
seed=2
CUDA_VISIBLE_DEVICES="7" XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 python stoix/systems/ramdp_vpg/ff_reinforce.py \
    env=gymnax/seaquest \
    logger.loggers.tensorboard.enabled=True \
    system.gamma=0.9999 \
    arch.seed=${seed} \
    arch.total_timesteps=3e8 \
    arch.num_evaluation=50 \
    network=transformer_compute \
    network.actor_network.pre_torso.hidden_dim=64 \
    network.actor_network.pre_torso.mlp_dim=64 \
    network.actor_network.pre_torso.min_steps=${c_max} \
    network.actor_network.pre_torso.max_steps=${c_max} \
    logger.base_exp_path=results_minatar_fixed_budget_sweep/seaquest-ff_reinforce-transformer-budget_${c_max}-hidden_dim_64-lr_0.0003-critic_lr_0.0003-seed_${seed} \
    system.ent_coef=0.01 \
    +env.wrapper._target_=stoa.FlattenObservationWrapper
```

## Fixed budget sweep
```
python ramdp_experiments/minatar_fixed_budget_sweep.py --systems ff_reinforce --budget 1,2,4,8,16 --seeds 3 --runs-per-gpu 4 --yes --architectures cnn --hidden-dim=64 --total-timesteps 3e8 --lr 3e-4 --no-skip-existing --delightful false,true --envs seaquest --use-layer-norm false,true --use-input-layer-norm false,true --gpus 0,1,2,3,4,5

python ramdp_experiments/minatar_fixed_budget_sweep.py --systems ff_reinforce --budget 1,2,4 --seeds 3 --runs-per-gpu 1 --yes --architectures transformer --hidden-dim=64 --total-timesteps 3e8 --lr 3e-4 --delightful false --envs seaquest --use-input-layer-norm true --gpus 6,7
```





# PPO Only
## Lightsout env
```
# IRU & Unshared IRU
## Fixed budget
python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 1,2,4,8,16 --seeds 10 --runs-per-gpu 2 --yes --architectures iru_unshared,iru --hidden-dim 16 --total-timesteps 1e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --delightful false --grid-sizes 3x3 --use-input-layer-norm true --episode-length 6 --gpus 0,1,2,3 --wandb true --wandb-project lightsout_sweep-ppo_only-iru

## Adaptive budget
Learning `Q(s, c)` because it's more robust to general `c`, also more fair as it has same number of parameters regardless of `c`
### Learn Q(s)
python ramdp_experiments/lightsout_sweep.py --systems ff_ppo_fac,ff_ppo_naive,ff_ppo_reinforce --max-steps 16 --seeds 10 --runs-per-gpu 2 --yes --architectures iru_unshared,iru --hidden-dim=16 --total-timesteps 1e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --delightful false --grid-sizes 3x3 --use-input-layer-norm true --episode-length 6 --gpus 4,6,7 --wandb true --wandb-project lightsout_sweep-ppo_only

### Learn Q(s, c)
python ramdp_experiments/lightsout_sweep.py --systems ff_ppo_cond_fac,ff_ppo_cond_naive --max-steps 16 --seeds 10 --runs-per-gpu 2 --yes --architectures iru_unshared,iru --hidden-dim=16 --total-timesteps 1e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --delightful false --grid-sizes 3x3 --use-input-layer-norm true --episode-length 6 --gpus 0,4,6,7 --wandb true --wandb-project lightsout_sweep-ppo_only

# Transformer

```

### TF architecture search
```
python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 1,4,8,16 --seeds 1 --runs-per-gpu 2 --yes --architectures transformer --hidden-dim 8 --mlp-dim 256 --num-layers 2 --num-heads 1,2 --total-timesteps 1e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --delightful false --grid-sizes 3x3 --use-input-layer-norm true --episode-length 6 --gpus 2,3 --wandb true --wandb-project lightsout_sweep-ppo_only-tf_arch

python ramdp_experiments/lightsout_sweep.py --systems ff_ppo_reinforce,ff_ppo_cond_fac,ff_ppo_cond_naive --max-steps 16 --seeds 1 --runs-per-gpu 2 --yes --architectures transformer --hidden-dim 8 --mlp-dim 256 --num-layers 2 --num-heads 1 --total-timesteps 1e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --delightful false --grid-sizes 3x3 --use-input-layer-norm true --episode-length 6 --gpus 4,5,6,7 --wandb true --wandb-project lightsout_sweep-ppo_only-tf_arch


######### Vulcan
python ramdp_experiments/lightsout_sweep.py --systems ff_ppo_reinforce,ff_ppo_cond_fac,ff_ppo_cond_naive --max-steps 16 --seeds 1 --runs-per-gpu 4 --yes --architectures transformer --hidden-dim 8 --mlp-dim 256 --num-layers 2 --num-heads 1 --total-timesteps 1e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --delightful false --grid-sizes 3x3 --use-input-layer-norm true --episode-length 6 --gpus 3,4,5 --wandb true --wandb-project lightsout_sweep-ppo_only-tf_arch-nope

python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 1,4,8 --seeds 1 --runs-per-gpu 2 --architectures transformer --hidden-dim 8,16 --mlp-dim 32 --num-layers 2 --num-heads 1,2 --total-timesteps 1e8 --clip-value-loss false --lr 1e-4 --critic-lr 1e-4 --delightful false --grid-sizes 3x3 --use-input-layer-norm true --episode-length 6 --gpus 0 --total-num-envs 2048 --rollout-length 160 --epochs 4 --num-minibatches 8 --wandb true --wandb-project lightsout_sweep-ppo_only-tf_arch-nope-vulcan --server vulcan --output-dir /home/chanb/scratch/logs/Stoix/results_lightsout_fixed_budget_sweep



#### CURRENT
python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 4,1,2,8 --seeds 1 --runs-per-gpu 4 --architectures transformer --hidden-dim 8,16 --mlp-dim 16,256 --num-layers 2 --num-heads 1,2 --total-timesteps 1e7 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --delightful false --grid-sizes 3x3 --use-input-layer-norm true --episode-length 6 --gpus 0 --total-num-envs 1024 --rollout-length 6 --epochs 8 --num-minibatches 8 --wandb true --wandb-project lightsout_sweep-ppo_only-tf_arch-nope-vulcan --server vulcan --output-dir /home/chanb/scratch/logs/Stoix/results_lightsout_fixed_budget_sweep --yes


### CHOSEN ARCHITECTURE
python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 4,1,2,8,16 --seeds 10 --runs-per-gpu 4 --architectures transformer --hidden-dim 8 --mlp-dim 16 --num-layers 2 --num-heads 1 --total-timesteps 1e7 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --delightful false --grid-sizes 3x3 --use-input-layer-norm true --episode-length 6 --gpus 0 --total-num-envs 1024 --rollout-length 6 --epochs 8 --num-minibatches 8 --wandb true --wandb-project lightsout-main-ppo_only-tf_arch-vulcan --server vulcan --output-dir /home/chanb/scratch/logs/Stoix/results_lightsout_fixed_budget_sweep --yes

### Check best hyperparam with different PPO setting
python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 4,1,2,8 --seeds 1 --runs-per-gpu 4 --architectures transformer --hidden-dim 8 --mlp-dim 16 --num-layers 2 --num-heads 1 --total-timesteps 1e7 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --delightful false --grid-sizes 3x3 --use-input-layer-norm true --episode-length 6 --gpus 0 --total-num-envs 512 --rollout-length 6 --epochs 8 --num-minibatches 4 --wandb true --wandb-project lightsout_sweep-ppo_only-tf_arch-nope-vulcan --server vulcan --output-dir /home/chanb/scratch/logs/Stoix/results_lightsout_fixed_budget_sweep --yes


### Q variants
python ramdp_experiments/lightsout_sweep.py --systems ff_ppo_reinforce,ff_ppo_cond_fac,ff_ppo_cond_naive --max-steps 16 --seeds 1 --runs-per-gpu 3 --architectures transformer --hidden-dim 8 --mlp-dim 16 --num-layers 2 --num-heads 1 --total-timesteps 1e7 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --delightful false --grid-sizes 3x3 --use-input-layer-norm true --episode-length 6 --gpus 0 --total-num-envs 1024 --rollout-length 6 --epochs 8 --num-minibatches 8 --wandb true --wandb-project lightsout_sweep-ppo_only-tf_arch-nope-vulcan --server vulcan --output-dir /home/chanb/scratch/logs/Stoix/results_lightsout_fixed_budget_sweep --yes

```


## minatar env
```
python ramdp_experiments/minatar_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 1,4,8 --seeds 1 --runs-per-gpu 2 --yes --architectures cnn+transformer --hidden-dim 16 --mlp-dim 16 --num-layers 2 --num-heads 2 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --delightful false --envs seaquest --use-input-layer-norm true --gpus 0 --wandb true --wandb-project seaquest_sweep-ppo_only-tf_arch

python ramdp_experiments/minatar_sweep.py --systems ff_ppo_reinforce,ff_ppo_cond_fac,ff_ppo_cond_naive --max-steps 8 --seeds 1 --runs-per-gpu 2 --yes --architectures cnn+transformer --hidden-dim 16 --mlp-dim 16 --num-layers 2 --num-heads 2 --total-timesteps 3e8 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --delightful false --envs seaquest --use-input-layer-norm true --gpus 1 --wandb true --wandb-project seaquest_sweep-ppo_only-tf_arch
```


```
# FLATTEN OBS
python ramdp_experiments/minatar_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 4,1,2,8 --seeds 1 --runs-per-gpu 4 --architectures transformer --hidden-dim 16,32 --mlp-dim 32,256 --num-layers 2 --num-heads 2 --total-timesteps 1e7 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --delightful false --envs asterix,breakout,freeway,seaquest,space_invaders --use-input-layer-norm true --gpus 0 --total-num-envs 128 --rollout-length 128 --epochs 8 --num-minibatches 16 --wandb true --wandb-project minatar_sweep-ppo_only-tf_arch-nope-vulcan --server vulcan --output-dir /home/chanb/scratch/logs/Stoix/results_minatar_fixed_budget_sweep --yes

# IMG OBS
python ramdp_experiments/minatar_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 4,1,2,8 --seeds 1 --runs-per-gpu 4 --architectures cnn+transformer --hidden-dim 16,32 --mlp-dim 32 --num-layers 2 --num-heads 2 --total-timesteps 1e7 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --delightful false --envs asterix --use-input-layer-norm true --gpus 0 --total-num-envs 128 --rollout-length 128 --epochs 2 --num-minibatches 64 --wandb true --wandb-project minatar_sweep-ppo_only-tf_arch-nope-vulcan --server vulcan --output-dir /home/chanb/scratch/logs/Stoix/results_minatar_fixed_budget_sweep --yes

python ramdp_experiments/minatar_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 4,1,2,8 --seeds 1 --runs-per-gpu 4 --architectures cnn+transformer --hidden-dim 16,32 --mlp-dim 32 --num-layers 2 --num-heads 2 --total-timesteps 1e7 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --delightful false --envs breakout --use-input-layer-norm true --gpus 0 --total-num-envs 128 --rollout-length 128 --epochs 8 --num-minibatches 16 --wandb true --wandb-project minatar_sweep-ppo_only-tf_arch-nope-vulcan --server vulcan --output-dir /home/chanb/scratch/logs/Stoix/results_minatar_fixed_budget_sweep --yes

python ramdp_experiments/minatar_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 4,1,2,8 --seeds 1 --runs-per-gpu 4 --architectures cnn+transformer --hidden-dim 16,32 --mlp-dim 32 --num-layers 2 --num-heads 2 --total-timesteps 1e7 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --delightful false --envs freeway --use-input-layer-norm true --gpus 0 --total-num-envs 128 --rollout-length 128 --epochs 16 --num-minibatches 2 --wandb true --wandb-project minatar_sweep-ppo_only-tf_arch-nope-vulcan --server vulcan --output-dir /home/chanb/scratch/logs/Stoix/results_minatar_fixed_budget_sweep --yes

python ramdp_experiments/minatar_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 4,1,2,8 --seeds 1 --runs-per-gpu 4 --architectures cnn+transformer --hidden-dim 16,32 --mlp-dim 32 --num-layers 2 --num-heads 2 --total-timesteps 1e7 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --delightful false --envs seaquest --use-input-layer-norm true --gpus 0 --total-num-envs 128 --rollout-length 128 --epochs 16 --num-minibatches 2 --wandb true --wandb-project minatar_sweep-ppo_only-tf_arch-nope-vulcan --server vulcan --output-dir /home/chanb/scratch/logs/Stoix/results_minatar_fixed_budget_sweep --yes

python ramdp_experiments/minatar_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 4,1,2,8 --seeds 1 --runs-per-gpu 4 --architectures cnn+transformer --hidden-dim 16,32 --mlp-dim 32 --num-layers 2 --num-heads 2 --total-timesteps 1e7 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --delightful false --envs space_invaders --use-input-layer-norm true --gpus 0 --total-num-envs 128 --rollout-length 128 --epochs 16 --num-minibatches 2 --wandb true --wandb-project minatar_sweep-ppo_only-tf_arch-nope-vulcan --server vulcan --output-dir /home/chanb/scratch/logs/Stoix/results_minatar_fixed_budget_sweep --yes
```




```
python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 1,4 --seeds 3 --runs-per-gpu 4 --architectures transformer --hidden-dim 8 --mlp-dim 32 --num-layers 2 --num-heads 2 --total-timesteps 5e7 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --delightful false --grid-sizes 3x3 --use-input-layer-norm true --episode-length 6 --gpus 0 --total-num-envs 1024 --rollout-length 6 --epochs 2,8 --num-minibatches 8,16 --wandb true --wandb-project lightsout-main-ppo_only-tf_arch-vulcan-5e7 --server vulcan --output-dir /home/chanb/scratch/logs/Stoix/results_lightsout_fixed_budget_sweep --yes


python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 1,2,4,8,16 --seeds 10 --runs-per-gpu 2 --yes --architectures iru --hidden-dim 16 --num-layers 2 --total-timesteps 5e7 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --epochs 8 --num-minibatches 16 --grid-sizes 4x4 --use-input-layer-norm false --episode-length 6 --gpus 0,1,2,3 --wandb true --wandb-project lightsout_sweep-ppo_only-iru




python ramdp_experiments/minatar_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 4,1 --seeds 3 --runs-per-gpu 4 --architectures cnn+transformer --hidden-dim 16 --mlp-dim 16 --num-layers 2 --num-heads 4 --total-timesteps 5e7 --clip-value-loss false --lr 1e-4 --critic-lr 3e-4 --envs asterix --use-input-layer-norm true --gpus 4 --total-num-envs 128 --rollout-length 64 --epochs 2 --num-minibatches 64 --wandb true --wandb-project minatar_sweep-ppo_only-cnn_tf_arch

python ramdp_experiments/minatar_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 4,1 --seeds 3 --runs-per-gpu 1 --architectures cnn+transformer --hidden-dim 16 --mlp-dim 16 --num-layers 2 --num-heads 4 --total-timesteps 5e7 --clip-value-loss false --lr 1e-4 --critic-lr 3e-4 --envs asterix --use-input-layer-norm true --gpus 4,5,6,7 --total-num-envs 128 --rollout-length 64 --epochs 2 --num-minibatches 64 --wandb true --wandb-project minatar_sweep-ppo_only-cnn_tf_arch


python ramdp_experiments/jumanji_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 4,1 --seeds 3 --runs-per-gpu 1 --architectures cnn+transformer --hidden-dim 16 --mlp-dim 16 --num-layers 2 --num-heads 4 --total-timesteps 5e7 --clip-value-loss false --lr 1e-4 --critic-lr 3e-4 --envs slidingtile --slidingtile-grid-size 3,4 --slidingtile-num-random-moves 5,20 --use-input-layer-norm true --gpus 4,5,6,7 --total-num-envs 128 --rollout-length 64 --epochs 8 --num-minibatches 16 --wandb true --wandb-project jumanji_sweep-ppo_only-cnn_tf_arch


python ramdp_experiments/jumanji_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 4,1 --seeds 3 --runs-per-gpu 1 --architectures cnn+transformer --hidden-dim 16 --mlp-dim 16 --num-layers 2 --num-heads 4 --total-timesteps 5e7 --clip-value-loss false --lr 1e-4 --critic-lr 3e-4 --envs sokoban --sokoban-generator toy --use-input-layer-norm true --gpus 4,5,6,7 --total-num-envs 128 --rollout-length 64 --epochs 8 --num-minibatches 16 --wandb true --wandb-project jumanji_sweep-ppo_only-cnn_tf_arch
```