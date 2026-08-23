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