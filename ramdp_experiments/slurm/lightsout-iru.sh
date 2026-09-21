#!/bin/bash
#SBATCH --account=aip-schuurma
#SBATCH --time=02:59:00
#SBATCH --mem=24GB
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:2
#SBATCH --array=1-1
#SBATCH --output=/home/chanb/scratch/logs/ramdp/Stoix/%x_%A_%a.out

module load StdEnv/2023
module load cuda/12.2

mkdir -p $SLURM_TMPDIR/tmp
export CUDA_MPS_LOG_DIRECTORY=$SLURM_TMPDIR/tmp
nvidia-cuda-mps-control -d

cd /home/chanb/research/iclr_2027/Stoix

echo "hostname: $(hostname)"
echo "starting at: $(date)"

# project_name=final-qkv
project_name=iru_unshared_sweep-qkv

python ramdp_experiments/lightsout_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 1,2,4,8,16 --seeds 10 --architectures iru_unshared --hidden-dim 16 --num-layers 1 --total-timesteps 1e8 --grid-sizes 3x3 --episode-length 5 --eval-episode-length 9 --difficulty-threshold 0.5 --lr 3e-4 --critic-lr 3e-4 --epochs 4 --num-minibatches 8 --use-input-layer-norm true --clip-value-loss false --critic-before-actor false --ent-coef 0.01 --clip-eps 0.2 --gamma 0.99 --actor-weight-decay 0.001,0.0001 --critic-weight-decay 0.0 --gae-lambda 0.95 --standardize-advantages true --use-rmsnorm true --use-sandwich-norm false --wandb true --wandb-project lightsout-iru_unshared-${project_name} --output-dir /home/chanb/scratch/logs/ramdp/lightsout-iru_unshared-${project_name} --runs-per-gpu 3 --gpus 0,1 --server vulcan --yes --no-skip-existing --use-expectile-value-loss false

# python ramdp_experiments/lightsout_sweep.py --systems ff_ppo_reinforce --max-steps 16 --seeds 10 --architectures iru_unshared --hidden-dim 16 --num-layers 1 --total-timesteps 1e8 --grid-sizes 3x3 --episode-length 5 --eval-episode-length 9 --difficulty-threshold 0.5 --lr 3e-4 --critic-lr 3e-4 --epochs 4 --num-minibatches 8 --use-input-layer-norm true --clip-value-loss false --critic-before-actor false --ent-coef 0.01 --clip-eps 0.2 --gamma 0.99 --actor-weight-decay 0.001,0.0001 --critic-weight-decay 0.0 --gae-lambda 0.95 --standardize-advantages true --use-rmsnorm true --use-sandwich-norm false --halting-ent-coef 0.01,0.0 --halting-temperature 0.5,1 --wandb true --wandb-project lightsout-iru_unshared-${project_name} --output-dir /home/chanb/scratch/logs/ramdp/lightsout-iru_unshared-${project_name} --runs-per-gpu 3 --gpus 0,1 --server vulcan --yes --no-skip-existing --use-expectile-value-loss true

wait

echo "finished with exit code $? at: $(date)"


# /home/bryanpu1/projects/iclr_2027/Stoix/stoix/systems/ramdp_vpg/ff_ppo.py env=lightsout/lightsout_3x3 env.scenario.name=lightsout-3x3 env.scenario.task_name=lightsout_3x3 env.kwargs.episode_length=6 env.kwargs.difficulty_threshold=0.5 network=iru_unshared_compute system.gamma=0.99 arch.total_timesteps=1e+08 arch.seed=9 arch.num_evaluation=50 network.actor_network.pre_torso.hidden_dim=16 ++network.actor_network.pre_torso.num_layers=1 network.actor_network.pre_torso.max_steps=16 network.actor_network.pre_torso.min_steps=16 system.actor_lr=0.0003 system.critic_lr=0.0003 system.ent_coef=0.01 logger.base_exp_path=/home/bryanpu1/projects/iclr_2027/Stoix/results_lightsout_fixed_budget_sweep/lightsout-3x3-ff_ppo_reinforce-iru_unshared-budget_16-hidden_dim_16-lr_0.0003-critic_lr_0.0003-num_layers_1-epochs_4-minibatches_16-clip_0.2-l2_critic-input_ln-seed_9 system.epochs=4 system.num_minibatches=16 system.clip_eps=0.2 system.clip_value_loss=False logger.loggers.wandb.enabled=True logger.loggers.wandb.project=lightsout_sweep-ppo_only ++network.actor_network.pre_torso.use_input_layer_norm=True system.qac_variant=reinforce +env.wrapper._target_=stoa.FlattenObservationWrapper