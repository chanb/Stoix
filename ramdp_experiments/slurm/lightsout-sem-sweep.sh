#!/bin/bash
#SBATCH --account=aip-schuurma
#SBATCH --time=10:00:00
#SBATCH --mem=24GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=0-15
#SBATCH --output=/home/chanb/scratch/logs/ramdp/Stoix/%x_%A_%a.out

# Lightsout 5x4 SEM sweep, fixed budgets, 2 runs per GPU. Other hparams: sem_round2_hparams.md.
# Tasks 0-7: sem_group_size {16,64} x block_init {orthogonal,trunc_normal} x num_state_tokens {8,16}.
# Task 8: sem 16, trunc_normal, 8 tokens, critic = own CoT torso looping 5 steps.
# Task 9: sem 16, trunc_normal, 8 tokens, SEM only on the readout (sem_readout_only).
# Task 10: as task 9, but a CNN encoder (cnn+transformer, 8 channel tokens, CNN critic as in round 2).
# Task 11: as task 10, but variable budget (learned halting, 1-5 steps); the budgets argument is not used.
# Task 12: as task 11, but stop_gradient_halting_input=true.
# Task 13: as task 10, but both CNNs use VALID padding and a 2x2 max-pool (stride 1) before a 1x1 last conv.
# Task 14: as task 10, but rollout_length 16 and 4096 envs (same 65,536-step batch, shorter rollouts).
# Task 15: as task 11 (variable budget), but rollout_length 16 and 4096 envs.
# Usage: sbatch [--array=...] lightsout-sem-sweep.sh [budgets=1,5] [seeds=5]
# New architecture variants: submit one job per budget so each can be monitored on its own, e.g.
#   sbatch --time=05:00:00 --array=13 -J ls-sem-b1 lightsout-sem-sweep.sh 1
#   sbatch --time=07:00:00 --array=13 -J ls-sem-b5 lightsout-sem-sweep.sh 5
# SMOKE=1 runs 5 of the 50 iterations (1/10 of total_timesteps), no wandb, to a separate dir.

budget=${1:-1,5}
seeds=${2:-5}

module load StdEnv/2023
module load cuda/12.2

mkdir -p $SLURM_TMPDIR/tmp
export CUDA_MPS_LOG_DIRECTORY=$SLURM_TMPDIR/tmp
nvidia-cuda-mps-control -d

cd /home/chanb/research/iclr_2027/Stoix

t=${SLURM_ARRAY_TASK_ID}
readout_only=false; arch=transformer; rollout=64; envs=1024
sweep_args="ramdp_experiments/lightsout_fixed_budget_sweep.py --budget ${budget} --critic-cot-steps 0"
if [ "${t}" = "8" ]; then
    sem=16; init=trunc_normal; nst=8
    sweep_args="ramdp_experiments/lightsout_fixed_budget_sweep.py --budget ${budget} --critic-cot-steps 5"
elif [ "${t}" = "9" ]; then
    sem=16; init=trunc_normal; nst=8; readout_only=true
elif [ "${t}" = "10" ] || [ "${t}" = "13" ] || [ "${t}" = "14" ]; then
    sem=16; init=trunc_normal; nst=8; readout_only=true; arch=cnn+transformer
    [ "${t}" = "13" ] && sweep_args="${sweep_args} --cnn-pool true"
    [ "${t}" = "14" ] && rollout=16 && envs=4096
elif [ "${t}" = "11" ] || [ "${t}" = "12" ] || [ "${t}" = "15" ]; then
    sem=16; init=trunc_normal; nst=8; readout_only=true; arch=cnn+transformer
    [ "${t}" = "15" ] && rollout=16 && envs=4096
    sweep_args="ramdp_experiments/lightsout_sweep.py --min-steps 1 --max-steps 5 --halting-lr 1e-4 --halting-weight-decay 0 --halting-ent-coef 0.01 --halting-temperature 1.0 --halting-hidden-dims none --stop-gradient-halting-input $([ "${t}" = "12" ] && echo true || echo false)"
else
    sems=(16 64); inits=(orthogonal trunc_normal); nsts=(8 16)
    sem=${sems[$((t / 4))]}; init=${inits[$((t / 2 % 2))]}; nst=${nsts[$((t % 2))]}
fi

timesteps=1e8
out=/home/chanb/scratch/logs/ramdp/lightsout-sem-sweep
wandb_args="--wandb true --wandb-project stoix-sem-lightsout_sweep"
if [ "${SMOKE:-0}" = "1" ]; then
    export NUM_EVALUATION=5
    timesteps=1e7
    out=${out}-smoke
    wandb_args="--wandb false"
    .venv/bin/python -m pytest -q tests/test_torso_compute_transformer.py || exit 1
fi

echo "hostname: $(hostname)"
echo "sem: ${sem} block_init: ${init} num_state_tokens: ${nst} readout_only: ${readout_only} arch: ${arch} rollout: ${rollout} envs: ${envs} budget: ${budget} seeds: ${seeds} smoke: ${SMOKE:-0}"
echo "starting at: $(date)"

python ${sweep_args} --seeds ${seeds} \
    --systems ff_ppo_reinforce --architectures ${arch} --runs-per-gpu 2 --gpus 0 \
    --grid-sizes 5x4 --episode-length 10 --eval-episode-length 20 --difficulty-threshold 0.5 \
    --total-timesteps ${timesteps} --total-num-envs ${envs} --rollout-length ${rollout} --gamma 0.999 --max-grad-norm 1.0 \
    --lr 1e-4 --critic-lr 1e-4 --actor-weight-decay 0 --critic-weight-decay 0 \
    --epochs 2 --num-minibatches 64 --clip-eps 0.3 --clip-value-loss false --use-expectile-value-loss false \
    --vf-coef 1.0 --ent-coef 0.01 --gae-lambda 0.95 --standardize-advantages true --recompute-advantages false \
    --critic-before-actor false --use-dpo-loss false --latent-kl-coef 0 --clip-halting-head true \
    --hidden-dim 128 --mlp-dim 512 --num-layers 2 --num-heads 4 --use-input-layer-norm true --use-rmsnorm true \
    --use-sandwich-norm false --latent-rms-norm true --input-injection true --residual-scaling true \
    --sem-group-size ${sem} --block-init ${init} --num-state-tokens ${nst} --sem-readout-only ${readout_only} \
    --critic-layer-sizes 256,256,256 --critic-activation gelu \
    ${wandb_args} --output-dir ${out} --server slurm --yes

echo "finished with exit code $? at: $(date)"
