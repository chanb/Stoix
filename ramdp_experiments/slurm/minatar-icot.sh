#!/bin/bash
#SBATCH --account=<ACCOUNT>
#SBATCH --time=05:00:00
#SBATCH --mem=24GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=0-4
#SBATCH --output=<LOG_DIR>/logs/ramdp/Stoix/%x_%A_%a.out

# Fixed-budget iCoT on MinAtar with gamma=0.999, expectile=0.5. One array task
# per game, 5 seeds, 2 concurrently on one GPU (3 waves). Budget is the first
# argument; set --time per budget at submission. Measured at 2 runs/GPU
# (independent of gamma): budget 1 takes ~40-47 min/run (~2 h per task),
# budget 5 ~68-73 min/run (~3.2 h per task) - sublinear in budget:
#   sbatch --time=03:00:00 ramdp_experiments/slurm/minatar-icot-gamma0.999.sh 1
#   sbatch --time=05:00:00 ramdp_experiments/slurm/minatar-icot-gamma0.999.sh 5
# Optional second argument overrides gamma (default 0.999); runs land in the
# same output dir / wandb project, distinguished by gamma in the run name:
#   sbatch --time=05:00:00 ramdp_experiments/slurm/minatar-icot-gamma0.999.sh 5 0.9999
# Pass "adaptive" instead of a budget for the variable-compute-budget (learned
# halting, max_steps=5) runs; use the budget-5 time:
#   sbatch --time=05:00:00 ramdp_experiments/slurm/minatar-icot-gamma0.999.sh adaptive 0.99999
# Several budgets can share one task (comma-separated) to fit a longer
# partition tier instead of queueing many short jobs, e.g. 15 runs, ~7-8 h:
#   sbatch --time=12:00:00 ramdp_experiments/slurm/minatar-icot-gamma0.999.sh 2,3,4 0.99999

budget=${1:?usage: sbatch [--time=...] $0 <budget|adaptive> [gamma]}
gamma=${2:-0.999}

module load StdEnv/2023
module load cuda/12.2

mkdir -p $SLURM_TMPDIR/tmp
export CUDA_MPS_LOG_DIRECTORY=$SLURM_TMPDIR/tmp
nvidia-cuda-mps-control -d

cd /home/chanb/research/iclr_2027/Stoix

games=(asterix breakout freeway seaquest space_invaders)
game=${games[$SLURM_ARRAY_TASK_ID]}

echo "hostname: $(hostname)"
echo "game: ${game} budget: ${budget} gamma: ${gamma}"
echo "starting at: $(date)"

project_name=nov06-g0.999

if [ "${budget}" = "adaptive" ]; then
    # Variable compute budget: learned halting, up to 5 steps.
    sweep_args="ramdp_experiments/jumanji_sweep.py --max-steps 5 --halting-ent-coef 0.01 --halting-temperature 5.0"
else
    sweep_args="ramdp_experiments/jumanji_fixed_budget_sweep.py --budget ${budget}"
fi

python ${sweep_args} --systems ff_ppo_reinforce --seeds 5 --runs-per-gpu 2 --architectures cnn+transformer --hidden-dim 64 --mlp-dim 128 --num-layers 2 --num-heads 4 --total-timesteps 1e8 --total-num-envs 128 --rollout-length 64 --envs ${game} --gamma ${gamma} --lr 3e-4 --critic-lr 3e-4 --epochs 4 --num-minibatches 8 --clip-eps 0.2 --clip-value-loss false --ent-coef 0.01 --gae-lambda 0.95 --standardize-advantages true --critic-before-actor false --use-input-layer-norm true --actor-weight-decay 0.0 --max-grad-norm 0.5 --use-expectile-value-loss true --expectile 0.5 --wandb true --wandb-project minatar-icot-${project_name} --output-dir /home/chanb/scratch/logs/ramdp/minatar-icot-${project_name} --gpus 0 --server vulcan --yes

echo "finished with exit code $? at: $(date)"
