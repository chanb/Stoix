#!/bin/bash
#SBATCH --account=aip-schuurma
#SBATCH --time=23:59:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
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


#### EXPECTILE LOSS
project_name=shallow_cnn
# python ramdp_experiments/jumanji_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 1,2,3 --seeds 1 --base-seed 0 --runs-per-gpu 1 --architectures cnn+transformer --hidden-dim 128 --qkv-dim 512 --mlp-dim 512 --num-layers 2 --num-heads 16 --total-timesteps 3e9 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --envs sokoban --sokoban-generator unfiltered-train --sokoban-eval-generator unfiltered-valid --gpus 0 --rollout-length 20 --total-num-envs 128 --epochs 2 --num-minibatches 4 --ent-coef 0.01 --gamma 0.999 --actor-weight-decay 0.001 --critic-before-actor false --standardize-advantages true --use-input-layer-norm true --use-rmsnorm true --gae-lambda 0.95 --max-grad-norm 10000.0 --wandb true --wandb-project sokoban-${project_name} --output-dir /home/chanb/scratch/logs/ramdp/sokoban-icot-${project_name} --yes --no-skip-existing --server vulcan --use-expectile-value-loss true --expectile 0.9 &

# python ramdp_experiments/jumanji_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 4 --seeds 1 --base-seed 0 --runs-per-gpu 1 --architectures cnn+transformer --hidden-dim 128 --qkv-dim 512 --mlp-dim 512 --num-layers 2 --num-heads 16 --total-timesteps 3e9 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --envs sokoban --sokoban-generator unfiltered-train --sokoban-eval-generator unfiltered-valid --gpus 0 --rollout-length 20 --total-num-envs 128 --epochs 2 --num-minibatches 4 --ent-coef 0.01 --gamma 0.999 --actor-weight-decay 0.001 --critic-before-actor false --standardize-advantages true --use-input-layer-norm true --use-rmsnorm true --gae-lambda 0.95 --max-grad-norm 10000.0 --wandb true --wandb-project sokoban-${project_name} --output-dir /home/chanb/scratch/logs/ramdp/sokoban-icot-${project_name} --yes --no-skip-existing --server vulcan --use-expectile-value-loss true --expectile 0.9 &

# python ramdp_experiments/jumanji_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 5 --seeds 1 --base-seed 0 --runs-per-gpu 1 --architectures cnn+transformer --hidden-dim 128 --qkv-dim 512 --mlp-dim 512 --num-layers 2 --num-heads 16 --total-timesteps 3e9 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --envs sokoban --sokoban-generator unfiltered-train --sokoban-eval-generator unfiltered-valid --gpus 0 --rollout-length 20 --total-num-envs 128 --epochs 2 --num-minibatches 4 --ent-coef 0.01 --gamma 0.999 --actor-weight-decay 0.001 --critic-before-actor false --standardize-advantages true --use-input-layer-norm true --use-rmsnorm true --gae-lambda 0.95 --max-grad-norm 10000.0 --wandb true --wandb-project sokoban-${project_name} --output-dir /home/chanb/scratch/logs/ramdp/sokoban-icot-${project_name} --yes --no-skip-existing --server vulcan --use-expectile-value-loss true --expectile 0.9 &

# python ramdp_experiments/jumanji_sweep.py --systems ff_ppo_reinforce --max-steps 5 --seeds 1 --base-seed 0 --runs-per-gpu 1 --architectures cnn+transformer --hidden-dim 128 --qkv-dim 512 --mlp-dim 512 --num-layers 2 --num-heads 16 --total-timesteps 3e9 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --envs sokoban --sokoban-generator unfiltered-train --sokoban-eval-generator unfiltered-valid --gpus 0 --rollout-length 20 --total-num-envs 128 --epochs 2 --num-minibatches 4 --ent-coef 0.01 --gamma 0.999 --actor-weight-decay 0.001 --critic-before-actor false --standardize-advantages true --use-input-layer-norm true --use-rmsnorm true --gae-lambda 0.95 --max-grad-norm 10000.0 --halting-ent-coef 0.1 --halting-temperature 5.0 --wandb true --wandb-project sokoban-${project_name} --output-dir /home/chanb/scratch/logs/ramdp/sokoban-icot-${project_name} --yes --no-skip-existing --server vulcan --use-expectile-value-loss true --expectile 0.9 &

# --halting-hidden-dims 128 --stop-gradient-halting-input true  &


### MAX GRAD NORM


# python ramdp_experiments/jumanji_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 1,2,3 --seeds 1 --base-seed 0 --runs-per-gpu 1 --architectures cnn+transformer --hidden-dim 128 --qkv-dim 512 --mlp-dim 512 --num-layers 2 --num-heads 16 --total-timesteps 3e9 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --envs sokoban --sokoban-generator unfiltered-train --sokoban-eval-generator unfiltered-valid --gpus 0 --rollout-length 20 --total-num-envs 128 --epochs 2 --num-minibatches 4 --ent-coef 0.01 --gamma 0.999 --actor-weight-decay 0.01 --critic-before-actor false --standardize-advantages true --use-input-layer-norm true --use-rmsnorm true --gae-lambda 0.95 --max-grad-norm 5.0 --wandb true --wandb-project sokoban-${project_name} --output-dir /home/chanb/scratch/logs/ramdp/sokoban-icot-${project_name} --yes --no-skip-existing --server vulcan --use-expectile-value-loss true --expectile 0.9 &

# python ramdp_experiments/jumanji_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 4 --seeds 1 --base-seed 0 --runs-per-gpu 1 --architectures cnn+transformer --hidden-dim 128 --qkv-dim 512 --mlp-dim 512 --num-layers 2 --num-heads 16 --total-timesteps 3e9 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --envs sokoban --sokoban-generator unfiltered-train --sokoban-eval-generator unfiltered-valid --gpus 0 --rollout-length 20 --total-num-envs 128 --epochs 2 --num-minibatches 4 --ent-coef 0.01 --gamma 0.999 --actor-weight-decay 0.01 --critic-before-actor false --standardize-advantages true --use-input-layer-norm true --use-rmsnorm true --gae-lambda 0.95 --max-grad-norm 5.0 --wandb true --wandb-project sokoban-${project_name} --output-dir /home/chanb/scratch/logs/ramdp/sokoban-icot-${project_name} --yes --no-skip-existing --server vulcan --use-expectile-value-loss true --expectile 0.9 &

# python ramdp_experiments/jumanji_fixed_budget_sweep.py --systems ff_ppo_reinforce --budget 5 --seeds 1 --base-seed 0 --runs-per-gpu 1 --architectures cnn+transformer --hidden-dim 128 --qkv-dim 512 --mlp-dim 512 --num-layers 2 --num-heads 16 --total-timesteps 3e9 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --envs sokoban --sokoban-generator unfiltered-train --sokoban-eval-generator unfiltered-valid --gpus 0 --rollout-length 20 --total-num-envs 128 --epochs 2 --num-minibatches 4 --ent-coef 0.01 --gamma 0.999 --actor-weight-decay 0.01 --critic-before-actor false --standardize-advantages true --use-input-layer-norm true --use-rmsnorm true --gae-lambda 0.95 --max-grad-norm 5.0 --wandb true --wandb-project sokoban-${project_name} --output-dir /home/chanb/scratch/logs/ramdp/sokoban-icot-${project_name} --yes --no-skip-existing --server vulcan --use-expectile-value-loss true --expectile 0.9 &

python ramdp_experiments/jumanji_sweep.py --systems ff_ppo_reinforce --max-steps 5 --seeds 1 --base-seed 0 --runs-per-gpu 1 --architectures cnn+transformer --hidden-dim 128 --qkv-dim 512 --mlp-dim 512 --num-layers 2 --num-heads 16 --total-timesteps 3e9 --clip-value-loss false --lr 3e-4 --critic-lr 3e-4 --envs sokoban --sokoban-generator unfiltered-train --sokoban-eval-generator unfiltered-valid --gpus 0 --rollout-length 20 --total-num-envs 128 --epochs 2 --num-minibatches 4 --ent-coef 0.01 --gamma 0.999 --actor-weight-decay 0.01 --critic-before-actor false --standardize-advantages true --use-input-layer-norm true --use-rmsnorm true --gae-lambda 0.95 --max-grad-norm 5.0 --halting-ent-coef 0.1 --halting-temperature 1.0 --wandb true --wandb-project sokoban-${project_name} --output-dir /home/chanb/scratch/logs/ramdp/sokoban-icot-${project_name} --yes --no-skip-existing --server vulcan --use-expectile-value-loss true --expectile 0.9 &



wait

echo "finished with exit code $? at: $(date)"
