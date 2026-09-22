#!/bin/bash
#SBATCH --account=aip-schuurma
#SBATCH --time=23:59:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=/home/chanb/scratch/logs/ramdp/Stoix/%x_%j.out

# Generic single-command runner for sokoban-icot jobs.
# Expects the python command to run, base64-encoded, in the $CMD_B64 environment
# variable (plain CMD would get silently truncated by sbatch --export at the
# first comma), e.g. via:
#   CMD_B64="$(base64 -w0 <<< "python ...")"
#   sbatch --export=ALL,CMD_B64="${CMD_B64}" --job-name=<name> sokoban-icot-run.sh
# Submitted by submit_sokoban-icot.sh, one job per (seed, line) combination.

module load StdEnv/2023
module load cuda/12.2

mkdir -p $SLURM_TMPDIR/tmp
export CUDA_MPS_LOG_DIRECTORY=$SLURM_TMPDIR/tmp
nvidia-cuda-mps-control -d

cd /home/chanb/research/iclr_2027/Stoix

echo "hostname: $(hostname)"
echo "starting at: $(date)"

if [ -z "${CMD_B64:-}" ]; then
    echo "CMD_B64 environment variable is not set, nothing to run" >&2
    exit 1
fi

CMD="$(base64 -d <<< "${CMD_B64}")"
echo "running: $CMD"
eval "$CMD"

echo "finished with exit code $? at: $(date)"
