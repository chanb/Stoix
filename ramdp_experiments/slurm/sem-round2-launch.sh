#!/bin/bash
# Launch SEM round 2 on Vulcan: fixed budgets 1-5 + variable budget, 5 seeds, 2 runs/GPU.
# Smoke test (5/50 iterations, variable budget, 2 runs/GPU, 2026-11-07), upper bound per run
# since the CoT scan always runs max_steps: MinAtar ~1.9 h, Lightsout ~1.5 h, sliding ~1.55 h,
# Sokoban ~5.25 h. Budgets 1-4 in pairs (10 runs, 5 waves); budget 5 and adaptive alone
# (5 runs, 3 waves). Sokoban one variant per job.
cd /home/chanb/research/iclr_2027/Stoix
S=ramdp_experiments/slurm/sem-round2.sh

for v in 1,2 3,4 5 adaptive; do
    case ${v} in *,*) t_short=10:00:00; t_minatar=12:00:00 ;; *) t_short=06:00:00; t_minatar=08:00:00 ;; esac
    for e in lightsout slidingtile_bank; do
        sbatch --time=${t_short} -J sem-${e}-${v} $S ${e} ${v}
    done
    sbatch --time=${t_minatar} --array=0-4 -J sem-minatar-${v} $S minatar ${v}
done
for v in 1 2 3 4 5 adaptive; do
    sbatch --time=20:00:00 -J sem-sokoban-${v} $S sokoban ${v}
done
