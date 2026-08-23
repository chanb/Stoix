#!/usr/bin/env python
"""Fixed-computation-budget sweep for RAMDP systems on Sokoban.

Companion to snake_sweep.py: instead of letting the actor's compute torso
adaptively halt (min_steps=1, sampled/learned halting), every job here pins
`min_steps == max_steps == budget`, which - per the min_steps mechanism added
to the compute torsos (see stoix/networks/torso_compute*.py) - forbids
halting before `budget` steps and forces a halt at exactly `budget` steps.
So every example always takes exactly `budget` steps of computation, with no
adaptivity at all.

This answers "does more computation help on this task?" directly (sweeping
`budget` at a fixed, non-adaptive cost, comparable to snake_sweep.py's
adaptive c_max axis) and is also a simpler system to debug against, since
compute_time is deterministic (== budget) rather than a learned/sampled
quantity.

Grid axes:
  - system:      ff_reinforce (PonderNet-style REINFORCE) | ff_qac_fac | ff_qac_naive
  - architecture: mlp (AdaptiveComputationTimeTorso) | transformer (TransformerChainOfThoughtTorso)
  - budget:      fixed number of steps every example takes (max_steps == min_steps)
  - hidden_dim:  actor torso width (network.actor_network.pre_torso.hidden_dim)
  - lr:          shared actor_lr/critic_lr
  - seed:        5 seeds per config by default

gamma is fixed at 0.9999, matching snake_sweep.py. total_timesteps defaults
to 2e7 (a reduced sweep budget) rather than the 1e8 used for full runs --
re-run the winning config(s) at full budget afterwards.

Jobs are scheduled across GPUs with a fixed number of concurrent runs per
GPU (a GPU "slot" queue + thread pool), each run pinned via
CUDA_VISIBLE_DEVICES and logged to its own file under <output-dir>/logs/.

Usage:
  python ramdp_experiments/snake_fixed_budget_sweep.py --dry-run                # preview the grid
  python ramdp_experiments/snake_fixed_budget_sweep.py --limit 6 --dry-run       # preview a slice
  python ramdp_experiments/snake_fixed_budget_sweep.py                           # run the full sweep
  python ramdp_experiments/snake_fixed_budget_sweep.py --systems ff_reinforce --architectures mlp \\
      --budget 1,8 --hidden-dim 16 --lr 3e-4 --seeds 1             # small pilot / debug run
"""

from __future__ import annotations

import argparse
import itertools
import json
import queue
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parent.parent

SYSTEM_TO_SCRIPT = {
    "ff_reinforce": "stoix/systems/ramdp_vpg/ff_reinforce.py",
    "ff_qac_fac": "stoix/systems/ramdp_vpg/ff_qac.py",
    "ff_qac_naive": "stoix/systems/ramdp_vpg/ff_qac.py",
}
SYSTEM_TO_QAC_VARIANT = {"ff_qac_fac": "fac", "ff_qac_naive": "naive"}
ARCH_TO_NETWORK = {
    "ff_reinforce": {"mlp": "mlp_compute", "transformer": "transformer_compute", "cnn": "cnn_compute"},
    "ff_qac_fac": {"mlp": "mlp_compute_qac", "transformer": "transformer_compute_qac", "cnn": "cnn_compute_qac"},
    "ff_qac_naive": {"mlp": "mlp_compute_qac", "transformer": "transformer_compute_qac", "cnn": "cnn_compute_qac"},
}


@dataclass
class Job:
    system: str
    arch: str
    budget: int
    hidden_dim: int
    lr: float
    critic_lr: float
    seed: int
    total_timesteps: float
    output_dir: Path

    @property
    def run_name(self) -> str:
        return (
            f"{self.system}-{self.arch}-budget_{self.budget}"
            f"-hidden_dim_{self.hidden_dim}-lr_{self.lr:g}-critic_lr_{self.critic_lr:g}-seed_{self.seed}"
        )

    def command(self, python_bin: str) -> List[str]:
        network = ARCH_TO_NETWORK[self.system][self.arch]
        cmd = [
            python_bin,
            SYSTEM_TO_SCRIPT[self.system],
            "env=jumanji/snake",
            f"network={network}",
            "logger.loggers.tensorboard.enabled=True",
            "logger.loggers.json.enabled=True",
            "system.gamma=0.9999",
            f"arch.total_timesteps={self.total_timesteps:g}",
            f"arch.seed={self.seed}",
            "arch.num_evaluation=50",
            f"network.actor_network.pre_torso.hidden_dim={self.hidden_dim}",
            # min_steps == max_steps == budget: no adaptivity, every example always
            # takes exactly `budget` steps (see stoix/networks/torso_compute*.py).
            f"network.actor_network.pre_torso.max_steps={self.budget}",
            f"network.actor_network.pre_torso.min_steps={self.budget}",
            f"system.actor_lr={self.lr:g}",
            f"system.critic_lr={self.critic_lr:g}",
            f"logger.base_exp_path={self.output_dir / self.run_name}",
        ]
        if self.system in SYSTEM_TO_QAC_VARIANT:
            cmd.append(f"system.qac_variant={SYSTEM_TO_QAC_VARIANT[self.system]}")
        if self.arch != "cnn":
            cmd.append(f"+wrapper._target_=stoa.FlattenObservationWrapper")
        else:
            cmd.append(f"network.actor_network.input_layer.channel_sizes=[32,32]")
            cmd.append(f"network.actor_network.input_layer.kernel_sizes=[2,2]")
            cmd.append(f"network.actor_network.input_layer.strides=[2,1]")
            cmd.append(f"network.actor_network.input_layer.hidden_sizes=[64]")
        return cmd

    def run_dir(self) -> Path:
        return self.output_dir / self.run_name


def build_grid(args: argparse.Namespace) -> List[Job]:
    jobs = []
    for system, arch, budget, hidden_dim, lr, critic_lr, seed in itertools.product(
        args.systems, args.architectures, args.budget, args.hidden_dim, args.lr, args.lr, range(args.seeds)
    ):
        jobs.append(
            Job(
                system=system,
                arch=arch,
                budget=budget,
                hidden_dim=hidden_dim,
                lr=lr,
                critic_lr=critic_lr,
                seed=seed,
                total_timesteps=args.total_timesteps,
                output_dir=args.output_dir,
            )
        )
    return jobs


def run_job(
    job: Job,
    gpu: int,
    python_bin: str,
    log_dir: Path,
    manifest_lock,
    manifest_path: Path,
    mem_fraction: float,
) -> dict:
    log_path = log_dir / f"{job.run_name}.log"
    cmd = job.command(python_bin)
    env = {
        "CUDA_VISIBLE_DEVICES": str(gpu),
        # JAX preallocates this fraction of the *visible* GPU's memory per process.
        # With `runs_per_gpu` processes sharing one physical GPU, each must be capped
        # to roughly 1/runs_per_gpu of the GPU or the later processes to allocate OOM.
        "XLA_PYTHON_CLIENT_MEM_FRACTION": str(mem_fraction),
    }
    import os

    full_env = os.environ.copy()
    full_env.update(env)

    start = time.time()
    with open(log_path, "w") as log_file:
        log_file.write(
            f"# GPU={gpu} XLA_PYTHON_CLIENT_MEM_FRACTION={mem_fraction:g}\n# CMD={' '.join(cmd)}\n\n"
        )
        log_file.flush()
        proc = subprocess.run(
            cmd, cwd=REPO_ROOT, env=full_env, stdout=log_file, stderr=subprocess.STDOUT
        )
    elapsed = time.time() - start

    result = {
        **asdict(job),
        "run_name": job.run_name,
        "output_dir": str(job.output_dir),
        "gpu": gpu,
        "returncode": proc.returncode,
        "elapsed_sec": round(elapsed, 1),
        "log": str(log_path),
    }
    with manifest_lock:
        with open(manifest_path, "a") as f:
            f.write(json.dumps(result) + "\n")
    status = "OK" if proc.returncode == 0 else f"FAILED (rc={proc.returncode})"
    print(f"[{status}] {job.run_name} on GPU {gpu} ({elapsed:.0f}s) -> {log_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--systems",
        default="ff_reinforce,ff_qac_fac,ff_qac_naive",
        help="Comma-separated subset of {ff_reinforce, ff_qac_fac, ff_qac_naive}.",
    )
    parser.add_argument(
        "--architectures", default="mlp,transformer", help="Comma-separated subset of {mlp, transformer, cnn}."
    )
    parser.add_argument(
        "--budget",
        default="1,2,4,8,16",
        help="Comma-separated fixed compute budgets - each sets max_steps == min_steps to this value.",
    )
    parser.add_argument("--hidden-dim", default="16,32", help="Comma-separated actor torso widths.")
    parser.add_argument("--lr", default="1e-4,3e-4,1e-3", help="Comma-separated shared actor_lr/critic_lr values.")
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds per config, seeded 0..seeds-1.")
    parser.add_argument("--total-timesteps", type=float, default=2e7, help="arch.total_timesteps per run.")
    parser.add_argument("--gpus", default="auto", help="Comma-separated GPU ids, or 'auto' to detect via nvidia-smi.")
    parser.add_argument("--runs-per-gpu", type=int, default=2, help="Concurrent runs per GPU.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results_snake_fixed_budget_sweep",
        help="Where per-run logger.base_exp_path and logs/ + manifest.jsonl are written.",
    )
    parser.add_argument("--python", default=str(REPO_ROOT / ".venv" / "bin" / "python"), help="Python interpreter.")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N jobs (for a pilot / sanity check).")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip jobs whose run directory already exists (default: on; use --no-skip-existing to force rerun).",
    )
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned jobs and exit without running anything.")
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation prompt before launching.")
    args = parser.parse_args()

    args.systems = args.systems.split(",")
    args.architectures = args.architectures.split(",")
    args.budget = [int(x) for x in args.budget.split(",")]
    args.hidden_dim = [int(x) for x in args.hidden_dim.split(",")]
    args.lr = [float(x) for x in args.lr.split(",")]

    for s in args.systems:
        assert s in SYSTEM_TO_SCRIPT, f"unknown system {s!r}, expected one of {list(SYSTEM_TO_SCRIPT)}"
    for a in args.architectures:
        assert a in ("mlp", "transformer", "cnn"), f"unknown architecture {a!r}"
    for b in args.budget:
        assert b >= 1, f"budget must be >= 1, got {b}"

    if args.gpus == "auto":
        try:
            out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout
            gpu_ids = list(range(len(out.strip().splitlines())))
        except Exception:
            gpu_ids = [0]
    else:
        gpu_ids = [int(x) for x in args.gpus.split(",")]

    jobs = build_grid(args)
    if args.limit is not None:
        jobs = jobs[: args.limit]

    if args.skip_existing:
        remaining = [j for j in jobs if not j.run_dir().exists()]
        n_skipped = len(jobs) - len(remaining)
        jobs = remaining
    else:
        n_skipped = 0

    concurrency = len(gpu_ids) * args.runs_per_gpu
    mem_fraction = 0.95 / args.runs_per_gpu
    print(
        f"GPUs: {gpu_ids} x {args.runs_per_gpu} runs/GPU = {concurrency} concurrent "
        f"(XLA_PYTHON_CLIENT_MEM_FRACTION={mem_fraction:g} per process)"
    )
    print(f"Grid: {len(jobs)} jobs to run" + (f" ({n_skipped} skipped as already-existing)" if n_skipped else ""))
    print(f"  systems={args.systems} architectures={args.architectures}")
    print(f"  budget (min_steps=max_steps)={args.budget} hidden_dim={args.hidden_dim} lr={args.lr} seeds=0..{args.seeds - 1}")
    print(f"  total_timesteps={args.total_timesteps:g} output_dir={args.output_dir}")

    if args.dry_run:
        for j in jobs:
            print(" ".join(j.command(args.python)))
        return

    if not jobs:
        print("Nothing to run.")
        return

    if not args.yes:
        resp = input(f"Launch {len(jobs)} jobs with concurrency {concurrency}? [y/N] ")
        if resp.strip().lower() != "y":
            print("Aborted.")
            return

    log_dir = args.output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.jsonl"

    import threading

    manifest_lock = threading.Lock()

    gpu_slots: "queue.Queue[int]" = queue.Queue()
    for gpu in gpu_ids:
        for _ in range(args.runs_per_gpu):
            gpu_slots.put(gpu)

    def worker(job: Job) -> dict:
        gpu = gpu_slots.get()
        try:
            return run_job(job, gpu, args.python, log_dir, manifest_lock, manifest_path, mem_fraction)
        finally:
            gpu_slots.put(gpu)

    failures = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(worker, job): job for job in jobs}
        for future in as_completed(futures):
            result = future.result()
            if result["returncode"] != 0:
                failures.append(result)

    print(f"\nDone. {len(jobs) - len(failures)}/{len(jobs)} succeeded.")
    if failures:
        print(f"{len(failures)} failed, see manifest.jsonl and logs/ for details:")
        for f in failures:
            print(f"  {f['run_name']} (rc={f['returncode']}) -> {f['log']}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    sys.exit(main())
