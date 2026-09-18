"""Launch model/seed pairs across multiple GPUs concurrently.

Each task runs in process isolation with its assigned CUDA_VISIBLE_DEVICES.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path
import queue
import torch


def run_single_task(gpu_id: int, model: str, seed: int, config_path: str, base_runs_dir: str) -> None:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"[GPU {gpu_id}] >>> Start {model}, seed={seed}", flush=True)

    train_cmd = [
        sys.executable,
        "-m",
        "experiments.classification.cli",
        "--config",
        config_path,
        "train",
        "--model",
        model,
        "--seed",
        str(seed),
    ]
    subprocess.run(train_cmd, check=True, env=env)

    run_dir = f"{base_runs_dir}/{model}/seed_{seed}"
    eval_cmd = [
        sys.executable,
        "-m",
        "experiments.classification.cli",
        "--config",
        config_path,
        "evaluate",
        "--run-dir",
        run_dir,
        "--split",
        "test",
    ]
    subprocess.run(eval_cmd, check=True, env=env)
    print(f"[GPU {gpu_id}] <<< Finished {model}, seed={seed}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True, choices=("pointnet2", "mamba3d"))
    parser.add_argument("--seeds", nargs="+", required=True, type=int)
    parser.add_argument("--config", default="config/experiments/classification.json")
    parser.add_argument("--gpus", nargs="+", type=int, default=None, help="List of GPU device IDs to use, e.g. 0 1")
    args = parser.parse_args()

    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as stream:
        cfg = json.load(stream)
    base_runs_dir = cfg.get("output", {}).get("runs_dir", "runs/classification")

    if args.gpus is None:
        count = torch.cuda.device_count() if torch.cuda.is_available() else 1
        available_gpus = list(range(count))
    else:
        available_gpus = args.gpus

    tasks = [(model, seed) for model in args.models for seed in args.seeds]
    print(f"Total tasks: {len(tasks)}, Available GPUs: {available_gpus}", flush=True)

    # Use a Queue to assign GPUs to tasks
    gpu_pool = queue.Queue()
    for gid in available_gpus:
        gpu_pool.put(gid)

    def worker(task):
        model, seed = task
        gid = gpu_pool.get()
        try:
            run_single_task(gid, model, seed, str(config_path), base_runs_dir)
        finally:
            gpu_pool.put(gid)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(available_gpus)) as executor:
        futures = [executor.submit(worker, t) for t in tasks]
        for f in concurrent.futures.as_completed(futures):
            f.result()

    print("=== All parallel training tasks completed! ===", flush=True)


if __name__ == "__main__":
    main()
