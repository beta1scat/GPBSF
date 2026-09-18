"""Launch each model/seed pair in a separate Python process.

The upstream projects expose conflicting top-level package names; process
isolation prevents a PointNet++ import from contaminating a Mamba3D run.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True, choices=("pointnet2", "mamba3d"))
    parser.add_argument("--seeds", nargs="+", required=True, type=int)
    parser.add_argument("--config", default="config/experiments/classification.json")
    args = parser.parse_args()
    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as stream:
        cfg = json.load(stream)
    base_runs_dir = cfg.get("output", {}).get("runs_dir", "runs/classification")

    for model in args.models:
        for seed in args.seeds:
            print(f"=== Start {model}, seed={seed} ===", flush=True)
            command = [
                sys.executable,
                "-m",
                "experiments.classification.cli",
                "--config",
                args.config,
                "train",
                "--model",
                model,
                "--seed",
                str(seed),
            ]
            subprocess.run(command, check=True)
            run_dir = f"{base_runs_dir}/{model}/seed_{seed}"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "experiments.classification.cli",
                    "--config",
                    args.config,
                    "evaluate",
                    "--run-dir",
                    run_dir,
                    "--split",
                    "test",
                ],
                check=True,
            )
            print(f"=== Finished {model}, seed={seed} ===", flush=True)


if __name__ == "__main__":
    main()
