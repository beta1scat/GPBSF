"""Evaluate completed validation-selected runs on the immutable external dataset."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True, choices=("pointnet2", "mamba3d"))
    parser.add_argument("--seeds", nargs="+", required=True, type=int)
    parser.add_argument("--config", default="config/experiments/classification.json")
    parser.add_argument("--runs-dir", default=None, help="Base runs directory override")
    parser.add_argument("--manifest", default="data/experiment/external_manifest.jsonl")
    args = parser.parse_args()

    config_path = Path(args.config)
    import json
    with config_path.open("r", encoding="utf-8") as stream:
        cfg = json.load(stream)
    base_runs_dir = args.runs_dir or cfg.get("output", {}).get("runs_dir", "runs/classification")

    for model in args.models:
        for seed in args.seeds:
            run_dir = f"{base_runs_dir}/{model}/seed_{seed}"
            print(f"=== External evaluation: {model}, seed={seed} ===", flush=True)
            subprocess.run(
                [sys.executable, "-m", "experiments.classification.cli", "--config", args.config,
                 "evaluate-external", "--run-dir", run_dir, "--manifest", args.manifest],
                check=True,
            )


if __name__ == "__main__":
    main()
