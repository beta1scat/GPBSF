"""Evaluate completed validation-selected runs on the immutable external dataset."""

from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True, choices=("pointnet2", "mamba3d"))
    parser.add_argument("--seeds", nargs="+", required=True, type=int)
    parser.add_argument("--config", default="config/experiments/classification.json")
    parser.add_argument("--manifest", default="data/experiment/external_manifest.jsonl")
    args = parser.parse_args()
    for model in args.models:
        for seed in args.seeds:
            run_dir = f"runs/classification/{model}/seed_{seed}"
            print(f"=== External evaluation: {model}, seed={seed} ===", flush=True)
            subprocess.run(
                [sys.executable, "-m", "experiments.classification.cli", "--config", args.config,
                 "evaluate-external", "--run-dir", run_dir, "--manifest", args.manifest],
                check=True,
            )


if __name__ == "__main__":
    main()
