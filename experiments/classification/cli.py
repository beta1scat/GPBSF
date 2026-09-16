"""CLI for reproducible PointNet++--Mamba3D Chapter 4 experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .dataset import ManifestPointCloudDataset
from .engine import evaluate, evaluate_external, evaluate_legacy_mamba3d, read_config, train
from .external import GROUPS, build_external_manifest
from .labels import CLASS_NAMES
from .metrics import bootstrap_mean


def _validate(config: dict) -> None:
    root = Path(__file__).resolve().parents[2]
    manifest = root / config["dataset"]["manifest"]
    counts = {}
    families = {}
    for split in ("train", "val", "test"):
        dataset = ManifestPointCloudDataset(manifest, split, config["training"]["num_points"], False, 0)
        counts[split] = len(dataset)
        for record in dataset.records:
            previous = families.setdefault(record["family_id"], split)
            if previous != split:
                raise RuntimeError(f"Family leakage: {record['family_id']} appears in {previous} and {split}")
    print(json.dumps({"manifest": str(manifest), "counts": counts, "families": len(families)}, indent=2))


def _aggregate(runs_dir: Path, repetitions: int, seed: int) -> dict:
    summaries = []
    for path in sorted(runs_dir.glob("*/seed_*/test_summary.json")):
        summaries.append(json.loads(path.read_text(encoding="utf-8")))
    if not summaries:
        raise FileNotFoundError(f"No test_summary.json below {runs_dir}")
    grouped = {}
    for summary in summaries:
        grouped.setdefault(summary["model"], []).append(summary)
    results = {}
    for model, rows in grouped.items():
        metrics = {}
        for metric in ("accuracy", "macro_f1", "macro_recall", "inference_ms_per_sample"):
            values = [float(row[metric]) for row in rows]
            low, high = bootstrap_mean(values, repetitions, seed)
            metrics[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "bootstrap_95_ci": [low, high],
                "values": values,
            }
        recalls = {}
        for class_name in CLASS_NAMES:
            values = [float(row["per_class_recall"][class_name]) for row in rows]
            low, high = bootstrap_mean(values, repetitions, seed)
            recalls[class_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "bootstrap_95_ci": [low, high],
                "values": values,
            }
        results[model] = {"run_count": len(rows), "metrics": metrics, "per_class_recall": recalls}
    output = runs_dir / "aggregate.json"
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return results


def _aggregate_external(runs_dir: Path, manifest_stem: str, repetitions: int, seed: int) -> dict:
    summaries = []
    for path in sorted(runs_dir.glob(f"*/seed_*/external/{manifest_stem}/external_summary.json")):
        summaries.append(json.loads(path.read_text(encoding="utf-8")))
    if not summaries:
        raise FileNotFoundError(f"No external_summary.json below {runs_dir} for manifest {manifest_stem!r}")
    grouped = {}
    for summary in summaries:
        grouped.setdefault(summary["model"], []).append(summary)
    results = {}
    for model, rows in grouped.items():
        per_group = {}
        for group in GROUPS:
            metrics, recalls = {}, {}
            for metric in ("accuracy", "macro_f1", "macro_recall", "inference_ms_per_sample"):
                values = [float(row["groups"][group][metric]) for row in rows]
                low, high = bootstrap_mean(values, repetitions, seed)
                metrics[metric] = {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0, "bootstrap_95_ci": [low, high], "values": values}
            for class_name in CLASS_NAMES:
                values = [float(row["groups"][group]["per_class_recall"][class_name]) for row in rows]
                low, high = bootstrap_mean(values, repetitions, seed)
                recalls[class_name] = {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0, "bootstrap_95_ci": [low, high], "values": values}
            per_group[group] = {"sample_count": rows[0]["groups"][group]["sample_count"], "metrics": metrics, "per_class_recall": recalls}
        results[model] = {"run_count": len(rows), "groups": per_group}
    output = runs_dir / f"external_aggregate_{manifest_stem}.json"
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return results


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", default="config/experiments/classification.json")
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("validate-data")
    manifest_parser = commands.add_parser("build-external-manifest")
    manifest_parser.add_argument("--dataset-root", default="data/experiment")
    manifest_parser.add_argument("--output", default="data/experiment/external_manifest.jsonl")
    train_parser = commands.add_parser("train")
    train_parser.add_argument("--model", required=True, choices=("pointnet2", "mamba3d"))
    train_parser.add_argument("--seed", required=True, type=int)
    train_parser.add_argument("--run-dir")
    evaluate_parser = commands.add_parser("evaluate")
    evaluate_parser.add_argument("--run-dir", required=True)
    evaluate_parser.add_argument("--split", choices=("val", "test"), default="test")
    external_parser = commands.add_parser("evaluate-external")
    external_parser.add_argument("--run-dir", required=True)
    external_parser.add_argument("--manifest", default="data/experiment/external_manifest.jsonl")
    legacy_parser = commands.add_parser("evaluate-legacy-mamba3d")
    legacy_parser.add_argument("--checkpoint", default="models/ckpt-best.pth")
    legacy_parser.add_argument("--legacy-model-config", default="config/bgspcd.yaml")
    legacy_parser.add_argument("--output-dir", default="runs/classification/mamba3d/legacy_ckpt_best")
    legacy_parser.add_argument("--external-manifest", default="data/experiment/external_manifest.jsonl")
    legacy_parser.add_argument("--evaluation-seed", type=int, default=3407)
    aggregate_parser = commands.add_parser("aggregate")
    aggregate_parser.add_argument("--runs-dir", default="runs/classification")
    aggregate_parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    aggregate_parser.add_argument("--seed", type=int, default=20260915)
    external_aggregate_parser = commands.add_parser("aggregate-external")
    external_aggregate_parser.add_argument("--runs-dir", default="runs/classification")
    external_aggregate_parser.add_argument("--manifest-stem", default="external_manifest")
    external_aggregate_parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    external_aggregate_parser.add_argument("--seed", type=int, default=20260915)
    return result


def main() -> None:
    args = parser().parse_args()
    config = read_config(args.config)
    if args.command == "validate-data":
        _validate(config)
    elif args.command == "build-external-manifest":
        print(json.dumps(build_external_manifest(args.dataset_root, args.output), ensure_ascii=False, indent=2))
    elif args.command == "train":
        print(train(config, args.model, args.seed, args.run_dir))
    elif args.command == "evaluate":
        print(json.dumps(evaluate(config, args.run_dir, args.split), ensure_ascii=False, indent=2))
    elif args.command == "evaluate-external":
        print(json.dumps(evaluate_external(config, args.run_dir, args.manifest), ensure_ascii=False, indent=2))
    elif args.command == "evaluate-legacy-mamba3d":
        print(json.dumps(evaluate_legacy_mamba3d(config, args.checkpoint, args.legacy_model_config, args.output_dir, args.external_manifest, args.evaluation_seed), ensure_ascii=False, indent=2))
    elif args.command == "aggregate-external":
        print(json.dumps(_aggregate_external(Path(args.runs_dir), args.manifest_stem, args.bootstrap_repetitions, args.seed), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(_aggregate(Path(args.runs_dir), args.bootstrap_repetitions, args.seed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
