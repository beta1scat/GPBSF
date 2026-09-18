"""Evaluate GPBSF fitting against synthetic geometry retained in the manifest.

This script deliberately separates two quantities that were previously
conflated: observed-to-fitted residual and fitted-to-independent-ground-truth
surface distance.  It does not use the old CPD self-consistency value as an
absolute reconstruction metric.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.bgspcd.dataset import _apply_pose, _sample_surface  # noqa: E402
from experiments.classification.labels import LABEL_BY_NAME  # noqa: E402


def _records(manifest: Path, split: str, limit: Optional[int]):
    result = []
    with manifest.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            if record["split"] == split:
                result.append(record)
            if limit and len(result) >= limit:
                break
    if not result:
        raise ValueError(f"No {split!r} samples in {manifest}")
    return result


def _matrix(transform) -> np.ndarray:
    return np.asarray(getattr(transform, "A", transform), dtype=np.float64)


def _valid_fit_parameters(params) -> bool:
    """Check that a fitter result is usable before counting it as a success."""
    if not params or len(params) != 4:
        return False
    try:
        dimensions = np.asarray(params[:3], dtype=np.float64)
        transform = _matrix(params[3])
    except (TypeError, ValueError):
        return False
    return bool(
        np.all(np.isfinite(dimensions))
        and np.all(dimensions > 0.0)
        and transform.shape == (4, 4)
        and np.all(np.isfinite(transform))
    )


def _predicted_geometry(primitive: str, params):
    values = [float(value) for value in params[:-1]]
    transform = _matrix(params[-1])
    if primitive == "cuboid":
        dimensions = {"length": 2.0 * values[0], "width": 2.0 * values[1], "height": 2.0 * values[2]}
    elif primitive == "frustum":
        dimensions = {"top_radius": values[0], "bottom_radius": values[1], "height": values[2]}
    elif primitive == "ellipsoid":
        dimensions = {"axis_x": values[0], "axis_y": values[1], "axis_z": values[2]}
    else:
        raise ValueError(primitive)
    return dimensions, transform


def _dimension_vector(primitive: str, values: dict) -> np.ndarray:
    if primitive == "cuboid":
        return np.sort([values["length"], values["width"], values["height"]])
    if primitive == "frustum":
        return np.concatenate(
            (np.sort([values["top_radius"], values["bottom_radius"]]), np.asarray([values["height"]]))
        )
    if primitive == "ellipsoid":
        return np.sort([values["axis_x"], values["axis_y"], values["axis_z"]])
    raise ValueError(primitive)


def _chamfer(left: np.ndarray, right: np.ndarray) -> float:
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise ImportError("Fitting evaluation requires scipy for nearest-neighbour surface distance.") from exc
    left_tree, right_tree = cKDTree(left), cKDTree(right)
    return float(left_tree.query(right, k=1)[0].mean() + right_tree.query(left, k=1)[0].mean())


def _summary(rows: list[dict]) -> dict:
    valid = [row for row in rows if row["fit_success"]]
    metrics_valid = [row for row in valid if row["metrics_valid"]]
    result = {
        "sample_count": len(rows),
        "fit_success_rate": len(valid) / len(rows) if rows else 0.0,
        "metrics_valid_count": len(metrics_valid),
        "metrics_valid_rate_among_fits": len(metrics_valid) / len(valid) if valid else 0.0,
    }
    for key in ("center_error_m", "size_relative_error", "surface_chamfer_m", "observed_to_fitted_m", "runtime_ms"):
        values = [float(row[key]) for row in valid if row[key] is not None]
        result[key] = {"median": float(np.median(values)) if values else None, "p95": float(np.quantile(values, 0.95)) if values else None}
    return result


def evaluate(args: argparse.Namespace) -> None:
    try:
        import open3d as o3d
        from shape_fitting import FittingByBGS
    except ImportError as exc:
        raise ImportError("Fitting evaluation requires the GPBSF Open3D fitting dependencies.") from exc

    manifest = Path(args.manifest).resolve()
    dataset_root = manifest.parent
    records = _records(manifest, args.split, args.limit)
    fitter = FittingByBGS()
    rows = []
    for index, record in enumerate(records, start=1):
        with np.load(dataset_root / record["path"], allow_pickle=False) as sample:
            observed = np.asarray(sample["points"], dtype=np.float64)
        point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(observed))
        point_cloud.estimate_normals()
        primitive = record["primitive"]
        started = __import__("time").perf_counter()
        try:
            params = fitter.fitting(point_cloud, str(LABEL_BY_NAME[primitive]), visual=False)
        except Exception as exc:  # A failure is a result, not a silently dropped sample.
            params, error = [], f"{type(exc).__name__}: {exc}"
        else:
            error = fitter.last_error
        fit_success = _valid_fit_parameters(params)
        if params and not fit_success and error is None:
            error = "invalid_fit_parameters"
        runtime_ms = 1000.0 * (__import__("time").perf_counter() - started)
        row = {
            "sample_id": record["sample_id"],
            "primitive": primitive,
            "difficulty": record.get("difficulty", "not_stratified"),
            "source_dataset": record.get("source_dataset", "unspecified"),
            "observation_mode": record.get("observation_mode", "unspecified"),
            "fit_success": fit_success,
            "metrics_valid": False,
            "fitting_method": fitter.last_method,
            "error": error,
            "metric_error": None,
            "runtime_ms": runtime_ms,
            "center_error_m": None,
            "size_relative_error": None,
            "surface_chamfer_m": None,
            "observed_to_fitted_m": None,
        }
        if params:
            try:
                predicted_dimensions, predicted_pose = _predicted_geometry(primitive, params)
                truth_dimensions, truth_pose = record["dimensions"], np.asarray(record["pose"], dtype=np.float64)
                predicted_vector = np.asarray(_dimension_vector(primitive, predicted_dimensions), dtype=np.float64)
                truth_vector = np.asarray(_dimension_vector(primitive, truth_dimensions), dtype=np.float64)
                row["center_error_m"] = float(np.linalg.norm(predicted_pose[:3, 3] - truth_pose[:3, 3]))
                row["size_relative_error"] = float(np.mean(np.abs(predicted_vector - truth_vector) / truth_vector))
                rng = np.random.default_rng(int(record["seed"]) + 701)
                truth_surface = _apply_pose(_sample_surface(primitive, rng, args.surface_points, truth_dimensions), truth_pose)
                predicted_surface = _apply_pose(_sample_surface(primitive, rng, args.surface_points, predicted_dimensions), predicted_pose)
                row["surface_chamfer_m"] = _chamfer(predicted_surface, truth_surface)
                row["observed_to_fitted_m"] = _chamfer(observed, predicted_surface)
                row["metrics_valid"] = True
            except Exception as exc:
                # A metric or manifest error must not be reclassified as a
                # geometric fitting failure.  Both states are retained in the
                # CSV so the reported success rate remains auditable.
                row["metric_error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        if index % 25 == 0 or index == len(records):
            print(f"Processed {index}/{len(records)} samples")

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    difficulties = sorted({row["difficulty"] for row in rows})
    primitives = sorted({row["primitive"] for row in rows})
    sources = sorted({row["source_dataset"] for row in rows})
    by_difficulty = {difficulty: _summary([row for row in rows if row["difficulty"] == difficulty]) for difficulty in difficulties}
    by_primitive = {primitive: _summary([row for row in rows if row["primitive"] == primitive]) for primitive in primitives}
    by_source = {source: _summary([row for row in rows if row["source_dataset"] == source]) for source in sources}
    summary = {
        "overall": _summary(rows),
        "by_difficulty": by_difficulty,
        "by_primitive": by_primitive,
        "by_source_dataset": by_source,
        "protocol": {
            "routing": "ground_truth_oracle",
            "meaning": "isolates geometric fitting from classification error",
            "fit_success_definition": "the fitter returned finite geometric parameters",
            "metrics_valid_definition": "all ground-truth metric calculations completed",
        },
    }
    output.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/bgspcd_v3_camera/manifest.jsonl")
    parser.add_argument("--split", default="test", choices=("val", "test"))
    parser.add_argument("--output", default="runs/fitting/oracle_test.csv")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--surface-points", type=int, default=4096)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
