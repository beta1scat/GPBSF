"""Build the V4-Robust mixed BGSPCD training corpus.

V4 keeps the public legacy training distribution as an explicitly recorded
source and adds camera-conditioned synthetic observations.  It deliberately
does not modify V3 or the original text dataset.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Mapping

import numpy as np

from .dataset import (
    LABELS,
    PRIMITIVES,
    _add_corruption,
    _allocate_family_splits,
    _apply_pose,
    _sample_dimensions,
    _sample_pose,
    _sample_surface_with_normals,
    _seed,
)


def _resolve_config_output(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config_path.resolve().parents[2] / path).resolve()


def _render_depth_visible(points, normals, camera: Mapping[str, float]):
    """Return the nearest front-facing point per image pixel without occlusion."""
    width, height = int(camera["width"]), int(camera["height"])
    focal = 0.5 * width / math.tan(0.5 * float(camera["horizontal_fov_radians"]))
    depth = points[:, 2]
    facing = np.einsum("ij,ij->i", normals, -points) > 0.0
    valid = facing & (depth > float(camera["near_plane_m"]))
    indices = np.flatnonzero(valid)
    if not indices.size:
        return np.empty((0, 3), dtype=np.float64), 0.0, {}
    projected, projected_depth = points[indices], depth[indices]
    quantization = float(camera["depth_quantization_m"])
    if quantization > 0.0:
        projected_depth = np.round(projected_depth / quantization) * quantization
    u = np.rint(focal * projected[:, 0] / projected_depth + (width - 1) / 2.0).astype(np.int64)
    v = np.rint(focal * projected[:, 1] / projected_depth + (height - 1) / 2.0).astype(np.int64)
    frame = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    indices, projected_depth, u, v = indices[frame], projected_depth[frame], u[frame], v[frame]
    if not indices.size:
        return np.empty((0, 3), dtype=np.float64), 0.0, {}
    pixel = v * width + u
    order = np.lexsort((projected_depth, pixel))
    kept = np.r_[True, pixel[order][1:] != pixel[order][:-1]]
    visible_indices = indices[order[kept]]
    visible = points[visible_indices].copy()
    if quantization > 0.0:
        visible[:, 2] = np.round(visible[:, 2] / quantization) * quantization
    record = {
        "width": width, "height": height, "fx": focal, "fy": focal,
        "cx": (width - 1) / 2.0, "cy": (height - 1) / 2.0,
        "world_to_camera": np.eye(4).tolist(), "depth_quantization_m": quantization,
    }
    return visible, float(visible.size / 3 / points.shape[0]), record


def _sphere_occlude(points: np.ndarray, rng: np.random.Generator, fraction: float):
    """Model an opaque foreground sphere in camera coordinates.

    The ray/sphere test creates a physically plausible circular depth shadow,
    unlike V3's image-aligned rectangular crop.
    """
    if fraction <= 0.0 or points.shape[0] < 2:
        return points, 0.0
    anchor = points[int(rng.integers(points.shape[0]))]
    anchor_depth = float(anchor[2])
    center_depth = anchor_depth * float(rng.uniform(0.35, 0.75))
    direction = anchor / np.linalg.norm(anchor)
    center = direction * center_depth
    radius = max(1e-4, center_depth * math.sqrt(min(fraction, 0.45)) * 0.7)
    ray = points / np.linalg.norm(points, axis=1, keepdims=True)
    along = ray @ center
    closest_sq = np.sum(center * center) - np.square(along)
    intersection = (closest_sq <= radius * radius) & (along > 0.0)
    # The sphere must lie in front of the measured point along the same ray.
    intersection &= along < np.linalg.norm(points, axis=1)
    selected = points[~intersection]
    if selected.shape[0] == 0:
        return points, 0.0
    return selected, float(intersection.mean())


def _mode_cloud(points, normals, mode: Mapping[str, object], rng, camera):
    if bool(mode.get("full_observation", False)):
        return points.copy(), 1.0, 0.0, {"mode": "full"}
    visible, visible_fraction, camera_record = _render_depth_visible(points, normals, camera)
    if not visible.shape[0]:
        return visible, visible_fraction, 0.0, camera_record
    fraction_range = mode.get("occluder_fraction", [0.0, 0.0])
    requested = float(rng.uniform(float(fraction_range[0]), float(fraction_range[1])))
    observed, actual = _sphere_occlude(visible, rng, requested)
    return observed, visible_fraction, actual, camera_record


def _write_npz(destination: Path, points: np.ndarray, label: int, metadata: Mapping[str, object]):
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, points=points.astype(np.float32, copy=False), label=np.int64(label), **metadata)


def _legacy_records(legacy_root: Path, output_root: Path, seed: int):
    """Convert the published legacy corpus once to compact XYZ NPZ files.

    The original files contain 10k XYZ+normal rows.  Caching deterministic
    2048-point XYZ subsets avoids parsing large ASCII files at every epoch and
    ensures no normals become an accidental model-specific input advantage.
    """
    train_names = [line.strip() for line in (legacy_root / "bgspcd_train.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    test_names = [line.strip() for line in (legacy_root / "bgspcd_test.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    split_by_name = {name: "test" for name in test_names}
    per_class = {primitive: [] for primitive in PRIMITIVES}
    name_to_primitive = {"cube": "cuboid", "cone": "frustum", "ellipsoid": "ellipsoid"}
    for name in train_names:
        prefix = name.rsplit("_", 1)[0]
        per_class[name_to_primitive[prefix]].append(name)
    for primitive, names in per_class.items():
        rng = np.random.default_rng(_seed(seed, LABELS[primitive], 707))
        names = sorted(names)
        validation_count = int(round(len(names) * 0.10))
        selected = set(rng.choice(np.asarray(names), size=validation_count, replace=False).tolist())
        for name in names:
            split_by_name[name] = "val" if name in selected else "train"

    records = []
    for name in train_names + test_names:
        prefix = name.rsplit("_", 1)[0]
        primitive = name_to_primitive[prefix]
        source = legacy_root / prefix / name
        target = output_root / "legacy_cache" / split_by_name[name] / primitive / f"legacy_{name}.npz"
        if not target.exists():
            raw = np.loadtxt(source, dtype=np.float32, usecols=(0, 1, 2))
            if raw.ndim != 2 or raw.shape[0] == 0:
                raise ValueError(f"Legacy sample has no valid XYZ points: {source}")
            rng = np.random.default_rng(_seed(seed, len(records), 719))
            selected = raw[rng.choice(raw.shape[0], size=2048, replace=raw.shape[0] < 2048)]
            unique_count = int(np.unique(selected, axis=0).shape[0])
            _write_npz(target, selected, LABELS[primitive], {
                "source_dataset": np.array("legacy_public"),
                "legacy_original_point_count": np.int64(raw.shape[0]),
                "duplicate_ratio": np.float64(1.0 - unique_count / 2048),
            })
        else:
            with np.load(target, allow_pickle=False) as cached:
                unique_count = int(np.unique(np.asarray(cached["points"]), axis=0).shape[0])
        records.append({
            "sample_id": f"legacy/{name}", "path": target.relative_to(output_root).as_posix(),
            "label": LABELS[primitive], "primitive": primitive, "family_id": f"legacy/{name}",
            "split": split_by_name[name], "source_dataset": "legacy_public", "observation_mode": "legacy",
            "unique_visible_points": unique_count, "final_unique_points": 2048,
            "duplicate_ratio": 1.0 - unique_count / 2048,
        })
    return records


def build_robust_dataset(args: argparse.Namespace) -> None:
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_root = Path(args.output).resolve() if args.output else _resolve_config_output(config_path, config["output_dir"])
    if output_root.exists():
        raise FileExistsError(f"Output exists: {output_root}. Choose a new --output path; V4 never overwrites data.")
    output_root.mkdir(parents=True)
    seed, families_per_primitive = int(config["master_seed"]), int(config["families_per_primitive"])
    modes = list(config["observation_modes"])
    point_count, minimum = int(config["points_per_sample"]), int(config["minimum_unique_visible_points"])
    if point_count != 2048 or minimum < point_count:
        raise ValueError("V4-Robust requires 2048 output points and at least that many unique rendered points")
    primitives = [primitive for primitive in PRIMITIVES for _ in range(families_per_primitive)]
    split_by_index = _allocate_family_splits(primitives, config["splits"], seed)
    records = []
    raw_count = int(config["surface_candidate_points"])

    for index, primitive in enumerate(primitives):
        family_rng = np.random.default_rng(_seed(seed, index, 811))
        dimensions = _sample_dimensions(primitive, family_rng, config["primitives"][primitive])
        family_id, split = f"v4/{primitive}_{index:06d}", split_by_index[index]
        for mode_index, mode in enumerate(modes):
            accepted = None
            for attempt in range(int(config["max_view_attempts"])):
                sample_seed = _seed(seed, index, mode_index, attempt, 829)
                rng = np.random.default_rng(sample_seed)
                pose = _sample_pose(rng, config["pose"])
                local, local_normals = _sample_surface_with_normals(primitive, rng, raw_count, dimensions)
                world, normals = _apply_pose(local, pose), local_normals @ pose[:3, :3].T
                observed, camera_fraction, occluder_fraction, camera_record = _mode_cloud(world, normals, mode, rng, config["camera"])
                if observed.shape[0] >= minimum:
                    accepted = (rng, sample_seed, pose, observed, camera_fraction, occluder_fraction, camera_record)
                    break
            if accepted is None:
                raise RuntimeError(f"Could not obtain {minimum} unique points for {family_id}/{mode['name']}")
            rng, sample_seed, pose, observed, camera_fraction, occluder_fraction, camera_record = accepted
            noise_range = mode["noise_std"]
            outlier_range = mode["outlier_ratio"]
            noise = float(rng.uniform(float(noise_range[0]), float(noise_range[1])))
            outlier = float(rng.uniform(float(outlier_range[0]), float(outlier_range[1])))
            output = _add_corruption(observed, rng, point_count, noise, outlier, float(config["outlier_bbox_expansion"]))
            if np.unique(output, axis=0).shape[0] != point_count:
                raise RuntimeError("V4 output contains duplicate points; check visibility threshold and corruption settings")
            sample_id = f"{primitive}_{index:06d}_{mode['name']}"
            relative = Path("samples") / split / primitive / f"{sample_id}.npz"
            _write_npz(output_root / relative, output, LABELS[primitive], {
                "primitive": np.array(primitive), "family_id": np.array(family_id), "split": np.array(split),
                "source_dataset": np.array("v4_camera"), "observation_mode": np.array(mode["name"]),
                "dimensions": np.array(json.dumps(dimensions, sort_keys=True)), "pose": pose,
                "seed": np.uint32(sample_seed),
                "camera": np.array(json.dumps(camera_record, sort_keys=True)), "noise_std": np.float64(noise),
                "outlier_ratio": np.float64(outlier), "occluder_fraction": np.float64(occluder_fraction),
            })
            records.append({
                "sample_id": f"v4/{sample_id}", "path": relative.as_posix(), "label": LABELS[primitive],
                "primitive": primitive, "family_id": family_id, "split": split, "source_dataset": "v4_camera",
                "observation_mode": mode["name"], "dimensions": dimensions, "pose": pose.tolist(),
                "camera": camera_record, "visible_fraction": float(observed.shape[0] / raw_count),
                "camera_visible_fraction": camera_fraction, "occluder_fraction": occluder_fraction,
                "noise_std": noise, "outlier_ratio": outlier, "unique_visible_points": int(observed.shape[0]),
                "final_unique_points": point_count, "duplicate_ratio": 0.0, "seed": sample_seed,
            })
        if (index + 1) % 50 == 0 or index + 1 == len(primitives):
            print(f"Generated V4 families: {index + 1}/{len(primitives)}", flush=True)

    if not args.without_legacy:
        legacy_root = _resolve_config_output(config_path, config["legacy_dataset_root"])
        records.extend(_legacy_records(legacy_root, output_root, seed))
    with (output_root / "manifest.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    summary = {
        "dataset_name": config["dataset_name"], "config": config, "sample_count": len(records),
        "counts_by_split": dict(Counter(record["split"] for record in records)),
        "counts_by_source": dict(Counter(record["source_dataset"] for record in records)),
        "counts_by_mode": dict(Counter(record["observation_mode"] for record in records)),
    }
    (output_root / "dataset_metadata.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/experiments/bgspcd_v4_robust.json")
    parser.add_argument("--output")
    parser.add_argument("--without-legacy", action="store_true")
    build_robust_dataset(parser.parse_args())


if __name__ == "__main__":
    main()
