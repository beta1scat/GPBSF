"""Generate and validate the reproducible BGSPCD synthetic dataset.

The unit of data separation is a ``family``: one primitive, its dimensions, and
its rigid pose.  Every noisy/occluded variant of a family is assigned to the
same split, preventing augmented siblings from leaking across train, validation,
and test sets.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


PRIMITIVES = ("cuboid", "frustum", "ellipsoid")
LABELS = {name: index for index, name in enumerate(PRIMITIVES)}
REQUIRED_MANIFEST_FIELDS = {
    "sample_id",
    "path",
    "label",
    "primitive",
    "family_id",
    "split",
    "difficulty",
    "seed",
    "dimensions",
    "pose",
    "visible_fraction",
    "camera_visible_fraction",
    "camera",
    "occluder_fraction",
    "noise_std",
    "outlier_ratio",
}


def _sample_range(rng: np.random.Generator, values: Sequence[float]) -> float:
    if len(values) != 2 or values[0] > values[1]:
        raise ValueError(f"Expected an ordered [minimum, maximum] range, got {values}")
    return float(rng.uniform(float(values[0]), float(values[1])))


def _random_rotation(rng: np.random.Generator, angle_range: Sequence[float]) -> np.ndarray:
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = _sample_range(rng, angle_range)
    skew = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def _sample_pose(rng: np.random.Generator, config: Mapping[str, Sequence[float]]) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = _random_rotation(rng, config["rotation_angle"])
    pose[:3, 3] = [
        _sample_range(rng, config["translation_x"]),
        _sample_range(rng, config["translation_y"]),
        _sample_range(rng, config["translation_z"]),
    ]
    return pose


def _sample_dimensions(
    primitive: str, rng: np.random.Generator, config: Mapping[str, Sequence[float]]
) -> Dict[str, float]:
    if primitive == "cuboid":
        return {name: _sample_range(rng, config[name]) for name in ("length", "width", "height")}
    if primitive == "frustum":
        bottom_radius = _sample_range(rng, config["bottom_radius"])
        return {
            "bottom_radius": bottom_radius,
            "top_radius": bottom_radius * _sample_range(rng, config["top_radius_ratio"]),
            "height": _sample_range(rng, config["height"]),
        }
    if primitive == "ellipsoid":
        return {name: _sample_range(rng, config[name]) for name in ("axis_x", "axis_y", "axis_z")}
    raise ValueError(f"Unsupported primitive: {primitive}")


def _sample_cuboid_surface_with_normals(
    rng: np.random.Generator, count: int, dims: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    half = np.array([dims["length"], dims["width"], dims["height"]], dtype=np.float64) / 2.0
    face_areas = np.array([half[1] * half[2], half[0] * half[2], half[0] * half[1]])
    axes = rng.choice(3, size=count, p=face_areas / face_areas.sum())
    signs = rng.choice(np.array([-1.0, 1.0]), size=count)
    points = rng.uniform(-1.0, 1.0, size=(count, 3)) * half
    points[np.arange(count), axes] = signs * half[axes]
    normals = np.zeros((count, 3), dtype=np.float64)
    normals[np.arange(count), axes] = signs
    return points, normals


def _sample_frustum_surface_with_normals(
    rng: np.random.Generator, count: int, dims: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    r0, r1, height = dims["bottom_radius"], dims["top_radius"], dims["height"]
    slant = math.sqrt((r0 - r1) ** 2 + height**2)
    areas = np.array([math.pi * r0**2, math.pi * r1**2, math.pi * (r0 + r1) * slant])
    components = rng.choice(3, size=count, p=areas / areas.sum())
    points = np.empty((count, 3), dtype=np.float64)
    normals = np.empty((count, 3), dtype=np.float64)

    for component, radius, z in ((0, r0, -height / 2.0), (1, r1, height / 2.0)):
        selected = np.flatnonzero(components == component)
        rho = radius * np.sqrt(rng.random(selected.size))
        theta = rng.uniform(0.0, 2.0 * math.pi, selected.size)
        points[selected] = np.column_stack((rho * np.cos(theta), rho * np.sin(theta), np.full(selected.size, z)))
        normals[selected] = np.array([0.0, 0.0, -1.0 if component == 0 else 1.0])

    selected = np.flatnonzero(components == 2)
    needed = selected.size
    accepted: List[np.ndarray] = []
    accepted_count = 0
    max_radius = max(r0, r1)
    while accepted_count < needed:
        batch_size = max(32, 2 * (needed - accepted_count))
        t = rng.random(batch_size)
        radius = r0 + (r1 - r0) * t
        keep = rng.random(batch_size) <= radius / max_radius
        values = t[keep]
        accepted.append(values)
        accepted_count += values.size
    t = np.concatenate(accepted)[:needed] if needed else np.empty(0)
    radius = r0 + (r1 - r0) * t
    theta = rng.uniform(0.0, 2.0 * math.pi, needed)
    points[selected] = np.column_stack(
        (radius * np.cos(theta), radius * np.sin(theta), -height / 2.0 + height * t)
    )
    normals[selected] = np.column_stack(
        (np.cos(theta), np.sin(theta), np.full(needed, (r0 - r1) / height))
    )
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return points, normals


def _sample_ellipsoid_surface_with_normals(
    rng: np.random.Generator, count: int, dims: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    directions = rng.normal(size=(count, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    axes = np.array([dims["axis_x"], dims["axis_y"], dims["axis_z"]], dtype=np.float64)
    points = directions * axes
    normals = points / np.square(axes)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return points, normals


def _sample_surface(
    primitive: str, rng: np.random.Generator, count: int, dimensions: Mapping[str, float]
) -> np.ndarray:
    return _sample_surface_with_normals(primitive, rng, count, dimensions)[0]


def _sample_surface_with_normals(
    primitive: str, rng: np.random.Generator, count: int, dimensions: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    samplers = {
        "cuboid": _sample_cuboid_surface_with_normals,
        "frustum": _sample_frustum_surface_with_normals,
        "ellipsoid": _sample_ellipsoid_surface_with_normals,
    }
    return samplers[primitive](rng, count, dimensions)


def _apply_pose(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return points @ pose[:3, :3].T + pose[:3, 3]


def _render_single_view(
    points: np.ndarray,
    normals: np.ndarray,
    rng: np.random.Generator,
    camera: Mapping[str, float],
    requested_occluder_fraction: float,
) -> Tuple[np.ndarray, float, float, dict]:
    """Render a partial depth point cloud with a pinhole camera and z-buffer.

    This follows the old generator's depth-rendering logic without depending on
    an interactive Open3D window.  The camera is at the world origin and looks
    along +Z; object pose randomization supplies viewpoint diversity.
    """
    width, height = int(camera["width"]), int(camera["height"])
    focal = 0.5 * width / math.tan(0.5 * float(camera["horizontal_fov_radians"]))
    depth = points[:, 2]
    facing = np.einsum("ij,ij->i", normals, -points) > 0.0
    valid = facing & (depth > float(camera["near_plane_m"]))
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size == 0:
        raise RuntimeError("No front-facing surface points lie in front of the camera")
    projected = points[valid_indices]
    depth_quantization = float(camera["depth_quantization_m"])
    projected_depth = projected[:, 2]
    if depth_quantization > 0.0:
        projected_depth = np.round(projected_depth / depth_quantization) * depth_quantization
    u = np.rint(focal * projected[:, 0] / projected_depth + (width - 1) / 2.0).astype(np.int64)
    v = np.rint(focal * projected[:, 1] / projected_depth + (height - 1) / 2.0).astype(np.int64)
    in_frame = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    valid_indices, projected_depth, u, v = (
        valid_indices[in_frame],
        projected_depth[in_frame],
        u[in_frame],
        v[in_frame],
    )
    if valid_indices.size == 0:
        raise RuntimeError("Object projects outside the configured camera frame")
    pixel = v * width + u
    order = np.lexsort((projected_depth, pixel))
    ordered_pixels = pixel[order]
    keep = np.r_[True, ordered_pixels[1:] != ordered_pixels[:-1]]
    visible_indices = valid_indices[order[keep]]
    camera_visible_fraction = float(visible_indices.size / points.shape[0])

    # A foreground rectangular mask in image coordinates models occlusion by a
    # second object. Unlike the previous random half-space crop, it respects
    # the camera image plane and keeps the depth-visible surface definition.
    occluded = np.zeros(visible_indices.size, dtype=bool)
    if requested_occluder_fraction > 0.0 and visible_indices.size > 1:
        visible_u, visible_v = u[order[keep]], v[order[keep]]
        width_fraction = min(0.95, math.sqrt(requested_occluder_fraction))
        object_width = max(1, int(visible_u.max() - visible_u.min() + 1))
        object_height = max(1, int(visible_v.max() - visible_v.min() + 1))
        rectangle_width = max(1, int(round(object_width * width_fraction)))
        rectangle_height = max(1, int(round(object_height * width_fraction)))
        anchor = int(rng.integers(visible_indices.size))
        center_u, center_v = int(visible_u[anchor]), int(visible_v[anchor])
        lower_u, upper_u = center_u - rectangle_width // 2, center_u + rectangle_width // 2
        lower_v, upper_v = center_v - rectangle_height // 2, center_v + rectangle_height // 2
        occluded = (visible_u >= lower_u) & (visible_u <= upper_u) & (visible_v >= lower_v) & (visible_v <= upper_v)
        if occluded.all():
            occluded[rng.integers(visible_indices.size)] = False
    selected = visible_indices[~occluded]
    selected_points = points[selected].copy()
    if depth_quantization > 0.0:
        selected_points[:, 2] = np.round(selected_points[:, 2] / depth_quantization) * depth_quantization
    record = {
        "width": width,
        "height": height,
        "fx": focal,
        "fy": focal,
        "cx": (width - 1) / 2.0,
        "cy": (height - 1) / 2.0,
        "world_to_camera": np.eye(4).tolist(),
        "depth_quantization_m": depth_quantization,
    }
    return selected_points, camera_visible_fraction, float(occluded.mean()), record


def _add_corruption(
    visible_points: np.ndarray,
    rng: np.random.Generator,
    point_count: int,
    noise_std: float,
    outlier_ratio: float,
    bbox_expansion: float,
) -> np.ndarray:
    outlier_count = min(point_count - 1, int(round(point_count * outlier_ratio)))
    inlier_count = point_count - outlier_count
    replace = visible_points.shape[0] < inlier_count
    selected = visible_points[rng.choice(visible_points.shape[0], size=inlier_count, replace=replace)].copy()
    if noise_std > 0.0:
        selected += rng.normal(0.0, noise_std, size=selected.shape)

    if outlier_count:
        lower = visible_points.min(axis=0)
        upper = visible_points.max(axis=0)
        center = (lower + upper) / 2.0
        half_extent = np.maximum((upper - lower) * bbox_expansion / 2.0, noise_std * 3.0)
        outliers = rng.uniform(center - half_extent, center + half_extent, size=(outlier_count, 3))
        selected = np.vstack((selected, outliers))

    rng.shuffle(selected)
    return selected.astype(np.float32, copy=False)


def _allocate_family_splits(
    primitives: Sequence[str], split_ratios: Mapping[str, float], seed: int
) -> Dict[int, str]:
    if not math.isclose(sum(split_ratios.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("Split ratios must sum to 1.0")
    split_names = list(split_ratios)
    assignments: Dict[int, str] = {}
    for primitive_index, primitive in enumerate(PRIMITIVES):
        indices = np.array([i for i, value in enumerate(primitives) if value == primitive], dtype=np.int64)
        rng = np.random.default_rng(np.random.SeedSequence([seed, primitive_index, 991]))
        rng.shuffle(indices)
        cumulative = 0
        for split_index, split_name in enumerate(split_names):
            if split_index == len(split_names) - 1:
                end = indices.size
            else:
                cumulative += int(round(indices.size * float(split_ratios[split_name])))
                end = min(cumulative, indices.size)
            start = 0 if split_index == 0 else sum(
                int(round(indices.size * float(split_ratios[name]))) for name in split_names[:split_index]
            )
            start = min(start, indices.size)
            for family_index in indices[start:end]:
                assignments[int(family_index)] = split_name
    if len(assignments) != len(primitives):
        raise RuntimeError("Failed to assign every family to exactly one split")
    return assignments


def _seed(master_seed: int, *parts: int) -> int:
    return int(np.random.SeedSequence([master_seed, *parts]).generate_state(1, dtype=np.uint32)[0])


def _resolve_output(config_path: Path, output_dir: str) -> Path:
    path = Path(output_dir)
    if path.is_absolute():
        return path
    # Config lives in config/experiments; relative output paths are repo-relative.
    return (config_path.resolve().parents[2] / path).resolve()


def generate_dataset(args: argparse.Namespace) -> None:
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    master_seed = args.seed if args.seed is not None else int(config["master_seed"])
    num_families = args.num_families or int(config["num_families"])
    variants = args.variants_per_difficulty or int(config["variants_per_difficulty"])
    point_count = args.points_per_sample or int(config["points_per_sample"])
    output_root = Path(args.output).resolve() if args.output else _resolve_output(config_path, config["output_dir"])
    manifest_path = output_root / "manifest.jsonl"

    if num_families < len(PRIMITIVES):
        raise ValueError(f"num_families must be at least {len(PRIMITIVES)}")
    if variants < 1 or point_count < 1:
        raise ValueError("variants_per_difficulty and points_per_sample must be positive")
    if manifest_path.exists():
        raise FileExistsError(
            f"Dataset already exists at {output_root}; choose a new --output directory explicitly"
        )
    output_root.mkdir(parents=True, exist_ok=True)

    primitives = [PRIMITIVES[index % len(PRIMITIVES)] for index in range(num_families)]
    split_by_family = _allocate_family_splits(primitives, config["splits"], master_seed)
    difficulties = list(config["difficulty"])
    records: List[dict] = []

    for family_index, primitive in enumerate(primitives):
        family_seed = _seed(master_seed, family_index, 17)
        family_rng = np.random.default_rng(family_seed)
        dimensions = _sample_dimensions(primitive, family_rng, config["primitives"][primitive])
        pose = _sample_pose(family_rng, config["pose"])
        family_id = f"{primitive}_{family_index:06d}"
        split = split_by_family[family_index]

        raw_count = max(
            point_count * max(int(config["surface_oversample_factor"]), 8),
            point_count * 8,
        )
        clean_local, clean_normals = _sample_surface_with_normals(primitive, family_rng, raw_count, dimensions)
        clean_world = _apply_pose(clean_local, pose)
        world_normals = clean_normals @ pose[:3, :3].T

        for difficulty_index, difficulty in enumerate(difficulties):
            difficulty_config = config["difficulty"][difficulty]
            for variant_index in range(variants):
                sample_seed = _seed(master_seed, family_index, difficulty_index, variant_index, 53)
                rng = np.random.default_rng(sample_seed)
                requested_occluder_fraction = _sample_range(rng, difficulty_config["occluder_fraction"])
                noise_std = _sample_range(rng, difficulty_config["noise_std"])
                outlier_ratio = _sample_range(rng, difficulty_config["outlier_ratio"])
                visible_points, camera_visible_fraction, measured_occluder_fraction, camera_record = _render_single_view(
                    clean_world,
                    world_normals,
                    rng,
                    config["camera"],
                    requested_occluder_fraction,
                )
                points = _add_corruption(
                    visible_points,
                    rng,
                    point_count,
                    noise_std,
                    outlier_ratio,
                    float(config["outlier_bbox_expansion"]),
                )

                sample_id = f"{family_id}_{difficulty}_v{variant_index:02d}"
                relative_path = Path("samples") / split / primitive / f"{sample_id}.npz"
                destination = output_root / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    destination,
                    points=points,
                    label=np.int64(LABELS[primitive]),
                    primitive=np.array(primitive),
                    family_id=np.array(family_id),
                    dimensions=np.array(json.dumps(dimensions, sort_keys=True)),
                    pose=pose,
                    split=np.array(split),
                    difficulty=np.array(difficulty),
                    seed=np.uint32(sample_seed),
                    visible_fraction=np.float64(visible_points.shape[0] / clean_world.shape[0]),
                    camera_visible_fraction=np.float64(camera_visible_fraction),
                    occluder_fraction=np.float64(measured_occluder_fraction),
                    camera=np.array(json.dumps(camera_record, sort_keys=True)),
                    noise_std=np.float64(noise_std),
                    outlier_ratio=np.float64(outlier_ratio),
                )
                records.append(
                    {
                        "sample_id": sample_id,
                        "path": relative_path.as_posix(),
                        "label": LABELS[primitive],
                        "primitive": primitive,
                        "family_id": family_id,
                        "split": split,
                        "difficulty": difficulty,
                        "seed": sample_seed,
                        "dimensions": dimensions,
                        "pose": pose.tolist(),
                        "visible_fraction": float(visible_points.shape[0] / clean_world.shape[0]),
                        "camera_visible_fraction": camera_visible_fraction,
                        "occluder_fraction": measured_occluder_fraction,
                        "camera": camera_record,
                        "noise_std": noise_std,
                        "outlier_ratio": outlier_ratio,
                    }
                )

        if (family_index + 1) % 100 == 0 or family_index + 1 == num_families:
            print(f"Generated {family_index + 1}/{num_families} families")

    with manifest_path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    metadata = {
        "dataset_name": config["dataset_name"],
        "master_seed": master_seed,
        "num_families": num_families,
        "variants_per_difficulty": variants,
        "points_per_sample": point_count,
        "sample_count": len(records),
        "labels": LABELS,
        "config": config,
    }
    (output_root / "dataset_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(records)} samples and manifest to {output_root}")


def _read_manifest(path: Path) -> Iterable[Tuple[int, dict]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line.strip():
                yield line_number, json.loads(line)


def validate_dataset(args: argparse.Namespace) -> None:
    dataset_root = Path(args.dataset_root).resolve()
    manifest_path = Path(args.manifest).resolve() if args.manifest else dataset_root / "manifest.jsonl"
    expected_points = int(args.points_per_sample)
    errors: List[str] = []
    sample_ids = set()
    paths = set()
    family_splits: Dict[str, str] = {}
    sample_count = 0

    for line_number, record in _read_manifest(manifest_path):
        sample_count += 1
        missing = REQUIRED_MANIFEST_FIELDS - set(record)
        if missing:
            errors.append(f"line {line_number}: missing fields {sorted(missing)}")
            continue
        sample_id = str(record["sample_id"])
        relative_path = Path(str(record["path"]))
        if sample_id in sample_ids:
            errors.append(f"line {line_number}: duplicate sample_id {sample_id}")
        sample_ids.add(sample_id)
        if relative_path.as_posix() in paths:
            errors.append(f"line {line_number}: duplicate path {relative_path.as_posix()}")
        paths.add(relative_path.as_posix())

        primitive = str(record["primitive"])
        if primitive not in LABELS or int(record["label"]) != LABELS.get(primitive):
            errors.append(f"line {line_number}: invalid label/primitive pair")
        family_id, split = str(record["family_id"]), str(record["split"])
        previous_split = family_splits.setdefault(family_id, split)
        if previous_split != split:
            errors.append(f"line {line_number}: family {family_id} occurs in {previous_split} and {split}")

        sample_path = (dataset_root / relative_path).resolve()
        try:
            sample_path.relative_to(dataset_root)
        except ValueError:
            errors.append(f"line {line_number}: path escapes dataset root: {relative_path}")
            continue
        if not sample_path.is_file():
            errors.append(f"line {line_number}: missing file {relative_path}")
            continue
        try:
            with np.load(sample_path, allow_pickle=False) as sample:
                if "points" not in sample or sample["points"].shape != (expected_points, 3):
                    shape = sample["points"].shape if "points" in sample else None
                    errors.append(
                        f"line {line_number}: expected points shape ({expected_points}, 3), got {shape}"
                    )
                if int(sample["label"]) != int(record["label"]):
                    errors.append(f"line {line_number}: NPZ label differs from manifest")
                if str(sample["family_id"]) != family_id or str(sample["split"]) != split:
                    errors.append(f"line {line_number}: NPZ family/split differs from manifest")
                if sample["pose"].shape != (4, 4):
                    errors.append(f"line {line_number}: pose is not 4x4")
                if "camera" not in sample or "occluder_fraction" not in sample:
                    errors.append(f"line {line_number}: missing camera visibility metadata")
                else:
                    try:
                        camera = json.loads(str(sample["camera"]))
                        if not {"fx", "fy", "cx", "cy", "world_to_camera"} <= set(camera):
                            errors.append(f"line {line_number}: incomplete camera metadata")
                    except (TypeError, json.JSONDecodeError):
                        errors.append(f"line {line_number}: invalid camera metadata")
        except (OSError, ValueError, KeyError) as exc:
            errors.append(f"line {line_number}: cannot read {relative_path}: {exc}")

    if sample_count == 0:
        errors.append("manifest contains no samples")
    if errors:
        preview = "\n".join(f"- {message}" for message in errors[:50])
        remainder = "" if len(errors) <= 50 else f"\n- ... {len(errors) - 50} more errors"
        raise RuntimeError(f"Validation failed with {len(errors)} error(s):\n{preview}{remainder}")
    print(
        f"Validation passed: {sample_count} samples, {len(family_splits)} families, "
        f"{expected_points} points per sample"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate a BGSPCD dataset")
    generate.add_argument("--config", default="config/experiments/bgspcd.json")
    generate.add_argument("--output", help="override output directory")
    generate.add_argument("--num-families", type=int, help="override family count")
    generate.add_argument("--variants-per-difficulty", type=int, help="override variants per difficulty")
    generate.add_argument("--points-per-sample", type=int, help="override fixed point count")
    generate.add_argument("--seed", type=int, help="override master seed")
    generate.set_defaults(handler=generate_dataset)

    validate = subparsers.add_parser("validate", help="validate files, labels, points, and split isolation")
    validate.add_argument("--dataset-root", default="data/bgspcd_v3_camera")
    validate.add_argument("--manifest", help="override manifest path")
    validate.add_argument("--points-per-sample", type=int, default=2048)
    validate.set_defaults(handler=validate_dataset)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
