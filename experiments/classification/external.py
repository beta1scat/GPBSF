"""Immutable external-test manifest and dataset for the archived experiment data.

This module deliberately has no training interface.  The four archived groups
are an external test set: checkpoints and hyperparameters must be fixed using
only BGSPCD-v3-camera before this module is invoked.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .labels import CLASS_NAMES, validate_label
from .preprocess import fixed_count, normalize_points


GROUPS = ("real_clutter", "real_flat", "sim_clutter", "sim_flat")
RAW_TYPE_TO_LABEL = {
    "0": 0,
    "01": 0,
    "1": 1,
    "11": 1,
    "12": 1,
    "13": 1,
    "14": 1,
    "2": 2,
}


def _load_label(path: Path) -> int:
    record = json.loads(path.read_text(encoding="utf-8"))
    raw_type = str(record.get("input_type"))
    try:
        return RAW_TYPE_TO_LABEL[raw_type]
    except KeyError as exc:
        supported = ", ".join(sorted(RAW_TYPE_TO_LABEL))
        raise ValueError(f"Unsupported input_type {raw_type!r} in {path}; supported: {supported}") from exc


def build_external_manifest(dataset_root, output_path) -> dict:
    """Create a deterministic manifest without moving or editing source data."""
    dataset_root = Path(dataset_root).resolve()
    output_path = Path(output_path).resolve()
    rows = []
    for group in GROUPS:
        group_root = dataset_root / group
        pcd_root, class_root = group_root / "pcd", group_root / "classes"
        if not pcd_root.is_dir() or not class_root.is_dir():
            raise FileNotFoundError(f"Expected pcd/ and classes/ below {group_root}")
        clouds = {item.stem: item for item in pcd_root.glob("*.ply") if item.is_file()}
        labels = {item.stem: item for item in class_root.glob("*.json") if item.is_file()}
        if clouds.keys() != labels.keys():
            missing_cloud = sorted(labels.keys() - clouds.keys())
            missing_label = sorted(clouds.keys() - labels.keys())
            raise ValueError(f"Unpaired files in {group}: missing clouds={missing_cloud}, missing labels={missing_label}")
        for stem in sorted(clouds):
            label = _load_label(labels[stem])
            primitive = CLASS_NAMES[label]
            validate_label(label, primitive)
            rows.append(
                {
                    "sample_id": f"{group}/{stem}",
                    "group": group,
                    "path": clouds[stem].relative_to(output_path.parent).as_posix(),
                    "label_path": labels[stem].relative_to(output_path.parent).as_posix(),
                    "label": label,
                    "primitive": primitive,
                }
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    counts = {group: sum(row["group"] == group for row in rows) for group in GROUPS}
    return {"manifest": str(output_path), "sample_count": len(rows), "groups": counts}


class ExternalPointCloudDataset:
    """Loads archived PLY observations using the classifier's fixed preprocessing."""

    def __init__(self, manifest_path, num_points: int, seed: int, group: str | None = None):
        self.manifest_path = Path(manifest_path).resolve()
        self.root = self.manifest_path.parent
        self.num_points = int(num_points)
        self.seed = int(seed)
        if group is not None and group not in GROUPS:
            raise ValueError(f"Unknown group {group!r}; expected one of {GROUPS}")
        self.records = []
        for line_number, line in enumerate(self.manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if group is not None and record.get("group") != group:
                continue
            validate_label(record["label"], record["primitive"])
            point_path = (self.root / record["path"]).resolve()
            try:
                point_path.relative_to(self.root)
            except ValueError as exc:
                raise ValueError(f"Manifest line {line_number} escapes manifest root") from exc
            if not point_path.is_file():
                raise FileNotFoundError(f"Missing sample listed on line {line_number}: {point_path}")
            self.records.append(record)
        if not self.records:
            raise ValueError(f"No external samples for group {group!r}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        import open3d as o3d
        import torch

        record = self.records[index]
        cloud = o3d.io.read_point_cloud(str(self.root / record["path"]))
        points = np.asarray(cloud.points, dtype=np.float32)
        if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] != 3:
            raise ValueError(f"Invalid or empty point cloud: {record['path']}")
        if not np.isfinite(points).all():
            raise ValueError(f"Non-finite coordinates in: {record['path']}")
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, index, 101]))
        points = normalize_points(fixed_count(points, self.num_points, rng))
        return torch.from_numpy(points), torch.tensor(record["label"], dtype=torch.long), record["sample_id"], record["group"]
