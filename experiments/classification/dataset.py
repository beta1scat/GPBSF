"""Manifest-backed dataset for the reproducible BGSPCD protocol."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np

from .labels import validate_label
from .preprocess import fixed_count, normalize_points


class ManifestPointCloudDataset:
    def __init__(self, manifest_path, split: str, num_points: int, training: bool, seed: int, preload: bool = True):
        self.manifest_path = Path(manifest_path).resolve()
        self.root = self.manifest_path.parent
        self.split = split
        self.num_points = int(num_points)
        self.training = bool(training)
        self.seed = int(seed)
        self.preload = bool(preload)
        if self.num_points < 1:
            raise ValueError("num_points must be positive")
        self.records: List[dict] = []
        with self.manifest_path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("split") != split:
                    continue
                validate_label(record["label"], record["primitive"])
                point_path = (self.root / record["path"]).resolve()
                try:
                    point_path.relative_to(self.root)
                except ValueError as exc:
                    raise ValueError(f"Manifest line {line_number} escapes dataset root") from exc
                if not point_path.is_file():
                    raise FileNotFoundError(f"Missing sample listed on line {line_number}: {point_path}")
                self.records.append(record)
        if not self.records:
            raise ValueError(f"No samples for split {split!r} in {self.manifest_path}")

        self.cached_points = None
        self.cached_labels = None
        if self.preload:
            import torch

            points_list = []
            labels_list = []
            for index, record in enumerate(self.records):
                point_path = self.root / record["path"]
                with np.load(point_path, allow_pickle=False) as sample:
                    points = np.asarray(sample["points"], dtype=np.float32)
                    label = int(sample["label"])
                validate_label(label, record["primitive"])
                rng = np.random.default_rng(np.random.SeedSequence([self.seed, index, 101]))
                points = normalize_points(fixed_count(points, self.num_points, rng))
                points_list.append(torch.from_numpy(points))
                labels_list.append(label)
            self.cached_points = torch.stack(points_list, dim=0)
            self.cached_labels = torch.tensor(labels_list, dtype=torch.long)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        import torch

        if self.preload and self.cached_points is not None and self.cached_labels is not None:
            return self.cached_points[index], self.cached_labels[index], self.records[index]["sample_id"]

        record = self.records[index]
        point_path = self.root / record["path"]
        with np.load(point_path, allow_pickle=False) as sample:
            points = np.asarray(sample["points"], dtype=np.float32)
            label = int(sample["label"])
        validate_label(label, record["primitive"])
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, index, 101]))
        points = normalize_points(fixed_count(points, self.num_points, rng))
        return torch.from_numpy(points), torch.tensor(label, dtype=torch.long), record["sample_id"]
