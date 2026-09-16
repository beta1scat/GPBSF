"""Shared point-cloud preprocessing.

The two upstream projects use different tensor layouts.  All normalization and
point selection occurs here, before an adapter changes layout, so both models
receive the identical BxNx3 point cloud.
"""

from __future__ import annotations

import numpy as np


def normalize_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected Nx3 points, got {points.shape}")
    centered = points - points.mean(axis=0, keepdims=True)
    radius = float(np.linalg.norm(centered, axis=1).max())
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("Point cloud has zero or non-finite radius")
    return (centered / radius).astype(np.float32, copy=False)


def fixed_count(points: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    """Select exactly ``count`` points without introducing model-specific FPS."""
    if count < 1:
        raise ValueError("count must be positive")
    points = np.asarray(points, dtype=np.float32)
    if points.shape[0] == count:
        return points
    replace = points.shape[0] < count
    indices = rng.choice(points.shape[0], size=count, replace=replace)
    return points[indices]


def augment_training_points(points, generator):
    """Apply the same lightweight geometric augmentation to either backend."""
    import torch

    scale = torch.empty((points.shape[0], 1, 3), device=points.device).uniform_(0.9, 1.1, generator=generator)
    shift = torch.empty((points.shape[0], 1, 3), device=points.device).uniform_(-0.05, 0.05, generator=generator)
    return points * scale + shift
