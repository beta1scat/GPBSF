"""Segment Anything Model (SAM) wrapper and depth projection utilities."""

from __future__ import annotations

import cv2
import numpy as np
from segment_anything import SamPredictor, sam_model_registry


class SegmentAnythingModel:
    """Wrapper around Segment Anything Model for prompt-based mask prediction."""

    def __init__(self, checkpoint_path: str, model_type: str = "vit_h", device: str = "cuda"):
        self.sam_model_type = model_type
        self.sam_checkpoint_path = checkpoint_path
        self.sam = sam_model_registry[self.sam_model_type](checkpoint=self.sam_checkpoint_path)
        self.sam = self.sam.to(device=device)
        self.predictor = SamPredictor(self.sam)

    def segment(self, image: np.ndarray, box: np.ndarray) -> np.ndarray:
        """Predict foreground binary mask within a bounding box prompt [x1, y1, x2, y2]."""
        self.predictor.set_image(image)
        masks, _, _ = self.predictor.predict(
            box=box[None, :],
            multimask_output=False,
        )
        return masks[0]


# Backward compatibility alias
SegmentAnythinModel = SegmentAnythingModel


def depth_to_pointcloud(
    depth_image: np.ndarray, fx: float, fy: float, cx: float, cy: float
) -> np.ndarray:
    """Convert depth map to 3D point cloud using pinhole camera intrinsics."""
    height, width = depth_image.shape
    u, v = np.meshgrid(np.arange(1, width + 1), np.arange(1, height + 1))
    z = depth_image
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    pointcloud = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=-1)
    nonzero_indices = np.all(pointcloud != [0, 0, 0], axis=1)
    return pointcloud[nonzero_indices]
