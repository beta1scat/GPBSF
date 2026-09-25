from __future__ import annotations

import numpy as np
from segment_anything import SamPredictor, sam_model_registry

from shape_fitting.pointcloud import depth_to_pointcloud


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

