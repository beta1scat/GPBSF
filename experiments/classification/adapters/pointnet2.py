"""PointNet++ SSG adapter using the pinned upstream implementation."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from .base import ClassificationAdapter


class PointNet2Adapter(ClassificationAdapter):
    name = "pointnet2"

    def __init__(self, repository_root: Path, device):
        import torch.nn.functional as functional

        self._functional = functional
        models_root = repository_root / "submodels" / "pointnet2" / "models"
        if not models_root.is_dir():
            raise FileNotFoundError(f"PointNet++ submodule is missing: {models_root}")
        sys.path.insert(0, str(models_root))
        try:
            module = importlib.import_module("pointnet2_cls_ssg")
        except ImportError as exc:
            raise ImportError(
                "Unable to import PointNet++. Initialize submodules and install PyTorch first."
            ) from exc
        self._model = module.get_model(num_class=3, normal_channel=False).to(device)

    @property
    def model(self):
        return self._model

    def logits(self, points_bnc):
        log_probabilities, _ = self._model(points_bnc.transpose(1, 2).contiguous())
        # The upstream implementation emits log-softmax scores.  They retain
        # the same argmax as logits and are valid inputs to NLL loss.
        return log_probabilities

    def loss(self, logits, labels):
        return self._functional.nll_loss(logits, labels)
