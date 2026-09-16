"""Mamba3D adapter using the pinned upstream implementation."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from pathlib import Path

from .base import ClassificationAdapter


DEFAULT_MODEL_CONFIG = {
    "NAME": "Mamba3D",
    "trans_dim": 384,
    "depth": 12,
    "drop_path_rate": 0.2,
    "cls_dim": 3,
    "num_heads": 6,
    "group_size": 32,
    "num_group": 128,
    "encoder_dims": 384,
    "bimamba_type": "v4",
    "center_local_k": 4,
    "ordering": False,
    "label_smooth": 0.0,
    "lr_ratio_cls": 1.0,
    "lr_ratio_lfa": 1.0,
}


class Mamba3DAdapter(ClassificationAdapter):
    name = "mamba3d"

    def __init__(self, repository_root: Path, device, model_config=None):
        import torch.nn.functional as functional

        self._functional = functional
        source_root = repository_root / "submodels" / "mamba3d"
        if not source_root.is_dir():
            raise FileNotFoundError(f"Mamba3D submodule is missing: {source_root}")
        bimamba_root = source_root / "models" / "bimamba_ssm"
        if not bimamba_root.is_dir():
            raise FileNotFoundError(f"Mamba3D BiMamba source is missing: {bimamba_root}")
        sys.path.insert(0, str(source_root))
        # Upstream mamba_simple.py imports ``ops`` as a top-level package after
        # appending an absolute development-machine path. Register precisely
        # that package from the checked-out source. Adding the whole BiMamba
        # directory to sys.path would also shadow the project's own ``utils``.
        ops_root = bimamba_root / "ops"
        if not ops_root.is_dir():
            raise FileNotFoundError(f"Mamba3D selective-scan ops are missing: {ops_root}")
        ops_package = ModuleType("ops")
        ops_package.__path__ = [str(ops_root)]
        ops_package.__package__ = "ops"
        sys.modules["ops"] = ops_package
        try:
            from easydict import EasyDict

            builder = importlib.import_module("tools.builder")
        except ImportError as exc:
            raise ImportError(
                "Unable to import Mamba3D. Install the pinned submodule requirements, "
                "including easydict, timm, causal-conv1d, and mamba-ssm."
            ) from exc
        config = dict(DEFAULT_MODEL_CONFIG)
        if model_config:
            config.update(model_config)
        config["NAME"] = "Mamba3D"
        config["cls_dim"] = 3
        self._model = builder.model_builder(EasyDict(config)).to(device)

    @property
    def model(self):
        return self._model

    def logits(self, points_bnc):
        return self._model(points_bnc)

    def loss(self, logits, labels):
        return self._functional.cross_entropy(logits, labels)
