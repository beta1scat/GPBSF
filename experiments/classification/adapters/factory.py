"""Create one backend per process to isolate upstream import namespaces."""

from __future__ import annotations

from pathlib import Path


def build_adapter(model_name: str, repository_root, device, model_config=None):
    root = Path(repository_root).resolve()
    if model_name == "pointnet2":
        from .pointnet2 import PointNet2Adapter

        return PointNet2Adapter(root, device)
    if model_name == "mamba3d":
        from .mamba3d import Mamba3DAdapter

        return Mamba3DAdapter(root, device, model_config=model_config)
    raise ValueError("model must be one of: pointnet2, mamba3d")
