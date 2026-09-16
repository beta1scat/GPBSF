"""Backend adapters for pinned external model implementations."""

from .base import ClassificationAdapter
from .factory import build_adapter

__all__ = ("ClassificationAdapter", "build_adapter")
