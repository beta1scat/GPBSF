"""Canonical primitive labels shared by every Chapter 4 experiment."""

from __future__ import annotations

CLASS_NAMES = ("cuboid", "frustum", "ellipsoid")
LABEL_BY_NAME = {name: index for index, name in enumerate(CLASS_NAMES)}


def validate_label(label: int, primitive: str) -> None:
    if primitive not in LABEL_BY_NAME:
        raise ValueError(f"Unsupported primitive {primitive!r}; expected one of {CLASS_NAMES}")
    if int(label) != LABEL_BY_NAME[primitive]:
        raise ValueError(
            f"Inconsistent label mapping: {primitive!r} must use "
            f"{LABEL_BY_NAME[primitive]}, got {label}"
        )
