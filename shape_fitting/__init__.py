"""Shape fitting sub-package.

Provides geometric primitive fitting (cuboid, truncated cone, ellipsoid)
algorithms for 3D point clouds.
"""

from .fitting import FittingByBGS, _oriented_bounding_box

__all__ = [
    "FittingByBGS",
    "_oriented_bounding_box",
]
