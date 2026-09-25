"""Shape fitting sub-package.

Provides geometric primitive fitting (cuboid, truncated cone, ellipsoid)
algorithms for 3D point clouds.
"""

from .pointcloud import compute_trimmed_distance
from .fitting import (
    FittingByBGS,
    _oriented_bounding_box,
    fit_frustum_cone_adaptive,
    compute_cone_residual,
)

__all__ = [
    "FittingByBGS",
    "_oriented_bounding_box",
    "fit_frustum_cone_adaptive",
    "compute_cone_residual",
    "compute_trimmed_distance",
]
