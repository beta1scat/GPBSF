"""Shape fitting sub-package.

Provides geometric primitive fitting (cuboid, truncated cone, ellipsoid)
algorithms for 3D point clouds.
"""

from .pointcloud import compute_trimmed_distance
from .fitting import (
    FittingByBGS,
    _oriented_bounding_box,
    align_vector_to_z,
    fit_cuboid_obb2,
    fit_cuboid_obb,
    fit_frustum_cone_adaptive,
    fit_frustum_cone_pca,
    fit_frustum_cone_normal,
    fit_frustum_cone_obb,
    fit_ellipsoid,
    compute_cone_residual,
)

__all__ = [
    "FittingByBGS",
    "_oriented_bounding_box",
    "align_vector_to_z",
    "fit_cuboid_obb2",
    "fit_cuboid_obb",
    "fit_frustum_cone_adaptive",
    "fit_frustum_cone_pca",
    "fit_frustum_cone_normal",
    "fit_frustum_cone_obb",
    "fit_ellipsoid",
    "compute_cone_residual",
    "compute_trimmed_distance",
]

