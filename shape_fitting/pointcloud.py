"""
Point cloud utilities, RANSAC fitting models, and synthetic shape generators.
"""

import numpy as np
import open3d as o3d
import scipy
import scipy.optimize
from scipy.spatial import cKDTree

np.set_printoptions(suppress=True)


# =============================================================================
# Point cloud normalization
# =============================================================================


def pc_normalize(pc):
    """Normalize a point cloud to unit sphere.

    Args:
        pc: (N, 3) array of points.

    Returns:
        Tuple of (normalized_points, scale_factor, centroid).
    """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc, m, centroid


# =============================================================================
# RANSAC
# =============================================================================


def random_partition(n, n_data):
    """Return n random indices and the remaining indices.

    Args:
        n: Number of indices to sample.
        n_data: Total number of data points.

    Returns:
        Tuple of (sampled_indices, remaining_indices).
    """
    all_idxs = np.arange(n_data)
    np.random.shuffle(all_idxs)
    return all_idxs[:n], all_idxs[n:]


def ransac(data, model, n, k, t, d, inliers_ratio=0.5, debug=False, return_all=False):
    """Fit model parameters to data using the RANSAC algorithm.

    Reference: http://en.wikipedia.org/w/index.php?title=RANSAC&oldid=116358182

    Args:
        data: (N, D) array of observed data points.
        model: Object implementing ``fit(data)`` and ``get_error(data, params)``.
        n: Minimum number of data points required to fit the model.
        k: Maximum number of iterations.
        t: Threshold for determining when a point fits the model.
        d: Minimum number of inliers to assert a good fit.
        inliers_ratio: Early termination when inlier fraction exceeds this.
        debug: Print debug information per iteration.
        return_all: If True, also return inlier indices.

    Returns:
        bestfit: Best model parameters, or None if fitting failed.
        best_inlier_idxs: (only if return_all) Indices of inliers.
    """
    iterations = 0
    bestfit = None
    besterr = np.inf
    best_inlier_idxs = None
    data_size = data.shape[0]
    inliers_condition = inliers_ratio * data_size

    while iterations < k:
        maybe_idxs, test_idxs = random_partition(n, data_size)
        maybeinliers = data[maybe_idxs, :]
        test_points = data[test_idxs]
        maybemodel = model.fit(maybeinliers)
        test_err = model.get_error(test_points, maybemodel)
        also_idxs = test_idxs[test_err < t]
        alsoinliers = data[also_idxs, :]
        alsoinliers_num = len(alsoinliers)

        if debug:
            print(f"test_err.min() {test_err.min()}")
            print(f"test_err.max() {test_err.max()}")
            print(f"np.mean(test_err) {np.mean(test_err)}")
            print(f"iteration {iterations}: len(alsoinliers) = {alsoinliers_num}")

        if alsoinliers_num > d:
            betterdata = np.concatenate((maybeinliers, alsoinliers))
            bettermodel = model.fit(betterdata)
            better_errs = model.get_error(betterdata, bettermodel)
            thiserr = np.mean(better_errs)
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
                best_inlier_idxs = np.concatenate((maybe_idxs, also_idxs))

        iterations += 1
        if alsoinliers_num + n > inliers_condition:
            break

    if bestfit is None:
        print("Fit failed in RANSAC")

    if return_all:
        return bestfit, best_inlier_idxs
    else:
        return bestfit


# =============================================================================
# Least-squares model classes for RANSAC
# =============================================================================


def fit_circle_kasa(points):
    """Direct closed-form algebraic circle fitting in 2D (Kåsa method).

    Solves (x - xc)^2 + (y - yc)^2 = r^2 via linear least squares:
      2*xc*x + 2*yc*y + (r^2 - xc^2 - yc^2) = x^2 + y^2

    Args:
        points: (N, 2) array of 2D points.

    Returns:
        Tuple of (center, radius) where center is (2,) np.ndarray, or None.
    """
    pts = np.asarray(points)
    if len(pts) < 4:
        return None
    x = pts[:, 0]
    y = pts[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x**2 + y**2
    try:
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        xc = float(sol[0] / 2.0)
        yc = float(sol[1] / 2.0)
        rad_sq = float(sol[2] + xc**2 + yc**2)
        if rad_sq <= 0 or np.isnan(rad_sq):
            return None
        r = float(np.sqrt(rad_sq))
        if np.isnan(r) or np.isinf(r) or r <= 0:
            return None
        return np.array([xc, yc]), r
    except Exception:
        return None


class EllipsoidLeastSquaresModel:
    """3D ellipsoid fitting model. Data shape: (N, 3).

    Analytical algebraic least-squares model with closed-form eigendecomposition
    to recover ellipsoid center, semi-axes, and rotation matrix without SymPy.
    """

    @staticmethod
    def get_design_matrix(pts: np.ndarray) -> np.ndarray:
        """Vectorized construction of quadratic design matrix (N, 10)."""
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        return np.column_stack([
            x**2, y**2, z**2,
            x * y, x * z, y * z,
            x, y, z,
            np.ones_like(x),
        ])

    def fit(self, data):
        data_size = len(data)
        if data_size > 100:
            data = data[np.random.choice(data_size, 100, replace=False)]
        A = self.get_design_matrix(data)
        try:
            _, _, V = scipy.linalg.svd(A, full_matrices=False)
            model = V[-1]
            if self.get_ellipsoid_params(model) is None:
                return None
            return model
        except Exception:
            return None

    def get_error(self, data, model):
        if model is None:
            return np.full(data.shape[0], np.inf)
        A = self.get_design_matrix(data)
        return np.abs(A @ model)

    def get_ellipsoid_params(self, model):
        """Extract ellipsoid parameters from implicit equation coefficients.

        Returns:
            Tuple (x0, y0, z0, a, b, c, R) where (x0,y0,z0) is the center,
            (a,b,c) are the semi-axes, and R is the 3x3 rotation matrix.
            Returns None if the model does not represent a valid ellipsoid.
        """
        if model is None:
            return None
        a, b, c, d, e, f, g, h, i, j = model

        # 3x3 Quadric coefficient matrix and 3x1 linear vector
        A0 = np.array([
            [a, d / 2.0, e / 2.0],
            [d / 2.0, b, f / 2.0],
            [e / 2.0, f / 2.0, c],
        ], dtype=np.float64)
        b0 = np.array([g, h, i], dtype=np.float64)

        # 1. Analytic center: x0 = -0.5 * A0^{-1} * b0
        try:
            center = -0.5 * np.linalg.solve(A0, b0)
        except np.linalg.LinAlgError:
            return None

        # 2. Translated constant: k = j + 0.5 * b0^T * center
        k = j + 0.5 * np.dot(b0, center)
        if np.isclose(k, 0.0):
            return None

        # 3. Normalized center quadric: x'^T * (A0 / -k) * x' = 1
        M = A0 / (-k)

        # 4. Eigendecomposition of real symmetric matrix M
        try:
            eigenvals, R = np.linalg.eigh(M)
        except np.linalg.LinAlgError:
            return None

        # Ellipsoid requires strictly positive eigenvalues
        if np.any(eigenvals <= 1e-4):
            return None

        # 5. Semi-axes lengths
        semi_axes = 1.0 / np.sqrt(eigenvals)

        # Physics/geometry sanity check in normalized coordinates:
        # Normalized points are bounded within [-1, 1], so semi-axes > 3.0 or aspect ratio > 5.0
        # indicates degenerate flat quadric or thin needle that ruins physical grasp estimation.
        if np.any(~np.isfinite(semi_axes)) or np.any(semi_axes <= 0.02) or np.any(semi_axes > 3.0):
            return None
        if (np.max(semi_axes) / np.min(semi_axes)) > 5.0:
            return None
        if np.linalg.norm(center) > 1.5:
            return None

        # Ensure right-handed coordinate frame
        if np.linalg.det(R) < 0:
            R[:, 0] = -R[:, 0]

        return (
            float(center[0]),
            float(center[1]),
            float(center[2]),
            float(semi_axes[0]),
            float(semi_axes[1]),
            float(semi_axes[2]),
            R,
        )


class NormalLeastSquaresModel:
    """Fits a cone-like surface normal distribution. Data shape: (N, 3) unit normals.

    Finds the axis vector that minimizes pairwise cosine angle differences,
    i.e. all normals make approximately the same angle with the axis.
    """

    def fit(self, data):
        init_guess = np.array([0.57735027, 0.57735027, 0.57735027])
        data_size = len(data)
        if data_size > 100:
            data = data[np.random.choice(data_size, 100, replace=False)]
        # Minimize pairwise cosine angle differences
        result = scipy.optimize.minimize(self._angle_diff, init_guess, args=(data,))
        vector = result.x / np.linalg.norm(result.x)
        angle = np.mean(np.arccos(np.dot(data, vector)))
        return vector, angle

    def get_error(self, data, model):
        vector, angle = model
        angles = np.arccos(np.dot(data, vector))
        return np.abs(angles - angle)

    @staticmethod
    def _angle_diff(X, normals):
        X = X / np.linalg.norm(X)
        size = len(normals)
        cos_theta = np.dot(normals, X)
        diff_matrix = cos_theta[:, np.newaxis] - cos_theta[np.newaxis, :]
        return np.sum(diff_matrix[np.triu_indices(size, k=1)] ** 2)


class ConeAxisLeastSquaresModel:
    """Fits a cone axis from surface normals by minimizing angle variance.

    Data shape: (N, 3) unit normals.
    """

    def fit(self, data):
        init_guess = np.array(
            [0.57735027, 0.57735027, 0.57735027]
        )  # [1,1,1] normalized
        # Minimize variance of angles between normals and candidate axis
        result = scipy.optimize.minimize(
            self._angle_diff_variance, init_guess, args=(data,)
        )
        vector = result.x / np.linalg.norm(result.x)
        angle = np.mean(np.arccos(np.dot(data, vector)))
        return vector, angle

    def get_error(self, data, model):
        vector, angle = model
        angles = np.arccos(np.clip(np.dot(data, vector), -1, 1))
        return np.abs(angles - angle)

    @staticmethod
    def _angle_diff_variance(X, normals):
        X = X / np.linalg.norm(X)
        angles = np.arccos(np.clip(np.dot(normals, X), -1, 1))
        return np.var(angles)


# =============================================================================
# Synthetic shape point cloud generators
# =============================================================================


def generate_cube_points(
    size=(10, 10, 10), delta=0.0, points_density=1, total_points=10000
):
    """Generate surface points on a cuboid.

    Args:
        size: (x, y, z) dimensions of the cuboid.
        delta: Random noise magnitude added to each coordinate.
        points_density: Points per unit area. If 0, use total_points instead.
        total_points: Total number of points (used when points_density == 0).

    Returns:
        List of [x, y, z] points on the cuboid surface.
    """
    assert min(size) > 0, "cube(x, y, z) should > 0"
    assert points_density >= 0, "number of points density should >= 0"
    assert total_points > 0, "number of points should > 0"

    half_size = np.array(size) / 2
    points = []

    area1 = size[0] * size[1]  # top/bottom
    area2 = size[1] * size[2]  # left/right (yz)
    area3 = size[0] * size[2]  # front/back (xz)
    total_area = 2 * (area1 + area2 + area3)

    def _noise():
        return np.random.uniform(-1, 1) * delta

    # Top and bottom surfaces
    if points_density != 0:
        num_points_tb = int(size[0] * size[1] * points_density)
    else:
        num_points_tb = int(total_points * (area1 / total_area))
    for _ in range(num_points_tb):
        x = np.random.uniform(-1, 1) * half_size[0]
        y = np.random.uniform(-1, 1) * half_size[1]
        points.append([x + _noise(), y + _noise(), half_size[2] + _noise()])
    for _ in range(num_points_tb):
        x = np.random.uniform(-1, 1) * half_size[0]
        y = np.random.uniform(-1, 1) * half_size[1]
        points.append([x + _noise(), y + _noise(), -half_size[2] + _noise()])

    # Left/right surfaces (yz)
    if points_density != 0:
        num_point_yz = int(size[1] * size[2] * points_density)
    else:
        num_point_yz = int(total_points * (area2 / total_area))
    for _ in range(num_point_yz):
        y = np.random.uniform(-1, 1) * half_size[1]
        z = np.random.uniform(-1, 1) * half_size[2]
        points.append([half_size[0] + _noise(), y + _noise(), z + _noise()])
    for _ in range(num_point_yz):
        y = np.random.uniform(-1, 1) * half_size[1]
        z = np.random.uniform(-1, 1) * half_size[2]
        points.append([-half_size[0] + _noise(), y + _noise(), z + _noise()])

    # Front/back surfaces (xz)
    if points_density != 0:
        num_point_xz = int(size[0] * size[2] * points_density)
    else:
        num_point_xz = int(total_points * (area3 / total_area))
    for _ in range(num_point_xz):
        x = np.random.uniform(-1, 1) * half_size[0]
        z = np.random.uniform(-1, 1) * half_size[2]
        points.append([x + _noise(), half_size[1] + _noise(), z + _noise()])
    for _ in range(num_point_xz):
        x = np.random.uniform(-1, 1) * half_size[0]
        z = np.random.uniform(-1, 1) * half_size[2]
        points.append([x + _noise(), -half_size[1] + _noise(), z + _noise()])

    return points


def generate_cone_points(
    r_bottom=10,
    r_top_ratio=0.5,
    height=20,
    delta=0.0,
    points_density=1,
    total_points=10000,
):
    """Generate surface points on a truncated cone (frustum).

    Args:
        r_bottom: Bottom radius.
        r_top_ratio: Ratio of top radius to bottom radius.
        height: Height of the frustum.
        delta: Random noise magnitude.
        points_density: Points per unit area. If 0, use total_points instead.
        total_points: Total number of points (used when points_density == 0).

    Returns:
        (M, 3) numpy array of points on the frustum surface.
    """
    assert r_bottom > 0, "cone r_bottom should > 0"
    assert height > 0, "cone height should > 0"
    assert points_density >= 0, "number of points density should >= 0"
    assert total_points > 0, "number of points should > 0"

    r_top = r_bottom * r_top_ratio
    half_height = height / 2
    points = []

    area_top = np.pi * r_top * r_top
    area_bottom = np.pi * r_bottom * r_bottom
    slant = np.sqrt((r_bottom - r_top) ** 2 + height**2)
    area_lateral = np.pi * (r_top + r_bottom) * slant
    total_area = area_top + area_bottom + area_lateral

    def _noise():
        return np.random.uniform(-1, 1) * delta

    # Top cap
    if points_density != 0:
        num_top = int(np.pi * r_top * r_top * points_density)
    else:
        num_top = int(total_points * (area_top / total_area))
    for _ in range(num_top):
        r = np.random.uniform() * r_top
        phi = 2 * np.pi * np.random.rand()
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        points.append([x + _noise(), y + _noise(), half_height + _noise()])

    # Bottom cap
    if points_density != 0:
        num_bottom = int(np.pi * r_bottom * r_bottom * points_density)
    else:
        num_bottom = int(total_points * (area_bottom / total_area))
    for _ in range(num_bottom):
        r = np.random.uniform() * r_bottom
        phi = 2 * np.pi * np.random.rand()
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        points.append([x + _noise(), y + _noise(), -half_height + _noise()])

    # Lateral surface
    if points_density != 0:
        num_lateral = int(np.pi * (r_top + r_bottom) * slant * points_density)
    else:
        num_lateral = int(total_points * (area_lateral / total_area))
    for _ in range(num_lateral):
        ratio = np.random.uniform(-1, 1)
        ratio_0_1 = (ratio + 1) / 2
        z = ratio * half_height
        r = ratio_0_1 * (r_top - r_bottom) + r_bottom
        phi = 2 * np.pi * np.random.rand()
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        points.append([x + _noise(), y + _noise(), z + _noise()])

    return np.array(points)


def generate_ellipsoid_points(a=10, b=10, c=10, total_points=10000):
    """Generate surface points on an ellipsoid.

    Parametric equations:
        x = a * sin(theta) * cos(phi)
        y = b * sin(theta) * sin(phi)
        z = c * cos(theta)

    Args:
        a, b, c: Semi-axis lengths.
        total_points: Number of points to generate.

    Returns:
        (total_points, 3) numpy array of surface points.
    """
    theta = np.pi * np.random.rand(total_points)
    phi = 2 * np.pi * np.random.rand(total_points)
    x = a * np.sin(theta) * np.cos(phi)
    y = b * np.sin(theta) * np.sin(phi)
    z = c * np.cos(theta)
    return np.column_stack((x, y, z))


# =============================================================================
# Geometric utilities
# =============================================================================



def depth_to_pointcloud(depth_image, fx, fy, cx, cy):
    """Convert a depth image to a 3D point cloud using pinhole camera intrinsics.

    Args:
        depth_image: (H, W) depth map (values in camera units, e.g. mm).
        fx, fy: Focal lengths in pixels.
        cx, cy: Principal point in pixels.

    Returns:
        (M, 3) array of 3D points (zero-depth points removed).
    """
    height, width = depth_image.shape
    u, v = np.meshgrid(np.arange(1, width + 1), np.arange(1, height + 1))
    z = depth_image
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    pointcloud = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=-1)
    nonzero_indices = np.all(pointcloud != [0, 0, 0], axis=1)
    return pointcloud[nonzero_indices]


def compute_trimmed_distance(
    pcd,
    fit_pcd,
    inlier_ratio: float = 0.90,
) -> float:
    """Compute robust trimmed point cloud distance (focusing on the closest inlier_ratio fraction).

    Filters out extreme boundary noise, flying pixels, and contact-surface artifacts.
    Evaluates how closely the fitted primitive surface matches the true visible workpiece surface.

    Args:
        pcd: Observed point cloud (o3d.geometry.PointCloud or (N, 3) ndarray).
        fit_pcd: Synthetic point cloud from fitted primitive (o3d.geometry.PointCloud or (M, 3) ndarray).
        inlier_ratio: Ratio of closest points to retain (default 0.90).

    Returns:
        Mean distance of the closest inlier_ratio fraction of points.
    """
    if pcd is None or fit_pcd is None:
        return float("inf")
    pts1 = np.asarray(pcd.points if hasattr(pcd, "points") else pcd, dtype=np.float64)
    pts2 = np.asarray(fit_pcd.points if hasattr(fit_pcd, "points") else fit_pcd, dtype=np.float64)
    if pts1.size == 0 or pts2.size == 0:
        return float("inf")

    tree = cKDTree(pts2)
    d1, _ = tree.query(pts1, k=1)
    d1_sorted = np.sort(d1)
    k = max(1, int(round(inlier_ratio * len(d1_sorted))))
    return float(np.mean(d1_sorted[:k]))


