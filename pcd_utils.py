import scipy
import numpy as np
import sympy as sp
import open3d as o3d
from spatialmath import SE3, SO3

np.set_printoptions(suppress=True)

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc, m, centroid

def generate_cube_points(size=(10, 10, 10), delta=0.0, points_density=0, total_points=10000):
    assert min(size) > 0, "cube(x, y, z) should > 0"
    assert points_density >= 0, "number of points density should >= 0"
    assert total_points > 0, "number of points should > 0"
    half_size = np.array(size) / 2
    points = []
    # top and bottom, 2 surfaces
    area1 = size[0] * size[1]
    area2 = size[1] * size[2]
    area3 = size[0] * size[2]
    total_area = 2 * (area1 + area2 + area3)
    num_points_tb = 0
    if points_density != 0:
        num_points_tb = int(size[0] * size[1] * points_density)
    else:
        num_points_tb = int(total_points * (area1 / total_area))
    for _ in range(num_points_tb):
        x = np.random.uniform(-1, 1) * half_size[0]
        y = np.random.uniform(-1, 1) * half_size[1]
        z = half_size[2]
        points.append([x + np.random.uniform(-1, 1) * delta, y + np.random.uniform(-1, 1) * delta, z + np.random.uniform(-1, 1) * delta])
    for _ in range(num_points_tb):
        x = np.random.uniform(-1, 1) * half_size[0]
        y = np.random.uniform(-1, 1) * half_size[1]
        z = -half_size[2]
        points.append([x + np.random.uniform(-1, 1) * delta, y + np.random.uniform(-1, 1) * delta, z + np.random.uniform(-1, 1) * delta])
    # vertical 4 surfaces
    num_point_yz = 0
    if points_density != 0:
        num_point_yz = int(size[1] * size[2] * points_density)
    else:
        num_point_yz = int(total_points * (area2 / total_area))
    for _ in range(num_point_yz):
        x = half_size[0]
        y = np.random.uniform(-1, 1) * half_size[1]
        z = np.random.uniform(-1, 1) * half_size[2]
        points.append([x + np.random.uniform(-1, 1) * delta, y + np.random.uniform(-1, 1) * delta, z + np.random.uniform(-1, 1) * delta])
    for _ in range(num_point_yz):
        x = -half_size[0]
        y = np.random.uniform(-1, 1) * half_size[1]
        z = np.random.uniform(-1, 1) * half_size[2]
        points.append([x + np.random.uniform(-1, 1) * delta, y + np.random.uniform(-1, 1) * delta, z + np.random.uniform(-1, 1) * delta])
    num_point_xz = 0
    if points_density != 0:
        num_point_xz = int(size[0] * size[2] * points_density)
    else:
        num_point_xz = int(total_points * (area3 / total_area))
    for _ in range(num_point_xz):
        y = half_size[1]
        x = np.random.uniform(-1, 1) * half_size[0]
        z = np.random.uniform(-1, 1) * half_size[2]
        points.append([x + np.random.uniform(-1, 1) * delta, y + np.random.uniform(-1, 1) * delta, z + np.random.uniform(-1, 1) * delta])
    for _ in range(num_point_xz):
        y = -half_size[1]
        x = np.random.uniform(-1, 1) * half_size[0]
        z = np.random.uniform(-1, 1) * half_size[2]
        points.append([x + np.random.uniform(-1, 1) * delta, y + np.random.uniform(-1, 1) * delta, z + np.random.uniform(-1, 1) * delta])
    return points

def generate_cone_points(r_bottom=10, r_top_ratio=0.5, height=20, delta=0.0, points_density=0, total_points=10000):
    assert r_bottom > 0, "cone r_top should > 0"
    assert height > 0, "cone height should > 0"
    assert points_density >= 0, "number of points density should >= 0"
    assert total_points > 0, "number of points should > 0"
    r_top = r_bottom * r_top_ratio
    half_height = height / 2
    points = []
    area1 = np.pi * r_top * r_top
    area2 = np.pi * r_bottom * r_bottom
    area3 = np.pi * (r_top + r_bottom) * np.sqrt((r_bottom - r_top)**2 + height**2)
    total_area = area1 + area2 + area3
    l = np.sqrt((r_bottom - r_top)**2 + height**2)
    # top and bottom surface
    num_points_t = 0
    if points_density != 0:
        num_points_t = int(np.pi * r_top * r_top * points_density)
    else:
        num_points_t = int(total_points * (area1 / total_area))
    for _ in range(num_points_t):
        z = half_height
        r = np.random.uniform() * r_top
        phi = 2 * np.pi * np.random.rand()
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        points.append([x + np.random.uniform(-1, 1) * delta, y + np.random.uniform(-1, 1) * delta, z + np.random.uniform(-1, 1) * delta])
    num_points_b = 0
    if points_density != 0:
        num_points_b = int(np.pi * r_bottom * r_bottom * points_density)
    else:
        num_points_b = int(total_points * (area2 / total_area))
    for _ in range(num_points_b):
        z = -half_height
        r = np.random.uniform() * r_bottom
        phi = 2 * np.pi * np.random.rand()
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        points.append([x + np.random.uniform(-1, 1) * delta, y + np.random.uniform(-1, 1) * delta, z + np.random.uniform(-1, 1) * delta])
    # outer surface
    num_points_l = 0
    if points_density != 0:
        num_points_l = int(np.pi * (r_top + r_bottom) * l * points_density)
    else:
        num_points_l = int(total_points * (area3 / total_area))
    for _ in range(num_points_l):
        ratio = np.random.uniform(-1, 1)
        ratio_0_1 = (ratio + 1) / 2
        z = ratio * half_height
        r = ratio_0_1 * (r_top - r_bottom) + r_bottom
        phi = 2 * np.pi * np.random.rand()
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        points.append([x + np.random.uniform(-1, 1) * delta, y + np.random.uniform(-1, 1) * delta, z + np.random.uniform(-1, 1) * delta])
    return np.array(points)

def generate_ellipsoid_points(a=10, b=10, c=10, total_points=10000):
    # 生成椭球体的点云
    # 椭球的参数方程：x = a*sin(t)*cos(p), y = b*sin(t)*sin(p), z = c*cos(t)
    # 椭球体面积估算方式：S = 4π(abc)^(2/3)
    theta = np.pi * np.random.rand(total_points)
    phi = 2 * np.pi * np.random.rand(total_points)
    x = a * np.sin(theta) * np.cos(phi)
    y = b * np.sin(theta) * np.sin(phi)
    z = c * np.cos(theta)
    points = np.column_stack((x, y, z))
    return points

def ransac(data, model, n, k, t, d, inliers_ratio=0.5, debug=False, return_all=False):
    """
    fit model parameters to data using the RANSAC algorithm

    This implementation written from pseudocode found at
    http://en.wikipedia.org/w/index.php?title=RANSAC&oldid=116358182

    {{{
    Given:
        data - a set of observed data points, shape is (n, 2/3)
        model - a model that can be fitted to data points
                model must implement two mehod:
                - def fit(data, ...): given data return parameters of model
                - def get_error(data, model): given data and model prameters return error between data and model
        n - the minimum number of data values required to fit the model
        k - the maximum number of iterations allowed in the algorithm
        t - a threshold value for determining when a data point fits a model
        d - the number of close data values required to assert that a model fits well to data
        inliers_ratio - the number of inliers gt than the raion will return
    Return:
        bestfit - model parameters which best fit the data (or nil if no good model is found)
        inliers_idxs(option) - the index of inliers of the data
    Pesudo code:
        iterations = 0
        bestfit = nil
        besterr = something really large
        while iterations < k {
            maybeinliers = n randomly selected values from data
            maybemodel = model parameters fitted to maybeinliers
            alsoinliers = empty set
            for every point in data not in maybeinliers {
                if point fits maybemodel with an error smaller than t add point to alsoinliers
            }
            if the number of elements in alsoinliers is > d {
                % this implies that we may have found a good model
                % now test how good it is
                bettermodel = model parameters fitted to all points in maybeinliers and alsoinliers
                thiserr = a measure of how well model fits these points
                if thiserr < besterr {
                    bestfit = bettermodel
                    besterr = thiserr
                }
            }
            increment iterations
        }
        return bestfit
    }}}
    """
    iterations = 0
    bestfit = None
    besterr = np.inf
    best_inlier_idxs = None
    data_size = data.shape[0]
    inliers_condition = inliers_ratio * data_size
    while iterations < k:
        maybe_idxs, test_idxs = random_partition(n, data_size)
        maybeinliers = data[maybe_idxs,:]
        test_points = data[test_idxs]
        maybemodel = model.fit(maybeinliers)
        test_err = model.get_error(test_points, maybemodel)
        also_idxs = test_idxs[test_err < t] # select indices of rows with accepted points
        alsoinliers = data[also_idxs,:]
        alsoinliers_num = len(alsoinliers)
        if debug:
            print('test_err.min()',test_err.min())
            print('test_err.max()',test_err.max())
            print('np.mean(test_err)',np.mean(test_err))
            print('iteration %d:len(alsoinliers) = %d'%(
                iterations,len(alsoinliers)))
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

def random_partition(n, n_data):
    """return n random rows of data (and also the other len(data)-n rows)"""
    all_idxs = np.arange( n_data )
    np.random.shuffle(all_idxs)
    return all_idxs[:n], all_idxs[n:]

class CircleLeastSquaresModel:
    """
        data shape is (n, 2)
    """
    def fit(self, data):
        x = data[:, 0]
        y = data[:, 1]
        size = data.shape[0]
        diff_x = 2 * (x[:, np.newaxis] - x[np.newaxis, :])[np.triu_indices(size, k=1)]
        diff_y = 2 * (y[:, np.newaxis] - y[np.newaxis, :])[np.triu_indices(size, k=1)]
        xy_2 = x**2 + y**2
        B = (xy_2[:, np.newaxis] - xy_2[np.newaxis, :])[np.triu_indices(size, k=1)]
        A = np.hstack((diff_x[:, np.newaxis], diff_y[:, np.newaxis]))
        center = np.linalg.lstsq(A, B, rcond=None)[0]
        r = np.mean(np.linalg.norm(np.hstack((x[:, np.newaxis], y[:, np.newaxis])) - center, axis=1))
        return (*center, r)
    def get_error(self, data, model):
        x = data[:, 0]
        y = data[:, 1]
        x0, y0, r = model
        return np.abs((x - x0)**2 + (y - y0)**2 - r**2)

class EllipseLeastSquaresModel:
    """
        data shape is (n, 2)
    """
    def fit(self, data):
        A = []
        for x in data:
            A.append([x[0]**2, x[0]*x[1], x[1]**2, x[0], x[1], 1])
        B = np.zeros(data.shape[0])
        _, _, V = np.linalg.svd(A)
        model = V[-1]
        if self.get_ellipse_params(model) is None:
            return None
        return model
    def get_error(self, data, model):
        if model is None:
            return np.array([np.inf]*data.shape[0])
        A = []
        for x in data:
            A.append([x[0]**2, x[0]*x[1], x[1]**2, x[0], x[1], 1])
        return np.abs((np.array(A) @ np.array(model)[:, np.newaxis])[:,0])
    def get_ellipse_params(self, model):
        a, b, c, d, e, f = model
        b2m4ac = b**2 - 4*a*c
        xc = (2*c*d - b*e) / b2m4ac
        yc = (2*a*e - b*d) / b2m4ac
        center = [xc, yc]
        t = atan(b / (c - a)) / (-2)
        tmpEq = a*xc**2 + b*xc*yc + c*yc**2 - f
        a_square = tmpEq / (a*cos(t)**2 + b*sin(t)*cos(t) + c*sin(t)**2)
        b_square = tmpEq / (c*cos(t)**2 - b*sin(t)*cos(t) + a*sin(t)**2)
        if (a_square < 0 or b_square < 0):
            return None
        a = sqrt(a_square)
        b = sqrt(b_square)
        return (xc, yc, a, b, t)

class EllipsoidLeastSquaresModel:
    """
        data shape is (n, 3)
    """
    def fit(self, data):
        A = []
        data_size = len(data)
        if data_size > 100:
            data = data[np.random.choice(data_size, 100, replace=False)]
        for x in data:
            A.append([x[0]**2, x[1]**2, x[2]**2, x[0] * x[1], x[0] * x[2], x[1] * x[2], x[0], x[1], x[2], 1])
        B = np.zeros(data.shape[0])
        U, S, V = scipy.linalg.svd(A)
        model = V[-1]
        if self.get_ellipsoid_params(model) is None:
            return None
        return model
    def get_error(self, data, model):
        if model is None:
            return np.array([np.inf]*data.shape[0])
        A = []
        for x in data:
            A.append([x[0]**2, x[1]**2, x[2]**2, x[0] * x[1], x[0] * x[2], x[1] * x[2], x[0], x[1], x[2], 1])
        return np.abs((np.array(A) @ np.array(model)[:, np.newaxis])[:,0])
    def get_ellipsoid_params(self, model):
        if model is None:
            return None
        a, b, c, d, e, f, g, h, i, j = model
        x, y, z = sp.symbols('x y z')
        expr = a*x**2 + b*y**2 + c*z**2 + d*x*y + e*x*z + f*y*z + g*x + h*y + i*z + j
        A0 = sp.Matrix([[a, d/2, e/2],
                        [d/2, b, f/2],
                        [e/2, f/2, c]])
        eigenvalues = sorted(A0.eigenvals())
        eigenvectors = sorted(A0.eigenvects(), key=lambda x: abs(x[0]))
        R = eigenvectors[2][2][0].col_insert(0, eigenvectors[1][2][0]).col_insert(0, eigenvectors[0][2][0])
        if np.linalg.det(np.asarray(R, dtype=np.float64)) + 1 < 1e-3:
            return None
        xp, yp, zp = sp.symbols('xp yp zp')
        xyz = sp.Matrix([[xp, yp, zp]]) * R.T
        trans = expr.subs({x: xyz[0], y: xyz[1], z: xyz[2]})
        if np.any(np.array(eigenvalues) < 0):
            trans = trans * (-1)
        var = (xp**2, yp**2, zp**2, xp*yp, xp*zp, yp*zp, xp, yp, zp, 1)
        expr_trans = trans.expand()
        coefficients_dict = expr_trans.as_coefficients_dict(*var)
        for term, coefficient in coefficients_dict.items():
            if abs(coefficient) < 1e-10:
                expr_trans = expr_trans.subs(term, 0)
        if coefficients_dict[1] > 0:
            return None
        coefficients_dict = expr_trans.as_coefficients_dict(*var)
        coeff_xp2 = float(coefficients_dict[xp**2])
        coeff_xp = float(coefficients_dict[xp])
        coeff_yp2 = float(coefficients_dict[yp**2])
        coeff_yp = float(coefficients_dict[yp])
        coeff_zp2 = float(coefficients_dict[zp**2])
        coeff_zp = float(coefficients_dict[zp])

        if coeff_zp2 < 0 or coeff_yp2 < 0 or coeff_xp2 < 0:
            return None

        x0 = -0.5 * coeff_xp / coeff_xp2
        y0 = -0.5 * coeff_yp / coeff_yp2
        z0 = -0.5 * coeff_zp / coeff_zp2
        x0t, y0t, z0t = (sp.Matrix([x0, y0, z0]).T * R.T).tolist()[0]

        Cx = coeff_xp2 * x0**2
        Cy = coeff_yp2 * y0**2
        Cz = coeff_zp2 * z0**2

        constJ = float(abs(coefficients_dict[1] - Cx - Cy - Cz))

        a = np.sqrt(constJ / coeff_xp2)
        b = np.sqrt(constJ / coeff_yp2)
        c = np.sqrt(constJ / coeff_zp2)

        return (x0t, y0t, z0t, a, b, c, R)

class NormalLeastSquaresModel:
    """
        data shape is (n, 3)
    """
    def fit(self, data):
        init_guess = np.array([0.57735027, 0.57735027, 0.57735027])
        data_size = len(data)
        if data_size > 100:
            data = data[np.random.choice(data_size, 100, replace=False)]
        result = scipy.optimize.minimize(self.angle_diff, init_guess, args=(data,)) # 优化搜索使夹角余弦差最小
        vector = result.x / np.linalg.norm(result.x)
        angle = np.mean(np.arccos(np.dot(data, vector)))
        return vector, angle
    def get_error(self, data, model):
        vector, angle = model
        angles = np.arccos(np.dot(data, vector))
        return np.abs(angles - angle)
    def angle_diff(self, X, normals):
        X = X / np.linalg.norm(X)
        size = len(normals)
        cos_theta_list_np = np.dot(normals, X)
        diff_matrix = cos_theta_list_np[:, np.newaxis] - cos_theta_list_np[np.newaxis, :]
        return np.sum(diff_matrix[np.triu_indices(size, k=1)] ** 2)

class ConeAxisLeastSquaresModel:
    """
        data shape is (n, 3)
    """
    def fit(self, data):
        data_size = len(data)
        init_guess = np.array([0.57735027, 0.57735027, 0.57735027]) # [1,1,1] normalized
        result = scipy.optimize.minimize(self.angle_diff_variance, init_guess, args=(data)) # 优化搜索使夹角余弦方差最小
        vector = result.x / np.linalg.norm(result.x)
        angle = np.mean(np.arccos(np.dot(data, vector)))
        return vector, angle
    def get_error(self, data, model):
        vector, angle = model
        angles = np.arccos(np.clip(np.dot(data, vector), -1, 1))
        return np.abs(angles - angle)
    def angle_diff_variance(self, X, normals):
        X = X / np.linalg.norm(X)
        angles = np.arccos(np.clip(np.dot(normals, X), -1, 1))
        # angles = angles[angles > 0.2]
        return np.var(angles)

def checkPickPoseFor2FingerGripper(pcd, poses, fingerRange, t = 10):
    """
    Check pick poses for 2-finger gripper

    Given:
        pcd - point cloud in open3d
        fingerRange - [w, h, d] contact range in x, y, z axis at pick pose coordinate
        poses - SE3 poses to check
        t - threshold of points in finger contact OBB
    Return:
        fileteredPoses - poses after filter
    """
    pts = np.asarray(pcd)
    filteredPoses = []
    for pose in poses:
        fingerObb = o3d.geometry.OrientedBoundingBox(pose.t, pose.R, np.array(fingerRange))
        idxs = fingerObb.get_point_indices_within_bounding_box(pcd.points)
        if len(idxs) > t:
            filteredPoses.append(pose)
    return filteredPoses

def filter_pose_by_axis_diff(poses, axis=2, ref_axis=[0,0,1], t=np.pi/2, sorted=False):
    def angle_between_axes(pose_axis, ref_axis):
        # 计算两个轴之间的夹角
        pose_axis = np.array(pose_axis)
        ref_axis = np.array(ref_axis)
        cos_angle = np.dot(pose_axis, ref_axis) / (np.linalg.norm(pose_axis) * np.linalg.norm(ref_axis))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # 避免数值误差导致的cos值超出[-1, 1]范围
        angle = np.arccos(cos_angle)
        return angle
    valid_poses = []
    angles = []
    for pose in poses:
        # 获取位姿矩阵的指定轴
        if isinstance(pose, SE3):
            pose_axis = pose.A[:3, axis]
        else:
            pose_axis = pose[:3, axis]
        angle = angle_between_axes(pose_axis, ref_axis)
        if angle <= t:
            valid_poses.append(pose)
            angles.append(angle)
    if sorted:
        # 按照角度进行排序
        sorted_indices = np.argsort(angles)
        valid_poses = [valid_poses[i] for i in sorted_indices]
    return valid_poses

def sort_pose_by_rot_diff(poses, ref_pose):
    def rotation_difference(pose1, pose2):
        # 计算两个位姿的旋转矩阵差异
        return pose1.angdist(pose2, metric=6)
    # 计算每个位姿与参考位姿的旋转矩阵差异
    differences = [rotation_difference(pose, ref_pose) for pose in poses]
    # 根据差异对位姿进行排序
    sorted_indices = np.argsort(differences)
    sorted_poses = [poses[i] for i in sorted_indices]
    return sorted_poses


def filter_pose_by_bin_side(poses, bin_size, ignore_size, bin_pose, threshold):
    # bin_pose is center poses
    half_size = np.asarray(bin_size) / 2
    half_ignore_size = np.asarray(ignore_size) / 2
    binXp = bin_pose.n
    binXn = -1*bin_pose.n
    binYp = bin_pose.o
    binYn = -1*bin_pose.o
    pose_filtered = []
    for pose in poses:
        pose_in_bin = bin_pose.inv() * pose
        if abs(pose_in_bin.t[0]) < half_ignore_size[0] and abs(pose_in_bin.t[1]) < half_ignore_size[1]:
            pose_filtered.append(pose)
            continue
        if half_size[0] - abs(pose_in_bin.t[0]) < half_size[1] - abs(pose_in_bin.t[1]):
            if pose_in_bin.t[0] > 0:
                angle_between_X = np.arccos(np.dot(bin_pose.n, pose_in_bin.n))
            else:
                angle_between_X = np.arccos(np.dot(-1*bin_pose.n, pose_in_bin.n))
        else:
            if pose_in_bin.t[1] > 0:
                angle_between_X = np.arccos(np.dot(bin_pose.o, pose_in_bin.n))
            else:
                angle_between_X = np.arccos(np.dot(-1*bin_pose.o, pose_in_bin.n))
        if angle_between_X < threshold:
            pose_filtered.append(pose)
    return pose_filtered

def points_to_point_distance(points, point):
    return np.linalg.norm(points - point, ord=2, axis=1)

def fit_circle(points, num_iterations, threshold=0.01):
    points = np.asarray(points)
    num_total_points = len(points)
    best_circle = None
    # best_circle_points = None
    best_num_inliers = 0
    remained_points_indices = None
    for _ in range(num_iterations):
        random_indices = np.random.choice(num_total_points, 3, replace=False)
        circle_points = points[random_indices]
        x1, x2, x3 = circle_points[0][0], circle_points[1][0], circle_points[2][0]
        y1, y2, y3 = circle_points[0][1], circle_points[1][1], circle_points[2][1]
        A = np.array([[2 * (x1 - x2), 2 * (y1 - y2)],
                      [2 * (x1 - x3), 2 * (y1 - y3)],
                      [2 * (x2 - x3), 2 * (y2 - y3)]])
        B = np.array([[x1**2 + y1**2 - x2**2 - y2**2],
                      [x1**2 + y1**2 - x3**2 - y3**2],
                      [x2**2 + y2**2 - x3**2 - y3**2]])
        X = np.linalg.lstsq(A, B, rcond=None)[0]
        center = np.array([X[0][0], X[1][0]])
        r = np.linalg.norm(circle_points[0] - center)
        outliers = np.where(abs(points_to_point_distance(points, center) - r) > threshold)[0]
        num_inliers = num_total_points - len(outliers)
        if num_inliers > best_num_inliers:
            best_circle = (center, r)
            best_num_inliers = num_inliers
            # best_circle_points = circle_points
            remained_points_indices = outliers
    return best_circle, best_num_inliers, remained_points_indices

def find_orthogonal_vectors(normal_vector):
    v1 = np.random.rand(3)
    v2 = np.random.rand(3)
    v1_proj = np.dot(v1, normal_vector) / np.linalg.norm(normal_vector) * normal_vector
    v1_ortho = v1 - v1_proj
    v2_ortho = np.cross(normal_vector, v1_ortho)
    return v1_ortho, v2_ortho


def depth_to_pointcloud(depth_image, fx, fy, cx, cy):
    height, width = depth_image.shape
    u, v = np.meshgrid(np.arange(1, width+1), np.arange(1, height+1))
    z = depth_image
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    pointcloud = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=-1)
    nonzero_indices = np.all(pointcloud != [0, 0, 0], axis=1)
    filteredPCD = pointcloud[nonzero_indices]
    return filteredPCD

def view_coordinate(ppose, pcd, length=100):
    scene = []
    # origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=length, origin=[0, 0, 0])
    # scene.append(origin)
    scene.append(pcd)
    for pose in ppose:
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=length/2, origin=[0, 0, 0])
        coord.transform(pose)
        # scene.add_3d_label(coordinate, "label at origin")
        scene.append(coord)
    o3d.visualization.draw_geometries(scene)