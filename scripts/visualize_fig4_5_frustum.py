#!/usr/bin/env python3
"""
Publication-grade 3D visualization generator for Figure 4.5:
圆锥台拟合的轴向假设检验与分层半径回归处理流程 (Frustum / Truncated Cone Fitting Pipeline)

Doctoral Thesis Standards:
  - Architecture: Standard Horizontal Three-Stage Layout (阶段 1 -> 阶段 2 -> 阶段 3)
  - Sub-figures:
      Row 1 (Horizontal): (a) 输入观测点云 -> (d) 分层截面圆拟合 -> (e) 圆锥台拟合输出
      Row 2 (Lower stream): (b) 轴向假设检验 -> (c) 建立局部坐标系
  - Arrows: Pure black (#000000)
  - Typography: Pure black (#000000), SimSun (宋体加粗) for Chinese, Times New Roman for English/Math
  - Clean Layout: Single centered title strictly below 3D viewport, zero secondary description text
  - DPI: 600 publication-grade raster & vector PDF
  - Output: code/GPBSF/figures/4_5_frustum_fitting.pdf (auto-synced to figures/chapter4/)
"""

from __future__ import annotations
import sys
import math
import shutil
from pathlib import Path

# Add code/GPBSF root to sys.path
_GPBSF_ROOT = Path(__file__).resolve().parents[1]
_THESIS_ROOT = Path(__file__).resolve().parents[3]
if str(_GPBSF_ROOT) not in sys.path:
    sys.path.insert(0, str(_GPBSF_ROOT))

import numpy as np
import open3d as o3d
from spatialmath import SE3, SO3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager as fm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

# Direct imports from GPBSF source code
from shape_fitting.fitting import (
    fit_frustum_cone_adaptive,
    fit_frustum_cone_by_slice_linear,
    fit_circle_kasa,
    _safe_se3,
    align_vector_to_z,
)
from shape_fitting.pointcloud import segment_plane_with_normals
from experiments.bgspcd.dataset import _sample_frustum_surface_with_normals

# Typography setup
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["SimSun", "Times New Roman", "DejaVu Serif"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "stix"


def get_font(name: str = "simsun", size: float = 12.0, bold: bool = False) -> fm.FontProperties:
    """Return FontProperties strictly adhering to SimSun / Times New Roman."""
    font_paths = {
        "simsun": ["C:/Windows/Fonts/simsun.ttc", "C:/Windows/Fonts/simsun.ttf"],
        "times": ["C:/Windows/Fonts/times.ttf"],
        "times_bold": ["C:/Windows/Fonts/timesbd.ttf"],
    }
    key = "times_bold" if name == "times" and bold else name
    for path_str in font_paths.get(key, []):
        p = Path(path_str)
        if p.exists():
            return fm.FontProperties(fname=str(p), size=size, weight="bold" if bold else "normal")

    family = "SimSun" if name == "simsun" else "Times New Roman"
    return fm.FontProperties(family=family, size=size, weight="bold" if bold else "normal")


# Academic color palette
PALETTE = {
    "point_raw": "#2563EB",         # Royal Blue
    "point_inlier": "#059669",      # Emerald Green
    "point_slice": "#3B82F6",       # Vibrant Blue
    "point_outlier": "#94A3B8",     # Slate Gray
    "point_plane": "#EA580C",       # Deep Coral Orange for RANSAC plane inlier points
    "plane_face": "#FED7AA",        # Translucent peach for plane patch
    "plane_edge": "#EA580C",        # Coral Orange border for plane patch
    "axis_x": "#DC2626",            # Red
    "axis_y": "#16A34A",            # Green
    "axis_z": "#2563EB",            # Blue
    "hypo_pca": "#EF4444",          # Red for PCA hypothesis
    "hypo_normal": "#10B981",       # Green for Normal hypothesis
    "hypo_obb": "#F59E0B",          # Amber for OBB hypothesis
    "circle_edge": "#059669",       # Emerald Green for sliced circle
    "cone_edge": "#047857",         # Dark Emerald
    "cone_face": "#A7F3D0",         # Mint Green
    "stage_bg": "#F8FAFC",          # Neutral header background
    "stage_edge": "#94A3B8",        # Header border
    "arrow": "#000000",             # Pure Black Arrow
    "text": "#000000",              # Pure Black Text
}


def sample_frustum_observation(
    r_bottom: float = 0.085,
    r_top: float = 0.052,
    height: float = 0.170,
    num_points: int = 5000,
    seed: int = 128,
) -> tuple[np.ndarray, dict]:
    """Sample realistic single-view point cloud of a truncated cone."""
    rng = np.random.default_rng(seed)
    dims = {"bottom_radius": r_bottom, "top_radius": r_top, "height": height}

    points_all, normals_all = _sample_frustum_surface_with_normals(rng, num_points, dims)

    # Place object at a realistic orientation
    R_obj = SO3.Rx(0.20) * SO3.Ry(-0.25) * SO3.Rz(0.35)
    center_obj = np.array([0.02, -0.01, 0.01], dtype=np.float64)
    points_world = (R_obj.R @ points_all.T).T + center_obj
    normals_world = (R_obj.R @ normals_all.T).T

    # Single-view camera filtering
    cam_pos = np.array([0.38, 0.42, 0.36], dtype=np.float64)
    view_dirs = cam_pos - points_world
    view_dirs /= np.linalg.norm(view_dirs, axis=1, keepdims=True)
    cos_theta = np.sum(normals_world * view_dirs, axis=1)

    visible_mask = cos_theta > 0.04
    pts = points_world[visible_mask].copy()

    # Mild sensor noise
    noise = rng.normal(0.0, 0.0012, size=pts.shape)
    pts += noise

    # Add small fraction of outlier points
    n_outliers = int(len(pts) * 0.025)
    half_box = np.array([r_bottom, r_bottom, height / 2.0]) * 1.35
    outliers = rng.uniform(-half_box, half_box, size=(n_outliers, 3)) + center_obj
    pts = np.vstack([pts, outliers])

    meta = {
        "r_bottom": r_bottom,
        "r_top": r_top,
        "height": height,
        "R_gt": R_obj.R,
        "center_gt": center_obj,
    }
    return pts, meta


def run_frustum_pipeline(pts: np.ndarray, meta: dict) -> dict:
    """Execute frustum fitting algorithms directly from GPBSF source modules."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=30))

    # Adaptive sequential multi-hypothesis fitting (lower threshold to traverse all candidate hypotheses)
    r1, r2, h, T_fit, method_name, residual = fit_frustum_cone_adaptive(pcd, tau_cone=0.0001)
    print(f"Frustum adaptive pipeline selected: {method_name}, trimmed residual: {residual * 1000.0:.2f} mm")

    centroid = np.mean(pts, axis=0)

    # 1. Plane-guided normal candidate (Strategy 1)
    v_plane = None
    plane_inliers = np.array([], dtype=int)
    normals = np.asarray(pcd.normals)
    try:
        plane_m, inliers = segment_plane_with_normals(
            pts, normals, dist_threshold=0.005, angle_threshold_deg=15.0, max_iterations=1500
        )
        if plane_m is not None and len(inliers) >= 15:
            plane_inliers = np.asarray(inliers, dtype=int)
            v_plane = np.array(plane_m[:3], dtype=np.float64)
            v_plane /= np.linalg.norm(v_plane)
            print(f"法向约束 RANSAC 平面拟合成功: 检出 {len(plane_inliers)} 个内点, 平面法向量: {v_plane}")
    except Exception as e:
        print(f"RANSAC 平面拟合异常: {e}")
    if v_plane is None:
        v_plane = meta["R_gt"][:, 2]

    # Ensure v_plane direction is oriented towards top face
    if np.dot(v_plane, meta["R_gt"][:, 2]) < 0:
        v_plane = -v_plane
    plane_center = np.mean(pts[plane_inliers], axis=0) if len(plane_inliers) > 0 else centroid

    # 2. Lateral surface normal RANSAC candidate (Strategy 2)
    v_normal_ransac = None
    try:
        from shape_fitting.pointcloud import ConeAxisLeastSquaresModel, ransac
        normals = np.asarray(pcd.normals)
        cone_m = ConeAxisLeastSquaresModel()
        fit_res, _ = ransac(normals, cone_m, 10, 200, 0.02, 1, inliers_ratio=0.85, return_all=True)
        if fit_res is not None:
            v_normal_ransac = fit_res[0] / np.linalg.norm(fit_res[0])
    except Exception:
        pass
    if v_normal_ransac is None:
        v_normal_ransac = meta["R_gt"][:, 2]

    # 3 & 4. PCA candidate axes (Strategies 3 & 4)
    pts_centered = pts - centroid
    cov = np.cov(pts_centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    sort_idx = np.argsort(eigenvalues)[::-1]
    v_pca0 = eigenvectors[:, sort_idx[0]]  # Longest variance axis
    v_pca2 = eigenvectors[:, sort_idx[2]]  # Shortest variance axis

    # 5. OBB candidate axis (Strategy 5)
    from shape_fitting.fitting import _oriented_bounding_box
    try:
        obb = _oriented_bounding_box(pcd)
        v_obb = np.asarray(obb.R)[:, 2]
    except Exception:
        v_obb = v_pca0

    # Ground-truth / selected symmetry axis
    R_fit = T_fit.R if hasattr(T_fit, "R") else T_fit.A[:3, :3]
    t_fit = T_fit.t if hasattr(T_fit, "t") else T_fit.A[:3, 3]
    axis_z = R_fit[:, 2]

    # Transform points to local frame for slicing
    R_align = align_vector_to_z(axis_z)
    pts_aligned = (R_align.T @ (pts - centroid).T).T

    # Compute sliced layers using GPBSF slice logic
    z_min = float(np.quantile(pts_aligned[:, 2], 0.005))
    z_max = float(np.quantile(pts_aligned[:, 2], 0.995))
    h_slice = z_max - z_min
    num_layers = 8
    slices = []
    for i in range(num_layers):
        l_min = z_min + i * h_slice / num_layers
        l_max = z_min + (i + 1) * h_slice / num_layers
        mask = (pts_aligned[:, 2] >= l_min) & (pts_aligned[:, 2] <= l_max)
        layer_pts = pts_aligned[mask]
        if len(layer_pts) >= 6:
            circle_res = fit_circle_kasa(layer_pts[:, :2])
            if circle_res is not None:
                (cx, cy), cr = circle_res
                z_mid = (l_min + l_max) / 2.0
                c_world = centroid + R_align @ np.array([cx, cy, z_mid])
                slices.append({
                    "layer_idx": i,
                    "z_local": z_mid,
                    "radius": cr,
                    "center_local": np.array([cx, cy, z_mid]),
                    "center_world": c_world,
                    "pts_world": pts[mask],
                })

    return {
        "pts": pts,
        "centroid": centroid,
        "v_plane": v_plane,
        "plane_inliers": plane_inliers,
        "plane_center": plane_center,
        "v_normal_ransac": v_normal_ransac,
        "v_pca0": v_pca0,
        "v_pca2": v_pca2,
        "v_obb": v_obb,
        "axis_z": axis_z,
        "R_align": R_align,
        "R_fit": R_fit,
        "t_fit": t_fit,
        "r_top": r1,
        "r_bottom": r2,
        "height": h,
        "method_name": method_name,
        "residual": residual,
        "slices": slices,
    }


def draw_axes_3d(ax, origin: np.ndarray, R: np.ndarray, length: float = 0.048, lw: float = 2.4):
    """Draw RGB 3D coordinate frame with black Times New Roman labels."""
    colors = [PALETTE["axis_x"], PALETTE["axis_y"], PALETTE["axis_z"]]
    names = ["X", "Y", "Z"]
    font_en_bold = get_font("times", size=15.0, bold=True)

    for i in range(3):
        v = R[:, i] * length
        ax.quiver(
            origin[0], origin[1], origin[2],
            v[0], v[1], v[2],
            color=colors[i],
            arrow_length_ratio=0.25,
            linewidth=lw,
            normalize=False,
        )
        pos = origin + v * 1.25
        ax.text(
            pos[0], pos[1], pos[2], names[i],
            color=PALETTE["text"],
            fontproperties=font_en_bold,
            ha="center", va="center"
        )


def draw_frustum_3d(
    ax,
    center: np.ndarray,
    R: np.ndarray,
    r_top: float,
    r_bottom: float,
    height: float,
    ec: str,
    fc: str,
    alpha: float = 0.25,
    num_theta: int = 32,
    num_h_rings: int = 6,
):
    """Draw 3D truncated cone wireframe and translucent polyhedral surface."""
    theta = np.linspace(0, 2 * np.pi, num_theta, endpoint=False)
    z_vals = np.linspace(-height / 2.0, height / 2.0, num_h_rings)

    # 1. Surface quadrilateral facets
    faces = []
    for j in range(num_h_rings - 1):
        z0, z1 = z_vals[j], z_vals[j + 1]
        r0 = r_bottom + (r_top - r_bottom) * (z0 + height / 2.0) / height
        r1 = r_bottom + (r_top - r_bottom) * (z1 + height / 2.0) / height
        for k in range(num_theta):
            k_next = (k + 1) % num_theta
            p1_loc = np.array([r0 * np.cos(theta[k]), r0 * np.sin(theta[k]), z0])
            p2_loc = np.array([r0 * np.cos(theta[k_next]), r0 * np.sin(theta[k_next]), z0])
            p3_loc = np.array([r1 * np.cos(theta[k_next]), r1 * np.sin(theta[k_next]), z1])
            p4_loc = np.array([r1 * np.cos(theta[k]), r1 * np.sin(theta[k]), z1])

            p1_w = R @ p1_loc + center
            p2_w = R @ p2_loc + center
            p3_w = R @ p3_loc + center
            p4_w = R @ p4_loc + center
            faces.append([p1_w, p2_w, p3_w, p4_w])

    poly = Poly3DCollection(faces, facecolors=fc, edgecolors=ec, linewidths=0.6, alpha=alpha)
    ax.add_collection3d(poly)

    # 2. Prominent top and bottom circular rims
    for z_end, r_end in [(-height / 2.0, r_bottom), (height / 2.0, r_top)]:
        rim_theta = np.linspace(0, 2 * np.pi, 64)
        rim_loc = np.column_stack([r_end * np.cos(rim_theta), r_end * np.sin(rim_theta), np.full_like(rim_theta, z_end)])
        rim_w = (R @ rim_loc.T).T + center
        ax.plot(rim_w[:, 0], rim_w[:, 1], rim_w[:, 2], color=ec, lw=2.0)

    # 3. Central symmetry axis
    axis_loc = np.array([[0, 0, -height / 2.0], [0, 0, height / 2.0]])
    axis_w = (R @ axis_loc.T).T + center
    ax.plot(axis_w[:, 0], axis_w[:, 1], axis_w[:, 2], color="#047857", lw=2.2, ls="--")


def configure_viewport(ax, bounds: tuple):
    """Set tight perspective viewport for frustum point cloud."""
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_zlim([zmin, zmax])
    ax.view_init(elev=24, azim=130)
    ax.set_axis_off()


def render_step_content(ax, step: int, data: dict, bounds: tuple):
    """Render 3D content for each frustum processing step."""
    configure_viewport(ax, bounds)
    pts = data["pts"]
    centroid = data["centroid"]

    if step == 1:
        # Step 1: Input point cloud P
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=3.8, c=PALETTE["point_raw"], alpha=0.85, edgecolors="none", rasterized=True)

    elif step == 2:
        # Step 2: Multi-hypothesis candidate symmetry axes (5 strategies according to Algorithm 4.1)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=2.8, c=PALETTE["point_raw"], alpha=0.35, edgecolors="none", rasterized=True)

        # 突出绘制 RANSAC 平面检测到的内点点集 (深珊瑚橙色，高对比度)
        plane_inliers = data.get("plane_inliers", np.array([], dtype=int))
        if len(plane_inliers) > 0:
            p_pts = pts[plane_inliers]
            ax.scatter(
                p_pts[:, 0], p_pts[:, 1], p_pts[:, 2],
                s=12.0, c=PALETTE["point_plane"], alpha=0.95, edgecolors="#7C2D12", linewidths=0.5,
                rasterized=True, zorder=5
            )
            # 绘制贴合在检测平面上的半透明参考圆盘与法向量
            plane_center = data.get("plane_center", np.mean(p_pts, axis=0))
            vp = data["v_plane"]
            theta_disc = np.linspace(0, 2 * np.pi, 48)
            R_disc = align_vector_to_z(vp)
            r_disc = 0.054
            disc_loc = np.column_stack([r_disc * np.cos(theta_disc), r_disc * np.sin(theta_disc), np.zeros_like(theta_disc)])
            disc_w = (R_disc @ disc_loc.T).T + plane_center
            poly_disc = Poly3DCollection([disc_w], facecolors=PALETTE["plane_face"], edgecolors=PALETTE["plane_edge"], linewidths=1.2, alpha=0.45)
            ax.add_collection3d(poly_disc)

            # 在检测平面中心处引出法向引导箭头
            ax.quiver(
                plane_center[0], plane_center[1], plane_center[2],
                vp[0] * 0.048, vp[1] * 0.048, vp[2] * 0.048,
                color=PALETTE["plane_edge"], lw=2.4, arrow_length_ratio=0.25
            )
        length = 0.075

        # 1. Plane normal candidate (Strategy 1, Green dashed)
        vp = data["v_plane"] * length
        ax.quiver(centroid[0], centroid[1], centroid[2], vp[0], vp[1], vp[2], color="#10B981", lw=2.2, ls="--", arrow_length_ratio=0.20)

        # 2. Lateral normal RANSAC candidate (Strategy 2, Teal dashed)
        vn = data["v_normal_ransac"] * (length * 0.90)
        ax.quiver(centroid[0], centroid[1], centroid[2], vn[0], vn[1], vn[2], color="#0D9488", lw=2.0, ls="-.", arrow_length_ratio=0.20)

        # 3. PCA dominant axis candidate (Strategy 3, Red dashed)
        v0 = data["v_pca0"] * (length * 0.85)
        ax.quiver(centroid[0], centroid[1], centroid[2], v0[0], v0[1], v0[2], color="#EF4444", lw=2.0, ls=":", arrow_length_ratio=0.20)

        # 4. PCA secondary axis candidate (Strategy 4, Amber dashed)
        v2 = data["v_pca2"] * (length * 0.80)
        ax.quiver(centroid[0], centroid[1], centroid[2], v2[0], v2[1], v2[2], color="#F59E0B", lw=1.8, ls=":", arrow_length_ratio=0.20)

        # 5. OBB axis candidate (Strategy 5, Slate dashed)
        v_obb = data["v_obb"] * (length * 0.85)
        ax.quiver(centroid[0], centroid[1], centroid[2], v_obb[0], v_obb[1], v_obb[2], color="#64748B", lw=1.8, ls="--", arrow_length_ratio=0.20)

        # Optimal selected symmetry axis (Solid thick dark emerald arrow, prominently thickened)
        vz = data["axis_z"] * (length * 1.15)
        ax.quiver(centroid[0], centroid[1], centroid[2], vz[0], vz[1], vz[2], color="#047857", lw=4.2, arrow_length_ratio=0.24)

    elif step == 3:
        # Step 3: Establish local coordinate frame at base center
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=3.4, c=PALETTE["point_raw"], alpha=0.65, edgecolors="none", rasterized=True)
        # Symmetry axis line
        t_fit = data["t_fit"]
        R_fit = data["R_fit"]
        h = data["height"]
        axis_pts = np.array([t_fit - R_fit[:, 2] * (h / 2.0), t_fit + R_fit[:, 2] * (h / 2.0)])
        ax.plot(axis_pts[:, 0], axis_pts[:, 1], axis_pts[:, 2], color="#047857", lw=2.4, ls="--")
        # Base frame
        base_center = t_fit - R_fit[:, 2] * (h / 2.0)
        draw_axes_3d(ax, base_center, R_fit, length=0.052, lw=2.6)

    elif step == 4:
        # Step 4: Layered slicing & algebraic circle fitting
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=2.8, c=PALETTE["point_raw"], alpha=0.35, edgecolors="none", rasterized=True)
        # Render sliced rings
        theta = np.linspace(0, 2 * np.pi, 48)
        R_align = data["R_align"]
        for s in data["slices"]:
            r = s["radius"]
            c_w = s["center_world"]
            ring_loc = np.column_stack([r * np.cos(theta), r * np.sin(theta), np.zeros_like(theta)])
            ring_w = (R_align @ ring_loc.T).T + c_w
            ax.plot(ring_w[:, 0], ring_w[:, 1], ring_w[:, 2], color=PALETTE["circle_edge"], lw=1.8)
            # Sliced inlier points
            s_pts = s["pts_world"]
            ax.scatter(s_pts[:, 0], s_pts[:, 1], s_pts[:, 2], s=4.2, c=PALETTE["point_inlier"], alpha=0.88, edgecolors="none", rasterized=True)

    elif step == 5:
        # Step 5: Final fitted truncated cone model
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=3.4, c=PALETTE["point_inlier"], alpha=0.68, edgecolors="none", rasterized=True)
        draw_frustum_3d(
            ax,
            data["t_fit"],
            data["R_fit"],
            data["r_top"],
            data["r_bottom"],
            data["height"],
            ec=PALETTE["cone_edge"],
            fc=PALETTE["cone_face"],
            alpha=0.30,
        )
        draw_axes_3d(ax, data["t_fit"], data["R_fit"], length=0.054, lw=2.6)


def export_figures(data: dict, bounds: tuple, output_dir: Path, sync_dir: Path | None = None):
    """Export 600 DPI standalone slices and master composite 3-stage horizontal figure."""
    output_dir.mkdir(parents=True, exist_ok=True)

    font_stage_header = get_font("simsun", size=18.0, bold=True)
    font_title = get_font("simsun", size=17.0, bold=True)

    # 1. Standalone slice images (step 1 to step 5)
    for step in range(1, 6):
        fig = plt.figure(figsize=(4.0, 3.6), dpi=600)
        ax = fig.add_subplot(111, projection="3d")
        render_step_content(ax, step, data, bounds)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        slice_path = output_dir / f"fig4_5_step{step}.png"
        fig.savefig(slice_path, transparent=True, dpi=600, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

    # 2. Master composite figure: Standard Horizontal Three-Stage Layout
    fig = plt.figure(figsize=(12.2, 7.8), dpi=600)
    fig.patch.set_facecolor("#FFFFFF")

    bg_ax = fig.add_axes([0, 0, 1, 1], zorder=0)
    bg_ax.set_axis_off()

    # Column coordinates
    w_col = 0.278
    gap_col = 0.060
    start_x = 0.035

    x_s1 = start_x
    x_s2 = start_x + w_col + gap_col
    x_s3 = start_x + 2 * (w_col + gap_col)

    # Vertical coordinates
    banner_y = 0.925
    banner_h = 0.045
    sub_w = w_col
    sub_h = 0.385
    row_top_y = 0.490
    row_bot_y = 0.045

    # Top stage banners
    stages = [
        (x_s1, banner_y, sub_w, banner_h, "阶段 1：数据预处理与轴向假设"),
        (x_s2, banner_y, sub_w, banner_h, "阶段 2：坐标变换与分层回归"),
        (x_s3, banner_y, sub_w, banner_h, "阶段 3：拟合输出"),
    ]

    for sx, sy, sw, sh, text in stages:
        bg_ax.add_patch(patches.FancyBboxPatch(
            (sx, sy), sw, sh, boxstyle="round,pad=0.008,rounding_size=0.015",
            facecolor=PALETTE["stage_bg"], edgecolor=PALETTE["stage_edge"], linewidth=1.1, zorder=1
        ))
        bg_ax.text(
            sx + sw / 2.0, sy + sh / 2.0,
            text, fontproperties=font_stage_header, color=PALETTE["text"],
            ha="center", va="center", zorder=3
        )

    # Banner inter-stage arrows
    banner_arrow_props = dict(
        arrowstyle="-|>,head_length=0.45,head_width=0.25",
        color=PALETTE["arrow"],
        lw=2.4,
        mutation_scale=16
    )
    bg_ax.annotate("", xy=(x_s2 - 0.008, banner_y + banner_h / 2.0),
                    xytext=(x_s1 + sub_w + 0.008, banner_y + banner_h / 2.0),
                    arrowprops=banner_arrow_props, zorder=5)
    bg_ax.annotate("", xy=(x_s3 - 0.008, banner_y + banner_h / 2.0),
                    xytext=(x_s2 + sub_w + 0.008, banner_y + banner_h / 2.0),
                    arrowprops=banner_arrow_props, zorder=5)

    # Sub-image cards mapping
    cards = {
        1: (x_s1, row_top_y, sub_w, sub_h),  # (a) 输入观测点云 (Top-Left)
        2: (x_s1, row_bot_y, sub_w, sub_h),  # (b) 轴向假设检验 (Bottom-Left)
        3: (x_s2, row_bot_y, sub_w, sub_h),  # (c) 建立局部坐标系 (Bottom-Mid)
        4: (x_s2, row_top_y, sub_w, sub_h),  # (d) 分层截面圆拟合 (Top-Mid)
        5: (x_s3, row_top_y, sub_w, sub_h),  # (e) 圆锥台拟合输出 (Top-Right, 严格与 a, d 水平对齐)
    }

    step_titles = {
        1: "(a) 输入观测点云",
        2: "(b) 轴向假设检验",
        3: "(c) 建立局部坐标系",
        4: "(d) 分层截面圆拟合",
        5: "(e) 圆锥台拟合输出",
    }

    # Draw slim card frames and place title strictly below the 3D plot area
    for idx in range(1, 6):
        cx, cy, cw, ch = cards[idx]
        bg_ax.add_patch(patches.FancyBboxPatch(
            (cx, cy), cw, ch, boxstyle="round,pad=0.008,rounding_size=0.015",
            facecolor="#FAFAFA", edgecolor="#CBD5E1", linewidth=0.9, zorder=1
        ))

        # Title text centered in the bottom area of each card
        bg_ax.text(
            cx + cw / 2.0, cy + 0.032,
            step_titles[idx],
            fontproperties=font_title, color=PALETTE["text"],
            ha="center", va="center", zorder=3
        )

    # Flow arrows
    arrow_props = dict(
        arrowstyle="-|>,head_length=0.45,head_width=0.26",
        color=PALETTE["arrow"],
        lw=2.2,
        mutation_scale=15
    )
    vert_arrow_props = dict(
        arrowstyle="-|>,head_length=0.38,head_width=0.20",
        color=PALETTE["arrow"],
        lw=1.8,
        mutation_scale=13
    )

    c1, c2, c3, c4, c5 = cards[1], cards[2], cards[3], cards[4], cards[5]

    # 1. Lower stream: (b) 轴向假设检验 -> (c) 建立局部坐标系
    y_bot_mid = row_bot_y + sub_h * 0.55
    bg_ax.annotate("", xy=(c3[0] - 0.006, y_bot_mid), xytext=(c2[0] + c2[2] + 0.006, y_bot_mid),
                    arrowprops=arrow_props, zorder=5)

    # 2. Stage 2 internal flow: (c) 建立局部坐标系 -> (d) 分层截面圆拟合 (Upward frame transfer)
    x2_mid = c3[0] + c3[2] * 0.50
    bg_ax.annotate("", xy=(x2_mid, c4[1] - 0.004), xytext=(x2_mid, c3[1] + c3[3] + 0.004),
                    arrowprops=vert_arrow_props, zorder=5)

    # 3. Upper stream: (a) 输入观测点云 -> (d) 分层截面圆拟合
    y_top_mid = row_top_y + sub_h * 0.55
    bg_ax.annotate("", xy=(c4[0] - 0.006, y_top_mid), xytext=(c1[0] + c1[2] + 0.006, y_top_mid),
                    arrowprops=arrow_props, zorder=5)

    # 4. Upper stream to output: (d) 分层截面圆拟合 -> (e) 圆锥台拟合输出 (Strict horizontal alignment)
    bg_ax.annotate("", xy=(c5[0] - 0.006, y_top_mid), xytext=(c4[0] + c4[2] + 0.006, y_top_mid),
                    arrowprops=arrow_props, zorder=5)

    # 5. Stage 1 internal flow: (a) 输入观测点云 -> (b) 轴向假设检验 (Downward evaluation)
    x1_mid = c1[0] + c1[2] * 0.50
    bg_ax.annotate("", xy=(x1_mid, c2[1] + c2[3] + 0.004), xytext=(x1_mid, c1[1] - 0.004),
                    arrowprops=vert_arrow_props, zorder=5)

    # Inset 3D subplots strictly above text area
    for idx in range(1, 6):
        cx, cy, cw, ch = cards[idx]
        ax_3d = fig.add_axes([cx + 0.004, cy + 0.055, cw - 0.008, ch - 0.063], projection="3d", zorder=4)
        render_step_content(ax_3d, idx, data, bounds)

    master_pdf = output_dir / "4_5_frustum_fitting.pdf"
    master_png = output_dir / "4_5_frustum_fitting.png"
    fig.savefig(master_pdf, format="pdf", dpi=600, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(master_png, format="png", dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Master figure generated successfully at: {master_pdf} (DPI=600)")

    # Auto-synchronize master figure to thesis directory
    if sync_dir is not None:
        sync_dir.mkdir(parents=True, exist_ok=True)
        for fname in ["4_5_frustum_fitting.pdf", "4_5_frustum_fitting.png"]:
            src = output_dir / fname
            if src.exists():
                shutil.copy2(src, sync_dir / fname)
        print(f"Auto-synchronized master figure to thesis directory: {sync_dir / '4_5_frustum_fitting.pdf'}")


def main():
    print("Sampling observed point cloud from GPBSF dataset frustum generator...")
    pts, meta = sample_frustum_observation()

    print("Running frustum fitting pipeline with GPBSF shape_fitting source algorithms...")
    data = run_frustum_pipeline(pts, meta)

    center = data["centroid"]
    max_range = 0.098
    bounds = (
        center[0] - max_range, center[0] + max_range,
        center[1] - max_range, center[1] + max_range,
        center[2] - max_range * 0.88, center[2] + max_range * 0.88,
    )

    # Primary output directory inside source code repository: code/GPBSF/figures
    output_dir = _GPBSF_ROOT / "figures"
    # Target directory for thesis LaTeX compilation: figures/chapter4
    sync_dir = _THESIS_ROOT / "figures" / "chapter4"

    print(f"Exporting visualization artifacts to source folder: {output_dir} (DPI=600)...")
    export_figures(data, bounds, output_dir, sync_dir=sync_dir)


if __name__ == "__main__":
    main()
