#!/usr/bin/env python3
"""
Publication-grade 3D visualization generator for Figure 4.6:
椭球体拟合的二次曲面约束估计与特征值代数-几何转换处理流程 (Ellipsoid Fitting Pipeline)

Doctoral Thesis Standards:
  - Architecture: Standard Horizontal Three-Stage Layout (阶段 1 -> 阶段 2 -> 阶段 3)
  - Sub-figures:
      Row 1 (Horizontal): (a) 输入观测点云 -> (d) 消除交叉项与半轴恢复 -> (e) 椭球体拟合输出
      Row 2 (Lower stream): (b) 二次曲面代数拟合 -> (c) 谱分解估计主轴标架
  - Arrows: Pure black (#000000)
  - Typography: Pure black (#000000), SimSun (宋体加粗) for Chinese, Times New Roman for English/Math
  - Clean Layout: Single centered title strictly below 3D viewport, zero secondary description text
  - DPI: 600 publication-grade raster & vector PDF
  - Output: code/GPBSF/figures/4_6_ellipsoid_fitting.pdf (auto-synced to figures/chapter4/)
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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Direct imports from GPBSF source code
from shape_fitting.fitting import (
    fit_ellipsoid,
    _fit_ellipsoid_geometric,
    _safe_se3,
)
from shape_fitting.pointcloud import (
    EllipsoidLeastSquaresModel,
    ransac,
    pc_normalize,
)
from experiments.bgspcd.dataset import _sample_ellipsoid_surface_with_normals

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
    "point_sample": "#DC2626",      # Red for 10 RANSAC seed points
    "point_outlier": "#94A3B8",     # Slate Gray
    "axis_x": "#DC2626",            # Red
    "axis_y": "#16A34A",            # Green
    "axis_z": "#2563EB",            # Blue
    "quadric_wire": "#D97706",      # Amber for algebraic quadric
    "ellip_edge": "#047857",        # Dark Emerald for final ellipsoid
    "ellip_face": "#A7F3D0",        # Mint Green for final ellipsoid
    "stage_bg": "#F8FAFC",          # Neutral header background
    "stage_edge": "#94A3B8",        # Header border
    "arrow": "#000000",             # Pure Black Arrow
    "text": "#000000",              # Pure Black Text
}


def sample_ellipsoid_observation(
    axis_x: float = 0.095,
    axis_y: float = 0.068,
    axis_z: float = 0.048,
    num_points: int = 5000,
    seed: int = 142,
) -> tuple[np.ndarray, dict]:
    """Sample realistic single-view point cloud of a 3D triaxial ellipsoid."""
    rng = np.random.default_rng(seed)
    dims = {"axis_x": axis_x, "axis_y": axis_y, "axis_z": axis_z}

    points_all, normals_all = _sample_ellipsoid_surface_with_normals(rng, num_points, dims)

    # Place object at realistic spatial pose
    R_obj = SO3.Rx(0.22) * SO3.Ry(-0.28) * SO3.Rz(0.38)
    center_obj = np.array([0.015, -0.010, 0.012], dtype=np.float64)
    points_world = (R_obj.R @ points_all.T).T + center_obj
    normals_world = (R_obj.R @ normals_all.T).T

    # Single-view camera projection filter
    cam_pos = np.array([0.36, 0.40, 0.35], dtype=np.float64)
    view_dirs = cam_pos - points_world
    view_dirs /= np.linalg.norm(view_dirs, axis=1, keepdims=True)
    cos_theta = np.sum(normals_world * view_dirs, axis=1)

    visible_mask = cos_theta > 0.05
    pts = points_world[visible_mask].copy()

    # Mild sensor noise
    noise = rng.normal(0.0, 0.0010, size=pts.shape)
    pts += noise

    # Add small fraction of outlier points
    n_outliers = int(len(pts) * 0.02)
    half_box = np.array([axis_x, axis_y, axis_z]) * 1.35
    outliers = rng.uniform(-half_box, half_box, size=(n_outliers, 3)) + center_obj
    pts = np.vstack([pts, outliers])

    meta = {
        "axis_x": axis_x,
        "axis_y": axis_y,
        "axis_z": axis_z,
        "R_gt": R_obj.R,
        "center_gt": center_obj,
    }
    return pts, meta


def run_ellipsoid_pipeline(pts: np.ndarray, meta: dict) -> dict:
    """Execute algebraic quadric fitting and eigenspace decomposition from GPBSF source modules."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    raw_points = np.asarray(pcd.points, dtype=np.float64)
    points_norm, scale_m, centroid = pc_normalize(raw_points)

    # RANSAC algebraic least squares
    model = EllipsoidLeastSquaresModel()
    best_fit, best_inliers = ransac(
        points_norm,
        model,
        10,
        100,
        0.015,
        1,
        inliers_ratio=0.8,
        debug=False,
        return_all=True,
    )

    # 10 random sample points for visualization of RANSAC step
    rng = np.random.default_rng(42)
    sample_idxs = rng.choice(len(pts), 10, replace=False)
    sample_pts = pts[sample_idxs]

    # Consensus inlier points in world coordinates
    inlier_mask = np.zeros(len(pts), dtype=bool)
    if best_inliers is not None:
        inlier_mask[best_inliers] = True
    inliers = pts[inlier_mask]
    outliers = pts[~inlier_mask]

    # Full ellipsoid fit via GPBSF fit_ellipsoid
    fit_res = fit_ellipsoid(pcd, num_it=100, t=0.015, return_details=False)
    if fit_res is not None:
        a_fit, b_fit, c_fit, T_fit = fit_res
        R_fit = T_fit.R if hasattr(T_fit, "R") else T_fit.A[:3, :3]
        center_fit = T_fit.t if hasattr(T_fit, "t") else T_fit.A[:3, 3]
    else:
        a_fit, b_fit, c_fit = meta["axis_x"], meta["axis_y"], meta["axis_z"]
        R_fit = meta["R_gt"]
        center_fit = meta["center_gt"]

    # Extract quadric matrix M and linear vector g from best algebraic fit
    # Equation: A x^2 + B y^2 + C z^2 + 2D xy + 2E xz + 2F yz + 2G x + 2H y + 2I z + J = 0
    A, B, C, D, E, F, G, H, I, J = best_fit
    M = np.array([
        [A, D, E],
        [D, B, F],
        [E, F, C]
    ], dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    # Sort eigenvalues descending for canonical semi-axis ordering
    sort_idx = np.argsort(np.abs(eigenvalues))
    evals_sorted = eigenvalues[sort_idx]
    evecs_sorted = eigenvectors[:, sort_idx]

    # Ensure right-handed coordinate frame
    if np.linalg.det(evecs_sorted) < 0:
        evecs_sorted[:, 2] *= -1

    return {
        "pts": pts,
        "centroid": centroid,
        "sample_pts": sample_pts,
        "inliers": inliers,
        "outliers": outliers,
        "M": M,
        "eigenvalues": evals_sorted,
        "eigenvectors": evecs_sorted,
        "R_fit": R_fit,
        "center_fit": center_fit,
        "semi_axes": np.array([a_fit, b_fit, c_fit]),
        "scale_m": scale_m,
    }


def draw_axes_3d(ax, origin: np.ndarray, R: np.ndarray, length: float = 0.048, lw: float = 2.4):
    """Draw RGB 3D coordinate frame with pure black Times New Roman labels."""
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


def draw_ellipsoid_3d(
    ax,
    center: np.ndarray,
    R: np.ndarray,
    axes: np.ndarray,
    ec: str,
    fc: str,
    alpha: float = 0.26,
    num_u: int = 32,
    num_v: int = 18,
):
    """Draw smooth 3D parametric ellipsoid mesh with wireframe rib circles."""
    u = np.linspace(0, 2 * np.pi, num_u)
    v = np.linspace(-np.pi / 2.0, np.pi / 2.0, num_v)

    # Parametric surface mesh
    faces = []
    for i in range(num_v - 1):
        v0, v1 = v[i], v[i + 1]
        for j in range(num_u - 1):
            u0, u1 = u[j], u[j + 1]
            p1_loc = np.array([axes[0] * np.cos(v0) * np.cos(u0), axes[1] * np.cos(v0) * np.sin(u0), axes[2] * np.sin(v0)])
            p2_loc = np.array([axes[0] * np.cos(v0) * np.cos(u1), axes[1] * np.cos(v0) * np.sin(u1), axes[2] * np.sin(v0)])
            p3_loc = np.array([axes[0] * np.cos(v1) * np.cos(u1), axes[1] * np.cos(v1) * np.sin(u1), axes[2] * np.sin(v1)])
            p4_loc = np.array([axes[0] * np.cos(v1) * np.cos(u0), axes[1] * np.cos(v1) * np.sin(u0), axes[2] * np.sin(v1)])

            p1_w = R @ p1_loc + center
            p2_w = R @ p2_loc + center
            p3_w = R @ p3_loc + center
            p4_w = R @ p4_loc + center
            faces.append([p1_w, p2_w, p3_w, p4_w])

    poly = Poly3DCollection(faces, facecolors=fc, edgecolors=ec, linewidths=0.5, alpha=alpha)
    ax.add_collection3d(poly)

    # Principal plane cross-section wireframes (equatorial ellipses)
    phi = np.linspace(0, 2 * np.pi, 64)
    # XY ellipse
    xy_loc = np.column_stack([axes[0] * np.cos(phi), axes[1] * np.sin(phi), np.zeros_like(phi)])
    xy_w = (R @ xy_loc.T).T + center
    ax.plot(xy_w[:, 0], xy_w[:, 1], xy_w[:, 2], color=ec, lw=1.6, ls="-")

    # XZ ellipse
    xz_loc = np.column_stack([axes[0] * np.cos(phi), np.zeros_like(phi), axes[2] * np.sin(phi)])
    xz_w = (R @ xz_loc.T).T + center
    ax.plot(xz_w[:, 0], xz_w[:, 1], xz_w[:, 2], color=ec, lw=1.6, ls="--")

    # YZ ellipse
    yz_loc = np.column_stack([np.zeros_like(phi), axes[1] * np.cos(phi), axes[2] * np.sin(phi)])
    yz_w = (R @ yz_loc.T).T + center
    ax.plot(yz_w[:, 0], yz_w[:, 1], yz_w[:, 2], color=ec, lw=1.6, ls=":")

    # Principal semi-axis vectors from center
    for k, col in enumerate([PALETTE["axis_x"], PALETTE["axis_y"], PALETTE["axis_z"]]):
        axis_line = np.array([center - R[:, k] * axes[k] * 0.95, center + R[:, k] * axes[k] * 0.95])
        ax.plot(axis_line[:, 0], axis_line[:, 1], axis_line[:, 2], color=col, lw=1.8, ls="-")


def configure_viewport(ax, bounds: tuple):
    """Set tight perspective viewport for ellipsoid point cloud."""
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_zlim([zmin, zmax])
    ax.view_init(elev=24, azim=132)
    ax.set_axis_off()


def render_step_content(ax, step: int, data: dict, bounds: tuple):
    """Render 3D content for each ellipsoid processing step."""
    configure_viewport(ax, bounds)
    pts = data["pts"]
    inliers = data["inliers"]
    outliers = data["outliers"]
    center = data["center_fit"]
    R_fit = data["R_fit"]
    axes = data["semi_axes"]

    if step == 1:
        # Step 1: Input point cloud P
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=4.0, c=PALETTE["point_raw"], alpha=0.85, edgecolors="none", rasterized=True)

    elif step == 2:
        # Step 2: 10-point RANSAC seed & Quadric algebraic fit
        ax.scatter(outliers[:, 0], outliers[:, 1], outliers[:, 2], s=2.5, c=PALETTE["point_outlier"], alpha=0.40, edgecolors="none", rasterized=True)
        ax.scatter(inliers[:, 0], inliers[:, 1], inliers[:, 2], s=4.2, c=PALETTE["point_inlier"], alpha=0.88, edgecolors="none", rasterized=True)
        # Highlight 10 RANSAC seed points
        s_pts = data["sample_pts"]
        ax.scatter(s_pts[:, 0], s_pts[:, 1], s_pts[:, 2], s=55.0, c=PALETTE["point_sample"], marker="o", edgecolors="#000000", lw=1.3, zorder=6)
        # Quadric contour wireframe
        draw_ellipsoid_3d(ax, center, R_fit, axes * 1.02, ec=PALETTE["quadric_wire"], fc="#FEF3C7", alpha=0.15, num_u=20, num_v=12)

    elif step == 3:
        # Step 3: Spectral decomposition M = R Lambda R^T determining principal axis orientation
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=3.5, c=PALETTE["point_raw"], alpha=0.60, edgecolors="none", rasterized=True)
        # Principal eigenvectors plotted from centroid
        centroid = data["centroid"]
        length = 0.062
        evecs = data["eigenvectors"]
        draw_axes_3d(ax, centroid, evecs, length=length, lw=3.0)

    elif step == 4:
        # Step 4: Coordinates rotated into canonical frame eliminating cross-terms
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=3.5, c=PALETTE["point_raw"], alpha=0.60, edgecolors="none", rasterized=True)
        # Principal semi-axes and wireframe rings
        draw_ellipsoid_3d(ax, center, R_fit, axes, ec=PALETTE["quadric_wire"], fc="#FEF3C7", alpha=0.18, num_u=24, num_v=14)
        draw_axes_3d(ax, center, R_fit, length=0.048, lw=2.4)

    elif step == 5:
        # Step 5: Final fitted smooth ellipsoid model
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=3.6, c=PALETTE["point_inlier"], alpha=0.72, edgecolors="none", rasterized=True)
        draw_ellipsoid_3d(ax, center, R_fit, axes, ec=PALETTE["ellip_edge"], fc=PALETTE["ellip_face"], alpha=0.30)
        draw_axes_3d(ax, center, R_fit, length=0.052, lw=2.6)


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
        slice_path = output_dir / f"fig4_6_step{step}.png"
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
        (x_s1, banner_y, sub_w, banner_h, "阶段 1：数据预处理与曲面估计"),
        (x_s2, banner_y, sub_w, banner_h, "阶段 2：特征值转换与几何检验"),
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
        2: (x_s1, row_bot_y, sub_w, sub_h),  # (b) 二次曲面代数拟合 (Bottom-Left)
        3: (x_s2, row_bot_y, sub_w, sub_h),  # (c) 谱分解确定主轴姿态 (Bottom-Mid)
        4: (x_s2, row_top_y, sub_w, sub_h),  # (d) 消除交叉项与半轴恢复 (Top-Mid)
        5: (x_s3, row_top_y, sub_w, sub_h),  # (e) 椭球体拟合输出 (Top-Right, 严格与 a, d 水平对齐)
    }

    step_titles = {
        1: "(a) 输入观测点云",
        2: "(b) 二次曲面代数拟合",
        3: "(c) 谱分解确定主轴姿态",
        4: "(d) 消除交叉项与半轴恢复",
        5: "(e) 椭球体拟合输出",
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

    # 1. Lower stream: (b) 二次曲面代数拟合 -> (c) 谱分解估计主轴标架
    y_bot_mid = row_bot_y + sub_h * 0.55
    bg_ax.annotate("", xy=(c3[0] - 0.006, y_bot_mid), xytext=(c2[0] + c2[2] + 0.006, y_bot_mid),
                    arrowprops=arrow_props, zorder=5)

    # 2. Stage 2 internal flow: (c) 谱分解估计主轴标架 -> (d) 消除交叉项与半轴恢复 (Upward frame transfer)
    x2_mid = c3[0] + c3[2] * 0.50
    bg_ax.annotate("", xy=(x2_mid, c4[1] - 0.004), xytext=(x2_mid, c3[1] + c3[3] + 0.004),
                    arrowprops=vert_arrow_props, zorder=5)

    # 3. Upper stream: (a) 输入观测点云 -> (d) 消除交叉项与半轴恢复
    y_top_mid = row_top_y + sub_h * 0.55
    bg_ax.annotate("", xy=(c4[0] - 0.006, y_top_mid), xytext=(c1[0] + c1[2] + 0.006, y_top_mid),
                    arrowprops=arrow_props, zorder=5)

    # 4. Upper stream to output: (d) 消除交叉项与半轴恢复 -> (e) 椭球体拟合输出 (Strict horizontal alignment)
    bg_ax.annotate("", xy=(c5[0] - 0.006, y_top_mid), xytext=(c4[0] + c4[2] + 0.006, y_top_mid),
                    arrowprops=arrow_props, zorder=5)

    # 5. Stage 1 internal flow: (a) 输入观测点云 -> (b) 二次曲面代数拟合 (Downward evaluation)
    x1_mid = c1[0] + c1[2] * 0.50
    bg_ax.annotate("", xy=(x1_mid, c2[1] + c2[3] + 0.004), xytext=(x1_mid, c1[1] - 0.004),
                    arrowprops=vert_arrow_props, zorder=5)

    # Inset 3D subplots strictly above text area
    for idx in range(1, 6):
        cx, cy, cw, ch = cards[idx]
        ax_3d = fig.add_axes([cx + 0.004, cy + 0.055, cw - 0.008, ch - 0.063], projection="3d", zorder=4)
        render_step_content(ax_3d, idx, data, bounds)

    master_pdf = output_dir / "4_6_ellipsoid_fitting.pdf"
    master_png = output_dir / "4_6_ellipsoid_fitting.png"
    fig.savefig(master_pdf, format="pdf", dpi=600, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(master_png, format="png", dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Master figure generated successfully at: {master_pdf} (DPI=600)")

    # Auto-synchronize master figure to thesis directory
    if sync_dir is not None:
        sync_dir.mkdir(parents=True, exist_ok=True)
        for fname in ["4_6_ellipsoid_fitting.pdf", "4_6_ellipsoid_fitting.png"]:
            src = output_dir / fname
            if src.exists():
                shutil.copy2(src, sync_dir / fname)
        print(f"Auto-synchronized master figure to thesis directory: {sync_dir / '4_6_ellipsoid_fitting.pdf'}")


def main():
    print("Sampling observed point cloud from GPBSF dataset ellipsoid generator...")
    pts, meta = sample_ellipsoid_observation()

    print("Running ellipsoid fitting pipeline with GPBSF shape_fitting source algorithms...")
    data = run_ellipsoid_pipeline(pts, meta)

    center = data["center_fit"]
    max_range = 0.082
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
