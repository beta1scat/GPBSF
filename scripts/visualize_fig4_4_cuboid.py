"""
Generate Figure 4.4 for Doctoral Thesis: Cuboid Fitting Process Illustration.
Directly invokes GPBSF source modules:
  - Dataset surface sampling: experiments.bgspcd.dataset._sample_cuboid_surface_with_normals
  - Algorithmic fitting & OBB: shape_fitting.fitting._oriented_bounding_box
  - Core 3D engine: Open3D (pcd.segment_plane) + Spatialmath (SE3, SO3)

Design Specifications:
  - Three-Stage Horizontal Pipeline (标准横向三段式排版):
      阶段 1：数据预处理  -->  阶段 2：坐标变换与边界估计  -->  阶段 3：拟合输出
  - Sub-images Layout:
      Upper row: (a) 输入观测点云  -->  (c) 建立局部坐标系  -->  (e) 长方体拟合输出
      Lower row: (b) 提取主平面内点  -->  (d) 投影与边界估计
  - Text: Strictly placed below each 3D plot, concise academic phrasing, no occlusion
  - Typography: SimSun for Chinese, Times New Roman for English & Math (五号字 / 比小四小一号)
  - DPI: 600 publication-grade raster & vector PDF
"""

from __future__ import annotations
import sys
import shutil
from pathlib import Path

# Add code/GPBSF root to sys.path so project modules can be directly imported
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
from shape_fitting.fitting import _oriented_bounding_box
from experiments.bgspcd.dataset import _sample_cuboid_surface_with_normals

# Typography setup: SimSun for Chinese, Times New Roman for English & Math
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


# Academic color palette (Refined minimal scheme)
PALETTE = {
    "point_raw": "#2563EB",         # Royal Blue
    "point_inlier": "#059669",      # Emerald Green
    "point_outlier": "#94A3B8",     # Slate Gray
    "point_projected": "#2563EB",   # Unified with raw points (Royal Blue)
    "axis_x": "#DC2626",            # Red
    "axis_y": "#16A34A",            # Green
    "axis_z": "#2563EB",            # Blue
    "box_edge": "#B45309",          # Amber
    "box_face": "#FDE68A",          # Light Amber
    "box_final_edge": "#047857",    # Forest Green
    "box_final_face": "#A7F3D0",    # Mint Green
    "stage_bg": "#F8FAFC",          # Stage Header Background
    "stage_edge": "#94A3B8",        # Stage Header Border
    "arrow": "#000000",             # Pure Black Arrow
    "text": "#000000",              # Pure Black Text
}


def sample_observation_from_dataset_source(
    length: float = 0.22,
    width: float = 0.14,
    height: float = 0.09,
    num_points: int = 5000,
    seed: int = 105,
) -> tuple[np.ndarray, dict]:
    """Sample cuboid surface points using GPBSF dataset surface sampler and filter visible faces."""
    rng = np.random.default_rng(seed)
    dims = {"length": length, "width": width, "height": height}

    points_all, normals_all = _sample_cuboid_surface_with_normals(rng, num_points, dims)

    cam_pos = np.array([0.38, 0.42, 0.35], dtype=np.float64)
    view_dirs = cam_pos - points_all
    view_dirs /= np.linalg.norm(view_dirs, axis=1, keepdims=True)
    cos_theta = np.sum(normals_all * view_dirs, axis=1)

    visible_mask = cos_theta > 0.05
    pts = points_all[visible_mask].copy()

    noise = rng.normal(0.0, 0.0015, size=pts.shape)
    pts += noise

    n_outliers = int(len(pts) * 0.03)
    half_box = np.array([length, width, height]) * 0.65
    outliers = rng.uniform(-half_box, half_box, size=(n_outliers, 3))
    pts = np.vstack([pts, outliers])

    return pts, dims


def run_pipeline_with_source_methods(pts: np.ndarray) -> dict:
    """Execute fitting using exact GPBSF Open3D plane segmentation and _oriented_bounding_box."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    diagonal = float(np.linalg.norm(np.ptp(pts, axis=0)))
    dist_threshold = max(diagonal * 0.015, 1e-4)

    # 1. Plane segmentation
    plane_model, inliers_idx = pcd.segment_plane(
        distance_threshold=dist_threshold, ransac_n=3, num_iterations=500
    )
    inliers_set = set(inliers_idx)
    inliers = pts[inliers_idx]
    outliers_idx = [i for i in range(len(pts)) if i not in inliers_set]
    outliers = pts[outliers_idx]

    # 2. Dominant plane OBB
    plane_cloud = pcd.select_by_index(inliers_idx)
    obb_plane = _oriented_bounding_box(plane_cloud)
    R_plane = np.asarray(obb_plane.R, dtype=np.float64)
    o_pi = np.asarray(obb_plane.center, dtype=np.float64)
    plane_extents = np.asarray(obb_plane.extent, dtype=np.float64)

    # 3. Project to local frame
    pts_local = (pts - o_pi) @ R_plane

    # 4. Quantile bounds
    min_b = np.quantile(pts_local, 0.005, axis=0)
    max_b = np.quantile(pts_local, 0.995, axis=0)
    extent = np.maximum(max_b - min_b, max(diagonal * 0.02, 1e-3))
    local_center = (min_b + max_b) / 2.0

    # 5. Recover center and pose
    cube_center = o_pi + R_plane @ local_center
    T_cube = SE3.Rt(SO3(R_plane), cube_center)

    return {
        "pts": pts,
        "inliers": inliers,
        "outliers": outliers,
        "o_pi": o_pi,
        "R_pi": R_plane,
        "plane_center": o_pi,
        "plane_extents": plane_extents,
        "pts_local": pts_local,
        "min_b": min_b,
        "max_b": max_b,
        "extents": extent,
        "cube_center": cube_center,
        "R_cube": R_plane,
        "T_cube": T_cube,
    }


def draw_axes_3d(ax, origin: np.ndarray, R: np.ndarray, length: float = 0.048, lw: float = 2.4):
    """Draw RGB 3D coordinate frame with enlarged Times New Roman labels."""
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


def draw_box_3d(ax, center: np.ndarray, R: np.ndarray, extents: np.ndarray, ec: str, fc: str, alpha: float, lw: float = 1.6, ls: str = "-"):
    """Draw oriented 3D bounding box with wireframe and transparent faces."""
    hx, hy, hz = np.asarray(extents) / 2.0
    corners_local = np.array([
        [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
        [-hx, -hy,  hz], [hx, -hy,  hz], [hx, hy,  hz], [-hx, hy,  hz],
    ])
    corners_world = (R @ corners_local.T).T + center
    faces = [
        [corners_world[0], corners_world[1], corners_world[2], corners_world[3]],
        [corners_world[4], corners_world[5], corners_world[6], corners_world[7]],
        [corners_world[0], corners_world[1], corners_world[5], corners_world[4]],
        [corners_world[2], corners_world[3], corners_world[7], corners_world[6]],
        [corners_world[1], corners_world[2], corners_world[6], corners_world[5]],
        [corners_world[0], corners_world[3], corners_world[7], corners_world[4]],
    ]
    poly = Poly3DCollection(faces, facecolors=fc, edgecolors=ec, linewidths=lw, linestyles=ls, alpha=alpha)
    ax.add_collection3d(poly)


def configure_viewport(ax, bounds: tuple):
    """Set tight equal scale and borderless 3D perspective view for enlarged point cloud."""
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_zlim([zmin, zmax])
    ax.view_init(elev=26, azim=132)
    ax.set_axis_off()


def render_step_content(ax, step: int, data: dict, bounds: tuple):
    """Render enlarged 3D graphics for each specific processing step."""
    configure_viewport(ax, bounds)
    pts = data["pts"]
    inliers = data["inliers"]
    outliers = data["outliers"]

    if step == 1:
        # Step 1: Input point cloud P
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=3.2, c=PALETTE["point_raw"], alpha=0.82, edgecolors="none", rasterized=True)

    elif step == 2:
        # Step 2: Dominant plane extraction
        ax.scatter(outliers[:, 0], outliers[:, 1], outliers[:, 2], s=2.2, c=PALETTE["point_outlier"], alpha=0.45, edgecolors="none", rasterized=True)
        ax.scatter(inliers[:, 0], inliers[:, 1], inliers[:, 2], s=4.0, c=PALETTE["point_inlier"], alpha=0.92, edgecolors="none", rasterized=True)

    elif step == 3:
        # Step 3: Dominant plane OBB and local frame
        ax.scatter(inliers[:, 0], inliers[:, 1], inliers[:, 2], s=3.6, c=PALETTE["point_inlier"], alpha=0.82, edgecolors="none", rasterized=True)
        ax.scatter(outliers[:, 0], outliers[:, 1], outliers[:, 2], s=1.5, c=PALETTE["point_outlier"], alpha=0.18, edgecolors="none", rasterized=True)
        draw_box_3d(ax, data["plane_center"], data["R_pi"], data["plane_extents"], ec=PALETTE["box_edge"], fc=PALETTE["box_face"], alpha=0.18, lw=1.6, ls="--")
        draw_axes_3d(ax, data["o_pi"], data["R_pi"], length=0.050, lw=2.4)

    elif step == 4:
        # Step 4: Projected cloud and quantile bounding box
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=3.0, c=PALETTE["point_projected"], alpha=0.74, edgecolors="none", rasterized=True)
        draw_box_3d(ax, data["cube_center"], data["R_pi"], data["extents"], ec=PALETTE["box_edge"], fc=PALETTE["box_face"], alpha=0.16, lw=1.6, ls="-.")
        draw_axes_3d(ax, data["o_pi"], data["R_pi"], length=0.044, lw=2.0)

    elif step == 5:
        # Step 5: Final fitted cuboid and recovered pose
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=3.0, c=PALETTE["point_inlier"], alpha=0.68, edgecolors="none", rasterized=True)
        draw_box_3d(ax, data["cube_center"], data["R_cube"], data["extents"], ec=PALETTE["box_final_edge"], fc=PALETTE["box_final_face"], alpha=0.28, lw=2.2, ls="-")
        draw_axes_3d(ax, data["cube_center"], data["R_cube"], length=0.060, lw=2.6)


def export_figures(data: dict, bounds: tuple, output_dir: Path, sync_dir: Path | None = None):
    """Export 600 DPI standalone slices and master composite 3-stage horizontal figure."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Typography: scaled to appear as ~10.5 pt (五号, 比小四小一号) when printed
    font_stage_header = get_font("simsun", size=18.0, bold=True)
    font_title = get_font("simsun", size=17.0, bold=True)
    font_arrow = get_font("simsun", size=15.0, bold=True)

    # 1. Standalone slices (600 DPI)
    for step in range(1, 6):
        fig = plt.figure(figsize=(4.0, 3.6), dpi=600)
        ax = fig.add_subplot(111, projection="3d")
        render_step_content(ax, step, data, bounds)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        slice_path = output_dir / f"fig4_4_step{step}.png"
        fig.savefig(slice_path, transparent=True, dpi=600, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

    # 2. Master composite figure: Standard Horizontal Three-Stage Layout (标准横向三段式排版)
    # Stage 1: 数据预处理  -->  Stage 2: 坐标变换与边界估计  -->  Stage 3: 拟合输出
    fig = plt.figure(figsize=(12.2, 7.8), dpi=600)
    fig.patch.set_facecolor("#FFFFFF")

    bg_ax = fig.add_axes([0, 0, 1, 1], zorder=0)
    bg_ax.set_axis_off()

    # Column coordinates for 3 stages
    # Stage 1: x1 to x1 + w1
    # Stage 2: x2 to x2 + w2
    # Stage 3: x3 to x3 + w3
    w_col = 0.278
    gap_col = 0.060
    start_x = 0.035

    x_s1 = start_x
    x_s2 = start_x + w_col + gap_col
    x_s3 = start_x + 2 * (w_col + gap_col)

    # Vertical coordinates
    # Top banner for the 3 stages
    banner_y = 0.925
    banner_h = 0.045

    # Sub-card dimensions
    sub_w = w_col
    sub_h = 0.385
    row_top_y = 0.490
    row_bot_y = 0.045

    # -------------------------------------------------------------
    # A. Top Stage Banners (横向三阶段大标题)
    # -------------------------------------------------------------
    stages = [
        (x_s1, banner_y, sub_w, banner_h, "阶段 1：数据预处理"),
        (x_s2, banner_y, sub_w, banner_h, "阶段 2：坐标变换与边界估计"),
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

    # Inter-stage horizontal big arrows at top banner level
    banner_arrow_props = dict(
        arrowstyle="-|>,head_length=0.45,head_width=0.25",
        color=PALETTE["arrow"],
        lw=2.4,
        mutation_scale=16
    )
    # Stage 1 -> Stage 2
    bg_ax.annotate("", xy=(x_s2 - 0.008, banner_y + banner_h / 2.0),
                    xytext=(x_s1 + sub_w + 0.008, banner_y + banner_h / 2.0),
                    arrowprops=banner_arrow_props, zorder=5)
    # Stage 2 -> Stage 3
    bg_ax.annotate("", xy=(x_s3 - 0.008, banner_y + banner_h / 2.0),
                    xytext=(x_s2 + sub_w + 0.008, banner_y + banner_h / 2.0),
                    arrowprops=banner_arrow_props, zorder=5)

    # -------------------------------------------------------------
    # B. Sub-Image Cards Placement
    # -------------------------------------------------------------
    # Stage 1: (a) at top, (b) at bottom
    # Stage 2: (d) at top, (c) at bottom
    # Stage 3: (e) at top, strictly aligned with (a) and (d)
    cards = {
        1: (x_s1, row_top_y, sub_w, sub_h),  # (a) 输入观测点云 (Top-Left)
        2: (x_s1, row_bot_y, sub_w, sub_h),  # (b) 提取主平面内点 (Bottom-Left)
        3: (x_s2, row_bot_y, sub_w, sub_h),  # (c) 建立局部坐标系 (Bottom-Mid)
        4: (x_s2, row_top_y, sub_w, sub_h),  # (d) 投影与边界估计 (Top-Mid)
        5: (x_s3, row_top_y, sub_w, sub_h),  # (e) 长方体拟合输出 (Top-Right, 严格与 a, d 水平对齐)
    }

    step_titles = {
        1: "(a) 输入观测点云",
        2: "(b) 提取主平面内点",
        3: "(c) 建立局部坐标系",
        4: "(d) 投影与边界估计",
        5: "(e) 长方体拟合输出",
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

    # -------------------------------------------------------------
    # C. Flow Arrows Linking Sub-images (Rigorous Algorithmic Flow)
    # -------------------------------------------------------------
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

    # 1. Lower stream: (b) 提取主平面内点 -> (c) 建立局部坐标系
    y_bot_mid = row_bot_y + sub_h * 0.55
    bg_ax.annotate("", xy=(c3[0] - 0.006, y_bot_mid), xytext=(c2[0] + c2[2] + 0.006, y_bot_mid),
                    arrowprops=arrow_props, zorder=5)

    # 2. Stage 2 internal flow: (c) 建立局部坐标系 -> (d) 投影与边界估计 (Upward frame transfer)
    x2_mid = c3[0] + c3[2] * 0.50
    bg_ax.annotate("", xy=(x2_mid, c4[1] - 0.004), xytext=(x2_mid, c3[1] + c3[3] + 0.004),
                    arrowprops=vert_arrow_props, zorder=5)

    # 3. Upper stream: (a) 输入观测点云 -> (d) 投影与边界估计
    y_top_mid = row_top_y + sub_h * 0.55
    bg_ax.annotate("", xy=(c4[0] - 0.006, y_top_mid), xytext=(c1[0] + c1[2] + 0.006, y_top_mid),
                    arrowprops=arrow_props, zorder=5)

    # 4. Upper stream to output: (d) 投影与边界估计 -> (e) 长方体拟合输出 (Strict horizontal alignment)
    bg_ax.annotate("", xy=(c5[0] - 0.006, y_top_mid), xytext=(c4[0] + c4[2] + 0.006, y_top_mid),
                    arrowprops=arrow_props, zorder=5)

    # 5. Stage 1 internal flow: (a) 输入观测点云 -> (b) 提取主平面内点 (Downward plane segmentation)
    x1_mid = c1[0] + c1[2] * 0.50
    bg_ax.annotate("", xy=(x1_mid, c2[1] + c2[3] + 0.004), xytext=(x1_mid, c1[1] - 0.004),
                    arrowprops=vert_arrow_props, zorder=5)

    # -------------------------------------------------------------
    # D. Inset 3D subplots strictly above text area (Zero Occlusion)
    # -------------------------------------------------------------
    for idx in range(1, 6):
        cx, cy, cw, ch = cards[idx]
        # Text occupies [cy, cy + 0.050], 3D viewport strictly in [cy + 0.055, cy + ch - 0.008]
        ax_3d = fig.add_axes([cx + 0.004, cy + 0.055, cw - 0.008, ch - 0.063], projection="3d", zorder=4)
        render_step_content(ax_3d, idx, data, bounds)

    master_pdf = output_dir / "4_4_cuboid_fitting.pdf"
    master_png = output_dir / "4_4_cuboid_fitting.png"
    fig.savefig(master_pdf, format="pdf", dpi=600, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(master_png, format="png", dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Master figure generated successfully at: {master_pdf} (DPI=600)")

    # Auto-synchronize master figure to thesis directory for seamless compilation
    if sync_dir is not None:
        sync_dir.mkdir(parents=True, exist_ok=True)
        for fname in ["4_4_cuboid_fitting.pdf", "4_4_cuboid_fitting.png"]:
            src = output_dir / fname
            if src.exists():
                shutil.copy2(src, sync_dir / fname)
        print(f"Auto-synchronized master figure to thesis directory: {sync_dir / '4_4_cuboid_fitting.pdf'}")


def main():
    print("Sampling observed point cloud from GPBSF dataset surface generator...")
    pts, dims = sample_observation_from_dataset_source()

    print("Running fitting pipeline with GPBSF shape_fitting source algorithms...")
    data = run_pipeline_with_source_methods(pts)

    center = data["cube_center"]
    max_range = 0.118
    bounds = (
        center[0] - max_range, center[0] + max_range,
        center[1] - max_range, center[1] + max_range,
        center[2] - max_range * 0.85, center[2] + max_range * 0.85,
    )

    # Primary output directory inside the source code repository: code/GPBSF/figures
    output_dir = _GPBSF_ROOT / "figures"
    # Target directory for thesis LaTeX compilation: figures/chapter4
    sync_dir = _THESIS_ROOT / "figures" / "chapter4"

    print(f"Exporting visualization artifacts to source folder: {output_dir} (DPI=600)...")
    export_figures(data, bounds, output_dir, sync_dir=sync_dir)


if __name__ == "__main__":
    main()
