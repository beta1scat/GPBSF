#!/usr/bin/env python3
"""
Evaluate Chamfer Distance (CD) and Residual metrics for each candidate principal axis
in the truncated cone (frustum) fitting pipeline (Figure 4.5).
"""

from __future__ import annotations
import sys
from pathlib import Path

# Add code/GPBSF root and scripts dir to sys.path
_SCRIPTS_DIR = Path(__file__).resolve().parent
_GPBSF_ROOT = _SCRIPTS_DIR.parent
for _p in [str(_GPBSF_ROOT), str(_SCRIPTS_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import open3d as o3d
from spatialmath import SE3, SO3

try:
    from visualize_fig4_5_frustum import sample_frustum_observation
except ImportError:
    from scripts.visualize_fig4_5_frustum import sample_frustum_observation
from shape_fitting.fitting import (
    fit_frustum_cone_pca,
    fit_frustum_cone_normal,
    fit_frustum_cone_obb,
    compute_cone_residual,
)
from shape_fitting.pointcloud import generate_cone_points
from experiments.fitting.evaluate import _chamfer


def main():
    print("=" * 80)
    print("圆锥台拟合（图4.5）5类主轴假说的倒角距离 (Chamfer Distance) 与拟合残差评估")
    print("=" * 80)

    pts, meta = sample_frustum_observation()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=30))

    # Downsample dense point clouds to ~2048 points matching fit_frustum_cone_adaptive
    n_pts = len(pcd.points)
    step = max(1, n_pts // 2048) if n_pts > 2048 else 1
    pcd_fit = pcd.uniform_down_sample(every_k_points=step)

    # 1. Ground truth surface points
    r_bottom_gt = meta["r_bottom"]
    r_top_gt = meta["r_top"]
    h_gt = meta["height"]
    pts_gt_local = generate_cone_points(
        r_bottom=r_bottom_gt,
        r_top_ratio=r_top_gt / r_bottom_gt,
        height=h_gt,
        delta=0.0,
        points_density=0,
        total_points=4096,
    )
    R_gt = meta["R_gt"]
    c_gt = meta["center_gt"]
    pts_gt_world = (R_gt @ pts_gt_local.T).T + c_gt

    # 2. Candidate hypotheses (ordered by ascending computational runtime)
    hypotheses = [
        ("pca_z0", "PCA 第一主轴 (最大方差, ~0.1ms)", lambda: fit_frustum_cone_pca(pcd_fit, z_dir=0)),
        ("pca_z2", "PCA 第三主轴 (最小方差, ~0.1ms)", lambda: fit_frustum_cone_pca(pcd_fit, z_dir=2)),
        ("obb", "有向包围盒主轴 (OBB, ~0.8ms)", lambda: fit_frustum_cone_obb(pcd_fit)),
        ("normal", "端面平面引导 (Plane RANSAC, ~5ms)", lambda: fit_frustum_cone_normal(pcd_fit, plane_t=0.005, normal_t=0.02, use_plane_normal=True)),
        ("normal_ransac", "侧面法向 RANSAC 轴 (~30ms)", lambda: fit_frustum_cone_normal(pcd_fit, plane_t=0.01, normal_t=0.02, use_plane_normal=False)),
    ]

    results = []

    for key, desc, solver in hypotheses:
        try:
            r1, r2, h, T = solver()
        except Exception as e:
            results.append({
                "key": key,
                "desc": desc,
                "error": str(e),
                "surface_cd_mm": float("inf"),
                "obs_cd_mm": float("inf"),
                "residual_mm": float("inf"),
            })
            continue

        if h <= 1e-4 or r1 <= 1e-4 or r2 <= 1e-4:
            results.append({
                "key": key,
                "desc": desc,
                "error": "Degenerate geometry",
                "surface_cd_mm": float("inf"),
                "obs_cd_mm": float("inf"),
                "residual_mm": float("inf"),
            })
            continue

        # Generate predicted surface points
        pts_pred_local = generate_cone_points(
            r_bottom=r2,
            r_top_ratio=r1 / r2 if r2 > 1e-6 else 1.0,
            height=h,
            delta=0.0,
            points_density=0,
            total_points=4096,
        )
        T_mat = T.A if hasattr(T, "A") else np.asarray(T, dtype=np.float64)
        R_fit = T_mat[:3, :3]
        t_fit = T_mat[:3, 3]
        pts_pred_world = (R_fit @ pts_pred_local.T).T + t_fit

        # Metric 1: Surface Chamfer Distance (vs Ground Truth full surface)
        surface_cd_m = _chamfer(pts_pred_world, pts_gt_world)
        surface_cd_mm = surface_cd_m * 1000.0

        # Metric 2: Observed Chamfer Distance (vs Observed partial point cloud)
        obs_cd_m = _chamfer(np.asarray(pcd.points), pts_pred_world)
        obs_cd_mm = obs_cd_m * 1000.0

        # Metric 3: Fitting Residual (Robust trimmed distance used by early-exit check)
        residual_m = compute_cone_residual(pcd, r1, r2, h, T)
        residual_mm = residual_m * 1000.0

        axis_vec = R_fit[:, 2]

        results.append({
            "key": key,
            "desc": desc,
            "error": None,
            "r1_mm": r1 * 1000.0,
            "r2_mm": r2 * 1000.0,
            "h_mm": h * 1000.0,
            "axis": axis_vec,
            "surface_cd_mm": surface_cd_mm,
            "obs_cd_mm": obs_cd_mm,
            "residual_mm": residual_mm,
        })

    # Print summary table
    print(f"\n真值参数 (Ground Truth):")
    print(f"  - 顶面半径 r_top:    {r_top_gt * 1000.0:.2f} mm")
    print(f"  - 底面半径 r_bottom: {r_bottom_gt * 1000.0:.2f} mm")
    print(f"  - 高度 height:       {h_gt * 1000.0:.2f} mm")
    print(f"  - 真值回转主轴:      [{R_gt[0, 2]:.4f}, {R_gt[1, 2]:.4f}, {R_gt[2, 2]:.4f}]\n")

    print(f"{'主轴假说策略':<16} | {'顶面/底面半径 (mm)':<18} | {'高度 (mm)':<10} | {'真值双向 CD (mm)':<16} | {'观测双向 CD (mm)':<16} | {'鲁棒截断残差 (mm)':<16}")
    print("-" * 106)

    min_res = min(r["residual_mm"] for r in results if r["error"] is None)

    for r in results:
        if r["error"]:
            print(f"{r['key']:<16} | 求解失败: {r['error']}")
            continue
        mark = " <-- 全局最优选定" if abs(r["residual_mm"] - min_res) < 1e-4 else ""
        radii_str = f"{r['r1_mm']:.1f} / {r['r2_mm']:.1f}"
        print(f"{r['key']:<16} | {radii_str:<18} | {r['h_mm']:<10.1f} | {r['surface_cd_mm']:<16.2f} | {r['obs_cd_mm']:<16.2f} | {r['residual_mm']:<16.2f}{mark}")

    print("\n指标定义说明:")
    print("  1. 真值双向 CD (Surface Chamfer Distance): 拟合所得全表面与真实几何全表面之间的双向 Chamfer 距离 (严格评估模型重建精度)。")
    print("  2. 观测双向 CD (Observed Chamfer Distance): 观测单视角点云与拟合几何表面之间的双向 Chamfer 距离。")
    print("  3. 鲁棒截断残差 (Trimmed Residual): 算法内部门限检验所依据的单向截断几何距离 (即 compute_cone_residual，门限 tau_m = tau_cone * 1e-3 进行比对)。")


if __name__ == "__main__":
    main()
