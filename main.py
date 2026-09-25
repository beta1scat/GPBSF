"""GPBSF: Geometric Primitive-Based Shape Fitting.

Provides standalone 3D point cloud fitting and optional interactive RGB-D demo.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

from shape_fitting import FittingByBGS, _oriented_bounding_box
from shape_fitting.pointcloud import (
    generate_cone_points,
    generate_cube_points,
    generate_ellipsoid_points,
    pc_normalize,
    compute_trimmed_distance,
)


def load_point_cloud(path: str | Path) -> o3d.geometry.PointCloud:
    """Load point cloud from .ply, .pcd, .xyz, .txt, or .npz file."""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Point cloud file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in (".ply", ".pcd", ".xyz", ".pts"):
        pcd = o3d.io.read_point_cloud(str(path))
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            if "points" in data:
                pts = data["points"]
            else:
                pts = data[list(data.keys())[0]]
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(pts, dtype=np.float64)))
    elif suffix in (".txt", ".csv"):
        pts = np.loadtxt(str(path), delimiter="," if suffix == ".csv" else None)
        if pts.shape[1] > 3:
            pts = pts[:, :3]
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts.astype(np.float64)))
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    if not pcd.has_normals():
        pcd.estimate_normals()
    return pcd


def generate_primitive_surface(cls_code: str, params: list, n_points: int = 5000) -> o3d.geometry.PointCloud:
    """Generate synthetic point cloud on the fitted primitive surface."""
    if cls_code == "0":
        dims = np.array(params[:3], dtype=np.float64) * 2.0  # half-extents to full extents
        pts = generate_cube_points(dims, total_points=n_points)
    elif cls_code == "1":
        r1, r2, height, _ = params
        pts = generate_cone_points(
            r_bottom=r2,
            r_top_ratio=r1 / r2 if r2 > 1e-6 else 1.0,
            height=height,
            total_points=n_points,
        )
    elif cls_code == "2":
        a, b, c, _ = params
        pts = generate_ellipsoid_points(a, b, c, total_points=n_points)
    else:
        raise ValueError(f"Unsupported primitive code: {cls_code}")

    fit_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    T_mat = np.asarray(getattr(params[-1], "A", params[-1]), dtype=np.float64)
    fit_pcd.transform(T_mat)
    return fit_pcd


def fit_point_cloud(
    pcd: o3d.geometry.PointCloud,
    cls_code: str = "auto",
    tau_cone: float = 2.0,
    visualize: bool = False,
):
    """Fit geometric primitive(s) to a point cloud and return the best fit."""
    fitter = FittingByBGS()

    if cls_code == "auto":
        candidate_codes = ["0", "1", "2"]
    else:
        candidate_codes = [cls_code]

    best_code = None
    best_params = None
    best_pcd_fit = None
    best_residual = float("inf")
    results = {}

    type_name_map = {
        "0": "Cuboid",
        "1": "Frustum Cone (Adaptive)",
        "2": "Ellipsoid",
    }

    for code in candidate_codes:
        try:
            params = fitter.fitting(pcd, cls=code, visual=False, tau_cone=tau_cone)
        except Exception as exc:
            results[code] = {"status": f"failed ({exc})", "residual_mm": float("inf")}
            continue

        if not params or len(params) != 4:
            results[code] = {"status": "failed (empty parameters)", "residual_mm": float("inf")}
            continue

        pcd_fit = generate_primitive_surface(code, params)
        residual = compute_trimmed_distance(pcd, pcd_fit, inlier_ratio=0.90)

        results[code] = {
            "name": type_name_map.get(code, code),
            "status": "success",
            "method": fitter.last_method,
            "fallback": fitter.last_fallback_triggered,
            "residual_mm": residual,
            "params": params,
            "pcd_fit": pcd_fit,
        }

        if residual < best_residual:
            best_residual = residual
            best_code = code
            best_params = params
            best_pcd_fit = pcd_fit

        if cls_code == "auto" and residual <= tau_cone:
            print(f"Early exit: {type_name_map.get(code, code)} residual {residual:.3f} mm <= tau ({tau_cone:.1f} mm)")
            break

    print("\n--- Primitive Fitting Summary ---")
    for code, info in results.items():
        res_str = f"{info['residual_mm']:.3f} mm" if np.isfinite(info["residual_mm"]) else "N/A"
        print(f"  [{code}] {type_name_map.get(code, code):<25}: Status={info['status']:<10} 90% Trimmed Resid={res_str}")

    if best_params is None:
        print("\nAll fitting hypotheses failed.")
        return None

    print(f"\nOptimal Primitive: [{best_code}] {type_name_map.get(best_code, best_code)}")
    print(f"  Method Used: {results[best_code]['method']}")
    print(f"  Fallback: {results[best_code]['fallback']}")
    print(f"  90% Trimmed Distance: {best_residual:.3f} mm")

    T = best_params[-1]
    T_mat = np.asarray(getattr(T, "A", T), dtype=np.float64)
    print("  Estimated Parameters:")
    if best_code == "0":
        print(f"    Dimensions (L, W, H): {best_params[0]*2:.3f}, {best_params[1]*2:.3f}, {best_params[2]*2:.3f} (half-extents: {best_params[0]:.3f}, {best_params[1]:.3f}, {best_params[2]:.3f})")
    elif best_code == "1":
        print(f"    Top Radius: {best_params[0]:.3f}, Bottom Radius: {best_params[1]:.3f}, Height: {best_params[2]:.3f}")
    elif best_code == "2":
        print(f"    Semi-axes (a, b, c): {best_params[0]:.3f}, {best_params[1]:.3f}, {best_params[2]:.3f}")
    print(f"    Center (x, y, z): {T_mat[:3, 3].tolist()}")

    if visualize:
        best_pcd_fit.paint_uniform_color([0.2, 0.7, 0.2])  # green
        pcd.paint_uniform_color([0.2, 0.4, 0.8])  # blue
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=max(best_params[:3]) * 1.5)
        coord.transform(T_mat)
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=max(best_params[:3]) * 0.5)
        o3d.visualization.draw_geometries(
            [pcd, best_pcd_fit, coord, origin],
            window_name=f"GPBSF Fitting: {type_name_map.get(best_code, best_code)}",
        )

    return {
        "primitive_code": best_code,
        "primitive_name": type_name_map.get(best_code, best_code),
        "parameters": best_params,
        "residual_mm": best_residual,
        "pose": T_mat,
    }


def run_interactive_demo(
    image_dir: str = "data/real_flat/img",
    depth_dir: str = "data/real_flat/depth",
    sam_checkpoint: str = "models/sam_vit_h_4b8939.pth",
    mamba_config: str = "config/bgspcd.yaml",
    mamba_checkpoint: str = "models/ckpt-best.pth",
):
    """Run the interactive SAM RGB-D segmentation and fitting pipeline."""
    import cv2
    import torch
    from sam import SegmentAnythingModel, depth_to_pointcloud

    if not os.path.isfile(sam_checkpoint):
        raise FileNotFoundError(f"SAM checkpoint not found at {sam_checkpoint}.")

    sam_predictor = SegmentAnythingModel(sam_checkpoint)

    # Classification model loading
    repository_root = Path(__file__).resolve().parent
    mamba3d_root = repository_root / "submodels" / "mamba3d"
    if not mamba3d_root.is_dir():
        raise FileNotFoundError(f"Mamba3D submodule missing at {mamba3d_root}. Run git submodule update --init --recursive")
    sys.path.insert(0, str(mamba3d_root))
    from tools import builder
    from utils import misc
    from utils.config import cfg_from_yaml_file

    config = cfg_from_yaml_file(mamba_config)
    base_model = builder.model_builder(config.model)
    base_model.load_model_from_ckpt(mamba_checkpoint)
    base_model = base_model.cuda().eval()
    print("Mamba3D and SAM models loaded successfully.")

    fit_model = FittingByBGS()
    fx, fy, cx, cy = 2422.5631, 2422.6038, 962.2961, 631.7894

    files = sorted(os.listdir(image_dir))
    for file in files:
        if not file.lower().endswith((".png", ".jpg", ".bmp")):
            continue
        file_name = Path(file).stem
        image = cv2.imread(os.path.join(image_dir, file))
        if image is None:
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        depth_file = os.path.join(depth_dir, file_name + ".tiff")
        if not os.path.isfile(depth_file):
            depth_file = os.path.join(depth_dir, file_name + ".png")
        depth_image = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
        if depth_image is None:
            continue

        cv2.namedWindow("ROI", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("ROI", 1280, 800)
        roi = cv2.selectROI("ROI", image)
        cv2.destroyWindow("ROI")
        if roi == (0, 0, 0, 0):
            break

        x, y, w, h = roi
        mask = sam_predictor.segment(image_rgb, np.array([x, y, x + w, y + h]))
        segmented_depth = np.copy(depth_image)
        segmented_depth[~mask] = 0
        pointcloud = depth_to_pointcloud(segmented_depth, fx, fy, cx, cy)
        if len(pointcloud) < 50:
            print("Too few points extracted.")
            continue

        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc_normalize(pointcloud)[0]))
        pcd.estimate_normals()
        pcd.orient_normals_towards_camera_location([0, 0, 800])

        if len(pcd.points) > 5000:
            pcd = pcd.farthest_point_down_sample(5000)
        _, ind = pcd.remove_radius_outlier(nb_points=20, radius=2)
        pcd = pcd.select_by_index(ind)

        pts = np.asarray(pcd.points)[:, :3].reshape(1, -1, 3)
        pts_tensor = torch.from_numpy(pts).float().cuda()
        pts_fps = misc.fps(pts_tensor, 2048).cpu().numpy()[0]
        pts_fps = pc_normalize(pts_fps)[0]
        current_points = torch.from_numpy(pts_fps.reshape(1, -1, 3)).float().cuda()
        logits = base_model(current_points)
        pred_int = int(logits.argmax(-1).view(-1).cpu().numpy()[0])
        print(f"Predicted class: {pred_int}")

        fit_point_cloud(pcd, cls_code=str(pred_int), visualize=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcd", type=str, default=None, help="Path to input 3D point cloud file (.ply, .pcd, .npz, .txt)")
    parser.add_argument("--cls", type=str, default="auto", choices=("auto", "0", "1", "2"), help="Primitive class code (default: auto)")
    parser.add_argument("--tau-cone", type=float, default=2.0, help="Cone adaptive fitting residual threshold in mm (default: 2.0)")
    parser.add_argument("--visualize", action="store_true", help="Visualize fitting result with Open3D")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive SAM + RGB-D demo")
    args = parser.parse_args()

    if args.interactive:
        run_interactive_demo()
    elif args.pcd:
        pcd = load_point_cloud(args.pcd)
        fit_point_cloud(pcd, cls_code=args.cls, tau_cone=args.tau_cone, visualize=args.visualize)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
