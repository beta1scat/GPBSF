import os
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import torch

from sam import SegmentAnythingModel, depth_to_pointcloud
from shape_fitting import FittingByBGS, _oriented_bounding_box
from shape_fitting.pointcloud import (
    generate_cone_points,
    generate_cube_points,
    generate_ellipsoid_points,
    pc_normalize,
    compute_trimmed_distance,
)

REPOSITORY_ROOT = Path(__file__).resolve().parent
MAMBA3D_ROOT = REPOSITORY_ROOT / "submodels" / "mamba3d"
if not MAMBA3D_ROOT.is_dir():
    raise FileNotFoundError(
        f"Mamba3D submodule is missing at {MAMBA3D_ROOT}. "
        "Run: git submodule update --init --recursive"
    )
sys.path.insert(0, str(MAMBA3D_ROOT))
from tools import builder
from utils import misc
from utils.config import cfg_from_yaml_file


def main() -> None:
    # 1. SAM model loading
    sam_checkpoint = "models/sam_vit_h_4b8939.pth"
    if not os.path.isfile(sam_checkpoint):
        raise FileNotFoundError(
            f"SAM checkpoint not found at {sam_checkpoint}. Please download it according to README."
        )
    sam_predictor = SegmentAnythingModel(sam_checkpoint)

    # 2. Classification model loading
    config = cfg_from_yaml_file("config/bgspcd.yaml")
    base_model = builder.model_builder(config.model)
    base_model.load_model_from_ckpt("models/ckpt-best.pth")
    base_model = base_model.cuda()
    base_model.eval()
    print("Classification model loaded successfully.")

    # 3. Fitting model loading
    fit_model = FittingByBGS()

    base_path = "data/real_flat"
    fx = 2422.5631472657747
    fy = 2422.603833233446
    cx = 962.2960960737917
    cy = 631.7893849597299

    image_path = os.path.join(base_path, "img")
    depth_path = os.path.join(base_path, "depth")
    if not os.path.isdir(image_path) or not os.path.isdir(depth_path):
        print(f"Dataset path not found at {base_path}. Exiting.")
        return

    files = sorted(os.listdir(image_path))
    for file in files:
        print(f"\nProcessing file: {file}")
        file_name = file.split(".")[0]
        idx = 0

        while True:
            image = cv2.imread(os.path.join(image_path, file))
            if image is None:
                break
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            depth_image = cv2.imread(
                os.path.join(depth_path, file_name + ".tiff"), cv2.IMREAD_UNCHANGED
            )

            cv2.namedWindow("ROI", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("ROI", 1280, 800)
            roi = cv2.selectROI("ROI", image)
            cv2.destroyWindow("ROI")

            if roi == (0, 0, 0, 0):
                print("No box was drawn (or closed). Next image...")
                break

            x, y, w, h = roi
            input_box = np.array([x, y, x + w, y + h])
            mask = sam_predictor.segment(image_rgb, input_box)

            segmented_depth_image = np.copy(depth_image)
            segmented_depth_image[~mask] = 0
            pointcloud = depth_to_pointcloud(segmented_depth_image, fx, fy, cx, cy)
            print(f"Point cloud count: {len(pointcloud)}")

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pc_normalize(pointcloud)[0])
            pcd.estimate_normals()
            pcd.orient_normals_towards_camera_location([0, 0, 800])
            o3d.visualization.draw_geometries([pcd], point_show_normal=True)

            # Classification
            if len(pcd.points) > 5000:
                pcd = pcd.farthest_point_down_sample(5000)
            _, ind = pcd.remove_radius_outlier(nb_points=20, radius=2)
            pcd = pcd.select_by_index(ind)

            pts = np.asarray(pcd.points)
            print(f"Cleaned point count: {len(pts)}")
            points = pts[:, :3].reshape(1, -1, 3)
            points = torch.from_numpy(points).float().cuda()
            points = misc.fps(points, 2048).cpu()
            pts_fps = pc_normalize(points.numpy()[0])[0]
            current_points = torch.from_numpy(pts_fps.reshape(1, -1, 3)).float().cuda()
            logits = base_model(current_points)

            pred = logits.argmax(-1).view(-1)
            pred_int = pred.cpu().numpy()[0]
            print(f"Predicted class: {pred_int}")

            # Fitting
            params_list = []
            pcd_fit_list = []
            min_dist_list = []
            obb_ratio_list = []
            print(f"Pcd points: {len(pcd.points)} points")

            if pred_int == 1:
                type_list = ["1", "01"]
            elif pred_int == 2:
                type_list = ["2"]
            else:
                type_list = ["0"]

            for tp in type_list:
                print(f"Testing primitive subtype: {tp}")
                try:
                    params = fit_model.fitting(pcd, tp)
                except Exception as e:
                    print(f"Fitting failed: {e}, continue...")
                    params_list.append(None)
                    pcd_fit_list.append(None)
                    min_dist_list.append(np.inf)
                    obb_ratio_list.append(np.inf)
                    continue

                if not params:
                    print("params is empty, continue...")
                    params_list.append(None)
                    pcd_fit_list.append(None)
                    min_dist_list.append(np.inf)
                    obb_ratio_list.append(np.inf)
                    continue

                params_list.append(params)
                if tp in ("0", "01"):
                    points = generate_cube_points(np.array(params[:3]) * 2, total_points=5000)
                elif tp in ("1", "11", "12", "13", "14"):
                    r1, r2, height, _ = params
                    points = generate_cone_points(
                        r_bottom=r2,
                        r_top_ratio=r1 / r2 if r2 > 1e-6 else 1.0,
                        height=height,
                        total_points=5000,
                    )
                elif tp == "2":
                    points = generate_ellipsoid_points(*params[:3], total_points=5000)
                else:
                    print(f"Category not supported: {tp}")
                    continue

                pcd_fit = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
                pcd_fit_list.append(pcd_fit)

                obb_pcd_size = np.array(sorted(_oriented_bounding_box(pcd).extent))
                obb_pcd_fit_size = np.array(sorted(_oriented_bounding_box(pcd_fit).extent))
                obb_diff = sum(abs(obb_pcd_fit_size - obb_pcd_size))
                obb_ratio_list.append(obb_diff)

                dist_pcd_fit = o3d.geometry.PointCloud(pcd_fit)
                dist_pcd_fit.transform(params[-1])
                mean_dist = compute_trimmed_distance(pcd, dist_pcd_fit, inlier_ratio=0.90)
                min_dist_list.append(mean_dist)
                print(f"Subtype {tp} trimmed distance: {mean_dist:.4f} mm")

                if mean_dist <= 2.0:
                    print(f"Early exit triggered for subtype {tp} (trimmed_dist={mean_dist:.4f} <= 2.0 mm)")
                    break

            min_idx = np.argmin(min_dist_list)
            pcd_fit = pcd_fit_list[min_idx]
            params = params_list[min_idx]
            if params is None:
                print("Fitting failed for all types.")
                continue

            pcd_fit.transform(params[-1])
            o3d.visualization.draw_geometries([pcd, pcd_fit], point_show_normal=True)
            idx += 1


if __name__ == "__main__":
    main()
