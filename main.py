import os
import sys
from pathlib import Path

from sam import *
from shape_fitting import FittingByBGS, _oriented_bounding_box
from shape_fitting.pointcloud import (
    generate_cone_points,
    generate_cube_points,
    generate_ellipsoid_points,
    pc_normalize,
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
from utils.config import *
from utils import misc


'''SAM model loading'''
sam_predictor = SegmentAnythinModel("models/sam_vit_h_4b8939.pth")

'''Classification model loading'''
config = cfg_from_yaml_file("config/bgspcd.yaml")
base_model = builder.model_builder(config.model)
base_model.load_model_from_ckpt("models/ckpt-best.pth")

base_model = base_model.cuda()
base_model.eval()
print("Classification model loaded successfully.")


'''Fitting model loading'''
fit_model = FittingByBGS()

base_path = "data/real_flat"
fx = 2422.5631472657747
fy = 2422.603833233446
cx = 962.2960960737917
cy = 631.7893849597299
# # RealSense 415
# fx, fy = [895.176, 895.176]
# cx, cy = [630.254, 374.059]
image_path = os.path.join(base_path, "img")
depth_path = os.path.join(base_path, "depth")
seg_path = os.path.join(base_path, "segments")
pcd_path = os.path.join(base_path, "pcd")
masks_path = os.path.join(base_path, "masks")
files = sorted(os.listdir(image_path))
for file in files:
    print(file)
    fileName = file.split('.')[0]
    imageIsOk = False
    idx = 0
    while not imageIsOk:
        image = cv2.imread(os.path.join(image_path, file))
        image_bak = image.copy()
        print(f"image size: {image.shape}")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        depth_image = cv2.imread(os.path.join(depth_path, fileName + ".tiff"), cv2.IMREAD_UNCHANGED)
        image_bak = image.copy()
        cv2.namedWindow("ROI", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("ROI", 1280, 800)
        roi = cv2.selectROI("ROI", image_bak)
        cv2.destroyWindow("ROI")
        if roi == (0, 0, 0, 0):
            print("No box was drawn.")
            break
        x, y, w, h = roi
        input_box = np.array([x, y, x + w, y + h])
        mask = sam_predictor.segment(image_rgb, input_box)
        segmented_image = np.zeros_like(image_rgb)
        segmented_image[mask] = image_rgb[mask]
        segmented_depth_image = np.copy(depth_image)
        segmented_depth_image[~mask] = 0
        pointcloud = depth_to_pointcloud(segmented_depth_image, fx, fy, cx, cy)
        print(f"pts: {len(pointcloud)}")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_normalize(pointcloud)[0])
        pcd.estimate_normals()
        camera = [0,0,800]
        pcd.orient_normals_towards_camera_location(camera)
        o3d.visualization.draw_geometries([pcd], point_show_normal=True)
        '''Clasification'''
        if len(pcd.points) > 5000:
            pcd = pcd.farthest_point_down_sample(5000)
        cl, ind = pcd.remove_radius_outlier(nb_points=20, radius=2)

        pcd = pcd.select_by_index(ind)
        pts = np.asarray(pcd.points)
        print(f"pts: {len(pts)}")
        points = pts[:, :3].reshape(1, -1, 3)
        points = torch.from_numpy(points).float()
        points = points.cuda()
        points = misc.fps(points, 2048)
        points = points.cpu()
        pts_fps = pc_normalize(points.numpy()[0])[0]
        current_points = torch.from_numpy(pts_fps.reshape(1, -1, 3)).float()
        current_points = current_points.cuda()
        logits = base_model(current_points)

        pred = logits.argmax(-1).view(-1)
        pred_int = pred.cpu().numpy()[0]
        print(f"Predicted class: {pred_int}")


        '''Fitting'''
        params_list = []
        pcd_fit_list = []
        min_dist_list = []
        obb_ratio_list = []
        pcd = pcd
        print(f"Pcd points: {len(pcd.points)} points")
        if pred_int == 1:
            type_list = ['01', '11', '12', '13', '14']
        elif pred_int == 2:
            type_list = ['2']
        else:
            type_list = ['0']
        for tp in type_list:
            print("current type: " + tp)
            try:
                params = fit_model.fitting(pcd, tp)
            except Exception as e:
                print(f"Fitting failed： {e}, continue...")
                params_list.append(None)
                pcd_fit_list.append(None)
                min_dist_list.append(np.inf)
                obb_ratio_list.append(np.inf)
                continue
            if params == []:
                print("params is [], continue...")
                params_list.append(None)
                pcd_fit_list.append(None)
                min_dist_list.append(np.inf)
                obb_ratio_list.append(np.inf)
                continue
            params_list.append(params)
            if tp == '0' or tp == '01':
                points = generate_cube_points(np.array(params[:3]) * 2, total_points=5000) # size=(10, 10, 10)
            elif tp == '1' or tp == '11' or tp == '12' or tp == '13' or tp == '14':
                r1, r2, height, _  = params
                points = generate_cone_points(r_bottom=r2, r_top_ratio=r1/r2, height=height, total_points=5000) # r_bottom=10, r_top_ratio=0.5, height=20,
            elif tp == '2':
                points = generate_ellipsoid_points(*params[:3], total_points=5000) # a=10, b=10, c=10
            else:
                print("Category not supported.")
            pcd_fit = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
            print(f"Fitting points: {len(pcd_fit.points)} points")
            pcd_fit_list.append(pcd_fit)
            obb_pcd_size = np.array(sorted(_oriented_bounding_box(pcd).extent))
            obb_pcd_fit_size = np.array(sorted(_oriented_bounding_box(pcd_fit).extent))
            obb_ratio_list.append(sum(abs(obb_pcd_fit_size - obb_pcd_size)))
            print(f"obb_ratio: {sum(abs(obb_pcd_fit_size - obb_pcd_size))}")
            dist_pcd_fit = o3d.geometry.PointCloud(pcd_fit)
            dist_pcd_fit.transform(params[-1])
            min_dist1 = pcd.compute_point_cloud_distance(dist_pcd_fit)
            min_dist2 = dist_pcd_fit.compute_point_cloud_distance(pcd)
            print(f"sum(min_dist): {np.mean(min_dist1) + np.mean(min_dist2)}")
            min_dist_list.append(np.mean(min_dist1) + np.mean(min_dist2))
        min_idx = np.argmin(min_dist_list)
        pcd_fit = pcd_fit_list[min_idx]
        params = params_list[min_idx]
        if params == None:
            print("Fitting failed for all types.")
            continue
        pcd_fit.transform(params[-1])
        o3d.visualization.draw_geometries([pcd, pcd_fit], point_show_normal=True)

        idx = idx + 1



