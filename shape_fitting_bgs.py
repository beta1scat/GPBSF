# import sys
# from PyQt5 import QtWidgets
# QtWidgets.QApplication(sys.argv)
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..\..')))

# import cv2
# import json
# import importlib
import numpy as np
import open3d as o3d
# import torch

from pcd_utils import *
from fit_bgspcd_noros import *
# from fit_bgspcd_noros import fit_cuboid_obb, fit_frustum_cone_obb, fit_ellipsoid, fit_cuboid_obb2, \
#                              fit_frustum_cone_normal, generate_cone_points, generate_ellipsoid_points
# import pointnet2_cls_ssg as model
def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc, m

class FittingByBGS:
    def fitting(self, pcd, cls='0', visual=False):
        if len(pcd.points) > 5000:
            pcd_fit = pcd.farthest_point_down_sample(5000)
            # cl, ind = pcd_fps.remove_statistical_outlier(nb_neighbors=100, std_ratio=2.0)
            # pcd_fit = pcd_fps.select_by_index(ind)
        else:
            pcd_fit = pcd
        params = []
        if cls == '0':
            a, b, c, T_cube = fit_cuboid_obb2(pcd_fit, dist_threshold=2)
            coord_frame_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            coord_frame.transform(T_cube)
            obb = pcd_fit.get_minimal_oriented_bounding_box()
            cube = o3d.geometry.TriangleMesh.create_box(width=a*2, height=b*2, depth=c*2)
            cube.translate(-1*np.array([a,b,c]))
            cube.transform(T_cube)
            fit_cube_pcd = cube.sample_points_poisson_disk(5000)
            fit_cube_pcd.paint_uniform_color([0, 0, 1])
            if visual:
                o3d.visualization.draw_geometries([fit_cube_pcd, obb, pcd, coord_frame, coord_frame_origin])
            params = [a, b, c, T_cube]
            # with open(os.path.join(class_path, f"{fileName}.json"), "w") as outfile:
            #     json.dump({"pred_type": pred_choice.item(), "input_type": cls, "size": [a,b,c], "T": T_cube.A.tolist()}, outfile)
            # o3d.io.write_point_cloud(os.path.join(fit_pcd_path, f"{fileName}.ply"), fit_cube_pcd)
        elif cls == '1':
            # r1, r2, height, T_cone = fit_frustum_cone_normal(pcd_fit, plane_t=0.001, normal_t=0.02)
            r1, r2, height, T_cone = fit_frustum_cone_normal(pcd_fit, plane_t=0.005, normal_t=0.02, use_plane_normal=True)
            fit_cone_points = generate_cone_points(r_bottom=r2, r_top_ratio=r1/r2, height=height, delta=0.0, points_density=0, total_points=5000)
            fit_cone_pcd = o3d.geometry.PointCloud()
            fit_cone_pcd.points = o3d.utility.Vector3dVector(fit_cone_points)
            fit_cone_pcd.paint_uniform_color([0, 0, 1])
            fit_cone_pcd.transform(T_cone)
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            coord_frame.transform(T_cone)
            coord_frame_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
            if visual:
                o3d.visualization.draw_geometries([fit_cone_pcd, pcd, coord_frame, coord_frame_origin], point_show_normal=False)
            params = [r1, r2, height, T_cone]
            # with open(os.path.join(class_path, f"{fileName}.json"), "w") as outfile:
            #     json.dump({"pred_type": pred_choice.item(), "input_type": cls, "size": [r1, r2, height], "T": T_cone.A.tolist()}, outfile)
            # o3d.io.write_point_cloud(os.path.join(fit_pcd_path, f"{fileName}.ply"), fit_cone_pcd)
        elif cls == '2':
            try:
                a, b, c, T_ellip = fit_ellipsoid(pcd_fit, num_it=500, t=0.01)
            except Exception as e:
                return params
            # a, b, c, T_ellip = fit_ellipsoid(pcd_fit, 0.1)
            fit_ellipsoid_points = generate_ellipsoid_points(a, b, c, total_points=5000)
            fit_ellipsoid_pcd = o3d.geometry.PointCloud()
            fit_ellipsoid_pcd.points = o3d.utility.Vector3dVector(fit_ellipsoid_points)
            fit_ellipsoid_pcd.paint_uniform_color([0, 0, 1])
            fit_ellipsoid_pcd.transform(T_ellip)
            coord_frame_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            coord_frame.transform(T_ellip)
            if visual:
                o3d.visualization.draw_geometries([fit_ellipsoid_pcd, pcd_fit, coord_frame, coord_frame_origin], point_show_normal=False)
            params = [a, b ,c ,T_ellip]
            # with open(os.path.join(class_path, f"{fileName}.json"), "w") as outfile:
            #     json.dump({"pred_type": pred_choice.item(), "input_type": cls, "size": [a, b, c], "T": T_ellip.A.tolist()}, outfile)
            # o3d.io.write_point_cloud(os.path.join(fit_pcd_path, f"{fileName}.ply"), fit_ellipsoid_pcd)
        elif cls == '01':
            a, b, c, T_cube = fit_cuboid_obb(pcd_fit)
            coord_frame_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            coord_frame.transform(T_cube)
            obb = pcd_fit.get_minimal_oriented_bounding_box()
            cube = o3d.geometry.TriangleMesh.create_box(width=a*2, height=b*2, depth=c*2)
            cube.translate(-1*np.array([a,b,c]))
            cube.transform(T_cube)
            fit_cube_pcd = cube.sample_points_poisson_disk(5000)
            fit_cube_pcd.paint_uniform_color([0, 0, 1])
            if visual:
                o3d.visualization.draw_geometries([fit_cube_pcd, obb, pcd, coord_frame, coord_frame_origin])
            params = [a, b, c, T_cube]
            # with open(os.path.join(class_path, f"{fileName}.json"), "w") as outfile:
            #     json.dump({"pred_type": pred_choice.item(), "input_type": cls, "size": [a,b,c], "T": T_cube.A.tolist()}, outfile)
            # o3d.io.write_point_cloud(os.path.join(fit_pcd_path, f"{fileName}.ply"), fit_cube_pcd)
        elif cls == '11':
            r1, r2, height, T_cone = fit_frustum_cone_obb(pcd_fit)
            fit_cone_points = generate_cone_points(r_bottom=r2, r_top_ratio=r1/r2, height=height, delta=0.0, points_density=0, total_points=5000)
            fit_cone_pcd = o3d.geometry.PointCloud()
            fit_cone_pcd.points = o3d.utility.Vector3dVector(fit_cone_points)
            fit_cone_pcd.paint_uniform_color([0, 0, 1])
            fit_cone_pcd.transform(T_cone)
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            coord_frame.transform(T_cone)
            coord_frame_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
            if visual:
                o3d.visualization.draw_geometries([fit_cone_pcd, pcd, coord_frame, coord_frame_origin], point_show_normal=False)
            params = [r1, r2, height, T_cone]
            # with open(os.path.join(class_path, f"{fileName}.json"), "w") as outfile:
            #     json.dump({"pred_type": pred_choice.item(), "input_type": cls, "size": [r1, r2, height], "T": T_cone.A.tolist()}, outfile)
            # o3d.io.write_point_cloud(os.path.join(fit_pcd_path, f"{fileName}.ply"), fit_cone_pcd)
        elif cls == '12':
            # r1, r2, height, T_cone = fit_frustum_cone_normal(pcd_fit, plane_t=0.001, normal_t=0.02)
            r1, r2, height, T_cone = fit_frustum_cone_normal(pcd_fit, plane_t=0.01, normal_t=0.02, use_plane_normal=False)
            fit_cone_points = generate_cone_points(r_bottom=r2, r_top_ratio=r1/r2, height=height, delta=0.0, points_density=0, total_points=5000)
            fit_cone_pcd = o3d.geometry.PointCloud()
            fit_cone_pcd.points = o3d.utility.Vector3dVector(fit_cone_points)
            fit_cone_pcd.paint_uniform_color([0, 0, 1])
            fit_cone_pcd.transform(T_cone)
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            coord_frame.transform(T_cone)
            coord_frame_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
            if visual:
                o3d.visualization.draw_geometries([fit_cone_pcd, pcd, coord_frame, coord_frame_origin], point_show_normal=False)
            params = [r1, r2, height, T_cone]
            # with open(os.path.join(class_path, f"{fileName}.json"), "w") as outfile:
            #     json.dump({"pred_type": pred_choice.item(), "input_type": cls, "size": [r1, r2, height], "T": T_cone.A.tolist()}, outfile)
            # o3d.io.write_point_cloud(os.path.join(fit_pcd_path, f"{fileName}.ply"), fit_cone_pcd)
        elif cls == '13':
            # r1, r2, height, T_cone = fit_frustum_cone_normal(pcd_fit, plane_t=0.001, normal_t=0.02)
            r1, r2, height, T_cone = fit_frustum_cone_pca(pcd_fit, use_poly=False, is_debug=False, z_dir=0)
            fit_cone_points = generate_cone_points(r_bottom=r2, r_top_ratio=r1/r2, height=height, delta=0.0, points_density=0, total_points=5000)
            fit_cone_pcd = o3d.geometry.PointCloud()
            fit_cone_pcd.points = o3d.utility.Vector3dVector(fit_cone_points)
            fit_cone_pcd.paint_uniform_color([0, 0, 1])
            fit_cone_pcd.transform(T_cone)
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            coord_frame.transform(T_cone)
            coord_frame_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
            if visual:
                o3d.visualization.draw_geometries([fit_cone_pcd, pcd, coord_frame, coord_frame_origin], point_show_normal=False)
            params = [r1, r2, height, T_cone]
            # with open(os.path.join(class_path, f"{fileName}.json"), "w") as outfile:
            #     json.dump({"pred_type": pred_choice.item(), "input_type": cls, "size": [r1, r2, height], "T": T_cone.A.tolist()}, outfile)
            # o3d.io.write_point_cloud(os.path.join(fit_pcd_path, f"{fileName}.ply"), fit_cone_pcd)
        elif cls == '14':
            # r1, r2, height, T_cone = fit_frustum_cone_normal(pcd_fit, plane_t=0.001, normal_t=0.02)
            r1, r2, height, T_cone = fit_frustum_cone_pca(pcd_fit, use_poly=False, is_debug=False, z_dir=2)
            fit_cone_points = generate_cone_points(r_bottom=r2, r_top_ratio=r1/r2, height=height, delta=0.0, points_density=0, total_points=5000)
            fit_cone_pcd = o3d.geometry.PointCloud()
            fit_cone_pcd.points = o3d.utility.Vector3dVector(fit_cone_points)
            fit_cone_pcd.paint_uniform_color([0, 0, 1])
            fit_cone_pcd.transform(T_cone)
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            coord_frame.transform(T_cone)
            coord_frame_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
            if visual:
                o3d.visualization.draw_geometries([fit_cone_pcd, pcd, coord_frame, coord_frame_origin], point_show_normal=False)
            params = [r1, r2, height, T_cone]
            # with open(os.path.join(class_path, f"{fileName}.json"), "w") as outfile:
            #     json.dump({"pred_type": pred_choice.item(), "input_type": cls, "size": [r1, r2, height], "T": T_cone.A.tolist()}, outfile)
            # o3d.io.write_point_cloud(os.path.join(fit_pcd_path, f"{fileName}.ply"), fit_cone_pcd)
        else:
            print(f"类型错误")
        return params


if __name__ == "__main__":
    # base_path = "/root/ros_ws/src/data/saved/scatter_pll"
    # base_path = "/root/ros_ws/src/data/sim/scatter_pll"
    # base_path = "/root/ros_ws/src/data/00000"
    # base_path = "/root/ros_ws/src/data/sim_flat"
    # base_path = "/root/ros_ws/src/data/sim_clutter"
    # base_path = "/root/ros_ws/src/data/real_flat"
    base_path = "/root/code/Mamba3D/data/results/real_flat"

    pcd_path = os.path.join(base_path, "pcd")
    fit_pcd_path = os.path.join(base_path, "fit")
    class_path = os.path.join(base_path, "classes")

    fittingModel = FittingByBGS()
    # pcd = o3d.io.read_point_cloud(os.path.join(pcd_path, ""))
    # o3d.visualization.draw_geometries([pcd], point_show_normal=False)
    # params = fittingModel.fitting(pcd, cls='0')
    # print(params)
    files = sorted(os.listdir(pcd_path))
    start_idx = np.where(np.array(files) == '0_0.ply')[0]
    files = files[start_idx[0]:]
    for file in files:
        isOK = False
        # if file != "0_4.ply":
        #     continue
        while not isOK:
            print(file)
            fileName, suffix = file.split('.')
            if suffix == "txt":
                continue
            pcd = o3d.io.read_point_cloud(os.path.join(pcd_path, file))
            # pcd.estimate_normals()
            cl, ind = pcd.remove_statistical_outlier(nb_neighbors=100, std_ratio=2)
            pcd = pcd.select_by_index(ind)
            labels = np.array(pcd.cluster_dbscan(eps=10, min_points=10, print_progress=True))
            print(labels)
            largest_cluster_idx = np.argmax(np.bincount(labels[labels >= 0]))  # 最大簇的索引
            largest_cluster_points = np.asarray(pcd.points)[labels == largest_cluster_idx]
            largest_cluster_pcd = o3d.geometry.PointCloud()
            largest_cluster_pcd.points = o3d.utility.Vector3dVector(largest_cluster_points)
            largest_cluster_pcd.estimate_normals()
            camera = [0,0,800]
            largest_cluster_pcd.orient_normals_towards_camera_location(camera)
            pcd = largest_cluster_pcd
            o3d.visualization.draw_geometries([pcd], point_show_normal=True)
            real_class = input("Real class:")
            params = fittingModel.fitting(pcd, cls=real_class)
            print(f"params = {params}")
            isOkStr = input("This file is OK? 'n' for 'not'")
            if isOkStr == "n":
                continue
            isOK = True