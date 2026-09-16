
import os
import cv2
import torch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from segment_anything import SamPredictor, sam_model_registry

class SegmentAnythinModel():
    def __init__(self, path):
        self.sam_model_type = "vit_h"  # or vit_b, vit_l based on the model you have
        self.sam_checkpoint_path = path  # 替换为你的模型路径ref_code\examples\data
        self.sam = sam_model_registry[self.sam_model_type](checkpoint=self.sam_checkpoint_path)
        self.sam = self.sam.to(device = "cuda")
        self.predictor = SamPredictor(self.sam)

    def segment(self, image, box):
        self.predictor.set_image(image)
        masks, _, _ = self.predictor.predict(
            box=box[None, :],  # 使用框提示进行分割，需要增加一个维度来匹配输入形状
            multimask_output=False  # 如果为 True，将返回多个可能的分割结果
        )
        return masks[0]

def mouse_callback(event, x, y, flags, param):
    # img, img_bak, click_pos = param
    if event == cv2.EVENT_LBUTTONDOWN:  # 鼠标左键点击事件
        # 记录点击的位置
        param[2].append((x, y))
        # 在图像上绘制圆点
        cv2.circle(param[0], (x, y), 5, (0, 0, 255), -1)  # 红色圆点
        cv2.imshow("Image with Clicks", param[0])  # 显示更新后的图像

    if event == cv2.EVENT_RBUTTONDOWN:  # 鼠标左键点击事件
        param[0] = param[1].copy()
        param[2].clear()
        cv2.imshow("Image with Clicks", param[0])  # 显示更新后的图像

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

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc, m

if __name__ == "__main__":
    sam_predictor = SegmentAnythinModel("models/sam_vit_h_4b8939.pth")

    base_path = os.path.join(os.path.dirname(__file__), "data", "real_flat")
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
            # click_positions = []
            image_bak = image.copy()
            # 使用 selectROI 在缩小后的图像上选择 ROI
            cv2.namedWindow("ROI", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("ROI", 1280, 800)
            roi = cv2.selectROI("ROI", image_bak)
            cv2.destroyWindow("ROI")
            if roi == (0, 0, 0, 0):
                print("No box was drawn.")
                break
            # 获取缩小后选择的 ROI 坐标
            x, y, w, h = roi

            # # 将 ROI 坐标转换回原始图像的坐标
            # x = int(x)
            # y = int(y)
            # w = int(w)
            # h = int(h)
            input_box = np.array([x, y, x + w, y + h])
            # input_box = torch.from_numpy(np.array([x, y, x + w, y + h]))

            # 使用框提示进行分割
            # image_torch = torch.from_numpy(image_rgb)
            mask = sam_predictor.segment(image_rgb, input_box)
            # 选择分割结果，并展示
            segmented_image = np.zeros_like(image_rgb)
            segmented_image[mask] = image_rgb[mask]

            segmented_depth_image = np.copy(depth_image)
            segmented_depth_image[~mask] = 0

            plt.figure(figsize=(10, 10))
            plt.subplot(1, 3, 1)
            plt.imshow(image_rgb)
            # plt.gca().add_patch(plt.Circle(click_positions[0], 5,
            #                                 edgecolor='red', facecolor='none', lw=2))
            plt.title("Original Image with Box")

            plt.subplot(1, 3, 2)
            plt.imshow(segmented_image)
            plt.title("Segmented Image")

            plt.subplot(1, 3, 3)
            plt.imshow(segmented_depth_image)
            plt.title("Segmented depth Image")
            plt.show()

            pointcloud = depth_to_pointcloud(segmented_depth_image, fx, fy, cx, cy)
            print(f"点数：{len(pointcloud)}")
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pointcloud)
            pcd.estimate_normals()
            camera = [0,0,800]
            pcd.orient_normals_towards_camera_location(camera)
            o3d.visualization.draw_geometries([pcd], point_show_normal=True)
            isSave = input("Is save result? 'n' for 'not'")
            if isSave == "n":
                continue
            o3d.io.write_point_cloud(os.path.join(pcd_path, f"{fileName}_{idx}.ply"), pcd)
            print(os.path.join(masks_path, f"{fileName}_{idx}.txt"))
            # np.savetxt(os.path.join(masks_path, f"{fileName}_{idx}.txt"), masks[0], fmt='%s')
            # print(type(masks[0]))
            # print(masks[0].shape)
            # print(masks[0])
            # cv2.imwrite(os.path.join(seg_path, f"{fileName}_{idx}_c.png"), cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))
            # cv2.imwrite(os.path.join(seg_path, f"{fileName}_{idx}_d.png"), np.uint8(cv2.normalize(segmented_depth_image, None, 0, 255, cv2.NORM_MINMAX)))
            idx = idx + 1
