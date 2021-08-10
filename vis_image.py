import math
import numpy as np
import random
import cv2
from reporjections import *
from load_data import *


# reproject and draw point cloud, 3d bounding boxes and 2d bounding boxes onto image
def vis_image(
    output_path,
    img_path,
    pts_path,
    calib_path,
    label_path,
    lidar=True,
    boxes_2d=True,
    boxes_3d=True
):

    # read in input files
    img = read_img(img_path)
    points = read_pts(pts_path)
    gt_bboxes_2d, gt_bboxes_3d = read_label(label_path)
    intrinsics, extrinsics, lidar2img = read_calib(calib_path)
    
    # reproject point cloud onto image
    if lidar:
        img = repro_lidar(img, points, lidar2img)
    
    # reproject 3d bounding boxes onto image
    if boxes_3d:
        lidar_pallet = []
        for box in gt_bboxes_3d:
            if len(lidar_pallet) == 0:
                lidar_pallet = pallet[:]
                random.shuffle(lidar_pallet)
            color = lidar_pallet.pop()
            img = repro_box(img, box, intrinsics, color)

    # reproject 2d bounding boxes onto image
    if boxes_2d:
        for box_2d in gt_bboxes_2d:
            x1, y1, x2, y2 = box_2d
            w = abs(x2 - x1)
            line_thickness = math.ceil(w/1000)
            img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), thickness=line_thickness)
    
    return img