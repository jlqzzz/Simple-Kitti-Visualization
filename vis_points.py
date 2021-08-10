import numpy as np
import cv2
from reporjections import *
from load_data import *
from views import *
import copy


# reproject and draw point cloud and 3d bounding boxes onto a black background
def vis_points(
    output_path,
    pts_path,
    label_path,
    calib_path,
    boxes_3d=True,
    view_dict=TPV
):

    # read in points and bounding boxes
    points = read_pts(pts_path)
    gt_bboxes_2d, gt_bboxes_3d = read_label(label_path)
    
    # create black background
    IMG_H = 1500
    IMG_W = 3000
    img = np.zeros((IMG_H, IMG_W, 3), np.uint8)

    # create transformation matrix
    lidar2img = create_calib(IMG_W, IMG_H, **view_dict)
    
    # reproject point cloud onto image
    img = repro_lidar(img, points, lidar2img)
    
    # reporject 3d bounding boxes onto image
    if boxes_3d:
        for box in gt_bboxes_3d:
            # correct for camera/lidar cordinate system misalignment
            box2 = copy.deepcopy(box)
            box2[0] = box[2]
            box2[1] = -box[0]
            box2[2] = -box[1]
            img = repro_box(img, box2, lidar2img, color=(255,255,255))
    
    return img