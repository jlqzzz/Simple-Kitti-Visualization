import cv2
import numpy as np

# turn a 3x4 to a 4x4 matrix
def extend_matrix(mat):
    mat = np.concatenate([mat, np.array([[0., 0., 0., 1.]])], axis=0)
    return mat

# read image
def read_img(img_path):
    img = cv2.imread(img_path)
    img = img.astype(np.float32)
    return img

# read point cloud
def read_pts(pts_path, reduce_points = True, FOV = 65.0):
    pointcloud = np.fromfile(pts_path, dtype=np.float32, count=-1).reshape([-1,4])
    reduce_points = True
    pointcloud = pointcloud[np.sqrt((pointcloud[:, 0] ** 2) + (pointcloud[:, 1] ** 2)) < 100]
    if reduce_points:
        FOVThresh = np.tan(FOV*np.pi/180.0)
        pointcloud = pointcloud[np.abs(pointcloud[:, 0])*FOVThresh - np.abs(pointcloud[:, 1]) >= 0]
        pointcloud = pointcloud[pointcloud[:, 0] >= 0]
    return pointcloud

# read kitti calib file
def read_calib(calib_path):
    calib = {}
    with open(calib_path, 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line) == 0: continue
            key, value = line.split(':', 1)
            try:
                calib[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
    
    rect = np.array([float(info) for info in calib['R0_rect']]).reshape([3, 3])
    rect_4x4 = np.zeros([4, 4], dtype=rect.dtype)
    rect_4x4[3, 3] = 1.
    rect_4x4[:3, :3] = rect
    Trv2c = extend_matrix(np.array([float(info) for info in calib['Tr_velo_to_cam']]).reshape([3, 4]))
    intrinsics = extend_matrix(np.array([float(info) for info in calib['P2']]).reshape([3, 4]))
    extrinsics = rect_4x4 @ Trv2c
    lidar2img = intrinsics @ extrinsics

    return intrinsics, extrinsics, lidar2img


# Turn kitti label box into a [3, 8](x, y, z for each corner of box) cuboid
def corner_calc(bbox):
    """Lift the ground truth into a (3, 8) cuboid.

    Arguments:
        bbox {List} -- GT label

    Returns:
        corners_3d {array} -- (3, 8) cuboid.
    """
    rot_y = float(bbox[6])
    l = float(bbox[2])
    w = float(bbox[1])
    h = float(bbox[0])
    centrex = float(bbox[3])
    centrey = float(bbox[4])
    centrez = float(bbox[5])
    rot_mtx = np.float32([
        [np.cos(rot_y), 0.0, np.sin(rot_y)],
        [0.0, 1.0, 0.0],
        [-np.sin(rot_y), 0.0, np.cos(rot_y)]
    ])
    x_corners = [-l/2, l/2, l/2, l/2, l/2, -l/2, -l/2, -l/2]  # -l/2
    y_corners = [-h, -h, 0.0, 0.0, -h, -h, 0.0, 0.0]  # -h
    z_corners = [-w/2, -w/2, -w/2, w/2, w/2, w/2, w/2, -w/2]  # -w/2
    corners_3d = np.array([x_corners, y_corners, z_corners])
    corners_3d = rot_mtx.dot(corners_3d)
    corners_3d += np.array([centrex, centrey, centrez]).reshape((3, 1))
    corners_3d = np.float32(corners_3d)
    return corners_3d

# read label info
def read_label(label_path):
    gt_bboxes_2d = []
    gt_bboxes_3d = []
    with open(label_path) as label_file:
        annos = [x.strip("\n") for x in label_file.readlines()]
        for ann in annos:
            ann = ann.split(" ")
            ann = [float(x) for x in ann[1:15]]
            gt_bboxes_2d.append(ann[3:7])
            gt_bboxes_3d.append(corner_calc(ann[7:]))
    
    return gt_bboxes_2d, gt_bboxes_3d