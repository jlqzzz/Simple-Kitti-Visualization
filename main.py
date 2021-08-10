from vis_image import vis_image
from vis_points import vis_points
from views import *
import cv2

# define frame name and get path for each file
frame = '000032'
img_path = 'data/' + frame + '.png'
pts_path = 'data/' + frame + '.bin'
calib_path = 'data/' + frame + '_calib.txt'
label_path = 'data/' + frame + '_label.txt'

# define output path for image visualisation and create image
output_path = "output/vis_image.png"
img = vis_image(
    output_path,
    img_path,
    pts_path,
    calib_path,
    label_path,
    lidar=False,
    boxes_2d=False,
    boxes_3d=True
)

cv2.imwrite(output_path, img)
print("Saved img:", output_path)

# define output path for points visualisation and create image
output_path = "output/vis_points.png"
img = vis_points(
    output_path,
    pts_path,
    label_path,
    calib_path,
    boxes_3d=True,
    view_dict=TPV # TPV(third person view), FPV(first person view), BEV(birds eye view)
)

cv2.imwrite(output_path, img)
print("Saved img:", output_path)