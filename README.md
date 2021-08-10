# Simple Kitti Visualizations

Simple and easy to follow point cloud and bounding box visualizations for kitti data. The only dependencies are OpenCV and Numpy.

Create re-projections onto images and 3 different views of a point-cloud with  a black background (birds eye view, first person view,  third person view).

## Installation
Clone the repository.

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the requirements.

```bash
pip install -r requirements.txt
```

## Usage

In main.py, change the filepaths to the location of your kitti files. Run vis_image or vis_points on your data.

### example:

``` python
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

```
## Results

#### 3D Bounding Boxes on Image
![img_3dbox.png](output/img_3dbox.png)

#### 2D Bounding Boxes on Image
![img_2dbox.png](output/img_2dbox.png)

#### Point Cloud on Image
![img_points.png](output/img_points.png)

#### Third Person View Point Cloud
![img_3dbox.png](output/pts_TPV.png)

#### First Person View Point Cloud
![img_3dbox.png](output/pts_BEV.png)

#### Birds Eye View Point Cloud
![img_3dbox.png](output/pts_FPV.png)