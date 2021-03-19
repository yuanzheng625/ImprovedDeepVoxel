# This is a minimal example to work with poses and depth images, in case you want to use the datasets for another project.
# Just move this into the root directory of any of the synthetic datasets (such as cube/), where the intrinsics.txt file, rgb, pose and depth directories reside.

import numpy as np
import os
from glob import glob
import cv2

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

data_root = '/Users/zyuan/workspace/deepvoxels/vase'

pose_fpaths = sorted(glob(os.path.join(data_root, "pose/*.txt")))
depth_fpaths = sorted(glob(os.path.join(data_root, "depth/*.png")))

intrinsics = np.genfromtxt(os.path.join(data_root, "intrinsics.txt"), max_rows=1)
f, cx, cy = intrinsics[:3]

with open(os.path.join(data_root, "intrinsics.txt"), 'r') as file:
    f, cx, cy = list(map(float, file.readline().split()))[:3]
    grid_barycenter = list(map(float, file.readline().split()))
    near_plane = float(file.readline())
    scale = float(file.readline())
    height, width = map(float, file.readline().split())

##### Zheng: don't worry about 512 or 64, because x, y and (intrinsic depending on x and y) cancalled out
# The intrinsics are for a sensor of shape (width, height) = (640, 480), where f is the vertical focal length.
cx *= 512. / width
cy *= 512. / height
f *= 512 / height

world_to_grid = np.eye(4)
world_to_grid[:3, 3] = grid_barycenter

v, u = np.mgrid[:512, :512]

assert len(pose_fpaths) == len(depth_fpaths)

grid_min = np.empty((0,4))
grid_max = np.empty((0,4))

for i in range(len(pose_fpaths)):
    pose = np.genfromtxt(pose_fpaths[i]).reshape(4, 4)
    depth = cv2.imread(depth_fpaths[i], cv2.IMREAD_UNCHANGED).astype(np.float32) * 1e-4

    x_cam = depth * (u - cx) / f
    y_cam = depth * (v - cy) / f
    cam_coords = np.stack((x_cam, y_cam, depth, np.ones_like(depth)), axis=-1)

    cam_coords = cam_coords[depth != 1]

    world_coords = np.matmul(pose, cam_coords.reshape(-1, 4, 1)).squeeze()

    # grid coords
    grid_metric = np.matmul(world_to_grid, world_coords.reshape(-1, 4, 1)).squeeze()

    view_min = np.min(grid_metric, axis=0)
    view_max = np.max(grid_metric, axis=0)

    grid_min = np.vstack((grid_min, view_min))
    grid_max = np.vstack((grid_max, view_max))

    if i % 10 == 0:
        print('processed {} views'.format(i))


grid_min = np.min(grid_min, axis=0)
grid_max = np.max(grid_max, axis=0)

print(grid_min)
print(grid_max)