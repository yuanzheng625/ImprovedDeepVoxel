import os
import numpy as np
from glob import glob
import data_util
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull
import random


def subsample_simplices(simplices):
    subsample = []
    excluded = []

    for face in simplices:
        v1, v2, v3 = face
        cand = []
        if v1 not in subsample and v1 not in excluded:
            cand.append(v1)
        if v2 not in subsample and v2 not in excluded:
            cand.append(v2)
        if v3 not in subsample and v3 not in excluded:
            cand.append(v3)

        if not cand:
            continue
        random.shuffle(cand)
        subsample.append(cand[0])

        if len(cand) == 2:
            excluded.append(cand[1])
        if len(cand) == 3:
            excluded.append(cand[1])
            excluded.append(cand[2])

    return subsample


class CameraViewVis():
    def __init__(self,
                 root_dir):
        super().__init__()

        self.pose_dir = os.path.join(root_dir, 'pose')
        self.all_poses = sorted(glob(os.path.join(self.pose_dir, '*.txt')))

        # Buffer files
        print("Buffering files...")
        self.all_views = []
        for i in range(self.__len__()):
            if not i % 10:
                print(i)
            # based on deepvoxel implmentation, pose is camera_to_world, i.e. for y = pose*x, x is camera, y is world
            self.all_views.append(data_util.load_pose(self.all_poses[i]))

        self.gaze = self.get_gaze_vector()
        self.pos = self.get_camera_pos()

    def __len__(self):
        return len(self.all_poses)

    def get_gaze_vector(self):
        gaze = np.stack([pose[:3, 2] for pose in self.all_views], axis=0)
        gaze /= np.linalg.norm(gaze, axis=1, ord=2, keepdims=True)
        return gaze  # [n, 3]

    def get_camera_pos(self):
        pos = np.stack([pose[:3, 3] for pose in self.all_views], axis=0)
        return pos  # [n, 3]

    def vis_3d(self):

        fig = plt.figure(dpi=500)
        ax = fig.gca(projection='3d')

        ax.quiver(self.pos[::1, 0], self.pos[::1, 1], self.pos[::1, 2],
                  self.gaze[::1, 0], self.gaze[::1, 1], self.gaze[::1, 2], length=0.1)

        plt.show()

    def vis_3d_pos_only(self):

        fig = plt.figure(dpi=500)
        ax = fig.gca(projection='3d')

        ax.scatter(self.pos[::1, 0], self.pos[::1, 1], self.pos[::1, 2])

        plt.show()

    def get_convex_hull(self, points):
        hull = ConvexHull(points)

        return hull

    def plot_convex_hull(self, points, hull):
        fig = plt.figure(dpi=500)
        ax = fig.gca(projection='3d')

        #shell = np.array([[self.pos[v][0], self.pos[v][1], self.pos[v][2]] for v in hull.vertices])
        #print('before {}, after {}'.format(self.pos.shape[0], shell.shape[0]))
        #ax.scatter(shell[:, 0], shell[:, 1], shell[:, 2])
        for simplex in hull.simplices:
            plt.plot(points[simplex, 0], points[simplex, 1], points[simplex, 2], 'k-')

        ax.scatter(points[:,0], points[:,1], points[:,2], c='magenta')

        plt.show()

    def subsample(self, hull, points):
        mask = subsample_simplices(hull.simplices)
        sampled = points[mask]
        return sampled

    def peel_hull(self, hull, points):
        assert points.shape[0] >= hull.vertices.shape[0]
        mask = np.ones((points.shape[0],), dtype=bool)
        mask[hull.vertices] = False
        inner_points = points[mask]

        return inner_points

    def pos_histo(self):
        self.pos  # nx3
        pass

    def gaze_histo(self):
        pass


def main():
    data_root = '/Users/zyuan/workspace/deepvoxels/stanford_fountain'
    c_vis = CameraViewVis(data_root)
    # c_vis.vis_3d()
    # c_vis.vis_3d_pos_only()

    print(c_vis.pos.shape)
    hull0 = c_vis.get_convex_hull(c_vis.pos)
    c_vis.plot_convex_hull(c_vis.pos, hull0)

    inner1 = c_vis.peel_hull(hull0, c_vis.pos)
    print(inner1.shape)
    hull1 = c_vis.get_convex_hull(inner1)
    c_vis.plot_convex_hull(inner1, hull1)

    inner2 = c_vis.peel_hull(hull1, inner1)
    print(inner2.shape)
    hull2 = c_vis.get_convex_hull(inner2)
    c_vis.plot_convex_hull(inner2, hull2)

    inner3 = c_vis.peel_hull(hull2, inner2)
    print(inner3.shape)
    hull3 = c_vis.get_convex_hull(inner3)
    c_vis.plot_convex_hull(inner3, hull3)

    return


    sampled1 = c_vis.subsample(hull0, c_vis.pos)
    print(sampled1.shape)

    hull1 = c_vis.get_convex_hull(sampled1)
    sampled2 = c_vis.subsample(hull1, sampled1)
    print(sampled2.shape)
    hull2 = c_vis.get_convex_hull(sampled2)
    sampled3 = c_vis.subsample(hull2, sampled2)
    print(sampled3.shape)
    hull3 = c_vis.get_convex_hull(sampled3)
    sampled4 = c_vis.subsample(hull3, sampled3)
    print(sampled4.shape)
    hull4 = c_vis.get_convex_hull(sampled4)
    c_vis.plot_convex_hull(sampled4, hull4)
    sampled5 = c_vis.subsample(hull4, sampled4)
    print(sampled5.shape)
    hull5 = c_vis.get_convex_hull(sampled5)
    sampled6 = c_vis.subsample(hull5, sampled5)
    print(sampled6.shape)
    hull6 = c_vis.get_convex_hull(sampled6)
    c_vis.plot_convex_hull(sampled6, hull6)
    sampled7 = c_vis.subsample(hull6, sampled6)
    print(sampled7.shape)
    hull7 = c_vis.get_convex_hull(sampled7)
    sampled8 = c_vis.subsample(hull7, sampled7)
    print(sampled8.shape)
    hull8 = c_vis.get_convex_hull(sampled8)
    c_vis.plot_convex_hull(sampled8, hull8)

    return 0


if __name__ == '__main__':
    main()
