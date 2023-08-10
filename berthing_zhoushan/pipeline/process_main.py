import numpy as np
from pipeline.read_radar_frames import read_radar
import matplotlib.pyplot as plt
from points_cluster import dbscan
from ego_velo_estimate import cal_v_comp


def main():
    # data loading
    radar_file = './test.h5'
    concate_num_radar = 2
    offset_frame = 42
    radar_g = read_radar(radar_file, concate_num_radar, offset_frame)

    max_iterations = 1000
    goal_inlier_rate = 0.8
    velo_reso = 0.27604014
    error_threshold = velo_reso / 2

    fig, axes = plt.subplots(1, 1)

    while True:

        radar_points = radar_g.__next__().T
        # radar_points[:, 0] = - radar_points[:, 0]

        v_comp, v_boat, v_abs = cal_v_comp(radar_points, max_iterations, goal_inlier_rate, error_threshold)
        radar_points[:, 3] = v_abs

        # DBSCAN cluster on Cartesian or Polar
        mask = np.ones(radar_points.shape[1])
        # mask = np.logical_and(mask, [1, 1, 1, 1, 0, 0, 0, 0, 0])  # select out xyzv
        mask = np.logical_and(mask, [1, 1, 1, 1, 0, 0, 0, 0, 0])  # select out xyzv

        # via Cartesian
        # el, ml, ev, mv = (2, 5, 2, 3)
        el, ml, ev, mv = (2, 3, 0.2, 5)
        labelsC, n_clustersC = dbscan(radar_points[:, mask], el, ml, ev, mv)

        # # via Polar
        # # el, ml, ev, mv = (0.072, 2, 0.7, 2)
        # el, ml, ev, mv = (0.25, 2, 0.7, 2)
        # labelsP, n_clustersP = P_dbscan(radar_points[:, mask], el, ml, ev, mv)

        # print(labelsC, n_clustersC, labelsP, n_clustersP)

        # show
        plt.cla()
        plt.plot(radar_points[:, 0], radar_points[:, 1], 'o', c='#000000')
        for i in range(n_clustersC):
            one_cluster = radar_points[labelsC == i]
            plt.plot(one_cluster[:, 0], one_cluster[:, 1], 'o')
        axes.set_aspect('equal')
        plt.xlim([-10, 10])
        plt.ylim([0, 20])

        # plt.show()
        plt.pause(1e-9 + concate_num_radar * 0.1)


if __name__ == '__main__':
    main()
