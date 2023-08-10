import sklearn.cluster as skc  # 密度聚类
from sklearn import preprocessing
import math

import numpy as np
from pipeline.read_radar_frames import read_radar
import matplotlib.pyplot as plt


def main():

    # data loading
    radar_file = './test.h5'
    concate_num_radar = 1
    offset_frame = 444

    radar_g = read_radar(radar_file, concate_num_radar, offset_frame)
    radar_points = radar_g.__next__()
    radar_points = radar_points.T
    print(radar_points.shape)

    # DBSCAN cluster on Cartesian or Polar
    mask = np.ones(radar_points.shape[1])
    mask = np.logical_and(mask, [1, 1, 1, 1, 0, 0, 0, 0, 0])

    # via Cartesian
    el, ml, ev, mv = (1.5 / 8.5, 5, 1.6, 2)
    labelsC, n_clustersC = dbscan(radar_points[:, mask], el, ml, ev, mv)

    # via Polar
    el, ml, ev, mv = (0.072, 2, 0.7, 2)
    labelsP, n_clustersP = P_dbscan(radar_points[:, mask], el, ml, ev, mv)

    print(labelsC, n_clustersC, labelsP, n_clustersP)

    # show
    fig, axes = plt.subplots(1, 1)
    plt.plot(radar_points[:, 0], radar_points[:, 1], 'o', c='#000000')
    for i in range(n_clustersP):
        one_cluster = radar_points[labelsP == i]
        plt.plot(one_cluster[:, 0], one_cluster[:, 1], 'o')
    axes.set_aspect('equal')
    plt.show()


def P_dbscan(X, el, ml, ev, mv):
    # Polar transform
    Cartesian = X
    gap = 0
    nbr_points = Cartesian.shape[0]
    Polar = []
    Sph = []
    for i in range(nbr_points):
        x, y, z = (Cartesian[i, 0], Cartesian[i, 1], Cartesian[i, 2])

        a = np.arctan((y + gap) / x) * 180 / np.pi
        a = (0 if a < 0 else 180) - a
        r = np.sqrt(x ** 2 + (y + gap) ** 2)
        Polar.append([r, a])

        XsqPlusYsq = x ** 2 + y ** 2
        r = math.sqrt(XsqPlusYsq + z ** 2)  # r
        elev = math.atan2(z, math.sqrt(XsqPlusYsq))  # theta
        az = math.atan2(y, x)  # phi
        Sph.append([r, elev, az])

    Polar = np.array(Polar)
    Sph = np.array(Sph)

    # normalization along r and a
    Polar = preprocessing.scale(Polar)
    Sph = preprocessing.scale(Sph)

    Polar = np.hstack((Polar, Cartesian[:, 2:]))
    Sph = np.hstack((Sph, Cartesian[:, 3:]))

    return dbscan(Polar, el, ml, ev, mv)
    # return dbscan(Sph, el, ml, ev, mv)


def dbscan(X, el, ml, ev, mv):
    # clustering via location
    db1 = skc.DBSCAN(eps=el, min_samples=ml).fit(X[:, :3])
    labels1 = db1.labels_
    # results
    n_clusters_1 = len(set(labels1)) - (1 if -1 in labels1 else 0)  # 获取分簇的数目

    # clustering via velocity
    db2 = skc.DBSCAN(eps=ev, min_samples=mv).fit(X[:, 3:])
    labels2 = db2.labels_
    n_clusters_2 = len(set(labels2)) - (1 if -1 in labels2 else 0)  # 获取分簇的数目
    print(n_clusters_2)

    # associating
    N = X.shape[0]
    labels3 = -1 * np.ones(N)
    n = 0
    D = {}
    for i in range(N):
        if labels1[i] != -1 and labels2[i] != -1:
            key = str(labels1[i]) + str(labels2[i])
            if key not in D:
                D[key] = n
                n += 1
            labels3[i] = D[key]

    labels_f = labels3
    n_clusters_f = len(set(labels_f)) - (1 if -1 in labels_f else 0)  # 获取分簇的数目

    # return labels_f, n_clusters_f
    return labels1, n_clusters_1
    # return labels2, n_clusters_2


if __name__ == '__main__':
    main()
