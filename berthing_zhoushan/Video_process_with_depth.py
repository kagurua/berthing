import numpy as np
import cv2 as cv
import h5py
import matplotlib.pyplot as plt
from berthing_zhoushan.infer_one_img import gain_img_seg, load_model
from berthing_zhoushan.pipeline.read_radar_frames import read_radar
import pandas as pd
import seaborn as sns
import math
import os

# from scipy import interpolate
from skimage import transform, measure

import matplotlib as mpl
import matplotlib.pyplot as plt

from ransac_fit_line import *

from gaussion_sample import sample_img_depth

# load pre-calculated error function
import dill

import sklearn.cluster as skc  # 密度聚类
from scipy.spatial import ConvexHull
import cv2

saved_function_file = './error_function_saved.bin'
f_sigma_u, f_sigma_v, f_sigma_d = dill.load(open(saved_function_file, 'rb'))

class_names = ['sea', 'sky', 'shore', 'ship', 'pillar', 'bank', 'background']
class_names.insert(0, 'unseen')


# H = np.array([[939.221222778002, 635.037860632676, 142.041238485687, -207.017968997192],
#               [-12.0084530697522, 574.72009106674, -861.686459337377, -494.926955223144],
#               [-0.0392615918420586, 0.976062351478972, 0.213917772593507, -0.201345093070659]])  # boat(not correct,
# # don't know why)


def read_video(filePath, both_delay_frame_img, skip_num_image):
    cap = cv.VideoCapture(filePath)
    if not cap.isOpened():
        print("Cannot open video")
        exit()
    frame_id = 0
    while True:
        # 逐帧捕获
        ret, frame = cap.read()
        # 如果正确读取帧，ret为True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frame_id += 1
        if frame_id < both_delay_frame_img:
            continue
        if frame_id % skip_num_image != 0:
            continue

        yield frame
    cap.release()
    return None


# TODO 0824
# adjust H by xk
def get_H():
    A = np.array([964.0495477017739, 0, 0, 0, 964.7504668505122, 0, 613.3463725824778, 377.1040664564536, 1]).reshape(
        (3, 3)).T
    # xk = np.array([-1.34639980735600, 0.0104533536651000, 0.00294198274867599, 0.00356628356156506,
    # -0.114767810602876, -0.119002343660951])
    xk = np.array([-1.37639980735600, 0.0104533536651000, 0.00294198274867599, 0.00356628356156506, -0.114767810602876,
                   -0.119002343660951])  # -1.34639980735600

    Rx = np.array([1, 0, 0, 0, np.cos(xk[0]), np.sin(xk[0]), 0, -np.sin(xk[0]), np.cos(xk[0])]).reshape((3, 3))
    Ry = np.array([np.cos(xk[1]), 0, -np.sin(xk[1]), 0, 1, 0, np.sin(xk[1]), 0, np.cos(xk[1])]).reshape((3, 3))
    Rz = np.array([np.cos(xk[2]), -np.sin(xk[2]), 0, np.sin(xk[2]), np.cos(xk[2]), 0, 0, 0, 1]).reshape((3, 3))
    R = np.dot(Rx, Ry, Rz)
    T = np.array([xk[3], xk[4], xk[5]]).reshape((-1, 1))
    B = np.concatenate([R, T], axis=1)

    H = np.dot(A, B)  # very precise
    # print(H)

    return H


def get_IRT():
    A = np.array([964.0495477017739, 0, 0, 0, 964.7504668505122, 0, 613.3463725824778, 377.1040664564536, 1]).reshape(
        (3, 3)).T
    # xk = np.array([-1.34639980735600, 0.0104533536651000, 0.00294198274867599, 0.00356628356156506,
    # -0.114767810602876, -0.119002343660951])
    xk = np.array([-1.37639980735600, 0.0104533536651000, 0.00294198274867599, 0.00356628356156506, -0.114767810602876,
                   -0.119002343660951])  # -1.34639980735600

    Rx = np.array([1, 0, 0, 0, np.cos(xk[0]), np.sin(xk[0]), 0, -np.sin(xk[0]), np.cos(xk[0])]).reshape((3, 3))
    Ry = np.array([np.cos(xk[1]), 0, -np.sin(xk[1]), 0, 1, 0, np.sin(xk[1]), 0, np.cos(xk[1])]).reshape((3, 3))
    Rz = np.array([np.cos(xk[2]), -np.sin(xk[2]), 0, np.sin(xk[2]), np.cos(xk[2]), 0, 0, 0, 1]).reshape((3, 3))
    R = np.dot(Rx, Ry, Rz)
    T = np.array([xk[3], xk[4], xk[5]]).reshape((-1, 1))

    return A, R, T


def get_external():
    xk = np.array([-1.37639980735600, 0.0104533536651000, 0.00294198274867599, 0.00356628356156506, -0.114767810602876,
                   -0.119002343660951])  # -1.34639980735600

    Rx = np.array([1, 0, 0, 0, np.cos(xk[0]), np.sin(xk[0]), 0, -np.sin(xk[0]), np.cos(xk[0])]).reshape((3, 3))
    Ry = np.array([np.cos(xk[1]), 0, -np.sin(xk[1]), 0, 1, 0, np.sin(xk[1]), 0, np.cos(xk[1])]).reshape((3, 3))
    Rz = np.array([np.cos(xk[2]), -np.sin(xk[2]), 0, np.sin(xk[2]), np.cos(xk[2]), 0, 0, 0, 1]).reshape((3, 3))
    R = np.dot(Rx, Ry, Rz)
    t = np.array([xk[3], xk[4], xk[5]]).reshape((-1, 1))

    return R.T, t.T.squeeze(0)


def gen_R(thetas):
    Rx = np.array([1, 0, 0, 0, np.cos(thetas[0]), np.sin(thetas[0]), 0, -np.sin(thetas[0]), np.cos(thetas[0])]).reshape(
        (3, 3))
    Ry = np.array([np.cos(thetas[1]), 0, -np.sin(thetas[1]), 0, 1, 0, np.sin(thetas[1]), 0, np.cos(thetas[1])]).reshape(
        (3, 3))
    Rz = np.array([np.cos(thetas[2]), -np.sin(thetas[2]), 0, np.sin(thetas[2]), np.cos(thetas[2]), 0, 0, 0, 1]).reshape(
        (3, 3))
    R = np.dot(Rx, Ry, Rz)

    return R


# project radar points to world coord
def comp_xy0(radar_points):
    theta = [0.175, -0.055, 0]  # fine theta tested for compensation from radar to world coordinate system
    R = gen_R(theta)

    p = radar_points[:, :3]
    r = np.array(
        [math.sin(theta[1]) * math.cos(theta[0]), - math.sin(theta[0]), math.cos(theta[1]) * math.cos(theta[0])])
    p = p - np.dot(p, r).reshape(-1, 1) @ r.reshape(1, -1)
    p = p @ R.T

    return p


def get_uv(radar_points, H):
    # project xyz to uv
    xyz = radar_points[:, 0:3].T
    xyz1 = np.concatenate([xyz, np.ones([1, xyz.shape[1]])])
    uv1 = np.dot(H, np.dot(xyz1, np.diag(1. / (np.dot(np.array([0, 0, 1]), np.dot(H, xyz1))))))
    uv = np.floor(uv1[0:2, :]).astype(int)

    return uv


def get_uvd(radar_points, H, video_frame=None):
    # project xyz to uv
    xyz = radar_points[:, 0:3].T
    xyz1 = np.concatenate([xyz, np.ones([1, xyz.shape[1]])])
    uv1 = np.dot(H, np.dot(xyz1, np.diag(1. / (1e-5 + np.dot(np.array([0, 0, 1]), np.dot(H, xyz1))))))
    # # discrete
    # uv = np.floor(uv1[0:2, :]).astype(int)
    uv = uv1[0:2, :]
    d = np.dot(np.array([0, 0, 1]), np.dot(H, xyz1))[None, :]  # (1, n)
    uvd = np.concatenate([uv, d])

    return uvd  # (3, n)


def get_xyz(uvd_points, I, R, t):
    # project uvd to xyz1
    udvdd = uvd_points[:, :3].T
    udvdd[:2, :] = udvdd[:2, :] * udvdd[2:, :]
    xyz = np.linalg.pinv(R) @ (np.linalg.pinv(I) @ udvdd - t)

    return xyz  # (3, n)


def get_points_semantic_ids(uv, frame, model):
    point_num = int(uv.shape[1])
    point_ids = np.arange(point_num)
    points_semantic_ids = np.zeros(point_num)

    # # remove_nearfield_points
    # nearfield_range = 2.5
    # points_semantic_ids[np.linalg.norm(radar_frame[0:3, :], axis=0) < nearfield_range] = -1
    # remove points outer image
    points_semantic_ids[uv[0, :] >= frame.shape[1]] = -1
    points_semantic_ids[uv[0, :] <= 0] = -1
    points_semantic_ids[uv[1, :] >= frame.shape[0]] = -1
    points_semantic_ids[uv[1, :] <= 0] = -1
    point_ids = point_ids[points_semantic_ids == 0]
    uv = uv[:, points_semantic_ids == 0]

    # get semantic infomation pointwise
    img_seg_maps, grid_map = gain_img_seg(model, frame)
    grid_map = grid_map.permute(1, 2, 0).cpu().numpy()
    # print(grid_map.shape, frame.shape)
    semantic_id_probs = img_seg_maps[:, uv[1, :], uv[0, :]].reshape([7, -1]).T

    semantic_ids = np.argmax(semantic_id_probs, axis=1)

    points_semantic_ids[point_ids] = semantic_ids
    # print(np.unique(points_semantic_ids))

    return points_semantic_ids, grid_map


def collect_xyzvs(radar_points, points_semantic_ids):
    # collect as xyzvs
    mask = np.ones(radar_points.shape[1])
    mask = np.logical_and(mask, [1, 1, 1, 1, 0, 0, 0, 0, 0])  # select out xyzv
    xyzv = radar_points[:, mask]
    xyzvs = pd.DataFrame(xyzv, columns=['x', 'y', 'z', 'v'])
    xyzvs_array = np.concatenate([xyzv, points_semantic_ids.reshape([-1, 1])], axis=1)
    xyzvs['s'] = [class_names[int(i + 1)] for i in points_semantic_ids]

    mask = (xyzvs_array[:, 4] > 1.5) & (xyzvs_array[:, 4] < 4.5)  # 2: shore, 3: ship, 4: pillar
    xyzvs = xyzvs.loc[mask]
    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_columns', None)
    # print(xyzvs)

    return xyzvs


def radar_points_filter(points):
    # todo: set to bev plane
    points[:, 2] = 0

    # todo: mask outside points
    mask0 = points[:, 1] < 20
    mask1 = abs(points[:, 0]) < 20
    mask_range_detect = np.bitwise_and(mask0, mask1)
    points = points[mask_range_detect, :]

    # todo: set semantic weights
    # points[:, 3:] = points[:, 3:] * np.array([0.3, 0.1, 0.2])  # ['shore', 'ship', 'pillar'] for radar points
    points[:, 3:] = points[:, 3:] * np.array([3, 10, 5])  # ['shore', 'ship', 'pillar'] for pseudo points

    return points


def convert_point_feature(xyzs):
    # todo; from s to onehot_s
    xyz = xyzs[:, 0:3]
    s = xyzs[:, -1]
    onehot_s = np.zeros((len(s), len(class_names)))
    for i in range(len(s)):
        onehot_s[i, int(s[i] + 1)] = 1
    selected_class_names = ['shore', 'ship', 'pillar']
    class_weights = np.array([1, 1, 1]) * 1
    selected_class_ids = [class_names.index(name) for name in selected_class_names]
    onehot_s = onehot_s[:, selected_class_ids] * class_weights
    xyzs = np.concatenate([xyz, onehot_s], axis=1)

    # todo: remove near-field points that is not pillar
    rm_range = 5
    mask1 = np.linalg.norm(xyzs[:, 0:3], axis=1) > rm_range
    mask2 = onehot_s[:, 2] != 0
    mask_attention = np.bitwise_or(mask1, mask2)
    cut_xyzs = xyzs[mask_attention, :]

    return cut_xyzs


def pseudo_point_detect(xyzs):

    points = radar_points_filter(convert_point_feature(xyzs))  # pre-processing

    # DBSCAN to detect
    eps = 0.5
    min_samples = 15  # 16
    db = skc.DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = db.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # 获取分簇的数目

    # Display
    fig, axes = plt.subplots(1, 1)
    axes.set_aspect('equal')
    aa_list = []
    label_list = []
    for i in np.unique(labels):
        one_frame = points[labels == i]
        aa, = plt.plot(one_frame[:, 0], one_frame[:, 1], '.')
        aa_list.append(aa)
        label_list.append(str(i))
    plt.legend(aa_list, label_list)
    axes.set_aspect('equal')
    plt.show()



if __name__ == '__main__':
    DRAW_ON_IMG = False
    DRAW_BEV = False

    # read data
    video_file = './pipeline/2021-01-06_15-18-18.mp4'
    radar_file = "./pipeline/test.h5"

    frame_rate_cam = 30
    fram_rate_radar = 10
    concate_num_radar = 5
    skip_num_image = concate_num_radar * frame_rate_cam / fram_rate_radar
    both_delay_frame = 0  # for radar
    both_delay_frame_img = both_delay_frame * frame_rate_cam / fram_rate_radar
    offset_frame = 45  # set radar faster than img, for synchronization 42

    video_g = read_video(video_file, both_delay_frame_img, skip_num_image)
    radar_g = read_radar(radar_file, concate_num_radar, offset_frame + both_delay_frame)

    # load deeplab model
    model = load_model()
    H = get_H()

    # # save_img_from_video
    # save_img_dir = '/media/dataset/cityscapes_berthing/sync_imgs_zhoushan/'
    count = 0
    # preprocessed_img_depth
    img_depth_dir = '/media/dataset/cityscapes_berthing/inference_scv3_zhoushan/model_v3/'

    # todo: evaluation of the extend sampling module
    lr_score_list = []
    lr_s_score_list = []
    mr_score_list = []
    count_list = []

    while True:

        try:
            video_frame = video_g.__next__()
        except StopIteration:
            break
        try:
            radar_points = radar_g.__next__().T
        except StopIteration:
            break

        if video_frame is None or radar_points is None:
            break

        # TODO: load preprocessed img depth
        img_depth_file = os.path.join(img_depth_dir + 'depth', "%05d" % count + '.npy')
        img_depth_vis_file = os.path.join(img_depth_dir + 'vis', "%05d" % count + '.jpg')
        # # save_img_from_video
        # cv.imwrite(img_depth_dir + "%05d" % count + '.jpg', video_frame)
        if count > 60:
            break
        count_list.append(count)
        count += 1
        depth_map = np.load(img_depth_file)
        depth_map = transform.resize(depth_map, (720, 1280), order=2)
        depth_vis = cv.imread(img_depth_vis_file)
        depth_vis = cv.resize(depth_vis, (1280, 720))

        # TODO: process img segs
        seg_probs, grid_map = gain_img_seg(model, video_frame)
        grid_map = grid_map.permute(1, 2, 0).cpu().numpy()
        # print(grid_map.shape, frame.shape)
        seg_map = np.argmax(seg_probs, axis=0)  # 0-6, keep 2, 3, 4
        foreground_seg_mask = np.zeros_like(seg_map)
        foreground_seg_mask[seg_map == 2] = 1
        foreground_seg_mask[seg_map == 3] = 1
        foreground_seg_mask[seg_map == 4] = 1

        # # fore_to_show = np.expand_dims(foreground_seg_mask, -1).astype(np.float32) * 255  # mask
        # fore_to_show = np.expand_dims(foreground_seg_mask, -1) * depth_vis  # masked depth
        # fore_to_show = fore_to_show.astype(np.uint8)
        # # fore_to_show = depth_vis  # masked depth
        # cv.imshow("win", fore_to_show)
        # if cv.waitKey(-1) == ord('q'):
        #     break

        # TODO: radar points to img, get uvd
        uvd = get_uvd(radar_points, H).T  # (n, 3)  todo: u in range (0, 1280) !!!
        # remove outliers
        mask = np.ones(radar_points.shape[0])
        mask[uvd[:, 0] < 0] = 0
        mask[uvd[:, 0] > 1280] = 0
        mask[uvd[:, 1] < 0] = 0
        mask[uvd[:, 1] > 720] = 0
        mask[uvd[:, 2] < 0] = 0
        uvd = uvd[mask == 1]
        radar_points = radar_points[mask == 1, :]
        # uvd = uvd[uvd[:, 0] > 0, :]
        # uvd = uvd[uvd[:, 0] < 1280, :]
        # uvd = uvd[uvd[:, 1] > 0, :]
        # uvd = uvd[uvd[:, 1] < 720, :]

        # todo: direct sample depth map: [uvd[:, 1].astype(np.int), uvd[:, 0].astype(np.int)]
        # mask by seg:
        radar_mask_seg = foreground_seg_mask[uvd[:, 1].astype(np.int), uvd[:, 0].astype(np.int)]
        uvd_fg = uvd[radar_mask_seg == 1, :]
        # todo: sample img_depth naive:
        img_depth = depth_map[uvd_fg[:, 1].astype(np.int), uvd_fg[:, 0].astype(np.int)]
        radar_depth = uvd_fg[:, 2]

        # todo: sample img_depth gauss:
        radar_points_r = np.linalg.norm(radar_points, axis=1)  # shape of n
        radar_points_t = np.arctan(radar_points[:, 0] / radar_points[:, 1])  # arctan() will return (-pi/2 - pi/2)
        radar_points_p = np.arctan(radar_points[:, 2] / np.linalg.norm(radar_points[:, :2], axis=1))

        sigma_u_array = f_sigma_u(radar_points_r, radar_points_t, radar_points_p)
        sigma_v_array = f_sigma_v(radar_points_r, radar_points_t, radar_points_p)

        # # show
        # plt.hist(sigma_u_array, label='sigma_u_array', alpha=0.5)
        # plt.hist(sigma_v_array, label='sigma_v_array', alpha=0.5)
        # plt.legend()
        # plt.show()

        sigma_uv = np.hstack((sigma_u_array.reshape(-1, 1), sigma_v_array.reshape(-1, 1)))

        sampled_radar_depth, sampled_img_depth, sampled_weight, sampled_seg = sample_img_depth(uvd, depth_map, sigma_uv,
                                                                                               seg_mask=foreground_seg_mask,
                                                                                               seg_map=seg_map)
        # seg_mask=None)
        sampled_img_depth_weight = np.concatenate([sampled_img_depth.reshape(-1, 1), sampled_weight.reshape(-1, 1)],
                                                  axis=1)

        # todo: assign by median
        ratio = np.median(radar_depth) / np.median(img_depth)
        ratio_s = np.median(sampled_radar_depth) / np.median(sampled_img_depth)

        # todo: assign by ransac linear(k, b) of vanilla sample with add-on regular points
        regressor_s = RANSAC(model=Weighted_LinearRegressor_0(), loss=square_error_loss, metric=mean_square_error,
                             n=10, k=100, t=1.5, d=100)
        regressor_s.fit(sampled_img_depth_weight, sampled_radar_depth.reshape(-1, 1))
        # get regress params
        params_s = regressor_s.best_fit.params
        inner_rate_s = regressor_s.best_fit_inner_num / sampled_radar_depth.shape[0]
        print("inner_rate_s:", inner_rate_s)
        lr_s_score_list.append(inner_rate_s)
        ratio_lr_s = params_s[0][0]

        # # todo: show points and regressed line(of extend sampling)
        # data_to_show = pd.DataFrame(
        #     np.concatenate([sampled_img_depth_weight, sampled_radar_depth.reshape(-1, 1), sampled_seg.reshape(-1, 1)],
        #                    axis=1), columns=['img_depth', 'weight', 'radar_depth', 'seg'])
        # g = sns.scatterplot(data=data_to_show, x='img_depth', y='radar_depth', size='weight', hue='seg')
        # g.set(xlim=(0, sampled_img_depth.max()), ylim=(0, sampled_radar_depth.max()))
        #
        # data_pred = pd.DataFrame(np.concatenate(
        #     [np.linspace(0, 100, 100).reshape(-1, 1), params_s[0][0] * np.linspace(0, 100, 100).reshape(-1, 1)],
        #     axis=1), columns=['x', 'y'])
        # sns.lineplot(x='x', y='y', data=data_pred, legend=False)
        # plt.show()

        # todo: assign by ransac linear(k, b) of vanilla sample with add-on regular points
        # addon_regulars = np.zeros([10]) + np.random.rand(10) - 0.5
        n_points = img_depth.shape[0]
        # addon_regulars_x = np.zeros(int(n_points / 2)) + (np.random.rand(int(n_points / 2)) - 0.5) * 0.7
        # addon_regulars_y = np.zeros(int(n_points / 2)) + (np.random.rand(int(n_points / 2)) - 0.5) * 0.7
        regressor = RANSAC(model=LinearRegressor_0(), loss=square_error_loss, metric=mean_square_error, t=1.5)
        # regressor.fit(np.concatenate([img_depth, addon_regulars_x]).reshape(-1, 1),
        #               np.concatenate([radar_depth, addon_regulars_y]).reshape(-1, 1))
        regressor.fit(img_depth.reshape(-1, 1), radar_depth.reshape(-1, 1))

        if regressor.best_fit is not None:
            # get regress params
            params = regressor.best_fit.params
            inner_rate = regressor.best_fit_inner_num / radar_depth.shape[0]
        else:  # use vanilla linear regression(without RANSAC)
            lr_regressor = LinearRegressor_0()
            lr_regressor.fit(img_depth.reshape(-1, 1), radar_depth.reshape(-1, 1))
            params = lr_regressor.params
            threshold = (square_error_loss(radar_depth.reshape(-1, 1),
                                           lr_regressor.predict(img_depth.reshape(-1, 1))) < 1.5)
            inner_rate = np.flatnonzero(threshold).flatten().shape[0] / radar_depth.shape[0]
        print("inner_rate:", inner_rate)
        lr_score_list.append(inner_rate)
        ratio_lr = params[0][0]

        # calculate median assign inner rate
        lr_regressor_median = LinearRegressor_0()
        lr_regressor_median.params = [[ratio]]
        threshold_median = (square_error_loss(radar_depth.reshape(-1, 1),
                                              lr_regressor_median.predict(img_depth.reshape(-1, 1))) < 1.5)
        inner_rate_median = np.flatnonzero(threshold_median).flatten().shape[0] / radar_depth.shape[0]
        mr_score_list.append(inner_rate_median)
        print("inner_rate_m:", inner_rate_median)

        print("Median Assign:", ratio)
        print("Median Assign(extend sampled):", ratio_s)
        print("Linear-Regression Assign:", ratio_lr)
        print("Weighted-Linear-Regression Assign(extend sampled):", ratio_lr_s)

        # # todo: show regression result(of vanilla sampling)
        # plt.style.use("seaborn-darkgrid")
        # fig, ax = plt.subplots(1, 1)
        # ax.set_box_aspect(1)
        # img_seg = seg_map[uvd_fg[:, 1].astype(np.int), uvd_fg[:, 0].astype(np.int)]
        # changecolor = mpl.colors.Normalize(vmin=2.0, vmax=4.0)
        # plt.scatter(img_depth, radar_depth, c=img_seg, cmap='viridis', norm=changecolor)
        # plt.xlim([-1, img_depth.max()])
        # plt.ylim([-1, radar_depth.max()])
        # line = np.linspace(0, [-1, img_depth.max()], num=100).reshape(-1, 1)
        # plt.plot(line, regressor.predict(line), c="peru")
        # plt.colorbar()
        # plt.show()

        # sample_mask = np.zeros_like(depth_map)
        # sample_mask[uvd[:, 1].astype(np.int), uvd[:, 0].astype(np.int)] = 1

        # todo: project img points to radar
        # block down-sample img seg/seg_mask and depth
        d_rate = 5
        depth_map_d = transform.resize(depth_map, (int(720 / d_rate), int(1280 / d_rate)))
        u = d_rate * (np.linspace(0, int(1280 / d_rate) - 1, int(1280 / d_rate)) + 0.5) - 0.5
        v = d_rate * (np.linspace(0, int(720 / d_rate) - 1, int(720 / d_rate)) + 0.5) - 0.5
        loc_u, loc_v = np.meshgrid(u, v)

        seg_probs_d = measure.block_reduce(seg_probs, block_size=(1, int(d_rate), int(d_rate)), func=np.sum, cval=0.0)
        seg_map_d = np.argmax(seg_probs_d, axis=0)  # equal to the weighted mode down-sampling
        foreground_seg_mask_d = np.zeros_like(seg_map_d)
        foreground_seg_mask_d[seg_map_d == 2] = 1
        foreground_seg_mask_d[seg_map_d == 3] = 1
        foreground_seg_mask_d[seg_map_d == 4] = 1

        seg_points = seg_map_d.flatten()
        point_seg = seg_points[foreground_seg_mask_d.flatten() == 1].reshape(-1, 1)
        # todo: use assign result
        depth_points = depth_map_d.flatten() * ratio_lr_s
        point_depth = depth_points[foreground_seg_mask_d.flatten() == 1].reshape(-1, 1)
        u_points = loc_u.flatten()
        point_u = u_points[foreground_seg_mask_d.flatten() == 1].reshape(-1, 1)
        v_points = loc_v.flatten()
        point_v = v_points[foreground_seg_mask_d.flatten() == 1].reshape(-1, 1)

        point_uvds = np.concatenate([point_u, point_v, point_depth, point_seg], axis=1)

        # project uvd to radar bev:
        I, R, t = get_IRT()
        point_xyz = get_xyz(point_uvds[:, :3], I, R, t).T

        point_xyzs = np.concatenate([point_xyz, point_uvds[:, 3:]], axis=1)
        # # todo: show reconstruct result
        # xyzs_img = pd.DataFrame(point_xyzs, columns=['x', 'y', 'z', 's'])
        # xyzs_img['t'] = 'img'
        # xyzs_img['s'] = [class_names[int(i + 1)] for i in xyzs_img['s']]
        # # sns.jointplot(data=xyzs, x='x', y='y', hue='s', xlim=[-10, 10], ylim=[0, 20])
        # # plt.show()
        #
        # # get semantic radar points
        # uv = get_uv(radar_points, H)
        # points_semantic_ids, grid_map = get_points_semantic_ids(uv, video_frame, model)
        # xyzvs = collect_xyzvs(radar_points, points_semantic_ids)
        # xyzs_radar = xyzvs.loc[:, ['x', 'y', 'z', 's']]
        # xyzs_radar['t'] = 'radar'
        # xyzs = pd.concat([xyzs_img, xyzs_radar])
        # xyzs = xyzs.reset_index(drop=True)
        # # show xyzs
        # g = sns.scatterplot(data=xyzs, x='x', y='y', hue='s', style='t')
        # g.set(xlim=(-10, 10), ylim=(0, 20))
        # plt.show()
        # # g = sns.scatterplot(data=xyzs_radar, x='x', y='y', hue='s')
        # # g.set(xlim=(-10, 10), ylim=(0, 20))
        # # plt.show()

        # todo: test detection via pesudo-points
        pseudo_point_detect(point_xyzs)

        # print('2333')

        # comp_xyz = comp_xy0(radar_points)

        # print(comp_xyz)
        # print(radar_points)

        # draw points with semantic info
        # plt.cla()
        # plt.plot(radar_points[:, 0], radar_points[:, 1], 'o', c='#000000')
        # for i in range(len(class_names)):
        #     if i in [-1, 0, 1, 2, 7]:
        #         continue
        #     one_cluster = radar_points[points_semantic_ids == i - 1]
        #     plt.plot(one_cluster[:, 0], one_cluster[:, 1], 'o')
        # axes.set_aspect('equal')
        # plt.xlim([-10, 10])
        # plt.ylim([0, 20])
        # plt.pause(1e-9 + concate_num_radar * 0.1)

        # # draw points on img
        # if DRAW_ON_IMG:
        #     for i in range(uv.shape[1]):
        #         point = (uv[0, i].astype(int), uv[1, i].astype(int))
        #         # color = [(radar_frame[3, i] + 5) / 10 * 255, 255 - (radar_frame[3, i] + 5) / 10 * 255, 0]
        #         color = [0, 255, 0]
        #         cv.circle(grid_map, point, 2, color, 8)
        #     cv.imshow('frame', grid_map)
        #     if cv.waitKey(1) == ord('q'):
        #         break
        #
        # if DRAW_BEV:
        #     sns.jointplot(data=xyzvs, x='x', y='y', hue='s', xlim=[-10, 10], ylim=[0, 20])
        #     plt.show()

    # todo: show evaluation
    plt.plot(count_list, lr_score_list, label='lr_score', color='r')
    plt.plot(count_list, lr_s_score_list, label='lr_s_score', color='g')
    # plt.plot(count_list, mr_score_list, label='mr_score', color='b')  # median regression
    plt.legend()
    plt.show()

    # 完成所有操作后，释放捕获器
    cv.destroyAllWindows()
