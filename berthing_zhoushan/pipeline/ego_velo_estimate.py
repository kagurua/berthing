import numpy as np
# from ransac import *
import random
from pipeline.read_radar_frames import read_radar
import matplotlib.pyplot as plt


def estimate(xyzv_s):
    return - np.dot(np.linalg.pinv(xyzv_s[:, :3]), xyzv_s[:, 3:4]).T[0]


def is_inlier(coeffs, xyzv_s, threshold):
    return np.abs(- coeffs.dot(xyzv_s[:3]) - xyzv_s[3]) < threshold


def run_ransac(data, estimate, is_inlier, sample_size, goal_inliers, max_iterations, stop_at_goal=True,
               random_seed=None):
    best_ic = 0
    best_model = None
    random.seed(random_seed)
    # random.sample cannot deal with "data" being a numpy array
    data = list(data)
    sample_size = min(sample_size, len(data))
    for i in range(max_iterations):
        s = random.sample(data, int(sample_size))
        # print(s)
        s = np.vstack(s)
        # print(s)
        m = estimate(s)
        ic = 0
        for j in range(len(data)):
            if is_inlier(m, data[j]):
                ic += 1

        if ic > best_ic:
            best_ic = ic
            best_model = m
            if ic > goal_inliers and stop_at_goal:
                break
    print('took iterations:', i + 1, 'best model:', best_model, 'explains:', best_ic, '/', int(1.25 * goal_inliers))
    return best_model, best_ic


def estimate_ego_v(data):
    return - np.matmul(np.linalg.pinv(data[:, :3]), data[:, 3:4])


def cal_v_comp(radar_frame, max_iterations, goal_rate, error_threshold):  # velo generate from boat moving
    xyz = radar_frame[:, :3]
    v = radar_frame[:, 3:4]
    xyz_normal = xyz / np.linalg.norm(xyz, ord=2, axis=1, keepdims=True)  # divide r
    xyzv = np.concatenate([xyz_normal, v], axis=1)
    total_num = xyz.shape[0]

    # RANSAC
    m, b = run_ransac(xyzv, estimate, lambda x, y: is_inlier(x, y, error_threshold), 10, total_num * goal_rate,
                      max_iterations)
    v_boat = np.array(m)
    v_rela = - np.dot(xyz_normal, v_boat)

    # # error_threshold = error_threshold * 2
    # v_comp = np.round(v_comp / error_threshold) * error_threshold

    v_original = radar_frame[:, 3]
    v_comp = v_original - v_original
    # error_threshold = error_threshold * 2
    # v_abs = np.round(v_abs / error_threshold) * error_threshold

    return v_rela, v_boat, v_comp


def cal_v_comp_all(radar_frame):
    xyz = radar_frame[:, :3]
    v = radar_frame[:, 3:4]  # approaching is '-'
    xyz_normal = xyz / np.linalg.norm(xyz, ord=2, axis=1, keepdims=True)  # divide r
    xyzv = np.concatenate([xyz_normal, v], axis=1)

    v_boat = estimate_ego_v(xyzv)
    v_rela = - np.matmul(xyz_normal, v_boat)
    v_comp = v - v_rela
    # print(v_boat)
    # print(v_comp)

    return v_boat


if __name__ == '__main__':
    # data loading
    radar_file = './pipeline/test.h5'
    concate_num_radar = 5
    offset_frame = int(8 / concate_num_radar)
    radar_g = read_radar(radar_file, concate_num_radar, offset_frame)

    max_iterations = 1000
    goal_inlier_rate = 0.8
    velo_reso = 0.27604014
    error_threshold = velo_reso
    # error_threshold = velo_reso / 2

    # v_rela, v_boat, v_comp = cal_v_comp(radar_frame, max_iterations, goal_inlier_rate, error_threshold)
    # print(v_comp, v_boat, v_abs)
    time_lenth = 600
    frame_ids = np.array(range(int(time_lenth / concate_num_radar) - offset_frame))
    v_boat_list = []
    for i in range(int(time_lenth / concate_num_radar) - offset_frame):
        radar_frame = radar_g.__next__().T
        # v_rela, v_boat, v_comp = cal_v_comp(radar_frame, max_iterations, goal_inlier_rate, error_threshold)
        v_boat = cal_v_comp_all(radar_frame)
        v_boat_list.append(v_boat)
    # v_boats = np.concatenate(v_boat_list, axis=1)
    v_boats = np.concatenate(v_boat_list, axis=0).reshape([-1, 3]).T

    W_lenth = 5
    weights = np.ones(W_lenth) / W_lenth
    v_boat_w = np.convolve(weights, v_boats[1, :], mode='same')

    fig, axes = plt.subplots(1, 1)
    plt.plot(frame_ids, v_boats[1, :])
    plt.plot(frame_ids, v_boat_w)
    plt.show()
