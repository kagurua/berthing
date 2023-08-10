import numpy as np
from probreg import filterreg
import transforms3d as t3d

from pipeline.read_radar_frames import read_radar
import matplotlib.pyplot as plt

from Video_process import read_video, get_H
from infer_one_img import load_model

import pandas as pd
import seaborn as sns

from my_particle import N_PARTICLE, STATE_SIZE, Particle, calc_final_state, motion_model, particle_filtering
from my_registration import collect_point2img_feature, convert_point_feature, to_cluster_feature
from pipeline.ego_velo_estimate import cal_v_comp_all

from legacy.detection_metric import current_detection

from loading_gt import load_gt
import os

os.chdir('/media/personal_data/wujx/Boat_berthing')

class_names = ['sea', 'sky', 'shore', 'ship', 'pillar', 'bank', 'background']
class_names.insert(0, 'unseen')


def cluster_semantic_registration(xyzs, win_xyzs, labels_frame):
    win_xyz, win_cluster_f = to_cluster_feature(win_xyzs, labels_frame, USE_WEIGHT=False)
    xyz, cluster_f = to_cluster_feature(xyzs, np.zeros(xyzs.shape[0]))

    # tf_param, _, _ = cpd.registration_cpd(xyz, win_xyz, update_scale=False)
    tf_param, _, _ = filterreg.registration_filterreg(xyz, win_xyz, cluster_f, win_cluster_f)

    # print("result: ", t3d.euler.mat2euler(tf_param.rot), tf_param.scale, tf_param.t)

    return tf_param


def radar_particle_filtering(particles, radar_points_far, reg_result, Q_est, Q_reg, DT):
    # cal u&z
    vx_est, vy_ext, _ = cal_v_comp_all(radar_points_far)
    u = np.array([vx_est, vy_ext]).reshape([2, 1])
    dx, dy, dtheta = reg_result
    z = np.array([dx, dy, dtheta]).reshape([3, 1])

    particles = particle_filtering(particles, u, z, Q_est, Q_reg, DT)

    xEst, T = calc_final_state(particles)

    dxDR = motion_model(None, u, DT)

    return xEst, T, dxDR


def call_func():
    # read data
    video_file = './sea_test_data/2021-01-06_15-18-18.mp4'
    radar_file = "./sea_test_data/test.h5"
    H = get_H()
    model = load_model()

    # TODO: data pipeline
    process_frame = 600
    window_lenth = 5
    frame_rate_cam = 30
    fram_rate_radar = 10
    concate_num_radar = 5
    skip_num_image = concate_num_radar * frame_rate_cam / fram_rate_radar
    both_delay_frame = 0  # for radar
    both_delay_frame_img = both_delay_frame * frame_rate_cam / fram_rate_radar
    offset_frame = 45  # set radar faster than img, for synchronization 42

    video_g = read_video(video_file, both_delay_frame_img, skip_num_image)
    radar_g = read_radar(radar_file, concate_num_radar, offset_frame + both_delay_frame)
    truth_g = load_gt()

    # TODO: registration params
    average_win_lenth = 5
    results = [np.array([0, 0, 0])]  # alpha, tx, ty
    tolerance = np.array([0.5, 0.5, 0.1])
    base_var = np.array([0.1, 0.1, 0.01])
    # base_var = np.array([0.001,0.001,0.001])

    # TODO: registration initial
    last_radar_points = radar_g.__next__().T
    last_video_frame = video_g.__next__()

    last_xyzvs = collect_point2img_feature(last_radar_points, last_video_frame, H, model)
    last_xyzs, last_xyzs_all = convert_point_feature(last_xyzvs)

    R0 = np.identity(3)
    T0 = np.zeros(3)
    xyzs_C0 = last_xyzs_all.copy()
    labelsC = np.zeros(last_xyzs.shape[0])

    # TODO: particle filtering params
    Q_est = np.diag([0.5, 0.5]) ** 2  # vx, vy
    # Q_est = np.diag([0.5, 0.5]) ** 2  # vx, vy
    rm_range = 2.5  # remove near_field for velo_estimate
    DT = 0.1 * concate_num_radar

    # TODO: particle filtering initial
    particles = [Particle() for _ in range(N_PARTICLE)]

    xEst = np.zeros((STATE_SIZE, 1))  # SLAM estimation
    xDR = np.zeros((STATE_SIZE, 1))  # Dead reckoning
    hxEst = xEst
    hxDR = xDR

    radar_points_far = last_radar_points[np.linalg.norm(last_radar_points[:, 0:3], axis=1) > rm_range, :]

    # # fig, axes = plt.subplots(1, 1)
    # fig = plt.figure()
    # ax1 = fig.add_subplot(121)
    # ax2 = fig.add_subplot(122)
    # ax1.set_aspect('equal')
    # ax2.set_aspect('equal')

    noo_scores = []
    frame_noos = []
    frame_false_alarm_rates = []
    frame_recalls = []

    for i in range(int(process_frame / concate_num_radar)):

        radar_points = radar_g.__next__().T
        video_frame = video_g.__next__()
        truth_points = None
        if i > 27:  # 26
            truth_points, frame_gt, _ = truth_g.__next__()
            print(frame_gt)

        xyzvs = collect_point2img_feature(radar_points, video_frame, H, model)
        xyzs, xyzs_all = convert_point_feature(xyzvs)

        win_xyzs = last_xyzs[labelsC > (i - window_lenth), :]
        labels_frame = labelsC[labelsC > (i - window_lenth)]
        labels_frame = np.max(labels_frame) - labels_frame
        labelsC = np.concatenate([labelsC, (i + 1) * np.ones(xyzs.shape[0])])

        # TODO(wujx): call registration process
        tf_param = cluster_semantic_registration(xyzs, win_xyzs, labels_frame)

        R, T = tf_param.rot.T, tf_param.t
        # R = np.identity(3)   # test no registration
        Rc = R  # coordinate's transform
        Tc = T @ R.T

        # TODO: define registration quality
        win_results = np.concatenate(results[max(0, len(results) - average_win_lenth): len(results)], axis=0).reshape(
            (-1, 3))
        reg_result = np.array([Tc[0], Tc[1], list(t3d.euler.mat2euler(Rc))[2]])

        var_factor = np.maximum(abs(np.average(win_results, axis=0) - reg_result) / tolerance, np.array([1, 1, 1]))
        reg_var = base_var * (var_factor ** 2)

        euler = list(t3d.euler.mat2euler(R))
        if euler[0] != 0 or euler[1] != 0 or abs(euler[2]) > 0.1:
            # print("wdnmd")
            R = np.identity(3)
            T = np.zeros(3)
            # reg_result[2] = 0
            reg_result = np.array([0, 0, 0])
            reg_var = np.array([1, 1, 1])

        print("reg_result:", reg_result)
        print("reg_var:", reg_var)

        # TODO: call particle-filtering process
        Q_reg = np.diag(reg_var[:-1]) ** 2
        xEst, filter_result, dxDR = radar_particle_filtering(particles, radar_points_far, reg_result, Q_est, Q_reg, DT)
        print("filter_result:", filter_result)
        results.append(filter_result)

        Rf = t3d.euler.euler2mat(0., 0., filter_result[2])
        Tf = np.array([filter_result[0], filter_result[1], 0]) @ Rf  # points' transform
        # Rf = R  # test no velo estimate
        # Tf = T

        R0 = np.dot(Rf, R0)
        T0 = T0 + np.dot(Tf, R0)

        last_xyzs = np.concatenate([np.concatenate([np.dot((last_xyzs[:, :3] - Tf), Rf.T), last_xyzs[:, 3:]], axis=1),
                                    xyzs])
        last_xyzs_all = np.concatenate(
            [np.concatenate([np.dot((last_xyzs_all[:, :3] - Tf), Rf.T), last_xyzs_all[:, 3:]], axis=1),
             xyzs_all])
        xyzs_C0 = np.concatenate([xyzs_C0, np.concatenate([(np.dot(xyzs_all[:, :3], R0) + T0), xyzs_all[:, 3:]],
                                                          axis=1)])  # convert points to the initial coordinate

        # # current processing
        # # display_current
        # ax1.plot(last_xyzs[:, 0], last_xyzs[:, 1], '.')
        # if truth_points is not None:
        #     ax1.plot(truth_points[:, 0], truth_points[:, 1], '.k')
        # # mask_time = labelsC > (np.max(labelsC) - 1)
        # # ax1.plot(last_xyzs[mask_time, 0], last_xyzs[mask_time, 1], '.')
        # ax1.set_xlim([-20, 20])
        # ax1.set_ylim([0, 20])
        # ax2.plot(xyzs[:, 0], xyzs[:, 1], '.')
        # if truth_points is not None:
        #     ax2.plot(truth_points[:, 0], truth_points[:, 1], '.k')
        # ax2.set_xlim([-20, 20])
        # ax2.set_ylim([0, 20])
        # # plt.xlim([-10, 10])
        # # plt.ylim([0, 20])
        # plt.pause(0.01)
        # # plt.cla()
        # ax1.cla()
        # ax2.cla()

        # TODO
        # current_motion_extraction()
        noo_score, frame_false_alarm_rate = current_detection(last_xyzs_all, labelsC, truth_points)

        if noo_score is not None:
            noo_scores.append(noo_score)

            frame_noo = sum(noo_score[noo_score > 0]) / sum(noo_score > 0)
            frame_recall = sum(noo_score > 0) / sum(noo_score > -1)
            print(frame_noo, frame_recall, frame_false_alarm_rate)

            frame_noos.append(frame_noo)
            frame_false_alarm_rates.append(frame_false_alarm_rate)
            frame_recalls.append(frame_recall)

        # store data history
        xDR[:2, [0]] = xDR[:2, [0]] + dxDR
        hxEst = np.hstack((hxEst, xEst))
        hxDR = np.hstack((hxDR, xDR))
        radar_points_far = radar_points[np.linalg.norm(radar_points[:, 0:3], axis=1) > rm_range, :]

    noo_scores = np.concatenate(noo_scores).reshape([-1, len(set(truth_points[:, -1]))])
    frame_noos = np.array(frame_noos)
    frame_false_alarm_rates = np.array(frame_false_alarm_rates)
    frame_recalls = np.array(frame_recalls)

    np.save("./tests/noo_scores1.npy", noo_scores)
    np.save("./tests/frame_noos1.npy", frame_noos)
    np.save("./tests/frame_false_alarm_rates1.npy", frame_false_alarm_rates)
    np.save("./tests/frame_recalls1.npy", frame_recalls)

    noo_scores[noo_scores == -1] = 0.0
    instance_noos = []
    for i in range(noo_scores.shape[1]):
        line = noo_scores[:, i]
        instance_noo = [sum(line), sum(line > 0)]
        instance_noos.append(instance_noo)
        print(instance_noo[0] / instance_noo[1])

    instance_noos = np.array(instance_noos)
    # ship_ids = [0, 6]
    ship_ids = [0]
    shore_ids = [1]
    pillar_ids = [2, 3, 4, 5]
    ship_noo = np.sum(instance_noos[ship_ids], axis=0)
    shore_noo = np.sum(instance_noos[shore_ids], axis=0)
    pillar_noo = np.sum(instance_noos[pillar_ids], axis=0)

    print(ship_noo, shore_noo, pillar_noo)
    print(ship_noo[0] / ship_noo[1],
          shore_noo[0] / shore_noo[1],
          pillar_noo[0] / pillar_noo[1])

    print(np.mean(frame_noos))
    print(np.mean(frame_false_alarm_rates))
    print(np.mean(frame_recalls))

    # # Display 3
    # plt.show()

    # # Display 1
    # fig, axes = plt.subplots(1, 1)
    # axes.set_aspect('equal')
    # for i in np.unique(labelsC):
    #     one_frame = xyzs_C0[labelsC == i]
    #     plt.plot(one_frame[:, 0], one_frame[:, 1], '.')
    #     plt.pause(0.1)
    # axes.set_aspect('equal')
    # plt.show()
    # Display 2
    pd_xyzs = pd.DataFrame(xyzs_C0[:, :3], columns=['x', 'y', 'z'])
    selected_class_names = ['shore', 'ship', 'pillar']
    s_raw = np.array([selected_class_names[int(i)] for i in np.argmax(xyzs_C0[:, 3:], axis=1)])
    s_raw[np.sum(xyzs_C0[:, 3:], axis=1) == 0] = 'unseen'
    pd_xyzs['s'] = s_raw
    fig, axes = plt.subplots(1, 1)
    axes.set_aspect('equal')
    # sns.jointplot(data=pd_xyzs, x='x', y='y', hue='s', xlim=[-10, 10], ylim=[0, 20])
    sns.jointplot(data=pd_xyzs, x='x', y='y', hue='s', s=0.4)
    print(pd_xyzs)
    plt.show()


call_func()
