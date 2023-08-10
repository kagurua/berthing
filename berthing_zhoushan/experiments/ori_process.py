from berthing_zhoushan.infer_one_img import gain_img_seg, load_model
from berthing_zhoushan.pipeline.read_radar_frames import read_radar
import os
from skimage import transform, measure
from ransac_fit_line import *
from gaussion_sample import sample_img_depth
import dill
from loading_gt import load_gt

from data_utils import *
from detection_utils import *

saved_function_file = '../error_function_saved.bin'
f_sigma_u, f_sigma_v, f_sigma_d = dill.load(open(saved_function_file, 'rb'))

class_names = ['sea', 'sky', 'shore', 'ship', 'pillar', 'bank', 'background']
class_names.insert(0, 'unseen')

if __name__ == '__main__':
    DRAW_ON_IMG = False
    DRAW_BEV = False

    # read data
    video_file = '../pipeline/2021-01-06_15-18-18.mp4'
    radar_file = "../pipeline/test.h5"

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

    # load deeplab model
    model = load_model()
    H = get_H()

    # preprocessed_img_depth
    img_depth_dir = '/media/dataset/cityscapes_berthing/inference_scv3_zhoushan/model_v3/'

    # todo: evaluation of the extend sampling module
    lr_score_list = []
    lr_s_score_list = []
    mr_score_list = []
    count_list = []

    noo_scores = []
    frame_noos = []
    frame_false_alarm_rates = []
    frame_recalls = []

    count = 0

    for i in range(int(process_frame / concate_num_radar)):

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
        truth_points = None

        if i > 27:  # 26
            truth_points, frame_gt, _ = truth_g.__next__()
            print(frame_gt)

        # TODO: load preprocessed img depth
        img_depth_file = os.path.join(img_depth_dir + 'depth', "%05d" % count + '.npy')
        img_depth_vis_file = os.path.join(img_depth_dir + 'vis', "%05d" % count + '.jpg')

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

        # # show uv distribution
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

        # todo: test detection via pesudo-points
        # pseudo_point_detect(point_xyzs)

        # TODO eval pesudo-points' detection results
        noo_score, frame_false_alarm_rate = pseudo_point_detect_and_eval(point_xyzs, truth_points, SHOW=True)

        if noo_score is not None:
            noo_scores.append(noo_score)

            frame_noo = sum(noo_score[noo_score > 0]) / sum(noo_score > 0)
            frame_recall = sum(noo_score > 0) / sum(noo_score > -1)
            print(frame_noo, frame_recall, frame_false_alarm_rate)

            frame_noos.append(frame_noo)
            frame_false_alarm_rates.append(frame_false_alarm_rate)
            frame_recalls.append(frame_recall)

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

    # # todo: show evaluation of linear regression
    # plt.plot(count_list, lr_score_list, label='lr_score', color='r')
    # plt.plot(count_list, lr_s_score_list, label='lr_s_score', color='g')
    # # plt.plot(count_list, mr_score_list, label='mr_score', color='b')  # median regression
    # plt.legend()
    # plt.show()

    # 完成所有操作后，释放捕获器
    cv.destroyAllWindows()
