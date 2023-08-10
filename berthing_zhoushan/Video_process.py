import numpy as np
import cv2 as cv
import h5py
import matplotlib.pyplot as plt
from berthing_zhoushan.infer_one_img import gain_img_seg, load_model
from berthing_zhoushan.pipeline.read_radar_frames import read_radar
import pandas as pd
import seaborn as sns
import math

class_names = ['sea', 'sky', 'shore', 'ship', 'pillar', 'bank', 'background']
class_names.insert(0, 'unseen')

# H = np.array([[939.221222778002, 635.037860632676, 142.041238485687, -207.017968997192],
#               [-12.0084530697522, 574.72009106674, -861.686459337377, -494.926955223144],
#               [-0.0392615918420586, 0.976062351478972, 0.213917772593507, -0.201345093070659]])  # boat(not correct,
# # don't know why)

# test data
data_grid = np.meshgrid(np.arange(200) - 100, np.arange(200))
x_array = data_grid[0].reshape((-1)) * 10
y_array = data_grid[1].reshape((-1)) * 10
z_array = np.ones_like(x_array) * -1
v_array = np.zeros_like(x_array)
xyzv_grid = np.vstack([x_array, y_array, z_array, v_array])


# print(xyzv_grid)


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


def get_external():
    xk = np.array([-1.37639980735600, 0.0104533536651000, 0.00294198274867599, 0.00356628356156506, -0.114767810602876,
                   -0.119002343660951])  # -1.34639980735600

    Rx = np.array([1, 0, 0, 0, np.cos(xk[0]), np.sin(xk[0]), 0, -np.sin(xk[0]), np.cos(xk[0])]).reshape((3, 3))
    Ry = np.array([np.cos(xk[1]), 0, -np.sin(xk[1]), 0, 1, 0, np.sin(xk[1]), 0, np.cos(xk[1])]).reshape((3, 3))
    Rz = np.array([np.cos(xk[2]), -np.sin(xk[2]), 0, np.sin(xk[2]), np.cos(xk[2]), 0, 0, 0, 1]).reshape((3, 3))
    R = np.dot(Rx, Ry, Rz)
    T = np.array([xk[3], xk[4], xk[5]]).reshape((-1, 1))

    return R.T, T.T.squeeze(0)


def gen_R(thetas):
    Rx = np.array([1, 0, 0, 0, np.cos(thetas[0]), np.sin(thetas[0]), 0, -np.sin(thetas[0]), np.cos(thetas[0])]).reshape(
        (3, 3))
    Ry = np.array([np.cos(thetas[1]), 0, -np.sin(thetas[1]), 0, 1, 0, np.sin(thetas[1]), 0, np.cos(thetas[1])]).reshape(
        (3, 3))
    Rz = np.array([np.cos(thetas[2]), -np.sin(thetas[2]), 0, np.sin(thetas[2]), np.cos(thetas[2]), 0, 0, 0, 1]).reshape(
        (3, 3))
    R = np.dot(Rx, Ry, Rz)

    return R


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


if __name__ == '__main__':
    DRAW_ON_IMG = False
    DRAW_BEV = False

    # read data
    video_file = './pipeline/2021-01-06_15-18-18.mp4'
    # video_file = '/media/dataset/Berthing_Data/1月海试视频/2021-01-06_15-34-10.mp4'
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

    # test data
    xyzv_grid = xyzv_grid.T.astype(float)
    theta = [0.175, -0.055, 0]
    R = gen_R(theta)
    xyzv_grid[:, :3] = xyzv_grid[:, :3] @ R
    uv = get_uv(xyzv_grid, H)

    # save_img_from_video
    save_img_dir = '/media/dataset/cityscapes_berthing/sync_imgs_zhoushan/'
    count = 0

    while True:

        video_frame = video_g.__next__()
        radar_points = radar_g.__next__().T

        if video_frame is None or radar_points is None:
            break

        cv.imwrite(save_img_dir + "%05d" % count + '.jpg', video_frame)
        count += 1

        uv = get_uv(radar_points, H)
        points_semantic_ids, grid_map = get_points_semantic_ids(uv, video_frame, model)
        xyzvs = collect_xyzvs(radar_points, points_semantic_ids)

        comp_xyz = comp_xy0(radar_points)

        # print(comp_xyz)
        # print(radar_points)

        # test grid data project
        for i in range(uv.shape[1]):
            point = (uv[0, i].astype(int), uv[1, i].astype(int))
            color = [0, 255, 0]
            cv.circle(video_frame, point, 2, color, 8)
        cv.imshow("", video_frame)
        if cv.waitKey(1) == ord('q'):
            break

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

        # draw points on img
        if DRAW_ON_IMG:
            for i in range(uv.shape[1]):
                point = (uv[0, i].astype(int), uv[1, i].astype(int))
                # color = [(radar_frame[3, i] + 5) / 10 * 255, 255 - (radar_frame[3, i] + 5) / 10 * 255, 0]
                color = [0, 255, 0]
                cv.circle(grid_map, point, 2, color, 8)
            cv.imshow('frame', grid_map)
            if cv.waitKey(1) == ord('q'):
                break

        if DRAW_BEV:
            sns.jointplot(data=xyzvs, x='x', y='y', hue='s', xlim=[-10, 10], ylim=[0, 20])
            plt.show()

    # 完成所有操作后，释放捕获器
    cv.destroyAllWindows()
