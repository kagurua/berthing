import numpy as np
import transforms3d as t3d
import matplotlib.pyplot as plt
import os
import cv2

# os.chdir('/media/personal_data/wujx/Boat_berthing/')
npy_src = "./gt_data/output_list.npy"
goo_src = "./gt_data/google_list_full00.npy"
# goo_src = "./gt_data/google_list_full0.npy"
# goo_src = "./gt_data/google_list_full.npy"
map_pic = "./gt_data/5m equals 72 pixel.jpg"
output_list = np.load(npy_src, allow_pickle=True)
google_list_full = np.load(goo_src, allow_pickle=True)
a_dict = {'small_ship': [1, 0],
          'coast': [0, 1],
          'pi_1': [2, 2],
          'pi_2': [2, 3],
          'pi_3': [2, 4],
          'pi_4': [2, 5],
          'big_ship': [1, 6],
          'side': [0, 7]}  # semantic and label_id
a_dict0 = {'small_ship0': [1, 0],
          'coast0': [0, 1],
          'pi_10': [2, 2],
          'pi_20': [2, 3],
          'pi_30': [2, 4],
          'pi_40': [2, 5],
          'big_ship0': [1, 6]}  # semantic and label_id
a_dict00 = {'small_ship00': [1, 0],
          'coast00': [0, 1],
          'pi_10': [2, 2],
          'pi_20': [2, 3],
          'pi_30': [2, 4],
          'pi_40': [2, 5],
          'big_ship0': [1, 6]}  # semantic and label_id

all_target_points = []
for item in google_list_full:
    x, y = item['pos']
    y = -y
    name = item['name']

    # if name == 'big_ship':
    #     continue

    if name == 'side':
        continue

    all_target_points.append([x, y] + a_dict00[name])
all_target_points = np.array(all_target_points).astype('float')
all_target_points[:, :2] = all_target_points[:, :2] * 5. / 72.

img = cv2.imread(map_pic)
rows, cols, channels = img.shape


def load_gt2(target_frame, deviation=np.array([0, 0, 0.])):
    now_points = None
    now_map = None

    resize_ratio = 5. / 72
    offset_range = 22

    for item in output_list:

        if item['frame'] > target_frame:
            print('Target frame:', target_frame, '(no data)')
            return now_points, now_map

        if item['frame'] < target_frame:
            continue

        x, y = item['pos']
        direction = item['direction']
        x += deviation[0]
        y += deviation[1]
        direction += deviation[-1]
        x = int(x)
        y = int(y)

        # get current map
        now_map = img.copy()
        rotate_matrix = cv2.getRotationMatrix2D((x, y), (np.pi / 2 - direction) / np.pi * 180, 1)
        translate_matrix = np.float32([[1, 0, -(x - offset_range / resize_ratio)],
                                       [0, 1, -(y - offset_range / resize_ratio)]])
        now_map = cv2.warpAffine(now_map, rotate_matrix, (cols, rows))
        now_map = cv2.warpAffine(now_map, translate_matrix,
                                 (int(2 * offset_range / resize_ratio) + 1, int(offset_range / resize_ratio) + 1))

        # cv2.imshow('img', now_map)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # get current points
        y = -y
        euler = [0.0, 0.0, -(np.pi / 2 - direction)]
        rot_matrix = t3d.euler.euler2mat(*euler)[:2, :2]
        translation = -np.array([x, y]).astype('float') * 5. / 72.
        now_points = all_target_points.copy()
        now_points[:, :2] = (now_points[:, :2] + translation) @ rot_matrix

        # # display map
        # display_range = 15
        # ar = np.zeros((now_map.shape[0], now_map.shape[1]))
        # for i in range(now_map.shape[0]):
        #     for j in range(now_map.shape[1]):
        #         if (i - offset_range / resize_ratio) ** 2 + (j - offset_range / resize_ratio) ** 2 < (
        #                 display_range / resize_ratio) ** 2:
        #             ar[i, j] = np.nan
        # now_map = now_map[..., ::-1]
        # fig, ax = plt.subplots()
        # ax.imshow(now_map)
        # ax.imshow(ar, alpha=0.2, cmap="RdBu")
        # ax.axis('on')  # 关掉坐标轴为 off
        # my_x_ticks = np.arange(0, int(2 * offset_range / resize_ratio) + 1, 1 / resize_ratio * 4)
        # my_y_ticks = np.arange(0, int(offset_range / resize_ratio) + 1, 1 / resize_ratio * 4)
        # plt.xticks(my_x_ticks)
        # plt.yticks(my_y_ticks)
        #
        # def format_func_x(value, tick_number):
        #     return int(value * resize_ratio - offset_range)
        #
        # def format_func_y(value, tick_number):
        #     return -int(value * resize_ratio - offset_range)
        #
        # ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func_x))
        # ax.yaxis.set_major_formatter(plt.FuncFormatter(format_func_y))
        # # display points
        # now_points[:, 1] = -now_points[:, 1]
        # now_points[:, :2] = now_points[:, :2] / resize_ratio
        # now_points[:, :2] = now_points[:, :2] + offset_range / resize_ratio
        # ax.set_xlim(0, int(2 * offset_range / resize_ratio) + 1)
        # ax.set_ylim(int(offset_range / resize_ratio) + 1, 0)
        # ax.plot(now_points[:, 0], now_points[:, 1], '.k')
        # plt.show()

        print('Target frame:', item['frame'])
        return now_points, now_map


def load_gt(delay_frame=0):
    # fig, axes = plt.subplots(1, 1)
    # axes.set_aspect('equal')
    pre_transform_list = None
    pre_translation = None
    pre_direction = None

    for item in output_list:
        x, y = item['pos']
        direction = item['direction']
        y = -y
        euler = [0.0, 0.0, -(np.pi / 2 - direction)]
        rot_matrix = t3d.euler.euler2mat(*euler)[:2, :2]
        translation = -np.array([x, y]).astype('float') * 5. / 72.
        # print(rot_matrix, translation)
        now_points = all_target_points.copy()

        now_points[:, :2] = (now_points[:, :2] + translation) @ rot_matrix
        # print(set(now_points[:, -1]))

        if pre_translation is not None:
            T = translation - pre_translation
            T = T @ t3d.euler.euler2mat(0.0, 0.0, -(np.pi / 2 - pre_direction))[:2, :2]
            T = -np.concatenate([T, [0.]])
            R = t3d.euler.euler2mat(0.0, 0.0, direction - pre_direction).T
            pre_transform_list = [R, T]

        print('Truth frame:', item['frame'])
        if item['frame'] < delay_frame:
            continue
        yield now_points, item['frame'], pre_transform_list

        pre_translation = translation
        pre_direction = direction


def load_gt(delay_frame=0):
    # fig, axes = plt.subplots(1, 1)
    # axes.set_aspect('equal')
    pre_transform_list = None
    pre_translation = None
    pre_direction = None

    for item in output_list:
        x, y = item['pos']
        direction = item['direction']
        y = -y
        euler = [0.0, 0.0, -(np.pi / 2 - direction)]
        rot_matrix = t3d.euler.euler2mat(*euler)[:2, :2]
        translation = -np.array([x, y]).astype('float') * 5. / 72.
        # print(rot_matrix, translation)
        now_points = all_target_points.copy()

        now_points[:, :2] = (now_points[:, :2] + translation) @ rot_matrix
        # print(set(now_points[:, -1]))

        if pre_translation is not None:
            T = translation - pre_translation
            T = T @ t3d.euler.euler2mat(0.0, 0.0, -(np.pi / 2 - pre_direction))[:2, :2]
            T = -np.concatenate([T, [0.]])
            R = t3d.euler.euler2mat(0.0, 0.0, direction - pre_direction).T
            pre_transform_list = [R, T]

        print('Truth frame:', item['frame'])
        if item['frame'] < delay_frame:
            continue
        yield now_points, item['frame'], pre_transform_list

        pre_translation = translation
        pre_direction = direction

    # print(all_target_points)

    # print(item['frame'])
    # # draw
    # fig, axes = plt.subplots(1, 1)
    # axes.set_aspect('equal')
    # axes.plot(now_points[:, 0], now_points[:, 1], '.k')
    # axes.set_xlim([-20, 20])
    # axes.set_ylim([0, 20])
    # # plt.pause(0.5)
    # # plt.cla()
    # plt.show()


def load_gt_new(delay_frame=45, concate_num_radar=5):
    pre_transform_list = None
    pre_translation = None
    pre_direction = None
    now_points = None

    for item in output_list:

        delay_frame += concate_num_radar
        while item['frame'] > delay_frame:
            print('Truth frame:', delay_frame - 1, '(no data)')
            delay_frame += concate_num_radar
            yield now_points, item['frame'], pre_transform_list

        x, y = item['pos']
        direction = item['direction']
        y = -y
        euler = [0.0, 0.0, -(np.pi / 2 - direction)]
        rot_matrix = t3d.euler.euler2mat(*euler)[:2, :2]
        translation = -np.array([x, y]).astype('float') * 5. / 72.
        # print(rot_matrix, translation)
        now_points = all_target_points.copy()

        now_points[:, :2] = (now_points[:, :2] + translation) @ rot_matrix

        if pre_translation is not None:
            T = translation - pre_translation
            T = T @ t3d.euler.euler2mat(0.0, 0.0, -(np.pi / 2 - pre_direction))[:2, :2]
            T = -np.concatenate([T, [0.]])
            R = t3d.euler.euler2mat(0.0, 0.0, direction - pre_direction).T
            pre_transform_list = [R, T]

        print('Truth frame:', item['frame'])
        yield now_points, item['frame'], pre_transform_list

        pre_translation = translation
        pre_direction = direction


if __name__ == '__main__':
    # g = load_gt()
    # while True:
    #     points, frame = g.__next__()
    #     print(points, frame)
    target_frame = 204
    load_gt2(target_frame, deviation=np.array([0, 0, 0.]))
