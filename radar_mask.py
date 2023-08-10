from __future__ import division

"""
inputs:
    cam_seg: segmentation result on cam img (matrix with size of ori img [H*W] plus channel [C] presenting the semantic number)
    rots, trans, intrins: extrinsics(rots, trans) and intrinsics(intrins)

outputs:
    bev_seg: segmentation result on bev of size [C x Z x X x Y]

"""
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from berthing_zhoushan.infer_one_img import gain_img_seg, load_model
from berthing_zhoushan.pipeline.read_radar_frames import read_radar_frame
import pandas as pd

from scipy.spatial.transform import Rotation

from my_segimg_cam2bev import convert_segimg_cam2bev

from berthing_zhoushan.Image_process import get_H_1280, get_H_1920

from berthing_zhoushan.DeepLab_utils.dataloaders.utils import *
from torchvision.utils import make_grid, save_image

import argparse
import os
import glob
import tqdm
import sys
import numpy as np
from PIL import Image
import tensorflow as tf
from SfMLearner import SfMLearner
from utils import normalize_depth_for_display
import cv2

import torch.nn.functional as F

from skimage import measure
from sklearn.linear_model import LinearRegression
import h5py

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# class_names = ['sea', 'sky', 'shore', 'ship', 'pillar', 'bank', 'background']
class_names = ['sea', 'sky', 'shore', 'ship', 'background']
class_names.insert(0, 'unseen')


def read_img(file_path):
    pic = cv2.imread(file_path)
    # print(pic.shape)
    return pic


def sync_img_by_name(file_dir, radar_faster_time):
    pre_file_name = '0.'
    target_time = 0.
    for root, dirs, files in os.walk(file_dir):
        files = sorted(files)
        for file_name in files:
            while (eval('.'.join(pre_file_name.split('.')[:-1])) + eval(
                    '.'.join(file_name.split('.')[:-1]))) > 2 * target_time:
                target_time = yield pre_file_name
                target_time = radar_faster_time + target_time
            pre_file_name = file_name


def gen_R_ori(thetas):
    Rx = np.array([1, 0, 0, 0, np.cos(thetas[0]), np.sin(thetas[0]), 0, -np.sin(thetas[0]), np.cos(thetas[0])]).reshape(
        (3, 3))
    Ry = np.array([np.cos(thetas[1]), 0, -np.sin(thetas[1]), 0, 1, 0, np.sin(thetas[1]), 0, np.cos(thetas[1])]).reshape(
        (3, 3))
    Rz = np.array([np.cos(thetas[2]), -np.sin(thetas[2]), 0, np.sin(thetas[2]), np.cos(thetas[2]), 0, 0, 0, 1]).reshape(
        (3, 3))
    R = np.dot(np.dot(Rx, Ry), Rz)

    return R


def gen_R_ori_correct(thetas):
    Rx = np.array([1, 0, 0, 0, np.cos(thetas[0]), -np.sin(thetas[0]), 0, np.sin(thetas[0]), np.cos(thetas[0])]).reshape(
        (3, 3))
    Ry = np.array([np.cos(thetas[1]), 0, np.sin(thetas[1]), 0, 1, 0, -np.sin(thetas[1]), 0, np.cos(thetas[1])]).reshape(
        (3, 3))
    Rz = np.array([np.cos(thetas[2]), -np.sin(thetas[2]), 0, np.sin(thetas[2]), np.cos(thetas[2]), 0, 0, 0, 1]).reshape(
        (3, 3))
    R = np.dot(np.dot(Rz, Ry), Rx)

    return R


def gen_R(thetas, type='xyz'):
    r = Rotation.from_euler(type, thetas)
    R = r.as_matrix()

    return R


def get_uvd(radar_points, H, video_frame=None):
    # project xyz to uv
    xyz = radar_points[:, 0:3].T
    xyz1 = np.concatenate([xyz, np.ones([1, xyz.shape[1]])])
    uv1 = np.dot(H, np.dot(xyz1, np.diag(1. / (1e-5 + np.dot(np.array([0, 0, 1]), np.dot(H, xyz1))))))
    # # discrete
    # uv = np.floor(uv1[0:2, :]).astype(int)
    uv = uv1[0:2, :]
    d = np.dot(np.array([0, 0, 1]), np.dot(H, xyz1))[None, :]  # (1, n)
    uvd = np.concatenate([uv, d]).T

    if video_frame is not None:
        # display
        uv_int = np.floor(uv)
        for i in range(uv_int.shape[1]):
            point = (uv_int[0, i].astype(int), uv_int[1, i].astype(int))
            color = [0, 255, 0]
            cv2.circle(video_frame, point, 2, color, 8)
        cv2.imshow("", video_frame)
        cv2.waitKey(100)

    return uvd  # (n, 3)


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
    semantic_id_probs = img_seg_maps[:, uv[1, :], uv[0, :]].reshape([5, -1]).T

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

    mask = (xyzvs_array[:, 4] > 2.5) & (xyzvs_array[:, 4] < 3.5)  # TODO: 2: shore, 3: ship, 4: pillar
    xyzvs = xyzvs.loc[mask]
    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_columns', None)
    # print(xyzvs)

    return xyzvs


def mask_radar_point_by_img(img_mask, radar_points_uv):
    """
    img_mask: dim, h, w
    use F.grid_sample to accomplish
    """

    # todo: construct a sampling_grid:
    sampling_grid = radar_points_uv.reshape(1, 1, -1, 2).clone()  # H=1, W=num_points
    input_map = img_mask.unsqueeze(0)
    # print(input_map.shape)
    # normalize uv to [-1, 1] for grid sampling API
    _, _, h, w = input_map.shape
    sampling_grid[..., 0] = 2 * sampling_grid[..., 0] / h - 1
    sampling_grid[..., 1] = 2 * sampling_grid[..., 1] / w - 1

    output = F.grid_sample(input_map, sampling_grid, mode='bilinear')  # input: nchw, grid:nHW2

    return output.squeeze(0).squeeze(1).permute(1, 0)  # ncHW -> Wc(c=5 for seg, c=1 for depth)


def bfs_seg_objs(seg_max_index):
    # TODO: 2: shore, 3: ship, 4: pillar
    result_dict = {
        'shore': [],
        'ship': [],
        'pillar': [],
    }
    activate_classes = {
        'shore': 2,
        'ship': 3,
        'pillar': 4,
    }
    # convert to numpy to use skimage
    seg_max_index = seg_max_index.cpu().numpy()

    for act_name, act_ind in activate_classes:
        bi_seg_mask = (seg_max_index == act_ind)
        obj_masks = measure.label(bi_seg_mask, connectivity=2)
        for i in range(0, max(obj_masks)):
            result_dict[act_name].append(obj_masks == i + 1)

    return result_dict


def optimize_depth(points_uvd, points_p, with_k=True, intrinsics=None):
    reg = LinearRegression(fit_intercept=True)  # if false, only k
    K, b = None, None
    # model: w0 + w1 * x = y
    # todo: convert uv_depth to uv_dist_p
    fx = intrinsics[0, 0]
    cx = intrinsics[0, 2]
    points_t = points_uvd[..., 2] / np.cos((points_uvd[..., 0] - cx) / fx)
    if with_k:
        reg.fit(points_p.reshape(-1, 1), points_t.reshape(-1))
        K = reg.coef_[0]
        b = reg.intercept_
    else:
        b = np.mean(points_t.reshape(-1) - points_p.reshape(-1))

    return K, b


def read_radar(filePath, concate_num_radar, offset_frame):
    with h5py.File(filePath, 'r') as hf:
        radarPoints = np.array(hf['radar_points'])
        frameInfo = np.array(hf['frame_info'])
        frame_num = frameInfo.shape[1]
        # TODO: fix bug
        radarPoints[0, :] = -radarPoints[0, :]
    frame_data = np.zeros((9, 0))
    for i in range(frame_num):
        if i < offset_frame:
            continue
        frame_loc = frameInfo[0, i]
        frame_len = frameInfo[1, i]
        frame_data = np.concatenate((frame_data, radarPoints[:, frame_loc:frame_loc + frame_len]), axis=1)
        if (i - offset_frame) % concate_num_radar != concate_num_radar - 1:
            continue
        print('Radar_frame:', i)
        yield frame_data
        frame_data = np.zeros((9, 0))
    return None


def read_video(filePath, both_delay_frame_img, skip_num_image):
    cap = cv2.VideoCapture(filePath)
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


if __name__ == '__main__':

    # os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    gpu_options = tf.GPUOptions()
    gpu_options.visible_device_list = "2"

    DRAW_ON_IMG = True
    DRAW_BEV = True
    # DATA_TYPE = 'shandong'
    DATA_TYPE = 'zhoushan'

    if DATA_TYPE == 'shandong':
        # todo: if shandong:
        # data path
        radar_file = 'berthing_zhoushan/pipeline/group2.h5'
        img_file_dir = '/media/dataset/cityscapes_berthing/seaTest_80m_group2'

        # params
        concate_num_radar = 1
        both_offset_frame = 500  # in second
        img_faster_time = 0.1 * 1e6  # in second, best:0.25
        radar_start_time = 1632803548225300
        radar_frame_gap = 100000

        # create and initiate generators
        radar_g = read_radar_frame(radar_file, concate_num_radar, both_offset_frame)
        img_g = sync_img_by_name(img_file_dir, img_faster_time)
        # initiate img_g
        _ = img_g.__next__()
    if DATA_TYPE == 'zhoushan':
        # read data
        video_file = 'berthing_zhoushan/pipeline/2021-01-06_15-18-18.mp4'
        radar_file = "berthing_zhoushan/pipeline/old/test.h5"

        # params
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
    model.to('cuda:3')

    H, I = get_H_1280()
    E = np.dot(np.linalg.inv(I), H)  # get R|T from H and I

    fig, axes = plt.subplots(1, 1)

    # todo: tensorflow model
    img_height = 512
    img_width = 2048
    ckpt_file = '/media/personal_data/wujx/Projects2022/cylindricalsfmlearner-master/checkpoint/model.latest'
    # setup graph
    print('Setting up TensorFlow graph...')
    tf.reset_default_graph()  # for reruns
    sfm = SfMLearner()
    sfm.setup_inference(img_height,
                        img_width,
                        mode='depth')

    # load model
    saver = tf.train.Saver([var for var in tf.model_variables()])
    with tf.Session() as sess:
        saver.restore(sess, ckpt_file)
        # TODO: changes start
        while True:
            if DATA_TYPE == 'shandong':
                radar_frame = radar_g.__next__()
                if radar_frame is None:
                    break
                # for radar_frame in radar_g:
                radar_data = radar_frame[0].T
                radar_time = radar_start_time + radar_frame_gap * radar_frame[1]
                img_file_path = os.path.join(img_file_dir, img_g.send(radar_time))
                img_data = read_img(img_file_path)
                # TODO: changes end
                video_frame = img_data
                radar_points = radar_data
            if DATA_TYPE == 'zhoushan':
                video_frame = video_g.__next__()
                radar_points = radar_g.__next__().T
                if radar_points is None:
                    break
            # TODO:wjx new fusion type: get camera seg map and inverse project to bev plane
            img_seg_maps, _ = gain_img_seg(model, video_frame)
            # soft_max
            t = np.exp(img_seg_maps)
            img_seg_maps = t / np.sum(t, axis=0, keepdims=True)
            # print(grid_map.shape, frame.shape)
            # semantic_id_probs = img_seg_maps[:, uv[1, :], uv[0, :]].reshape([5, -1]).T

            # TODO:wjx get img depth
            # firstly run tf net to gen image depth
            IM = Image.fromarray(cv2.cvtColor(video_frame, cv2.COLOR_RGB2BGR))
            # I = Image.open(fh)
            IM = IM.resize((img_width, img_height), Image.ANTIALIAS)
            IM = np.array(IM)
            # run model
            pred = sfm.inference(IM[None, :, :, :], sess, mode='depth')
            # # output depth image for display
            # depth = normalize_depth_for_display(pred['depth'][0, :, :, 0], cmap=cmap)
            # O = np.uint8(depth * 255)
            # secondly get depth map
            img_depth_map = pred['depth'][0, :, :, 0]

            # img_seg_maps
            # img_depth_map
            depth_img = torch.tensor(img_depth_map)  # todo
            seg_maps = torch.tensor(img_seg_maps)  # todo size of 5, h, w

            cam_H, cam_W, cam_C = video_frame.shape
            cam_downsample = 2  # downsample cam seg image
            fH, fW = cam_H // cam_downsample, cam_W // cam_downsample

            # TODO: reformat img_seg_maps & img_depth_map
            # todo:1.1 reshape depth_img
            depth_img = depth_img.unsqueeze(0).unsqueeze(0)
            depth_img = F.interpolate(depth_img, size=(fH, fW), mode='bilinear')
            depth_map = depth_img.squeeze(0)  # 1, H, W

            # todo:1.2 reshape seg_map
            seg_maps = seg_maps.unsqueeze(0)
            seg_maps = F.interpolate(seg_maps, size=(fH, fW), mode='bilinear')
            seg_maps = seg_maps.squeeze(0)  # 5, H, W

            # todo:2 project to img
            # radar_points_uvd = get_uvd(radar_points, H, video_frame=video_frame)  # (n, 2)
            radar_points_uvd = get_uvd(radar_points, H)  # (n, 2)
            in_img_index = (radar_points_uvd[:, 0] > 0) \
                           & (radar_points_uvd[:, 0] < cam_H) \
                           & (radar_points_uvd[:, 1] > 0) \
                           & (radar_points_uvd[:, 1] < cam_W)
            # # todo; if mask by depth
            depth_range = [5, 50]
            proper_depth_index = (radar_points_uvd[:, 2] > depth_range[0]) \
                           & (radar_points_uvd[:, 2] < depth_range[1])
            in_img_index = in_img_index & proper_depth_index

            radar_points_uvd = radar_points_uvd[in_img_index, :]
            radar_points_uvd[:, 0] = radar_points_uvd[:, 0] * fH / cam_H
            radar_points_uvd[:, 1] = radar_points_uvd[:, 1] * fW / cam_W

            radar_points_uvd = torch.tensor(radar_points_uvd, dtype=torch.float)  # todo

            # todo:3 mask radar with semantic
            radar_points_p = mask_radar_point_by_img(depth_map, radar_points_uvd[:, :2])  # img depth(on radar position)

            # todo: all masks
            radar_points_s = mask_radar_point_by_img(seg_maps, radar_points_uvd[:, :2])  # radar semantic
            # todo: foreground mask: 1. get img_mask from seg 2. get point_mask from img_mask 3. sample points by mask
            # class_names = ['sea', 'sky', 'shore', 'ship', 'background'] ? # TODO: 2: shore, 3: ship, 4: pillar
            seg_max, seg_max_index = torch.max(seg_maps, dim=0)
            fg_seg_mask = (seg_max_index > 1.5) & (seg_max_index < 4.5)
            fg_points_mask = mask_radar_point_by_img(fg_seg_mask[None, ...].type_as(radar_points_uvd), radar_points_uvd[:, :2])
            BORDER_THRES = 0.5
            fg_radar_points_uvd = radar_points_uvd[fg_points_mask.flatten() > BORDER_THRES, :]
            fg_radar_points_p = radar_points_p[fg_points_mask.flatten() > BORDER_THRES, :]

            # todo: use the above two to optimize depth_map by fg_radar_points, determine k, b_global(bg)
            k, bg = optimize_depth(fg_radar_points_uvd, fg_radar_points_p, with_k=True, intrinsics=I)
            # optimize
            depth_map = k * depth_map + bg

            # todo: object mask: bfs to seperate each object
            obj_seg_mask_dict = bfs_seg_objs(seg_max_index)

            for obj_class, obj_seg_mask_list in obj_seg_mask_dict:
                for obj_seg_mask in obj_seg_mask_list:
                    obj_points_mask = mask_radar_point_by_img(obj_seg_mask, radar_points_uvd[:, :2])
                    # get obj_frustum depth_surface and corresponding object points
                    obj_depth_surface = depth_map[:, obj_seg_mask]
                    obj_points_uvd = radar_points_uvd[obj_points_mask > BORDER_THRES, :]
                    obj_points_p = radar_points_p[obj_points_mask > BORDER_THRES, :]

                    # todo: use the above two to optimize obj_depth by obj_points, determine b_individual(bi)
                    _, bi = optimize_depth(obj_points_uvd, obj_points_p, with_k=False, intrinsics=I)
                    # optimize
                    obj_depth_surface = obj_depth_surface + bi

                    # todo: inverse_project to bev
