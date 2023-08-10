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
import os
import cv2

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


def get_uv(radar_points, H):
    # project xyz to uv
    xyz = radar_points[:, 0:3].T
    xyz1 = np.concatenate([xyz, np.ones([1, xyz.shape[1]])])
    uv1 = np.dot(H, np.dot(xyz1, np.diag(1. / (1e-5 + np.dot(np.array([0, 0, 1]), np.dot(H, xyz1))))))
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


if __name__ == '__main__':

    # os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    gpu_options = tf.GPUOptions()
    gpu_options.visible_device_list = "2"

    DRAW_ON_IMG = True
    DRAW_BEV = True

    # data path
    radar_file = 'berthing_zhoushan/pipeline/group2.h5'
    img_file_dir = '/media/dataset/cityscapes_berthing/seaTest_80m_group2'

    # params
    concate_num_radar = 1
    both_offset_frame = 500  # in second
    img_faster_time = 0.1 * 1e6  # in second, best:0.25
    radar_start_time = 1632803548225300
    radar_frame_gap = 100000

    H, I = get_H_1280()

    # # TODO:wjx get I
    # I = np.array([[1445.63012662922, 0, 0],
    #               [0, 1445.10705595391, 0],
    #               [920.902144019657, 564.645986732188, 1]]).T  # intrinsic of our logi camera 1920*1080

    # create and initiate generators
    radar_g = read_radar_frame(radar_file, concate_num_radar, both_offset_frame)
    img_g = sync_img_by_name(img_file_dir, img_faster_time)
    # initiate img_g
    _ = img_g.__next__()

    # load deeplab model
    model = load_model()
    model.to('cuda:3')

    # # TODO:
    # H = np.array([[1259.97462186079, 1161.18658038759, -44.9099061739195, -328.171613142775],
    #               [-68.6180889630474, 465.074702108821, -1478.56571474723, -145.212293418375],
    #               [-0.176568697163514, 0.982155492048013, -0.0647617527704749, -0.0253043652589756]])
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

    # start output video
    width_out = img_width
    height_out = img_height
    # fps = 10
    # videowriter = cv2.VideoWriter(outfile + '.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps,
    #                               (width_out, height_out))

    # load model
    saver = tf.train.Saver([var for var in tf.model_variables()])
    with tf.Session() as sess:
        saver.restore(sess, ckpt_file)
        # TODO: changes start
        for radar_frame in radar_g:
            radar_data = radar_frame[0].T
            radar_time = radar_start_time + radar_frame_gap * radar_frame[1]
            img_file_path = os.path.join(img_file_dir, img_g.send(radar_time))
            img_data = read_img(img_file_path)
            # TODO: changes end

            video_frame = img_data
            radar_points = radar_data

            # TODO:wjx old fusion type: project radar points to img plane and get uv location
            # uv = get_uv(radar_points, H)
            # points_semantic_ids, grid_map = get_points_semantic_ids(uv, video_frame, model)
            # xyzvs = collect_xyzvs(radar_points, points_semantic_ids)

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

            # TODO:wjx call inverse projection
            bev_seg = convert_segimg_cam2bev(img_seg_maps, E, I, radar_info=radar_points,
                                             depth_map=img_depth_map)  # C*Z*X*Y
            # sum among z
            bev_seg_z = bev_seg.sum(axis=1)
            # # select z range
            # bev_seg_z = bev_seg[:, 20, :, :]

            # convert seg_img to rgb_img
            bev_seg_img = make_grid(decode_seg_map_sequence(np.argmax(bev_seg_z[None, :], axis=1)),
                                    3, normalize=False, range=(0, 255)).permute(2, 1, 0).detach().numpy()  # H, W, 3

            # todo: display process START
            # # show bev seg
            # cv2.imshow('frame', bev_seg_img)
            # cv2.waitKey(0)
            # # if cv2.waitKey(1) == ord('q'):
            # #     break
            #
            # # show uv
            # uv = get_uv(radar_points, H)
            # points_semantic_ids, grid_map = get_points_semantic_ids(uv, video_frame, model)
            #
            # for i in range(uv.shape[1]):
            #     point = (uv[0, i].astype(int), uv[1, i].astype(int))
            #     color = [(radar_points[i, 1] + 5) / 80 * 255, 0, 255 - (radar_points[i, 1] + 5) / 80 * 255]
            #     # color = [0, 255, 0]
            #     cv2.circle(grid_map, point, 2, color, 8)
            # cv2.imshow('frame', grid_map)
            # cv2.waitKey(0)
            # # if cv2.waitKey(1) == ord('q'):
            # #     break
            # todo: display process END

        # cv2.destroyAllWindows()
