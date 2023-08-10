import numpy as np
import sklearn.cluster as skc  # 密度聚类
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import cv2

from data_utils import get_uv, get_H
from upsample_ordered_hull import *

class_names = ['sea', 'sky', 'shore', 'ship', 'pillar', 'bank', 'background']
class_names.insert(0, 'unseen')


def points_filter(points):
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


def cal_metric(targets, targets_gt):
    # cal IoU matrix
    IoU_matrix = []
    points_num = 0
    for target in targets:
        NOarea_pred = target['noarea']
        points_num += target['points'].shape[0]
        for target_gt in targets_gt:
            NOarea_gt = target_gt['noarea']
            if target['class'] != 3 and target['class'] != target_gt['class']:
                IoU = 0.
            else:
                IoU = cal_iou(NOarea_pred, NOarea_gt)
            IoU_matrix.append(IoU)
    IoU_matrix = np.array(IoU_matrix).reshape([len(targets), len(targets_gt)])

    FP_points_list = []
    # each pred set to gt
    for i in range(len(targets)):
        FP_points_num = 0
        line = IoU_matrix[i, :]
        new_line = np.zeros_like(line)
        max_iou = np.max(line)
        if max_iou > 0.1:
            new_line[np.argmax(line)] = max_iou
        elif max_iou > 0.0:
            # print("FP at:", targets[i]['location'])
            FP_points_num = targets[i]['points'].shape[0]
        FP_points_list.append(FP_points_num)
        IoU_matrix[i, :] = new_line

    # cal_each_gt match  (best way is to merge the pred noarea)
    # merge

    TP_NOO = []
    for i in range(len(targets_gt)):
        poly_gt = targets_gt[i]['noarea']
        line = IoU_matrix[:, i]

        indices = line > 0
        # poly_merge_list = [target['noarea'] for target, indice in zip(targets, indices) if indice]
        # NOO = cal_merge_IoU(poly_merge_list, poly_gt)
        target_list = [target for target, indice in zip(targets, indices) if indice]
        if len(target_list) == 0:
            TP_NOO.append(0.)
        else:
            poly_pred = cal_NOareas(target_list)
            NOO = cal_iou(poly_pred, poly_gt)
            TP_NOO.append(NOO)

    P_labels = [item['label'] for item in targets_gt]
    FP = sum(FP_points_list) / points_num
    # print(P_labels)
    # print(FP)
    # print(TP_NOO)
    # print(IoU_matrix)

    return TP_NOO, P_labels, FP


def cal_iou(poly1, poly2):
    color = (1, 0, 0)
    color2 = (0, 1, 0)

    img = np.zeros([1000, 2000, 2])
    triangle1 = (poly1 + np.array([20, 0])) * 50
    triangle1 = triangle1.astype(int)
    cv2.fillConvexPoly(img, triangle1, color)
    area1 = img.sum()

    img = np.zeros([1000, 2000, 3])
    triangle2 = (poly2 + np.array([20, 0])) * 50
    triangle2 = triangle2.astype(int)
    cv2.fillConvexPoly(img, triangle2, color2)
    area2 = img.sum()

    cv2.fillConvexPoly(img, triangle1, color)
    union_area = img.sum()
    inter_area = area1 + area2 - union_area
    IOU = inter_area / union_area

    # print(IOU)
    return IOU


def upsample_hulls(target_gts, mask_lines):
    # sample edge_points to maintain hull shape
    # hulls: NOO or truth_points, list of size[n, 2] points
    # mask_lines: Ax+By+C=0, list of [A,B,C]

    for target_gt in target_gts:
        hull = target_gt['noarea']  # NOarea_gt
        for mask_line in mask_lines:
            # todo: firstly, find hits index & points
            hits_index, hits_pairs = find_hits(hull, mask_line)
            if hits_index.shape[0] == 0:
                continue
            # todo: secondly, calculate hits
            hits = cal_hits(hits_pairs, mask_line)
            # todo: thirdly, insert hits to hull, NOTEs that hull changes in each inner loop
            hull = insert_hits(hull, hits, hits_index)
        target_gt['noarea'] = hull


def mask_hulls(target_gts):
    for target_gt in target_gts:
        hull = target_gt['noarea']  # NOarea_gt
        # todo: mask
        pp = 1e-5
        mask0 = hull[:, 1] <= 20 + pp
        mask1 = hull[:, 1] >= 0 - pp
        mask2 = abs(hull[:, 0]) <= 20 + pp
        mask_range_detect = np.bitwise_and(mask0, mask1, mask2)
        # todo: for pseudo points add fov mask
        # mask_fov = get_fov_mask(truth_points)
        # mask_final = np.bitwise_and(mask_fov, mask_range_detect)
        target_gt['noarea'] = hull[mask_range_detect, :]


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


def cal_NOarea(original_hull):
    original_hull_a = np.concatenate([np.array([0., 0.]).reshape(1, 2), original_hull], axis=0)
    out_hull_o = list(ConvexHull(original_hull_a).vertices)
    out_hull_c = out_hull_o.index(0)
    out_hull_cl = out_hull_o[out_hull_c - 1]
    out_hull_cr = out_hull_o[(out_hull_c + 1) % len(out_hull_o)]

    ids = [0, out_hull_cl]
    id = out_hull_cl + 1
    while True:
        if id > len(original_hull):
            id = id - len(original_hull)
        ids.append(id)
        if id == out_hull_cr:
            break
        id += 1
    NOarea = original_hull_a[ids, :]

    return NOarea


def cal_NOareas(target_list):
    all_points = []
    for target in target_list:
        all_points.append(target['points'])
    all_points = np.concatenate(all_points, axis=0)
    hull = ConvexHull(all_points)

    all_dimension = all_points[hull.vertices, :]

    return cal_NOarea(all_dimension)


def get_fov_mask(points_xyz, cam_H=720, cam_W=1280):
    points_uv = get_uv(points_xyz, get_H())
    # fov masks
    maskH0 = points_uv[1, :] > 0
    maskH1 = points_uv[1, :] < cam_H
    maskH = np.bitwise_and(maskH0, maskH1)
    maskW0 = points_uv[0, :] > 0
    maskW1 = points_uv[0, :] < cam_W
    maskW = np.bitwise_and(maskW0, maskW1)
    # maskW = np.ones_like(maskH)
    mask_fov = np.bitwise_and(maskH, maskW)

    return mask_fov


def pseudo_point_detect(xyzs):
    points = points_filter(convert_point_feature(xyzs))  # pre-processing

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


def pseudo_point_detect_and_eval(xyzs, truth_points, SHOW=True):
    noo_score = None
    false_alarm_rate = None

    if truth_points is not None:
        noo_score = -1 * np.ones(len(set(truth_points[:, -1])))

        # pre-process points
        points = points_filter(convert_point_feature(xyzs))  # pre-processing

        # draw points
        if SHOW:
            fig, axes = plt.subplots(1, 1)
            plt.cla()
            plt.plot(points[:, 0], points[:, 1], 'o', c='#000000')

        # DBSCAN to detect
        eps = 0.5
        min_samples = 15  # 16
        db = skc.DBSCAN(eps=eps, min_samples=min_samples).fit(points)
        labels = db.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # 获取分簇的数目

        # extract detection
        targets = []
        for i in range(n_clusters):
            one_cluster = points[labels == i]

            try:
                hull = ConvexHull(one_cluster[:, :2])
            except:
                continue

            target_dimension = one_cluster[hull.vertices, :2]
            target_pointnum = one_cluster.shape[0]
            target_class = np.mean(one_cluster[:, 3:], axis=0) / np.array([0.3, 0.1, 0.2])
            target_center = np.mean(one_cluster[:, :2], axis=0)

            if target_pointnum < 35 and max(target_class) < 0.3:
                continue

            target = {'hull': target_dimension, 'location': target_center, 'points': one_cluster[:, :2],
                      'class': np.argmax(target_class)}

            if np.max(target_class) < 0.1:
                target['class'] = 3

            NOarea = cal_NOarea(target['hull'])
            target['noarea'] = NOarea
            targets.append(target)

            if SHOW:
                plt.plot(one_cluster[:, 0], one_cluster[:, 1], 'o')
                plt.plot(target_center[0], target_center[1], '.r')
                # for simplex in hull.simplices:
                #     plt.plot(one_cluster[simplex, 0], one_cluster[simplex, 1], 'b-')
                for j in range(NOarea.shape[0]):
                    plt.plot([NOarea[j - 1, 0], NOarea[j, 0]], [NOarea[j - 1, 1], NOarea[j, 1]], 'b-')

        # pre-process gt points
        mask0 = truth_points[:, 1] < 20
        mask1 = abs(truth_points[:, 0]) < 20
        mask_range_detect = np.bitwise_and(mask0, mask1)
        # todo: for pseudo points add fov mask
        mask_fov = get_fov_mask(truth_points)
        mask_final = np.bitwise_and(mask_fov, mask_range_detect)
        # todo: disable pre point mask
        # truth_points = truth_points[mask_final, :]
        labels_gt = truth_points[:, -1]

        # extract gt
        targets_gt = []
        for label_gt in set(labels_gt):
            one_cluster_gt = truth_points[labels_gt == label_gt, :]
            if one_cluster_gt.shape[0] < 4:
                continue

            hull_gt = ConvexHull(one_cluster_gt[:, :2])
            target_gt_dimension = one_cluster_gt[hull_gt.vertices, :2]
            target_gt_class = int(one_cluster_gt[0, 2])
            target_gt_center = np.mean(one_cluster_gt[:, :2], axis=0)
            NOarea_gt = cal_NOarea(target_gt_dimension)

            target_gt = {'hull': target_gt_dimension, 'location': target_gt_center, 'points': one_cluster_gt[:, :2],
                         'class': target_gt_class, 'noarea': NOarea_gt, 'label': label_gt}

            targets_gt.append(target_gt)

            if SHOW:
                for simplex in hull_gt.simplices:
                    plt.plot(one_cluster_gt[simplex, 0], one_cluster_gt[simplex, 1], 'g-')
                for j in range(NOarea_gt.shape[0]):
                    plt.plot([NOarea_gt[j - 1, 0], NOarea_gt[j, 0]], [NOarea_gt[j - 1, 1], NOarea_gt[j, 1]], 'g-')

        # todo: process targets_gt to fulfill mask
        mask_lines = [[0, 1, 0], [0, 1, -20], [1, 0, 20], [1, 0, -20]]  # list of [A, B, C]
        upsample_hulls(targets_gt, mask_lines)
        # todo: process post mask (mind the precision)
        mask_hulls(targets_gt)  # ignore truncation

        # cal_metric
        noos, p_label, fp = cal_metric(targets, targets_gt)
        for target_id, noo in zip(p_label, noos):
            noo_score[int(target_id)] = noo
        false_alarm_rate = fp

        # print(noo_score, false_alarm_rate)
        if SHOW:
            axes.set_aspect('equal')
            plt.xlim([-20, 20])
            plt.ylim([0, 20])
            plt.show()

    return noo_score, false_alarm_rate
