import numpy as np
import matplotlib.pyplot as plt

import sklearn.cluster as skc  # 密度聚类
from scipy.spatial import ConvexHull
import cv2

class_list = ['shore', 'ship', 'pillar', 'unsure']


def current_motion_extraction():
    return None


def points_filter(points, frame_ids):
    points[:, 2] = 0
    rm_range = 5
    mask_range_near = np.linalg.norm(points[:, 0:3], axis=1) > rm_range
    # mask_item = np.sum(points[:, 3:], axis=1) != 0
    mask_item = points[:, 5] != 0
    mask_rm_noise = np.bitwise_or(mask_range_near, mask_item)

    # scope mask
    mask0 = points[:, 1] < 20
    mask1 = abs(points[:, 0]) < 20
    mask_range_detect = np.bitwise_and(mask0, mask1)
    # time mask
    time_lenth = 100
    mask_time = frame_ids > (np.max(frame_ids) - time_lenth)
    mask_detect = np.bitwise_and(mask_range_detect, mask_time)

    # cut out
    mask = np.bitwise_and(mask_rm_noise, mask_detect)
    points = points[mask, :]

    # set semantic weights
    points[:, 3:] = points[:, 3:] * np.array([0.3, 0.1, 0.2])  # ['shore', 'ship', 'pillar']

    return points


def current_detection_old(detection_points, frame_ids, truth_points, SHOW=True):
    noo_score = None
    false_alarm_rate = None

    points = detection_points.copy()

    points = points_filter(points, frame_ids)

    # DBSCAN
    eps = 0.5
    min_samples = 15  # 16

    db = skc.DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = db.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # 获取分簇的数目

    # if np.max(frame_ids) > 30:
    if truth_points is not None:
        # draw points with semantic info
        if SHOW:
            fig, axes = plt.subplots(1, 1)
            plt.cla()
            plt.plot(points[:, 0], points[:, 1], 'o', c='#000000')
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

            # print(target_center, class_list[np.argmax(target_class)], np.max(target_class), target_pointnum)

            if SHOW:
                plt.plot(one_cluster[:, 0], one_cluster[:, 1], 'o')
                plt.plot(target_center[0], target_center[1], '.r')
                for simplex in hull.simplices:
                    plt.plot(one_cluster[simplex, 0], one_cluster[simplex, 1], 'b-')
                for j in range(NOarea.shape[0]):
                    plt.plot([NOarea[j - 1, 0], NOarea[j, 0]], [NOarea[j - 1, 1], NOarea[j, 1]], 'b-')

            # TODO
            targets_gt = []

            noo_score = -1 * np.ones(len(set(truth_points[:, -1])))

            # mask0 = np.bitwise_and(truth_points[:, 1] < 20, truth_points[:, 1] > 0)

            # mask_range_detect = np.linalg.norm(truth_points[:, :2], axis=1) < 20
            mask0 = truth_points[:, 1] < 20
            mask1 = abs(truth_points[:, 0]) < 20
            mask_range_detect = np.bitwise_and(mask0, mask1)
            truth_points = truth_points[mask_range_detect, :]

            labels_gt = truth_points[:, -1]

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

        noos, p_label, fp = cal_metric(targets, targets_gt)
        for target_id, noo in zip(p_label, noos):
            noo_score[int(target_id)] = noo
        false_alarm_rate = fp

        if SHOW:
            axes.set_aspect('equal')
            plt.xlim([-20, 20])
            plt.ylim([0, 20])
            plt.show()

    # print(noo_score, false_alarm_rate)

    return noo_score, false_alarm_rate


def current_detection(detection_points, frame_ids, truth_points, SHOW=True):
    noo_score = None
    false_alarm_rate = None

    if truth_points is not None:

        noo_score = -1 * np.ones(len(set(truth_points[:, -1])))

        # pre-process points
        points = detection_points.copy()
        points = points_filter(points, frame_ids)

        # draw points with semantic info
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
                for simplex in hull.simplices:
                    plt.plot(one_cluster[simplex, 0], one_cluster[simplex, 1], 'b-')
                for j in range(NOarea.shape[0]):
                    plt.plot([NOarea[j - 1, 0], NOarea[j, 0]], [NOarea[j - 1, 1], NOarea[j, 1]], 'b-')

        # pre-process gt points
        # mask0 = np.bitwise_and(truth_points[:, 1] < 20, truth_points[:, 1] > 0)
        # mask_range_detect = np.linalg.norm(truth_points[:, :2], axis=1) < 20
        mask0 = truth_points[:, 1] < 20
        mask1 = abs(truth_points[:, 0]) < 20
        mask_range_detect = np.bitwise_and(mask0, mask1)
        truth_points = truth_points[mask_range_detect, :]
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


def current_detection2(detection_points, frame_ids, truth_points):

    # pre-process points
    points = detection_points.copy()
    points = points_filter(points, frame_ids)

    # DBSCAN to detect
    eps = 0.5
    min_samples = 15  # 16
    db = skc.DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = db.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # 获取分簇的数目
    noise_points = points[labels == -1]

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

    # pre-process gt points
    # mask0 = np.bitwise_and(truth_points[:, 1] < 20, truth_points[:, 1] > 0)
    # mask_range_detect = np.linalg.norm(truth_points[:, :2], axis=1) < 20
    mask0 = truth_points[:, 1] < 30
    mask1 = abs(truth_points[:, 0]) < 30
    mask_range_detect = np.bitwise_and(mask0, mask1)
    truth_points = truth_points[mask_range_detect, :]
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

    return noise_points, targets, targets_gt


def current_detection3(detection_points, frame_ids, truth_points):

    # pre-process points
    points = detection_points.copy()
    points = points_filter(points, frame_ids)

    # DBSCAN to detect
    eps = 0.5
    min_samples = 15  # 16
    db = skc.DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = db.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # 获取分簇的数目
    noise_points = points[labels == -1]

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

        NOarea = cal_NOarea3(target['hull'])
        target['noarea'] = NOarea
        targets.append(target)

    # pre-process gt points
    # mask0 = np.bitwise_and(truth_points[:, 1] < 20, truth_points[:, 1] > 0)
    # mask_range_detect = np.linalg.norm(truth_points[:, :2], axis=1) < 20
    mask0 = truth_points[:, 1] < 30
    mask1 = abs(truth_points[:, 0]) < 30
    mask_range_detect = np.bitwise_and(mask0, mask1)
    truth_points = truth_points[mask_range_detect, :]
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
        NOarea_gt = cal_NOarea3(target_gt_dimension)

        target_gt = {'hull': target_gt_dimension, 'location': target_gt_center, 'points': one_cluster_gt[:, :2],
                     'class': target_gt_class, 'noarea': NOarea_gt, 'label': label_gt}

        targets_gt.append(target_gt)

    return noise_points, targets, targets_gt


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


def cal_NOarea3(original_hull):
    original_hull_a = np.concatenate([np.array([0., 22.]).reshape(1, 2), original_hull], axis=0)
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


def cal_metric2(targets, targets_gt):
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

    return TP_NOO, P_labels, FP, FP_points_list


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


def cal_merge_IoU(poly_list, poly_gt):
    color = (1, 0, 0)

    img = np.zeros([1000, 2000, 3])
    triangle_gt = (poly_gt + np.array([20, 0])) * 50
    triangle_gt = triangle_gt.astype(int)
    cv2.fillConvexPoly(img, triangle_gt, color)
    area_gt = img.sum()

    img = np.zeros([1000, 2000, 3])
    for poly in poly_list:
        triangle = (poly + np.array([20, 0])) * 50
        triangle = triangle.astype(int)
        cv2.fillConvexPoly(img, triangle, color)
    area_merge = img.sum()

    cv2.fillConvexPoly(img, triangle_gt, color)
    union_area = img.sum()
    inter_area = area_gt + area_merge - union_area
    IOU = inter_area / union_area

    return IOU
