import numpy as np
import math


# import xml.etree.ElementTree as ET
# import glob
# from centernet_image_util import draw_dense_reg, draw_msra_gaussian, draw_umich_gaussian
# from centernet_image_util import get_affine_transform, affine_transform, gaussian_radius


# data_dir = r"*.jpg"
# a_file = glob.glob(data_dir)[0]
# print(a_file, a_file.replace(".jpg", ".xml"))
#
# tree = ET.parse(a_file.replace(".jpg", ".xml"))
# root = tree.getroot()
# size = root.find('size')
# width = int(size.find('width').text)
# height = int(size.find('height').text)
# print(f"原图宽：{width} 高：{height}")
#
# num_classes = 3
# output_h = height
# output_w = width
# hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
#
# anns = []
# for obj in root.iter('object'):
#     bbox = obj.find('bndbox')
#     cate = obj.find('name').text
#     # print(cate, bbox.find("xmin").text, bbox.find("xmax").text,
#     #       bbox.find("ymin").text, bbox.find("ymax").text)
#     xyxy = [int(bbox.find("xmin").text), int(bbox.find("ymin").text),
#           int(bbox.find("xmax").text),int(bbox.find("ymax").text)]
#     anns.append({"bbox" : xyxy,'category_id':int(cate)})
#
# num_objs = len(anns)
# flipped = False #是否经过全图翻转
#
# import matplotlib.pyplot as plt
# plt.figure(figsize=(19, 6))
# plt.ion()
# plt.subplot(131)
# img = plt.imread(a_file)
# plt.title('Origin_img')
# plt.imshow(img)
#
# for k in range(num_objs):
#     ann = anns[k]
#     bbox = ann['bbox']
#     cls_id = ann['category_id']
#     if flipped:
#         bbox[[0, 2]] = width - bbox[[2, 0]] - 1
#     # bbox[:2] = affine_transform(bbox[:2], trans_output)# 仿射变换
#     # bbox[2:] = affine_transform(bbox[2:], trans_output)
#     # bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)#裁剪
#     # bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
#     h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
#     if h > 0 and w > 0:
#         radius = gaussian_radius((math.ceil(h), math.ceil(w)))
#         radius = max(0, int(radius))
#         # radius = self.opt.hm_gauss if self.opt.mse_loss else radius
#         ct = np.array(
#             [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
#         ct_int = ct.astype(np.int32)
#         plt.subplot(133)
#         hm_out, gaussian = draw_umich_gaussian(hm[cls_id], ct_int, radius)
#         plt.title('Umich Heatmap')
#         # hm_out = draw_msra_gaussian(hm[cls_id], ct_int, radius)
#         # print(hm_out.shape)
#         # plt.title("Mara Heatmap")
#         plt.text(ct[0], ct[1], f"(class:{cls_id})", c='white')
#         plt.plot([bbox[0], bbox[2], bbox[2], bbox[0], bbox[0]], [bbox[1], bbox[1], bbox[3], bbox[3], bbox[1]])
#         plt.imshow(hm_out)
#         plt.subplot(132)
#         plt.title(f'Gaussian: bbox_h={h},bbox_w={w}, radius={radius}')
#         plt.imshow(gaussian)
#         plt.pause(2)


# def gaussian2D(shape, sigma_x=1, sigma_y=1):  # sigma_x stands for sigma_v !!!!
#     m, n = [(ss - 1.) / 2. for ss in shape]
#     x, y = np.ogrid[-m:m + 1, -n:n + 1]
#     # print(x)
#
#     hx = np.exp(-(x * x) / (2 * sigma_x * sigma_x)) / (math.sqrt(2 * math.pi) * sigma_x)
#     hy = np.exp(-(y * y) / (2 * sigma_y * sigma_y)) / (math.sqrt(2 * math.pi) * sigma_y)
#     h = hx @ hy
#     h[h < np.finfo(h.dtype).eps * h.max()] = 0  # ?
#     return h
#
#
# def draw_umich_gaussian(heatmap, center, radius, k=1, sigma_x=1, sigma_y=1):
#     height, width = heatmap.shape[0:2]
#
#     diameter = 2 * radius + 1
#     gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
#
#     x, y = int(center[0]), int(center[1])
#
#     height, width = heatmap.shape[0:2]
#
#     left, right = min(x, radius), min(width - x, radius + 1)
#     top, bottom = min(y, radius), min(height - y, radius + 1)
#
#     masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
#     masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
#     if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
#         np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
#     return heatmap
#
#
# def get_weight_gauss(edges, sigma_x=1, sigma_y=1):  # sigma_x stands for sigma_v !!!!
#     left, right, top, bottom = edges
#     # left, right, top, bottom = sigma_u, sigma_u + 1, sigma_v, sigma_v + 1
#     sample_gap = 2
#     x_u, y_u = np.ogrid[0:bottom:sample_gap, 0:right:sample_gap]
#     x_d, y_d = np.ogrid[0:-top - 1:-sample_gap, 0:-left - 1:-sample_gap]
#     x_d = x_d[::-1, :][:-1, :]
#     y_d = y_d[:, ::-1][:, :-1]
#     x = np.concatenate([x_d, x_u], axis=0)
#     y = np.concatenate([y_d, y_u], axis=0)
#
#     hx = np.exp(-(x * x) / (2 * sigma_x * sigma_x)) / (math.sqrt(2 * math.pi) * sigma_x)
#     hy = np.exp(-(y * y) / (2 * sigma_y * sigma_y)) / (math.sqrt(2 * math.pi) * sigma_y)
#     h = hx @ hy
#     h[h < np.finfo(h.dtype).eps * h.max()] = 0  # ?
#
#     return h


def cal_weight_gauss(x, y, sigma_x, sigma_y):
    hx = np.exp(-(x * x) / (2 * sigma_x * sigma_x)) / (math.sqrt(2 * math.pi) * sigma_x)
    hy = np.exp(-(y * y) / (2 * sigma_y * sigma_y)) / (math.sqrt(2 * math.pi) * sigma_y)
    h = hx @ hy
    h[h < np.finfo(h.dtype).eps * h.max()] = 0  # ?

    return h


def sample_img_depth(uvd, depth_map, sigma_uv, seg_mask=None, seg_map=None):
    height, width = depth_map.shape
    weight_list = []
    depth_list = []
    radar_depth_list = []
    point_seg_list = []
    for loc, loc_sigma in zip(uvd, sigma_uv):  # uv/sigma_uv: (n, 2)
        u, v, d = int(loc[0]), int(loc[1]), loc[2]
        sigma_u, sigma_v = int(loc_sigma[0]), int(loc_sigma[1])
        radius_u, radius_v = sigma_u, sigma_v  # todo: here we choose 1-sigma area (to speed-up)

        left, right = min(u, radius_u), min(width - u, radius_u + 1)
        top, bottom = min(v, radius_v), min(height - v, radius_v + 1)

        sample_gap = 5
        v_u, u_u = np.ogrid[0:bottom:sample_gap, 0:right:sample_gap]
        v_d, u_d = np.ogrid[0:-top - 1:-sample_gap, 0:-left - 1:-sample_gap]
        v_d = v_d[::-1, :][:-1, :]
        u_d = u_d[:, ::-1][:, :-1]
        v_s0 = np.concatenate([v_d, v_u], axis=0)  # m, 1
        u_s0 = np.concatenate([u_d, u_u], axis=1)  # 1, n

        sampled_gauss_weight = cal_weight_gauss(v_s0, u_s0, sigma_v, sigma_u)
        # print(sampled_gauss_weight.shape)

        # sample depth_map
        v_s = v_s0.reshape(-1) + v
        u_s = u_s0.reshape(-1) + u
        sampled_depth_map = depth_map[v_s[0]:v_s[-1] + 1:sample_gap, u_s[0]:u_s[-1] + 1:sample_gap]
        # print(sampled_depth_map.shape)

        depth = sampled_depth_map.flatten()
        weight = sampled_gauss_weight.flatten()
        radar_dpeth = np.ones_like(depth) * d

        if seg_mask is not None:
            ui, vi = np.meshgrid(u_s, v_s)
            point_seg_mask = seg_mask[vi.flatten(), ui.flatten()]
            depth = depth[point_seg_mask == 1]
            weight = weight[point_seg_mask == 1]
            radar_dpeth = radar_dpeth[point_seg_mask == 1]
            if seg_map is not None:
                point_seg = seg_map[vi.flatten(), ui.flatten()]
                point_seg = point_seg[point_seg_mask == 1]
                point_seg_list.append(point_seg)

        depth_list.append(depth)
        weight_list.append(weight)
        radar_depth_list.append(radar_dpeth)
    if seg_map is None:
        return np.concatenate(radar_depth_list), np.concatenate(depth_list), np.concatenate(weight_list)
    else:
        return np.concatenate(radar_depth_list), np.concatenate(depth_list), np.concatenate(weight_list), \
               np.concatenate(point_seg_list)


if __name__ == '__main__':
    sigma_u = 12
    sigma_v = 7
    shape = (int(2 * sigma_v + 1), int(2 * sigma_u + 1))  # sigma_v, sigma_u
    # h = gaussian2D(shape, sigma_v, sigma_u)
    #
    # left, right, top, bottom = sigma_u, sigma_u + 1, sigma_v, sigma_v + 1
    # sample_gap = 2
    # x_u, y_u = np.ogrid[0:bottom:sample_gap, 0:right:sample_gap]
    # x_d, y_d = np.ogrid[0:-top - 1:-sample_gap, 0:-left - 1:-sample_gap]
    # x_d = x_d[::-1, :][:-1, :]
    # y_d = y_d[:, ::-1][:, :-1]
    # x = np.concatenate([x_d, x_u], axis=0)
    # y = np.concatenate([x_d, x_u], axis=0)
    #
    # print(x)
    #
    # print(h)
