"""
inputs:
    cam_seg: segmentation result on cam img (matrix with size of ori img [H*W] plus channel [C] presenting the semantic number)
    rots, trans, intrins: extrinsics(rots, trans) and intrinsics(intrins)

outputs:
    bev_seg: segmentation result on bev of size [C x Z x X x Y]

"""
import torch

cuda_device = torch.device('cuda:3')


def cumsum_trick(x, geom_feats, ranks):
    x = x.cumsum(0)
    kept = torch.ones(x.shape[0], dtype=torch.bool, device=cuda_device)
    kept[:-1] = (ranks[1:] != ranks[:-1])

    x, geom_feats = x[kept], geom_feats[kept]
    x = torch.cat((x[:1], x[1:] - x[:-1]))

    return x, geom_feats


def radar2img(radar_points, rots, trans, intrins, fH, fW, cam_downsample):
    radar_points = torch.tensor(radar_points, dtype=torch.float, device=cuda_device)  # N, 9
    p_ego = radar_points[:, :3].permute(1, 0)
    # 1.
    p_cam = rots.matmul(p_ego) + trans
    # 2.
    p_img = intrins.matmul(p_cam)
    p_img_d = p_img[2, :]
    p_img = p_img / p_img_d
    # 3. add d to uv loc
    p_img[2, :] = p_img_d
    p_img = p_img.permute(1, 0)

    # TODO:wjx griddify u, v, d
    uv_size = [cam_downsample * 1.0, cam_downsample * 1.0]
    ubound = [0.0, fW, uv_size[0]]
    vbound = [0.0, fH, uv_size[1]]
    dbound = [4.0, 45.0, 1.0]  # --> (0, ..., 40)
    nx = [int((ubound[1] - ubound[0]) // ubound[2]),
          int((vbound[1] - vbound[0]) // vbound[2]),
          int((dbound[1] - dbound[0]) // dbound[2])]
    # remove outliers
    kept = (p_img[:, 0] >= ubound[0]) & (p_img[:, 0] < ubound[1]) \
           & (p_img[:, 1] >= vbound[0]) & (p_img[:, 1] < vbound[1]) \
           & (p_img[:, 2] >= dbound[0]) & (p_img[:, 2] < dbound[1])
    p_img = p_img[kept]
    # girdiffy
    p_img[:, 0] = ((p_img[:, 0] - ubound[0]) // ubound[2]).long()
    p_img[:, 1] = ((p_img[:, 1] - vbound[0]) // vbound[2]).long()
    p_img[:, 2] = ((p_img[:, 2] - dbound[0]) // dbound[2]).long()

    # TODO:wjx count depth frequency
    points_loc_std = p_img
    points_data = torch.ones(p_img.shape[0], dtype=torch.float, device=cuda_device)  # prepare initial freq
    # to make same grid location the same rank
    ranks = points_loc_std[:, 0] * (nx[1] * nx[2]) \
            + points_loc_std[:, 1] * nx[2] \
            + points_loc_std[:, 2]
    sorts = ranks.argsort()
    points_data, points_loc_std, ranks = points_data[sorts], points_loc_std[sorts], ranks[sorts]

    points_data, points_loc_std = cumsum_trick(points_data, points_loc_std,
                                               ranks)  # TODO:wjx a trick to sum x_feature in each grid location

    final = torch.zeros((nx[1], nx[0], nx[2]), device=cuda_device)  # H, W, 41(depth_distribution)
    points_loc_std = points_loc_std.long()
    final[points_loc_std[:, 1], points_loc_std[:, 0], points_loc_std[:, 2]] = points_data  # v, u, d

    # TODO:wjx normalize among D(Y)
    final = final.squeeze(0)
    final += 1e-5
    final_sum = torch.sum(final, dim=2, keepdim=True)
    final_norm = final / final_sum

    return final_norm

# # define the bev scale
# xbound = [-50.0, 50.0, 1.0]
# ybound = [0.0, 50.0, 1.0]
# zbound = [-10.0, 10.0, 1.0]
# d_bound = [4.0, 45.0, 1.0]
#
#
# def get_radar_depth(radar_points):
#     """
#     input:
#         radar_points: (N, 9) x,y,z,...
#     """
#
#     points_loc = radar_points[:, :3]
#     points_loc = torch.tensor(points_loc, dtype=torch.float)  # N, 3
#     points_data = torch.ones((points_loc.shape[0], 1),
#                              dtype=torch.float)  # (N, 1) stand for depth frequency (accumulate in each grid)
#
#     dx, bx, nx = gen_dx_bx(xbound, ybound, zbound)  # dx: voxel size; bx: bottom center; nx: total voxel num
#
#     # flatten indices
#     points_loc_std = ((points_loc - (bx - dx / 2.)) / dx).long()  # TODO:wjx make frustum location from meter to grid
#
#     # filter out points that are outside box
#     kept = (points_loc_std[:, 0] >= 0) & (points_loc_std[:, 0] < nx[0]) \
#            & (points_loc_std[:, 1] >= 0) & (points_loc_std[:, 1] < nx[1]) \
#            & (points_loc_std[:, 2] >= 0) & (points_loc_std[:, 2] < nx[2])
#     points_data = points_data[kept]
#     points_loc_std = points_loc_std[kept]
#
#     # to make same grid location the same rank
#     ranks = points_loc_std[:, 0] * (nx[1] * nx[2]) \
#             + points_loc_std[:, 1] * nx[2] \
#             + points_loc_std[:, 2]
#     sorts = ranks.argsort()
#     points_data, points_loc_std, ranks = points_data[sorts], points_loc_std[sorts], ranks[sorts]
#
#     points_data, points_loc_std = cumsum_trick(points_data, points_loc_std,
#                                                ranks)  # TODO:wjx a trick to sum x_feature in each grid location
#
#     # griddify (C x Z x X x Y)
#     final = torch.zeros((1, nx[2], nx[0], nx[1]))  # make 4D bev grid
#     final[:, points_loc_std[:, 2], points_loc_std[:, 0], points_loc_std[:, 1]] = points_data.permute(1, 0)  # 1, Z, X, Y
#     # TODO: ZXY ~ HWD
#
#     # TODO:wjx normalize among D(Y)
#     final = final.squeeze(0)
#     final += 1e-5
#     final_sum = torch.sum(final, dim=2, keepdim=True)
#     final_norm = final / final_sum
#
#     return final_norm

# convert_segimg_cam2bev(None, None, None)
