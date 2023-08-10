"""
inputs:
    cam_seg: segmentation result on cam img (matrix with size of ori img [H*W] plus channel [C] presenting the semantic number)
    rots, trans, intrins: extrinsics(rots, trans) and intrinsics(intrins)

outputs:
    bev_seg: segmentation result on bev of size [C x Z x X x Y]

"""
import torch
from my_radar_depth import radar2img

import torch.nn.functional as F
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

cuda_device = torch.device('cuda:3')


def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]]).cuda(cuda_device)
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]]).cuda(cuda_device)
    nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]]).cuda(cuda_device)

    return dx, bx, nx


def cumsum_trick(x, geom_feats, ranks):
    x = x.cumsum(0)
    kept = torch.ones(x.shape[0], dtype=torch.bool, device=cuda_device)
    kept[:-1] = (ranks[1:] != ranks[:-1])

    x, geom_feats = x[kept], geom_feats[kept]
    x = torch.cat((x[:1], x[1:] - x[:-1]))

    return x, geom_feats


def convert_segimg_cam2bev(cam_seg, extrins, intrins, radar_info=None, depth_map=None):
    """
    cam_seg:X*Y*C
    """

    # os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    # TODO:wjx convert to tensor
    cam_seg = torch.tensor(cam_seg, dtype=torch.float, device=cuda_device)
    extrins = torch.tensor(extrins, dtype=torch.float, device=cuda_device)
    intrins = torch.tensor(intrins, dtype=torch.float, device=cuda_device)
    cam_seg = cam_seg.permute(1, 2, 0)

    # TODO:wjx 221013 convert_segimg_cam2bev
    cam_H, cam_W, cam_C = cam_seg.shape
    # cam_H, cam_W, cam_C = 1080, 1920, 10
    # cam_H, cam_W, cam_C = 1080, 1920, 10
    cam_downsample = 2  # downsample cam seg image

    fH, fW = cam_H // cam_downsample, cam_W // cam_downsample

    # define the bev scale
    xbound = [-50.0, 50.0, 0.5]
    ybound = [-50.0, 50.0, 0.5]
    # ybound = [0.0, 50.0, 0.5]
    zbound = [-5.0, 5.0, 1.0]
    # zbound = [-10.0, 10.0, 0.5]
    d_bound = [4.0, 45.0, 1.0]

    # dx, bx, nx = gen_dx_bx(xbound, ybound, zbound)

    # step 1: create cam frustum(IN POINT CLOUD FORMAT, i.e. store their xyz, p.s. only their grid)
    def create_frustum():
        # make grid in image plane
        ds = torch.arange(*d_bound, dtype=torch.float, device=cuda_device).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape
        xs = torch.linspace(0, cam_W - 1, fW, dtype=torch.float, device=cuda_device).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, cam_H - 1, fH, dtype=torch.float, device=cuda_device).view(1, fH, 1).expand(D, fH, fW)

        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)

        return frustum
        # return nn.Parameter(frustum, requires_grad=False)

    # step 2: get frustum's grids real location
    def get_geometry(frustum, rots, trans, intrins):
        points = frustum
        # img(frustum)_to_cam_to_ego
        # formulations: 1. I * p_cam = p_img([u, v, 1] * z); 2. p_ego = R * p_cam
        points = torch.cat((points[:, :, :, :2] * points[:, :, :, 2:3], points[:, :, :, 2:3]), 3)  # (UZ, VZ, Z)
        combine = rots.matmul(torch.inverse(intrins))
        # combine = torch.inverse(intrins.matmul(rots))
        points = combine.view(1, 1, 1, 3, 3).matmul(points.unsqueeze(-1)).squeeze(-1)
        points += trans.view(1, 1, 1, 3)

        return points  # DHW3

    # adding d dim to cam seg img
    def cam_encode(cam_seg, D, depth_img=None):
        # ori cam_seg of shape XYC
        cam_seg = torch.tensor(cam_seg, device=cuda_device)
        depth_img = torch.tensor(depth_img, device=cuda_device)
        imH, imW, C = cam_seg.shape
        # TODO:wjx downsampling cam_seg
        x = cam_seg.permute(2, 0, 1).unsqueeze(0)
        x = F.interpolate(x, scale_factor=1. / cam_downsample)
        x = x.squeeze(0).permute(1, 2, 0)
        x = x.reshape(fH * fW, C)
        if depth_img is not None:
            # todo:firstly reshape depth_img
            depth_img = depth_img.unsqueeze(0).unsqueeze(0)
            depth_img = F.interpolate(depth_img, size=(fH, fW), mode='bilinear')
            depth_img = depth_img.reshape(fH, fW)
            # todo: transfer from h*w(depth) to h*w*depth_sampled
            # sample
            depth_img = depth_img.reshape(fH * fW, 1)
            print(depth_img.max(), depth_img.min())  # very small span of 0-1
            dbound = [4.0, 45.0, 1.0]  # --> (0, ..., 40)
            # assume that depth_img indicates the real depth
            depth_img[depth_img < dbound[0]] = dbound[0]
            depth_img[depth_img > dbound[1]] = dbound[1]
            depth_img = ((depth_img - dbound[0]) // dbound[2]).long()
            # # assume that depth_img is the normalized depth
            # depth_img = (((dbound[1] - dbound[0]) * depth_img) // dbound[2]).long()
            # scatter
            target = torch.zeros([fH * fW, D], device=cuda_device)
            source = torch.ones([fH * fW, 1], device=cuda_device)
            depth = target.scatter(dim=1, index=depth_img, src=source)
            print(depth.shape)
        elif radar_info is not None:
            # TODO:wjx get fv-depth from radar point cloud
            fv_depth = radar2img(radar_info, rots, trans, intrins, imH, imW, cam_downsample)
            depth = fv_depth.reshape(fH * fW, D)  # assert shape
        else:
            depth = torch.full([fH * fW, D], 1. / D, device=cuda_device)  # TODO:wjx assume even depth distribution
        new_x = depth.unsqueeze(1) * x.unsqueeze(2)  # H*W, C, D
        new_x = new_x.view(fH, fW, C, D)
        new_x = new_x.permute(3, 0, 1, 2)  # D, H, W, C

        return new_x

    # step 3: bev grid(voxel) pooling
    def voxel_pooling(geom_feats, x):
        """
            x: shape of D*H*W*C
            geom_feats: shape of D*H*W*3(xyz)

            geom_feats is frustums' loc(3 dim)
            x is frustums' feature(C dim)
        """
        D, H, W, C = x.shape
        Nprime = D * H * W
        B = 1  # batch_size=1

        # flatten x
        x = x.reshape(Nprime, C)

        dx, bx, nx = gen_dx_bx(xbound, ybound, zbound)  # dx: voxel size; bx: bottom center; nx: total voxel num

        # flatten indices
        geom_feats = ((geom_feats - (bx - dx / 2.)) / dx).long()  # TODO:wjx make frustum location from meter to grid
        geom_feats = geom_feats.view(Nprime, 3)
        # batch_ix = torch.cat([torch.full([Nprime // B, 1], ix, dtype=torch.long, device=cuda_device) for ix in range(B)])
        # geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < nx[0]) \
               & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < nx[1]) \
               & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]

        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (nx[1] * nx[2]) \
                + geom_feats[:, 1] * nx[2] \
                + geom_feats[:, 2]
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        x, geom_feats = cumsum_trick(x, geom_feats, ranks)  # TODO:wjx a trick to sum x_feature in each grid location

        # griddify (B x C x Z x X x Y)
        final = torch.zeros((nx[2], nx[0], nx[1], C), device=cuda_device)  # make 4D bev grid
        final[geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1], :] = x

        # # collapse Z
        # final = torch.cat(final.unbind(dim=2), 1)  # (C*Z) x X x Y, else: C x Z x X x Y

        return final

    rots, trans = extrins[:, :3], extrins[:, 3:]
    frustum = create_frustum()
    D, _, _, _ = frustum.shape
    x = cam_encode(cam_seg, D, depth_img=depth_map)  # img data frustum
    frustum_loc = get_geometry(frustum, rots, trans, intrins)
    # frustum_loc = frustum_loc.to(torch.float16)
    # x = x.to(torch.float16)
    # frustum_loc = frustum_loc.cpu()
    # x = x.cpu()
    torch.cuda.empty_cache()
    bev_seg = voxel_pooling(frustum_loc, x)

    bev_seg = bev_seg.permute(3, 0, 1, 2)  # CZXY
    return bev_seg.cpu().detach().numpy()

# convert_segimg_cam2bev(None, None, None)
