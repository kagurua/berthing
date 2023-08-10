import cv2 as cv
import numpy as np


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


def get_IRT():
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

    return A, R, T


def get_uv(radar_points, H):
    # project xyz to uv
    xyz = radar_points[:, 0:3].T
    xyz1 = np.concatenate([xyz, np.ones([1, xyz.shape[1]])])
    uv1 = np.dot(H, np.dot(xyz1, np.diag(1. / (np.dot(np.array([0, 0, 1]), np.dot(H, xyz1))))))
    uv = np.floor(uv1[0:2, :]).astype(int)

    return uv


def get_uvd(radar_points, H, video_frame=None):
    # project xyz to uv
    xyz = radar_points[:, 0:3].T
    xyz1 = np.concatenate([xyz, np.ones([1, xyz.shape[1]])])
    uv1 = np.dot(H, np.dot(xyz1, np.diag(1. / (1e-5 + np.dot(np.array([0, 0, 1]), np.dot(H, xyz1))))))
    # # discrete
    # uv = np.floor(uv1[0:2, :]).astype(int)
    uv = uv1[0:2, :]
    d = np.dot(np.array([0, 0, 1]), np.dot(H, xyz1))[None, :]  # (1, n)
    uvd = np.concatenate([uv, d])

    return uvd  # (3, n)


def get_xyz(uvd_points, I, R, t):
    # project uvd to xyz1
    udvdd = uvd_points[:, :3].T
    udvdd[:2, :] = udvdd[:2, :] * udvdd[2:, :]
    xyz = np.linalg.pinv(R) @ (np.linalg.pinv(I) @ udvdd - t)

    return xyz  # (3, n)