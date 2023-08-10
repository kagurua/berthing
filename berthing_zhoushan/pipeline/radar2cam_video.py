import numpy as np
import cv2 as cv
import h5py
import matplotlib.pyplot as plt

def read_radar(filePath, concate_num_radar, offset_frame):
    with h5py.File(filePath, 'r') as hf:
        radarPoints = np.array(hf['radar_points'])
        frameInfo = np.array(hf['frame_info'])
        frame_num = frameInfo.shape[1]

    frame_data = np.zeros((9, 0))
    for i in range(frame_num):
        if i < offset_frame:
            continue
        frame_loc = frameInfo[0, i]
        frame_len = frameInfo[1, i]
        frame_data = np.concatenate((frame_data, radarPoints[:, frame_loc:frame_loc + frame_len]), axis=1)
        if i % concate_num_radar != concate_num_radar - 1:
            continue
        yield frame_data
        print(i)
        frame_data = np.zeros((9, 0))


# cap = cv.VideoCapture('/media/personal_data/wujx/sea_test/202101/视频/2021-01-06_15-18-18.mp4')
cap = cv.VideoCapture('./2021-01-06_15-18-18.mp4')
radar_file = "./test.h5"
if not cap.isOpened():
    print("Cannot open video")
    exit()

frame_rate_cam = 30
fram_rate_radar = 10
concate_num_radar = 1
skip_num_image = concate_num_radar * frame_rate_cam / fram_rate_radar
both_delay_frame = 0  # for radar
both_delay_frame_img = both_delay_frame * frame_rate_cam / fram_rate_radar
offset_frame = 42  # set radar faster than img, for synchronization
radar_g = read_radar(radar_file, concate_num_radar, offset_frame + both_delay_frame)

H = np.array([[939.221222778002, 635.037860632676, 142.041238485687, -207.017968997192],
              [-12.0084530697522, 574.72009106674, -861.686459337377, -494.926955223144],
              [-0.0392615918420586, 0.976062351478972, 0.213917772593507, -0.201345093070659]])  # boat

# H = np.array([[686.147008498085, 561.356335208823, -13.9189565035784, 178.516443947872],
#               [-40.2234429753803, 396.462299425261, -710.591648034102, -96.8761714710670],
#               [-0.0629553179025540, 0.998012433615555, 0.00279469077826616, 0.0233620503112028]])  # car

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
    radar_frame = radar_g.__next__()

    xyz = radar_frame[0:3, :]
    xyz[0, :] = -xyz[0, :]  # fix bug
    xyz1 = np.concatenate([xyz, np.ones([1, xyz.shape[1]])])
    uv1 = np.dot(H, np.dot(xyz1, np.diag(1. / (np.dot(np.array([0, 0, 1]), np.dot(H, xyz1))))))
    uv = np.floor(uv1[0:2, :])

    # remove Near-field Interference
    rm_range = 2.5
    uv = uv[:, np.linalg.norm(xyz[0:3, :], axis=0) > rm_range]

    # remove points outer image
    uv = uv[:, uv[0, :] < frame.shape[1]]
    uv = uv[:, uv[0, :] > 0]
    uv = uv[:, uv[1, :] < frame.shape[0]]
    uv = uv[:, uv[1, :] > 0]

    for i in range(uv.shape[1]):
        point = (uv[0, i].astype(int), uv[1, i].astype(int))
        color = [(radar_frame[3, i] + 5)/10 * 255, 255 - (radar_frame[3, i] + 5)/10 * 255, 0]
        # color = [0, 255, 0]
        cv.circle(frame, point, 2, color, 8)


    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break


# 完成所有操作后，释放捕获器
cap.release()
cv.destroyAllWindows()
