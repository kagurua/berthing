import matplotlib.pyplot as plt
import numpy as np
import h5py
from Video_process import comp_xy0


def read_radar(filePath, concate_num_radar, offset_frame):
    with h5py.File(filePath, 'r') as hf:
        radarPoints = np.array(hf['radar_points'])
        frameInfo = np.array(hf['frame_info'])
        frame_num = frameInfo.shape[1]
        radarPoints[0, :] = -radarPoints[0, :]
    frame_data = np.zeros((9, 0))
    for i in range(frame_num):
        if i < offset_frame:
            continue
        frame_loc = frameInfo[0, i]
        frame_len = frameInfo[1, i]
        frame_data = np.concatenate((frame_data, radarPoints[:, frame_loc:frame_loc + frame_len]), axis=1)
        if i % concate_num_radar != concate_num_radar - 1:
            continue
        print(i)
        yield frame_data
        frame_data = np.zeros((9, 0))
    return None


if __name__ == "__main__":

    # radar pipeline
    process_frame = 600
    concate_num_radar = 5
    offset_frame = 45  # delay for synchronization, no need to change
    radar_file = "./pipeline/test.h5"  # radar file
    radar_g = read_radar(radar_file, concate_num_radar, offset_frame)

    # fig, axes = plt.subplots(1, 1)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_aspect('equal')

    for i in range(int(process_frame / concate_num_radar)):

        radar_points = radar_g.__next__().T  # get a radar frame(5 concatenated)

        plt.plot(radar_points[:, 0], radar_points[:, 1], '.') # display_radar_points

        comp_radar_points = comp_xy0(radar_points)

        # plt.xlim([-20, 20])
        # plt.ylim([0, 25])
        # axes.set_aspect('equal')
        # # stream display
        # plt.pause(0.1)
        # plt.cla()
        # # one by one display
        # # plt.show()

        # display_current
        ax1.plot(radar_points[:, 0], radar_points[:, 1], '.')
        ax1.set_xlim([-10, 10])
        ax1.set_ylim([0, 20])
        ax1.plot(comp_radar_points[:, 0], comp_radar_points[:, 1], '.')
        plt.pause(0.5)
        ax1.cla()
