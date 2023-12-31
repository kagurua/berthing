import sys
import time
import h5py
import numpy as np
import matplotlib.pyplot as plt


def read_radar_points(filePath, concate_num=5):
    with h5py.File(filePath, 'r') as hf:
        radarPoints = np.array(hf['radar_points'])
        frameInfo = np.array(hf['frame_info'])

        #   an interesting(rediculous) finding
        print('no matter what SHIT, this line just do not show!')
        print('While this SHIT will show')

        frame_num = frameInfo.shape[1]
    fig, axes = plt.subplots(1, 1)

    frame_data = np.zeros((9, 0))
    for i in range(frame_num):
        frame_loc = frameInfo[0, i]
        frame_len = frameInfo[1, i]
        frame_data = np.concatenate((frame_data, radarPoints[:, frame_loc:frame_loc + frame_len]), axis=1)

        if i % concate_num != (concate_num - 1):
            continue

        plt.cla()
        axes.set_xlim([-10, 10])
        axes.set_ylim([0, 20])

        axes.set_aspect('equal')

        plt.scatter(-frame_data[0, :], frame_data[1, :], c=frame_data[3, :])
        plt.clim(vmin=-2, vmax=2)
        plt.text(6, 19, "Time : " + "%05.1fs" % ((i + 1) / 10), fontsize=15)
        plt.grid(True)
        plt.xlabel('X Range/m')
        plt.ylabel('Y Range/m')
        plt.pause(0.00000000001 + concate_num * 0.1)

        frame_data = np.zeros((9, 0))
    plt.close(fig)


if __name__ == "__main__":
    # TODO: change path and name
    # radar_dataroot = "./experi_data/fine_data/"
    # for file_id in [1,3]:
    #     print("Now playing file_id:" + str(file_id))
    #     fileName = str(file_id) + ".h5"
    #     filePath = radar_dataroot + fileName
    #     read_radar_points(filePath, concate_num=5)

    fileName = "./test.h5"
    read_radar_points(fileName, concate_num=1)
