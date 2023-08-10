import numpy as np
import h5py
import matplotlib.pyplot as plt

def read_radar(filePath, concate_num_radar, offset_frame):
    with h5py.File(filePath, 'r') as hf:
        radarPoints = np.array(hf['radar_points'])
        frameInfo = np.array(hf['frame_info'])
        frame_num = frameInfo.shape[1]
        # TODO: fix bug
        radarPoints[0, :] = -radarPoints[0, :]
    frame_data = np.zeros((9, 0))
    for i in range(frame_num):
        if i < offset_frame:
            continue
        frame_loc = frameInfo[0, i]
        frame_len = frameInfo[1, i]
        frame_data = np.concatenate((frame_data, radarPoints[:, frame_loc:frame_loc + frame_len]), axis=1)
        if (i - offset_frame) % concate_num_radar != concate_num_radar - 1:
            continue
        print('Radar_frame:', i)
        yield frame_data
        frame_data = np.zeros((9, 0))
    return None


if __name__ == "__main__":

    radar_file = './test.h5'
    concate_num_radar = 5
    offset_frame = 400
    radar_g = read_radar(radar_file, concate_num_radar, offset_frame)

    frame_data = radar_g.__next__()

    # vmaxa = 0
    # vmina = 0
    # while True:
    #     try:
    #         frame_data = radar_g.__next__()
    #     except:
    #         break
    #     v = frame_data[3, :]
    #     vmax = max(v)
    #     vmin = min(v)
    #
    #     if vmax > vmaxa : vmaxa = vmax
    #     if vmin < vmina : vmina = vmin
    # print(vmaxa, vmina)

    fig, axes = plt.subplots(1, 1)
    ONCE = True

    count = 0

    while True:

        if count == 85:
            while True:
                pass
        count += 1

        try:
            frame_data = radar_g.__next__()
        except:
            break

        plt.cla()
        axes.set_xlim([-10, 10])
        axes.set_ylim([0, 20])

        axes.set_aspect('equal')

        plt.scatter(frame_data[0, :], frame_data[1, :], c=frame_data[3, :], cmap="Set2")
        # plt.clim(vmin=-2.2083211035897232, vmax=1.9322809656410078)
        plt.clim(vmin=-1.381, vmax=1.381)
        plt.grid(True)

        if ONCE:
            plt.colorbar()
            ONCE = False

        plt.xlabel('X Range/m')
        plt.ylabel('Y Range/m')

        # plt.show()
        plt.pause(0.00000000001 + concate_num_radar * 0.1)

    plt.close(fig)