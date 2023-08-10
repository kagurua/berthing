import numpy as np


def find_hits(hull, mask_line):
    # hull: n points xy size of (n, 2)
    # mask_line: [A, B, C]

    mask_line = np.array(mask_line).reshape(3, -1)
    hull1 = np.concatenate([hull, np.ones([hull.shape[0], 1])], axis=1)
    ud = np.dot(hull1, mask_line)  # up or down
    sb = (ud * np.roll(ud, -1)).flatten()  # same or both sides
    hits_index = np.argwhere(sb < 0).flatten()
    hits_pairs = np.concatenate([hull[sb < 0, :], np.roll(hull, -1, axis=0)[sb < 0, :]], axis=1)  # x1, y1, x2, y2

    return hits_index, hits_pairs


def cal_hits(hits_pairs, mask_line):
    # line 1:
    a1, b1, c1 = mask_line
    x1, y1, x2, y2 = hits_pairs[:, 0], hits_pairs[:, 1], hits_pairs[:, 2], hits_pairs[:, 3]
    # line 2(array):
    a2 = y1 - y2
    b2 = x2 - x1
    c2 = x1 * y2 - x2 * y1

    # intersection
    hits_x = (b2 * c1 - b1 * c2) / (b1 * a2 - b2 * a1)
    hits_y = (a2 * c1 - a1 * c2) / (a1 * b2 - a2 * b1)
    hits = np.concatenate([hits_x.reshape(-1, 1), hits_y.reshape(-1, 1)], axis=1)

    return hits


def insert_hits(hull, hits, hits_index):
    hull_inserted = np.insert(hull, hits_index + 1, hits, axis=0)

    return hull_inserted
