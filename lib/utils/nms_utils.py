import numpy as np
from .box_utils import nms2d

def classAgnosticNMS(dets, scores, overlapThresh=0.3, top_k=None):
    """Compute the NMS for a set of scored tubelets
    scored tubelets are numpy array with 4K+1 columns, last one being the score
    return the indices of the tubelets to keep
    """

    # If there are no detections, return an empty list
    if len(dets) == 0: return np.empty((0,), dtype=np.int32)
    if top_k is None: top_k = len(dets)

    pick = []

    # Coordinates of bounding boxes
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    # Compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    # area = (x2 - x1 + 1) * (y2 - y1 + 1)
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    I = np.argsort(scores)
    indices = np.empty(top_k, dtype=np.int32)
    counter = 0

    while I.size > 0:
        i = I[-1]
        indices[counter] = i
        counter += 1

        # Compute overlap
        xx1 = np.maximum(x1[i], x1[I[:-1]])
        yy1 = np.maximum(y1[i], y1[I[:-1]])
        xx2 = np.minimum(x2[i], x2[I[:-1]])
        yy2 = np.minimum(y2[i], y2[I[:-1]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        inter_area = w * h
        ious = inter_area/ (area[I[:-1]] + area[i] - inter_area)

        I = I[np.where(ious <= overlapThresh)[0]]

        if counter == top_k: break

    return indices[:counter]