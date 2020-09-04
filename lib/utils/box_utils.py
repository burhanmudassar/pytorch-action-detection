import sys
import os
import numpy as np
import torch

""" utilities functions for boxes, tubelets, tubes, and ap"""


### BOXES
""" boxes are represented as a numpy array with 4 columns corresponding to the coordinates (x1, y1, x2, y2)"""

def boxes_xywh_xyxy(boxes):

    assert len(boxes.shape) == 2
    boxes[:,2] += boxes[:,0]
    boxes[:,3] += boxes[:,1]

    return boxes


def area2d(b):
    """Compute the areas for a set of 2D boxes"""

    return (b[:,2]-b[:,0]+1) * (b[:,3]-b[:,1]+1)


def overlap2d(b1, b2):
    """Compute the overlaps between a set of boxes b1 and one box b2"""

    xmin = np.maximum(b1[:,0], b2[:,0])
    ymin = np.maximum(b1[:,1], b2[:,1])
    xmax = np.minimum(b1[:,2] + 1, b2[:,2] + 1)
    ymax = np.minimum(b1[:,3] + 1, b2[:,3] + 1)

    width = np.maximum(0, xmax - xmin)
    height = np.maximum(0, ymax - ymin)

    return width * height


def iou2d(b1, b2):
    """Compute the IoU between a set of boxes b1 and 1 box b2"""

    if b1.ndim == 1: b1 = b1[None, :]
    if b2.ndim == 1: b2 = b2[None, :]

    assert b2.shape[0] == 1

    ov = overlap2d(b1, b2)

    return ov / (area2d(b1) + area2d(b2) - ov)


def nms2d(boxes, overlap=0.3):
    """Compute the nms given a set of scored boxes,
    as numpy array with 5 columns <x1> <y1> <x2> <y2> <score>
    return the indices of the tubelets to keep
    """

    if boxes.size == 0:
        return np.array([],dtype=np.int32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    scores = boxes[:, 4]
    areas = (x2-x1+1) * (y2-y1+1)
    I = np.argsort(scores)
    indices = np.zeros(scores.shape, dtype=np.int32)

    counter = 0
    while I.size > 0:
        i = I[-1]
        indices[counter] = i
        counter += 1

        xx1 = np.maximum(x1[i],x1[I[:-1]])
        yy1 = np.maximum(y1[i],y1[I[:-1]])
        xx2 = np.minimum(x2[i],x2[I[:-1]])
        yy2 = np.minimum(y2[i],y2[I[:-1]])

        inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
        iou = inter / (areas[i] + areas[I[:-1]] - inter)
        I = I[np.where(iou <= overlap)[0]]

    return indices[:counter]


### TUBELETS
""" tubelets of length K are represented using numpy array with 4K columns """

def nms_tubelets(dets, overlapThresh=0.3, top_k=None):
    """Compute the NMS for a set of scored tubelets
    scored tubelets are numpy array with 4K+1 columns, last one being the score
    return the indices of the tubelets to keep
    """

    # If there are no detections, return an empty list
    if len(dets) == 0: return np.empty((0,), dtype=np.int32)
    if top_k is None: top_k = len(dets)

    pick = []

    K = (dets.shape[1] - 1) // 4

    # Coordinates of bounding boxes
    x1 = [dets[:, 4*k] for k in range(K)]
    y1 = [dets[:, 4*k + 1] for k in range(K)]
    x2 = [dets[:, 4*k + 2] for k in range(K)]
    y2 = [dets[:, 4*k + 3] for k in range(K)]

    # Compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    # area = (x2 - x1 + 1) * (y2 - y1 + 1)
    area = [(x2[k] - x1[k] + 1) * (y2[k] - y1[k] + 1) for k in range(K)]
    I = np.argsort(dets[:,-1])
    indices = np.empty(top_k, dtype=np.int32)
    counter = 0

    while I.size > 0:
        i = I[-1]
        indices[counter] = i
        counter += 1

        # Compute overlap
        xx1 = [np.maximum(x1[k][i], x1[k][I[:-1]]) for k in range(K)]
        yy1 = [np.maximum(y1[k][i], y1[k][I[:-1]]) for k in range(K)]
        xx2 = [np.minimum(x2[k][i], x2[k][I[:-1]]) for k in range(K)]
        yy2 = [np.minimum(y2[k][i], y2[k][I[:-1]]) for k in range(K)]

        w = [np.maximum(0, xx2[k] - xx1[k] + 1) for k in range(K)]
        h = [np.maximum(0, yy2[k] - yy1[k] + 1) for k in range(K)]

        inter_area = [w[k] * h[k] for k in range(K)]
        ious = sum([inter_area[k] / (area[k][I[:-1]] + area[k][i] - inter_area[k]) for k in range(K)])

        I = I[np.where(ious <= overlapThresh * K)[0]]

        if counter == top_k: break

    return indices[:counter]

### TUBES
""" tubes are represented as a numpy array with nframes rows and 5 columns (frame, x1, y1, x2, y2). frame number are 1-indexed, coordinates are 0-indexed """

def iou3d(b1, b2):
    """Compute the IoU between two tubes with same temporal extent"""

    assert b1.shape[0] == b2.shape[0]
    assert np.all(b1[:, 0] == b2[:, 0])

    ov = overlap2d(b1[:,1:5],b2[:,1:5])

    return np.mean(ov / (area2d(b1[:, 1:5]) + area2d(b2[:, 1:5]) - ov) )


def iou3dt(b1, b2, spatialonly=False):
    """Compute the spatio-temporal IoU between two tubes"""

    tmin = max(b1[0, 0], b2[0, 0])
    tmax = min(b1[-1, 0], b2[-1, 0])

    if tmax < tmin: return 0.0

    temporal_inter = tmax - tmin + 1
    temporal_union = max(b1[-1, 0], b2[-1, 0]) - min(b1[0, 0], b2[0, 0]) + 1

    tube1 = b1[int(np.where(b1[:, 0] == tmin)[0]) : int(np.where(b1[:, 0] == tmax)[0]) + 1, :]
    tube2 = b2[int(np.where(b2[:, 0] == tmin)[0]) : int(np.where(b2[:, 0] == tmax)[0]) + 1, :]

    return iou3d(tube1, tube2) * (1. if spatialonly else temporal_inter / temporal_union)


def nms3dt(tubes, overlap=0.5):
    """Compute NMS of scored tubes. Tubes are given as list of (tube, score)
    return the list of indices to keep
    """

    if not tubes:
        return np.array([], dtype=np.int32)

    I = np.argsort([t[1] for t in tubes])
    indices = np.zeros(I.size, dtype=np.int32)
    counter = 0

    while I.size > 0:
        i = I[-1]
        indices[counter] = i
        counter += 1
        ious = np.array([iou3dt(tubes[ii][0], tubes[i][0]) for ii in I[:-1]])
        I = I[np.where(ious <= overlap)[0]]

    return indices[:counter]

def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = np.clip((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]

def intersect_numpy(box_a, box_b):
    A = box_a.shape[0]
    B = box_b.shape[0]
    max_xy = np.minimum(np.repeat(np.expand_dims(box_a[:, 2:], axis=1), B, axis=1),
                       np.repeat(np.expand_dims(box_b[:, 2:], axis=0), A, axis=0))
    min_xy = np.maximum(np.repeat(np.expand_dims(box_a[:, :2], axis=1), B, axis=1),
                       np.repeat(np.expand_dims(box_b[:, :2], axis=0), A, axis=0))
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=None)
    return inter[:, :, 0] * inter[:, :, 1]

def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]

def jaccard_numpy(box_a, box_b):
    inter = intersect_numpy(box_a, box_b)
    area_a = np.repeat(np.expand_dims(((box_a[:, 2]-box_a[:, 0]) *
            (box_a[:, 3]-box_a[:, 1])), axis=1),inter.shape[1],axis=1)
    area_b = np.repeat(np.expand_dims(((box_b[:, 2]-box_b[:, 0]) *
            (box_b[:, 3]-box_b[:, 1])), axis=0),inter.shape[0],axis=0)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]
