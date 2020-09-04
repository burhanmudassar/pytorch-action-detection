# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from nms.gpu_nms import gpu_nms
from nms.cpu_nms import cpu_nms

def nms(dets, thresh, force_cpu=False, use_gpu_nms=True):
    """Dispatch to either CPU or GPU NMS implementations."""

    if dets.shape[0] == 0:
        return []
    if use_gpu_nms and not force_cpu:
        return gpu_nms(dets, thresh, device_id=0)
    else:
        return cpu_nms(dets, thresh)