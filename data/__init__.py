import cv2
import numpy as np
from .UCF24 import UCF24_Dataset
from .JHMDB21 import JHMDB21Dataset
from .UCFSports import UCFSportsDataset

def base_transform(image, size, mean):
    x = cv2.resize(image, (size, size)).astype(np.float32)
    # x = cv2.resize(np.array(image), (size, size)).astype(np.float32)
    x -= mean
    x = x.astype(np.float32)
    return x


class BaseTransform:
    def __init__(self, size, mean):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        return base_transform(image, self.size, self.mean), boxes, labels

def base_transform_tubelet(image, size, mean, std):
    image = [cv2.resize(img, (size, size)).astype(np.float32) for img in image]
    image = [img - mean for img in image]
    image = [img / std / 255.0 for img in image]
    return image

class BaseTransformTubeLet:
    def __init__(self, size, mean, std):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)


    def __call__(self, image, boxes=None, labels=None):
        return base_transform_tubelet(image, self.size, self.mean, self.std), boxes, labels


def DatasetBuilder(dataset='ucf24', path='data/ucf24/', split=1):
    if dataset == 'ucf24':
        return UCF24_Dataset(path=path, split=split)
    elif dataset == 'jhmdb':
        return JHMDB21Dataset(path=path, split=split)
    elif dataset == 'ucfsports':
        return UCFSportsDataset(path=path, split=split)
    else:
        raise IndexError
