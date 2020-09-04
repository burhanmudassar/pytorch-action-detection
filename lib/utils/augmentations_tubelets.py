
""" Agumentation code for SSD network

Original author: Ellis Brown, Max deGroot for VOC dataset
https://github.com/amdegroot/ssd.pytorch

"""

import torch
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random

def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels


class Lambda(object):
    """Applies a lambda as a transform."""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, boxes=None, labels=None):
        return self.lambd(img, boxes, labels)

# Done
class ConvertFromInts(object):
    def __call__(self, image, boxes=None, labels=None):
        image = [ img.astype(np.float32) for img in image ]
        return image, boxes, labels

# Done
class SubtractMeans(object):
    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)


    def __call__(self, image, boxes=None, labels=None):
        image = [img.astype(np.float32) - self.mean for img in image]
        image = [img / self.std / 255.0 for img in image]

        assert (boxes >= 0).any(), 'boxes cannot be negative'
        assert ((boxes[:, :, 2] - boxes[:, :, 0]) >= 0).any(), 'x2 should be greater than x1'

        return image, boxes, labels

# Done
class ToAbsoluteCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image[0].shape
        boxes *= np.asarray([[[width, height, width, height]]])

        return image, boxes, labels

# Done
class ToPercentCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image[0].shape
        boxes /= np.asarray([[[width, height, width, height]]])

        return image, boxes, labels

# Done
class Resize(object):
    def __init__(self, size=300):
        self.size = size

    def __call__(self, image, boxes=None, labels=None):
        image = [cv2.resize(img, (self.size,
                                 self.size)) for img in image]
        return image, boxes, labels

# Done
class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            r = random.uniform(self.lower, self.upper)
            for img in image:
                img[:, :, 1] *= r

        return image, boxes, labels

# Done
class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            r = random.uniform(-self.delta, self.delta)
            for img in image:
                img[:, :, 0] += r
                img[:, :, 0][img[:, :, 0] > 360.0] -= 360.0
                img[:, :, 0][img[:, :, 0] < 0.0] += 360.0
        return image, boxes, labels

# Done
class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = [shuffle(img) for img in image]
        return image, boxes, labels

# Done
class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, boxes=None, labels=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = [cv2.cvtColor(img, cv2.COLOR_BGR2HSV) for img in image]
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = [cv2.cvtColor(img, cv2.COLOR_HSV2BGR) for img in image]
        else:
            raise NotImplementedError
        return image, boxes, labels

# Done
class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image = [img * alpha for img in image]
        return image, boxes, labels

# Done
class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image = [img + delta for img in image]
        return image, boxes, labels

# Done
class ToCV2Image(object):
    def __call__(self, tensor, boxes=None, labels=None):
        # convert back from BCHW to BHWC
        return tensor.cpu().numpy().astype(np.float32).transpose((0, 2, 3, 1)), boxes, labels

# Done
class ToTensor(object):
    def __call__(self, cvimage, boxes=None, labels=None):
        imgs = [torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1).unsqueeze(0) for img in cvimage]
        return torch.cat(imgs,0), boxes, labels

# Done
class RandomSampleCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """
    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image[0].shape
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:
                return image, boxes, labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            temp_boxes = boxes.copy()
            temp_boxes = temp_boxes.reshape((temp_boxes.shape[0]*temp_boxes.shape[1], 4))

            # max trails (50)
            for _ in range(50):
                current_image = [img for img in image]

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(temp_boxes, rect)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # cut the crop from the image
                current_image = [img[rect[1]:rect[3], rect[0]:rect[2],:] for img in current_image]

                # keep overlap with gt box IF center in sampled patch

                centers = (boxes[:, :, :2] + boxes[:, :, 2:]) / 2.0


                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, :, 0]) * (rect[1] < centers[:, :, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, :, 0]) * (rect[3] > centers[:, :, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2
                mask = mask.all(1) # Collapse along tubelet dimension

                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                # take only matching gt boxes
                current_boxes = boxes[mask, :, :].copy()

                # take only matching gt labels
                current_labels = labels[mask]

                # should we use the box left and top corner or the crop's
                current_boxes[:, :, :2] = np.maximum(current_boxes[:, :, :2], rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :, :2] -= rect[:2]

                current_boxes[:, :, 2:] = np.minimum(current_boxes[:, :, 2:], rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :, 2:] -= rect[:2]

                assert (current_boxes >= 0).any(), 'boxes cannot be negative'
                assert ((current_boxes[:, :, 2] - current_boxes[:, :, 0]) >= 0).any(), 'x2 should be greater than x1'

                return current_image, current_boxes, current_labels

# Expands dimensions of image by ratio
# done
class Expand(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, boxes, labels):
        if random.randint(2):
            return image, boxes, labels

        height, width, depth = image[0].shape
        ratio = random.uniform(1, 4)
        left = random.uniform(0, width*ratio - width)
        top = random.uniform(0, height*ratio - height)

        new_images = []
        for i, img in enumerate(image):
            expand_image = np.zeros(
                (int(height*ratio), int(width*ratio), depth),
                dtype=img.dtype)
            expand_image[:, :, :] = self.mean
            expand_image[int(top):int(top + height),
                         int(left):int(left + width)] = img
            img = expand_image
            new_images.append(img)

        new_boxes = boxes.copy()
        new_boxes[:, :, :] += (int(left), int(top), int(left), int(top))

        assert (new_boxes >= 0).any(), 'boxes cannot be negative'
        assert ((new_boxes[:, :, 2] - new_boxes[:, :, 0]) >= 0).any(), 'x2 should be greater than x1'

        return new_images, new_boxes, labels

# Done
class RandomMirror(object):
    def __call__(self, image, boxes, classes):
        _, width, _ = image[0].shape
        if random.randint(2):
            image = [ img[:, ::-1] for img in image]
            new_boxes = boxes.copy()
            new_boxes[:, :, 0::2] = width - new_boxes[:, :, 2::-2]
            assert (new_boxes >= 0).any(), 'boxes cannot be negative'
            assert ((new_boxes[:, :, 2] - new_boxes[:, :, 0]) >= 0).any(), 'x2 should be greater than x1'
        else:
            new_boxes = boxes
        return image, new_boxes, classes

class RandomSpatialResolution(object):
    def __init__(self, max_factor=2):
        self.max_factor = max_factor

    # expects float image
    def __call__(self, image, boxes=None, labels=None):
        random_int = random.randint(self.max_factor+1)
        if random_int > 0:
            factor = 2 ** (random_int) # 2^1, 2^2
            height, width, _ = image[0].shape
            image = [cv2.resize(cv2.resize(im, (width // factor, height // factor), interpolation=cv2.INTER_AREA),
                                (width, height),
                                interpolation=cv2.INTER_NEAREST)
                     for im in image]

            # image = [img * alpha for img in image]
        return image, boxes, labels

# Did not need to do
class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image

# Done
class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, boxes, labels):
        im = image.copy()
        im, boxes, labels = self.rand_brightness(im, boxes, labels)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, boxes, labels = distort(im, boxes, labels)
        return self.rand_light_noise(im, boxes, labels)


class SSDAugmentation(object): ### BGR!!!
    def __init__(self, size=300, mean=(104, 117, 123), std=(0.225, 0.224, 0.229)):
        self.mean = mean
        self.size = size
        self.std = std
        self.augment = Compose([
            ConvertFromInts(),
            ToAbsoluteCoords(),
            PhotometricDistort(),
            Expand(self.mean),
            RandomSampleCrop(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(self.size),
            SubtractMeans(self.mean, self.std)
        ])

    def __call__(self, img, boxes, labels):
        return self.augment(img, boxes, labels)

class SSDAugmentationFeedback(object): ### BGR!!!
    def __init__(self, size=300, mean=(104, 117, 123), std=(0.225, 0.224, 0.229)):
        self.mean = mean
        self.size = size
        self.std = std
        self.augment = Compose([
            ConvertFromInts(),
            RandomSpatialResolution(2),
            ToAbsoluteCoords(),
            PhotometricDistort(),
            Expand(self.mean),
            RandomSampleCrop(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(self.size),
            SubtractMeans(self.mean, self.std)
        ])

    def __call__(self, img, boxes, labels):
        return self.augment(img, boxes, labels)

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

def base_transform_tubelet(image, size, mean):
    image = [cv2.resize(img, (size, size)).astype(np.float32) for img in image]
    image = [img - mean for img in image]
    image = [img.astype(np.float32) for img in image]
    return image

class BaseTransformTubeLet:
    def __init__(self, size, mean):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        return base_transform_tubelet(image, self.size, self.mean), boxes, labels
