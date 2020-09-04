""" Generates prior boxes for SSD netowrk

Original author: Ellis Brown, Max deGroot for VOC dataset
https://github.com/amdegroot/ssd.pytorch

"""

import torch
from math import sqrt as sqrt
from itertools import product as product

class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    Note:
    This 'layer' has changed between versions of the original SSD
    paper, so we include both versions, but note v2 is the most tested and most
    recent version of the paper.

    """
    def __init__(self, cfg, feature_maps):
        super(PriorBox, self).__init__()
        # self.type = cfg.name
        self.image_size = cfg.IMAGE_SIZE[0]
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = sum([len(ar) * 2 for ar in cfg.ASPECT_RATIOS])
        self.feature_maps = feature_maps
        self.min_sizes = [v[0] for v in cfg.SIZES]
        self.max_sizes = [v[1] for v in cfg.SIZES]
        self.steps = [v[0] for v in cfg.STEPS]
        self.aspect_ratios = cfg.ASPECT_RATIOS
        self.clip = True
        self.variance = cfg.VARIANCE
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f[0]), range(f[1])):
                f_k = self.image_size / self.steps[k]
                # unit center x,y
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # aspect_ratio: 1
                # rel size: min_size
                s_k = self.min_sizes[k]/self.image_size

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))

                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    if ar == 1:
                        mean += [cx, cy, s_k, s_k]
                        mean += [cx, cy, s_k_prime, s_k_prime]
                    else:
                        mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                        mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]

        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
