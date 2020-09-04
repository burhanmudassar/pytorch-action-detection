
""" SSD network Classes

Original author: Ellis Brown, Max deGroot for VOC dataset
https://github.com/amdegroot/ssd.pytorch

Updated by Gurkirt Singh for ucf101-24 dataset
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from lib.modeling.components.modules import Predictor2D
from lib.modeling.components.modules import TemporalAggregator
from lib.modeling.components.modules import TemporalAggregation_Mean
from lib.modeling.components.modules import TemporalAggregation_Cat
from lib.modeling.components.modules import TemporalAggregation_ConvCat


class ACT(nn.Module):
    """ACT Architecture adapted from SSD
    The network is composed of a backbone 2D CNN followed by couple 
    of extra layers and added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions - 4*K where K is the length of the clip
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        base: backbone(VGG16) layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, cfg, num_classes, fmap_size, feature_extractor, offmap_size, num_K, **kwargs):
        super().__init__()

        self.cfg = cfg
        self.num_classes = num_classes
        # TODO: implement __call__ in PriorBox
        # self.priorbox = PriorBox(resnetv1)
        # self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.priors = torch.Tensor(0)
        self.priorbox = None
        # self.num_priors = self.priors.size(0)
        self.size = 300
        self.K = num_K

        mbox_layers = add_predictor_layers(cfg,
                                          num_classes,
                                          num_K=self.K,
                                          fmap_size=fmap_size)

        self.softmax = nn.Softmax().cuda()

        self.feature_extractor = feature_extractor
        self.predictor = Predictor2D(cfg, mbox_layers[1], mbox_layers[2], self.num_classes)
        self.extras = TemporalAggregator(cfg, mbox_layers[0])

        if torch.cuda.is_available():
            self.feature_extractor = self.feature_extractor.cuda()
            self.predictor = self.predictor.cuda()
            self.extras = self.extras.cuda()

    def forward(self, x, phase='eval'):

        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,K,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        # Reshape the batch
        batch_size = x.size(0)
        x = x.view(x.size(0)*x.size(1), x.size(2), x.size(3), x.size(4))

        sources = self.feature_extractor(x)

        if phase == 'feature':
            return sources

        # Do Temporal Aggregation
        sources = self.extras([sources])

        # apply multibox head to source layers
        loc, conf = self.predictor(sources)

        conf = conf.view(conf.size(0), -1, self.num_classes)
        loc = loc.view(loc.size(0), conf.size(1), -1, 4)

        output = (loc,
                  conf,
                  self.priors
                  )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file, map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


def add_predictor_layers(model_cfg, num_classes, num_K=1, fmap_size=None):
    temporalagg_layers = []
    loc_layers = []
    conf_layers = []

    for k,v in enumerate(model_cfg.FEATURE_LAYER[1]):
        in_channels = v
        if isinstance(model_cfg.FEATURE_LAYER[0][k], str):          ## To deal with bottleneck block
            if 'B' in model_cfg.FEATURE_LAYER[0][k]:
                in_channels = v * 4

        elif model_cfg.TEMPORAL_LAYER[k] == 'mean':
            temporalagg_layers += [TemporalAggregation_Mean(model_cfg)]
        elif model_cfg.TEMPORAL_LAYER[k] == 'cat':
            in_channels *= num_K
            temporalagg_layers += [TemporalAggregation_Cat(model_cfg)]
        elif model_cfg.TEMPORAL_LAYER[k] == 'convcat':
            in_channels *= num_K
            temporalagg_layers += [TemporalAggregation_ConvCat(model_cfg, in_channels)]
        else:
            print("some temporal aggregation must be specified")
            raise NotImplementedError

        anchor_boxes = len(model_cfg.ASPECT_RATIOS[k]) * 2
        loc_layers += [nn.Conv2d(in_channels, num_K * anchor_boxes * 4, kernel_size=3, padding=1)] #K anchor regressions
        conf_layers += [nn.Conv2d(in_channels, anchor_boxes * num_classes, kernel_size=3, padding=1)]

    return temporalagg_layers, loc_layers, conf_layers

