
""" SSD network Classes

Original author: Ellis Brown, Max deGroot for VOC dataset
https://github.com/amdegroot/ssd.pytorch

Updated by Gurkirt Singh for ucf101-24 dataset
"""

import torch
import torch.nn as nn
import os
import copy

class SSD_Fusion(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, **kwargs):

        super(SSD_Fusion, self).__init__()
        self.cfg = kwargs['cfg']
        self.num_classes = kwargs['num_classes']
        self.size = 300
        self.K = kwargs['num_K']
        model_map = kwargs['model_map']
        kwargs.pop('model_map')
        self.rgb_model = model_map(**kwargs)
        kwargs['feature_extractor'] = copy.deepcopy(kwargs['feature_extractor'])
        self.flo_model = model_map(**kwargs)

        self.softmax = nn.Softmax()
        if torch.cuda.is_available():
            self.softmax.cuda()

    def forward(self, x, phase='eval'):

        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3*batch,300,300].

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
        batch_size = x.size(0)
        x_rgb = x[:, ::2, ...]
        x_flo = x[:, 1::2, ...]

        if phase == 'feature':
            feature_rgb = self.rgb_model(x_rgb, phase='feature')
            return feature_rgb


        if self.cfg.FUSION_LEVEL == 'feature':
            feature_rgb, o_rgb = self.rgb_model(x_rgb, phase='feature')
            feature_flo, o_flo = self.flo_model(x_flo, phase='feature')

            if self.cfg.FUSION_TYPE == 'mean':
                feature = [f_rgb + f_flo for f_rgb, f_flo in zip(feature_rgb, feature_flo)]
                offsets = [o_rgb + o_flo for o_rgb, o_flo in zip(o_rgb, o_flo)]

                feature = self.rgb_model.temporalagg(feature, offsets)
                feature[-1] = feature[-1].mean(-1, keepdim=True).mean(-2, keepdim=True)
                if self.rgb_model.global_feat:
                    feature = self.rgb_model.global_feat(feature)

                loc, conf = self.rgb_model.predictor(feature)
                loc = loc.view(loc.size(0), conf.size(1), -1, 4)
                priors = self.priors


        if self.cfg.FUSION_LEVEL == 'decision':
            out_rgb = self.rgb_model(x_rgb, phase=phase)
            out_flo = self.flo_model(x_flo, phase=phase)

            loc_rgb, conf_rgb, priors_rgb = out_rgb
            loc_flo, conf_flo, priors_flo = out_flo
            if self.cfg.FUSION_TYPE == 'mean':
                loc = loc_rgb
                conf = (conf_rgb + conf_flo) / 2
                priors = self.priors
            elif self.cfg.FUSION_TYPE == 'cat':
                loc = torch.cat([loc_rgb, loc_flo], dim=1)
                conf = torch.cat([conf_rgb, conf_flo], dim=1)
                priors = torch.cat([self.priors, self.priors], dim=0)

        return (
            loc,
            conf,
            priors
        )

