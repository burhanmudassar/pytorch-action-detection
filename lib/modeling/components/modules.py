import torch
import torch.nn as nn
from lib.modeling.layers.modules.l2norm import L2Norm
import torch.nn.functional as F

# Get all fmaps
class FeatureExtractor(nn.Module):
    '''
    Encapsulates the backbone and runs each layer in a for loop.
    '''
    def __init__(self, cfg, backbone, extras):
        super(FeatureExtractor, self).__init__()

        self.backbone = nn.ModuleList(backbone)
        self._bn_modules = [it for it in self.backbone.modules() if isinstance(it, nn.BatchNorm2d)]
        self.backbone_source = cfg.FEATURE_LAYER[0]
        self.extras = nn.ModuleList(extras)

        self.L2Norm = []
        for ind, k in enumerate(self.backbone_source):
            # If a feature map from pre-trained backbone is used then apply L2 Normalization
            if type(k) == int:
                self.L2Norm += [L2Norm(cfg.FEATURE_LAYER[1][ind], 20)]
        self.L2Norm = nn.ModuleList(self.L2Norm)

    def forward(self, x):
        sources = list()
        norm_ind = 0
        for k in range(len(self.backbone)):
            x = self.backbone[k](x)
            if k in self.backbone_source:
                s = self.L2Norm[norm_ind](x)
                sources.append(s)
                norm_ind += 1

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = v(x)
            sources.append(x)

        return sources

# Apply multibox on each fmap
class Predictor2D(nn.Module):
    def __init__(self, cfg, loc, conf, num_classes):
        super(Predictor2D, self).__init__()

        self.loc = nn.ModuleList(loc)
        self.conf = nn.ModuleList(conf)
        self.num_classes = num_classes

    def forward(self, sources):
        loc = list()
        conf = list()

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            o_l = l(x)
            o_c = c(x)

            loc.append(o_l.permute(0, 2, 3, 1).contiguous())
            conf.append(o_c.permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        output = (loc.view(loc.size(0), -1, 4),
                  conf.view(conf.size(0), -1, self.num_classes))

        return output

# Apply each fmaps temporal aggregator
class TemporalAggregator(nn.Module):
    def __init__(self, cfg, tempagg):
        super(TemporalAggregator, self).__init__()
        self.tempagg = nn.ModuleList(tempagg)
        self.K = cfg.K

    def forward(self, input):
        sources = input
        sources_new = []

        for x_ind, x in enumerate(sources):
            sources_new.append(self.tempagg[x_ind](x))

        return sources_new


        