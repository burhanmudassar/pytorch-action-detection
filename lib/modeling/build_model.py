import torch
from torch.autograd import Variable

from lib.modeling.components import build_backbone
from lib.modeling.components import build_extras
from lib.modeling.components.modules import FeatureExtractor
from lib.modeling.layers.functions import PriorBox
from lib.modeling.models.ACT import ACT
from lib.modeling.models.ACT_Fusion import SSD_Fusion

net_map = {
    'resnet50': build_backbone.build_resnet50,
    'mobilenetv2': build_backbone.build_mobilenetv2,
    'vgg16': build_backbone.build_vgg16bn
}

extras_map = {
    'resnet50': build_extras.add_extras_resnet,
    'mobilenetv2': build_extras.add_extras_mobilenet,
    'vgg16': build_extras.add_extras_vgg
}

model_map = {
    'ACT': ACT
}

def build_ssd(model_cfg):
    backbone = net_map[model_cfg.NETS](model_cfg)
    extra_layers = extras_map[model_cfg.NETS](model_cfg)

    img_size = model_cfg.IMAGE_SIZE
    # Create feature extractor and determine the size of the feature maps
    feature_extractor = FeatureExtractor(model_cfg, backbone, extra_layers)
    if torch.cuda.is_available():
        feature_extractor = feature_extractor.cuda()
    feature_extractor.eval()
    C = model_cfg.K
    if model_cfg.INPUT_TYPE == 'fusion':
        C = model_cfg.K * 2

    x = torch.rand(1, C, 3, img_size[0], img_size[1])
    x = torch.autograd.Variable(x, volatile=True) #.cuda()
    x = x.view(x.size(0) * x.size(1), x.size(2), x.size(3), x.size(4))
    feature_maps = feature_extractor(x)
    feature_maps[-1] = feature_maps[-1].mean(-1, keepdim=True).mean(-2, keepdim=True)
    fmap_size = [(o.size()[-2], o.size()[-1]) for o in feature_maps]
    print(fmap_size)

    kwargs = {
        'cfg':model_cfg,
        'feature_extractor':feature_extractor,
        'num_classes':model_cfg.NUM_CLASSES,  # Should be background + 1
        'num_K': model_cfg.K,
        'fmap_size':fmap_size,
    }

    if model_cfg.INPUT_TYPE == 'fusion':
        kwargs['model_map'] = model_map[model_cfg.SSDS]
        ssd = SSD_Fusion(**kwargs)
    else:
        ssd = model_map[model_cfg.SSDS](**kwargs)
    priorbox = PriorBox(model_cfg, fmap_size)

    return ssd, priorbox