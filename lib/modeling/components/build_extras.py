from numpy import kernel_version
import torch.nn as nn
from torchvision.models.resnet import Bottleneck

def _make_layer_resnet(block, inplanes, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * block.expansion,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )

    layers = []
    layers.append(block(inplanes, planes, stride, downsample))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(inplanes, planes))

    return nn.Sequential(*layers)

def _conv_dw(inp, oup, stride=1, padding=0, expand_ratio=1):
    return nn.Sequential(
        # pw
        nn.Conv2d(inp, oup * expand_ratio, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup * expand_ratio),
        nn.ReLU6(inplace=True),
        # dw
        nn.Conv2d(oup * expand_ratio, oup * expand_ratio, 3, stride, padding, groups=oup * expand_ratio, bias=False),
        nn.BatchNorm2d(oup * expand_ratio),
        nn.ReLU6(inplace=True),
        # pw-linear
        nn.Conv2d(oup * expand_ratio, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
    )

def add_extras_resnet(model_cfg):
    # B3 is a 3 block bottleneck similar to res4 and res5
    layers = []
    in_channels = 0
    for k,v in enumerate(zip(model_cfg.FEATURE_LAYER[0], model_cfg.FEATURE_LAYER[1])):
        if v[0] == 'B3':
            layers += [_make_layer_resnet(block=Bottleneck, inplanes=in_channels, planes=v[1], blocks=3, stride=2)]
            in_channels = v[1] * 4
        elif v[0] == 'B2':
            layers += [_make_layer_resnet(block=Bottleneck, inplanes=in_channels, planes=v[1], blocks=2, stride=2)]
            in_channels = v[1] * 4
        elif v[0] == 'B1':
            layers += [_make_layer_resnet(block=Bottleneck, inplanes=in_channels, planes=v[1], blocks=1, stride=3)]
            in_channels = v[1] * 4
        elif v[0] == 'P':
            layers += [nn.AvgPool2d(kernel_size=3, stride=None, padding=0)]
        else:
            in_channels = v[1]
            continue

    return layers

def add_extras_vgg(model_cfg):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = 0
    for k, v in enumerate(zip(model_cfg.FEATURE_LAYER[0], model_cfg.FEATURE_LAYER[1])):
        depth = v[1]
        if v[0] == 'S':
            layers += [ nn.Sequential(
                            nn.Conv2d(in_channels, int(depth / 2), kernel_size=1),
                            nn.ReLU(),
                            nn.Conv2d(int(depth / 2), depth, kernel_size=3, stride=2, padding=1),
                            nn.ReLU()
                             )
                      ]

        elif v[0] == '':
            layers += [ nn.Sequential(
                        nn.Conv2d(in_channels, int(depth / 2), kernel_size=1),
                        nn.ReLU(),
                        nn.Conv2d(int(depth / 2), depth, kernel_size=3),
                        nn.ReLU()
                         )
                      ]

        in_channels = depth
    return layers

def add_extras_mobilenet(model_cfg):
    # Four extra layers after feature extractor
    extra_layers = []
    in_channels = 0
    for k, v in enumerate(zip(model_cfg.FEATURE_LAYER[0], model_cfg.FEATURE_LAYER[1])):
        depth = v[1]
        if v[0] == 'S':
            extra_layers += [ _conv_dw(in_channels, depth, stride=2, padding=1, expand_ratio=1) ]
            in_channels = depth
        elif v[0] == 'P':
            extra_layers += [ _conv_dw(in_channels, depth, stride=2, padding=0, expand_ratio=1) ]
            in_channels = depth
        elif v[0] == '':
            extra_layers += [ _conv_dw(in_channels, depth, stride=1, expand_ratio=1) ]
            in_channels = depth
        else:
            in_channels = depth
    return extra_layers

