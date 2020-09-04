import torch.nn as nn
from torchvision.models import resnet
from torchvision.models import vgg
from torchvision.models import mobilenet_v2

def build_resnet50(cfg):
    backbone = resnet.resnet50(pretrained=True)
    features = list(backbone.children())[:-2]  # For ResNet-101 drop avg-pool and linear classifier

    return features

def build_mobilenetv2(cfg):
    backbone = mobilenet_v2(pretrained=True, progress=True)
    features = list(backbone.features.children())[:-1]
    return features

def build_vgg16bn(cfg):
    '''
    Match with Gurkirt SSD
    1. Their VGG does not end with MaxPool but torchvision does hence -1
    2. Pool5, conv6, conv6 are added
    3. Their feature sources is 23 (conv_4_3 -> 32 here) and -2 (conv7 -> 47)
    4. 23 has L2 Norm
    5. 4 extra layers
    '''
    backbone = vgg.vgg16_bn(pretrained=True)
    features = list(backbone.features.children())[:-1]


    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    features += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]


    return features