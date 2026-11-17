"""Module 3 - model architecture: ImageNet backbone (feature extractor) + new 6-class head."""
import torch.nn as nn
from torchvision import models

from . import config as C


def build(name, pretrained=True):
    n = len(C.CLASSES)
    if name == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights="IMAGENET1K_V1" if pretrained else None)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, n)
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n)
    elif name == "resnet50":
        m = models.resnet50(weights="IMAGENET1K_V2" if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, n)
    else:
        raise ValueError(name)
    return m


def head_parameters(m, name):
    return (m.fc if name == "resnet50" else m.classifier).parameters()


def last_conv(m, name):
    """Layer whose activations Grad-CAM reads."""
    return m.layer4[-1] if name == "resnet50" else m.features[-1]


def n_params(m):
    return sum(p.numel() for p in m.parameters())
