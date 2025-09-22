"""
Backbones for feature extraction

Author: Peter Thomas
Date: September 21, 2025
"""
import torch
from torchvision.models import resnet18, ResNet18_Weights


def get_resnet18_backbone():
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    # Remove the final classification layer
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    return model