import torch.nn as nn
from torchvision.models import resnet50


def build_classifier(num_classes: int):
    """ResNet50 classification head. Training can initialize from ImageNet weights."""
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
