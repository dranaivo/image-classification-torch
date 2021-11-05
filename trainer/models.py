'''Definitions for classification models
> AlexNet [ok]
> ResNet
> EfficientNet
'''
from torch import nn
from torchvision import models
from torchvision.models.alexnet import alexnet
import torch

def get_model(pretrained: bool=True, n_classes: int=1000) -> torch.nn.Module:
    ''' Create a classification model (Alexnet)

    The classifier head of the classification model is always initialized.
    As for the feature extractor, see the Args.

    Args:
        pretrained : if True, the backbone is pre-trained on Imagenet.
    Returns:
        The classification model (tmp: AlexNet)
    '''

    model = models.alexnet(pretrained=pretrained)
    # this is a copy of model.classifer
    classifier_ft = nn.Sequential(
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(256*6*6, out_features=64, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(64, out_features=64, bias=True),
        nn.ReLU(inplace=True),
        nn.Linear(64, out_features=n_classes, bias=True),
    )
    model.classifier = classifier_ft

    return model
