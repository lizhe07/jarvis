# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 11:42:08 2020

@author: Zhe
"""

from typing import List, Any

import torch
import torch.nn as nn


class AlexNet(nn.Module):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args
    ----
    class_num: int
        The number of classes.
    act_type: str
        Nonlinear activation type, can be ``'ReLU'``, ``'Hardtanh'``.

    """

    def __init__(
            self,
            in_channels: int = 3,
            class_num: int = 10,
            act_type: str = 'ReLU',
            ) -> None:
        super(AlexNet, self).__init__()

        def _act_layer(act_type):
            if act_type=='ReLU':
                return nn.ReLU(inplace=True)
            if act_type=='Hardtanh':
                return nn.Hardtanh(inplace=True)

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=11, stride=4, padding=2),
                _act_layer(act_type),
                nn.MaxPool2d(kernel_size=3, stride=2),
                ),
            nn.Sequential(
                nn.Conv2d(64, 192, kernel_size=5, padding=2),
                _act_layer(act_type),
                nn.MaxPool2d(kernel_size=3, stride=2),
                ),
            nn.Sequential(
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                _act_layer(act_type),
                ),
            nn.Sequential(
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                _act_layer(act_type),
                ),
            nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                _act_layer(act_type),
                nn.MaxPool2d(kernel_size=3, stride=2),
                )
            ])
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256*6*66, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, class_num),
        )

    def layer_activations(self, x: torch.Tensor) -> List[torch.Tensor]:
        activations = []
        for layer in self.layers:
            x = layer(x)
            activations.append(x)
        return activations

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_activations(x)[-1]
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def alexnet(**kwargs: Any) -> AlexNet:
    model = AlexNet(**kwargs)
    return model
