# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 14:16:01 2021

@author: Zhe
"""

from typing import List, Any

import torch
import torch.nn as nn

from . import ImageClassifier


class LeNet(ImageClassifier):
    r"""LeNet model architecture from the
    `"Backpropagation Applied to..." <https://www.mitpressjournals.org/doi/abs/10.1162/neco.1989.1.4.541>`_
    paper.

    """

    def __init__(
            self,
            **kwargs: Any,
            ) -> None:
        super(LeNet, self).__init__(**kwargs)
        in_channels, class_num = self.in_channels, self.class_num

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, 6, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                ),
            nn.Sequential(
                nn.Conv2d(6, 16, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                ),
            nn.Sequential(
                nn.Conv2d(16, 120, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.AdaptiveMaxPool2d(1),
                nn.Flatten(1),
                ),
            nn.Sequential(
                nn.Linear(120, 84),
                nn.ReLU(),
                ),
            ])
        self.fc = nn.Linear(84, out_features=class_num)

    def layer_activations(self, x: torch.Tensor) -> List[torch.Tensor]:
        r"""Returns activations of all layers.

        Args
        ----
        x: (N, C, H, W), tensor
            The normalized input images.

        Returns
        -------
        acts: list of tensors
            The activations for each layer (unitl the last fc layer).
        logits: tensor
            The logits.

        """
        acts = []
        for layer in self.layers:
            x = layer(x)
            acts.append(x)
        logits = self.fc(x)
        return acts, logits


def lenet(**kwargs: Any) -> LeNet:
    model = LeNet(**kwargs)
    return model
