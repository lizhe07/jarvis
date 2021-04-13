# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 14:53:08 2020

@author: Zhe
"""

from typing import Optional, List, Any

import torch
import torch.nn as nn


class Normalizer(nn.Module):

    def __init__(
            self,
            mean: List[float],
            std: List[float],
            ) -> None:
        super(Normalizer, self).__init__()

        assert len(mean)==1 or len(std)==1 or len(mean)==len(std), 'mean and std are inconsistent'
        assert min(std)>0, 'std has to be positive'
        self.mean = nn.Parameter(
            torch.tensor(mean, dtype=torch.float)[..., None, None],
            requires_grad=False
            )
        self.std = nn.Parameter(
            torch.tensor(std, dtype=torch.float)[..., None, None],
            requires_grad=False
            )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return (images-self.mean)/self.std


class ImageClassifier(nn.Module):
    r"""A base class for image classifier.

    Args
    ----
    in_channels: int
        The number of input channels.
    class_num: int
        The number of classes.
    mean, std: list of floats
        The mean and std parameters for input normalization.

    """

    def __init__(
            self,
            in_channels: int = 3,
            class_num: int = 10,
            mean: Optional[List[float]] = None,
            std: Optional[List[float]] = None,
            **kwargs: Any,
            ) -> None:
        super(ImageClassifier, self).__init__()
        self.in_channels, self.class_num = in_channels, class_num

        if mean is None:
            if in_channels==3:
                mean = [0.485, 0.456, 0.406]
            else:
                mean = [0.5]
        if std is None:
            if in_channels==3:
                std = [0.229, 0.224, 0.225]
            else:
                std = [0.2]
        self.normalizer = Normalizer(mean, std)

    def layer_activations(self, x: torch.Tensor) -> List[torch.Tensor]:
        raise NotImplementedError

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        *_, logits = self.layer_activations(self.normalizer(images))
        return logits


class WrappedClassifier(ImageClassifier):

    def __init__(
            self,
            model: nn.Module,
            **kwargs: Any,
            ) -> None:
        super(WrappedClassifier, self).__init__(**kwargs)
        self.raw_model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.raw_model(self.normalizer(images))
        return logits
