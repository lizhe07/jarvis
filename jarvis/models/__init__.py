# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 14:53:08 2020

@author: Zhe
"""

from typing import Optional, List, Any

import torch
import torch.nn as nn


class ImageClassifier(nn.Module):
    r"""A base class for image classifier.

    Args
    ----
    in_channels: int
        The number of input channels.
    class_num: int
        The number of classes.
    i_shift, i_scale: list of floats
        The shift and scale parameters for input preprocessing.

    """

    def __init__(
            self,
            in_channels: int = 3,
            class_num: int = 10,
            i_shift: Optional[List[float]] = None,
            i_scale: Optional[List[float]] = None,
            **kwargs: Any,
            ) -> None:
        super(ImageClassifier, self).__init__()
        self.in_channels, self.class_num = in_channels, class_num

        if i_shift is None:
            if in_channels==3:
                i_shift = [0.485, 0.456, 0.406]
            else:
                i_shift = [0.5]
        if i_scale is None:
            if in_channels==3:
                i_scale = [1/0.229, 1/0.224, 1/0.225]
            else:
                i_scale = [5.]
        self.i_shift = nn.Parameter(
            torch.tensor(i_shift, dtype=torch.float)[:, None, None],
            requires_grad=False
            )
        self.i_scale = nn.Parameter(
            torch.tensor(i_scale, dtype=torch.float)[:, None, None],
            requires_grad=False
            )

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        return (images-self.i_shift)*self.i_scale

    def layer_activations(self, images: torch.Tensor) -> List[torch.Tensor]:
        raise NotImplementedError

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        *_, logits = self.layer_activations(images)
        return logits
