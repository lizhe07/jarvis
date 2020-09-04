# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 12:15:55 2020

@author: Zhe
"""

import torch
import torch.nn as nn

class ResBlock(nn.Module):
    r"""ResNet block.

    Args
    ----
    block_type: str
        The block type, can be ``Basic`` or ``Bottleneck``.
    in_channels: int
        The input channel number.
    base_channels: int
        The base channel number for layers.
    stride: int
        The stride for the first convolution.

    """

    def __init__(self, block_type, in_channels, base_channels, stride=1):
        super(ResBlock, self).__init__()
        self.block_type = block_type
        if self.block_type=='Basic':
            expansion = 1
        if self.block_type=='Bottleneck':
            expansion = 4
        out_channels = expansion*base_channels
        self.in_channels, self.out_channels = in_channels, out_channels

        if self.block_type=='Basic':
            self.layer0 = nn.Sequential(
                nn.Conv2d(in_channels, base_channels,
                          kernel_size=3, padding=1, stride=stride, bias=False),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(),
                )
            self.layer1 = nn.Sequential(
                nn.Conv2d(base_channels, out_channels,
                          kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(base_channels),
                )
        if self.block_type=='Bottleneck':
            self.layer0 = nn.Sequential(
                nn.Conv2d(in_channels, base_channels,
                          kernel_size=1, padding=0, stride=stride, bias=False),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(),
                )
            self.layer1 = nn.Sequential(
                nn.Conv2d(base_channels, base_channels,
                          kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(),
                )
            self.layer2 = nn.Sequential(
                nn.Conv2d(base_channels, out_channels,
                          kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
                )

        if stride==1 and in_channels==out_channels:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, padding=0, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
                )

    def layer_activations(self, x):
        r"""Returns activations of all layers.

        Args
        ----
        x: (N, C, H, W), tensor
            The input to this block.

        Returns
        -------
        activations: list of tensors
            The activations for each layer.

        """
        if self.block_type=='Basic':
            out0 = self.layer0(x)
            out1 = torch.relu(self.layer1(out0)+self.shortcut(x))
            activations = [out0, out1]
        if self.block_type=='Bottleneck':
            out0 = self.layer0(x)
            out1 = self.layer1(out0)
            out2 = torch.relu(self.layer2(out1)+self.shortcut(x))
            activations = [out0, out1, out2]
        return activations

    def forward(self, x):
        r"""Implements the forward pass of the ResNet block.

        Args
        ----
        x: (N, C, H, W), tensor
            The input to this block.

        Returns
        -------
        The output of the block.

        """
        return self.layer_activations(x)[-1]


class ResNet(nn.Module):
    r"""ResNet model.

    In addition to an input section, four more sections are stacked to form a
    classical `ResNet model <https://arxiv.org/abs/1512.03385>`_. Each section
    is composed of a few ResNet blocks, either basic ones or bottleneck ones.
    Within each section the spatial size of feature maps does not change.

    Args
    ----
    block_nums: list
        The number of ResNet blocks in each section, usually of length 4.
    block_type: str
        The block type, can be ``Basic`` or ``Bottleneck``.
    class_num: int
        The number of classes.
    in_channels: int
        The input channel number.
    base_channels: int
        The base channel number.
    i_shift: (in_channels,), tensor
        The shift parameter for image preprocessing.
    i_scale: (in_channels,), tensor
        The scale parameter for image preprocessing.

    """

    def __init__(self, block_nums, block_type, class_num=10,
                 in_channels=3, base_channels=64,
                 i_shift=None, i_scale=None, **kwargs):
        super(ResNet, self).__init__()

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
        self.i_shift = torch.nn.Parameter(
            torch.tensor(i_shift, dtype=torch.float)[:, None, None],
            requires_grad=False
            )
        self.i_scale = torch.nn.Parameter(
            torch.tensor(i_scale, dtype=torch.float)[:, None, None],
            requires_grad=False
            )

        assert block_type in ['Basic', 'Bottleneck']
        self.section_num = len(block_nums)

        self.sections = nn.ModuleList([nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            )])

        in_channels = base_channels
        base_channels = [base_channels*(2**i) for i in range(self.section_num)]
        strides = [1]+[2]*(self.section_num-1)

        for i in range(self.section_num):
            section, in_channels = self._make_section(
                block_nums[i], block_type, in_channels, base_channels[i], strides[i]
                )
            self.sections.append(section)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, class_num)

    def _make_section(self, block_num, block_type, in_channels, base_channels, stride):
        r"""Constructs one ResNet section.

        Args
        ----
        block_num: int
            The number of ResNet blocks in the section.
        block_type: str
            The block type.
        in_channels: int
            The input channel number.
        base_channels: int
            The base channel number for blocks in the section.
        stride: int
            The stride of the first ResNet block.

        Returns
        -------
        section: nn.Module
            The constructed ResNet section.
        out_channels: int
            The output channel number.

        """
        blocks = [ResBlock(block_type, in_channels, base_channels, stride)]
        out_channels = blocks[0].out_channels
        for _ in range(block_num-1):
            blocks.append(ResBlock(block_type, out_channels, base_channels))
        section = nn.Sequential(*blocks)
        return section, out_channels

    def layer_activations(self, images):
        r"""Returns activations of all layers.

        Args
        ----
        images: (N, C, H, W), tensor
            The input images, with values in ``[0, 1]``.

        Returns
        -------
        activations: list of tensors
            The activations for each layer.

        """
        activations = [self.sections[0]((images-self.i_shift)*self.i_scale)]
        for i in range(self.section_num):
            for block in self.sections[i+1]:
                activations += block.layer_activations(activations[-1])
        activations.append(self.fc(self.avgpool(activations[-1]).flatten(1)))
        return activations

    def forward(self, images):
        r"""Implements the forward pass of the ResNet model.

        Args
        ----
        images: (N, C, H, W), tensor
            The input images, with values in ``[0, 1]``.

        Returns
        -------
        logits: (N, class_num), tensor
            The output logits.

        """
        logits = self.layer_activations(images)[-1]
        return logits


def resnet18(**kwargs):
    return ResNet([2, 2, 2, 2], 'Basic', **kwargs)


def resnet34(**kwargs):
    return ResNet([3, 4, 6, 3], 'Basic', **kwargs)


def resnet50(**kwargs):
    return ResNet([3, 4, 6, 3], 'Bottleneck', **kwargs)


def resnet101(**kwargs):
    return ResNet([3, 4, 23, 3], 'Bottleneck', **kwargs)


def resnet152(**kwargs):
    return ResNet([3, 8, 36, 3], 'Bottleneck', **kwargs)
