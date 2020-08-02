# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 21:59:48 2020

@author: Zhe
"""

import os, random
import torch, torchvision
import numpy as np
from torch.utils.data import Subset
from torchvision import transforms


def prepare_datasets(task, benchmarks_dir, valid_num=None):
    r"""Prepares vision datasets.

    Args
    ----
    task: str
        The name of the dataset, e.g. ``'CIFAR10'``.
    benchmarks_dir: str
        The directory of vision benchmarks.
    valid_num: int
        The validation dataset size. For ``'16ImageNet'`` task, it is the
        number of validation images per class. When `valid_num` is ``None``,
        only testing dataset will be returned.

    Returns
    -------
    dataset_train, dataset_valid, dataset_test: Dataset
        The training/validation/testing datasets for the task pathway. Each
        dataset item is a tuple `(image, label)`. `image` is a float tensor of
        shape `(3, H, W)` in [0, 1], and `label` is an integer.
    weight: (class_num,), array_like
        The class weight for training dataset. If the classes are balanced,
        ``None`` is returned.

    Examples
    --------
    >>> dataset_test = prepare_datasets(task, benchmarks_dir)

    >>> dataset_train, dataset_valid, dataset_test, weight = \
            prepare_datasets(task, benchmarks_dir, valid_num)

    """
    if task.startswith('CIFAR'):
        t_common = [
            transforms.ToTensor(),
        ]
        if task=='CIFAR10':
            dataset_test = torchvision.datasets.CIFAR10(
                benchmarks_dir, train=False, transform=transforms.Compose(t_common),
                )
        if task=='CIFAR100':
            dataset_test = torchvision.datasets.CIFAR100(
                benchmarks_dir, train=False, transform=transforms.Compose(t_common),
                )

        if valid_num is None:
            return dataset_test
        else:
            t_aug = [
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                transforms.RandomHorizontalFlip(),
                ]
            sample_num = 50000
            idxs_valid = random.sample(range(sample_num), valid_num)
            idxs_train = [i for i in range(sample_num) if i not in idxs_valid]
            if task=='CIFAR10':
                dataset_train = Subset(torchvision.datasets.CIFAR10(
                    benchmarks_dir, train=True, transform=transforms.Compose(t_aug+t_common),
                    ), idxs_train)
                dataset_valid = Subset(torchvision.datasets.CIFAR10(
                    benchmarks_dir, train=True, transform=transforms.Compose(t_common),
                    ), idxs_valid)
            if task=='CIFAR100':
                dataset_train = Subset(torchvision.datasets.CIFAR100(
                    benchmarks_dir, train=True, transform=transforms.Compose(t_aug+t_common),
                    ), idxs_train)
                dataset_valid = Subset(torchvision.datasets.CIFAR100(
                    benchmarks_dir, train=True, transform=transforms.Compose(t_common),
                    ), idxs_valid)

            weight = None
            return dataset_train, dataset_valid, dataset_test, None

    if task=='16ImageNet':
        t_common = [
            transforms.ToTensor(),
        ]
        dataset_test = torchvision.datasets.ImageFolder(
            f'{benchmarks_dir}/16imagenet_split/test',
            transform=transforms.Compose(t_common),
            )

        if valid_num is None:
            return dataset_test
        else:
            t_aug = [
                transforms.RandomCrop(256, padding=32, padding_mode='reflect'),
                transforms.RandomHorizontalFlip(),
            ]
            class_names = os.listdir(f'{benchmarks_dir}/16imagenet_split/train')
            sample_nums = [len(os.listdir(f'{benchmarks_dir}/16imagenet_split/train/{c_name}')) \
                           for c_name in class_names]
            idxs_valid = [random.sample(range(sample_num), valid_num) \
                          for sample_num in sample_nums]
            idxs_valid = np.concatenate([
                np.array(idxs_valid[i])+sum(sample_nums[:i]) for i in range(len(class_names))
                ]).tolist()
            idxs_train = [i for i in range(sum(sample_nums)) if i not in idxs_valid]
            dataset_train = Subset(torchvision.datasets.ImageFolder(
                f'{benchmarks_dir}/16imagenet_split/train',
                transform=transforms.Compose(t_aug+t_common),
                ), idxs_train)
            dataset_valid = Subset(torchvision.datasets.ImageFolder(
                f'{benchmarks_dir}/16imagenet_split/train',
                transform=transforms.Compose(t_common),
                ), idxs_valid)

            weight = 1/torch.tensor(
                np.array(sample_nums)-valid_num, dtype=torch.float
                )
            return dataset_train, dataset_valid, dataset_test, weight
