# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 21:59:48 2020

@author: Zhe
"""

import os, random
import torch, torchvision
import numpy as np
from torch.utils.data import Subset, DataLoader
from torchvision import transforms

from .models import resnet

MODELS = {
    'ResNet18': resnet.resnet18,
    'ResNet34': resnet.resnet34,
    'ResNet50': resnet.resnet50,
    'ResNet101': resnet.resnet101,
    'ResNet152': resnet.resnet152,
    }


def prepare_datasets(task, datasets_dir, valid_num=None, train_tensor=True):
    r"""Prepares vision datasets.

    Args
    ----
    task: str
        The name of the dataset, e.g. ``'CIFAR10'``.
    datasets_dir: str
        The directory of vision benchmarks.
    valid_num: int
        The validation dataset size. For ``'16ImageNet'`` task, it is the
        number of validation images per class. When `valid_num` is ``None``,
        only testing dataset will be returned.
    train_tensor: bool
        Training set returns PIL images instead of tensors if ``True``.

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
    >>> dataset_test = prepare_datasets(task, datasets_dir)

    >>> dataset_train, dataset_valid, dataset_test, weight = \
            prepare_datasets(task, datasets_dir, valid_num)

    """
    if task.startswith('CIFAR'):
        t_test = transforms.ToTensor()
        if task=='CIFAR10':
            dataset_test = torchvision.datasets.CIFAR10(
                datasets_dir, train=False, transform=t_test,
                )
        if task=='CIFAR100':
            dataset_test = torchvision.datasets.CIFAR100(
                datasets_dir, train=False, transform=t_test,
                )

        if valid_num is None:
            return dataset_test
        else:
            t_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                transforms.RandomHorizontalFlip(),
                ]+([transforms.ToTensor()] if train_tensor else []))
            sample_num = 50000
            idxs_valid = np.array(random.sample(range(sample_num), valid_num))
            idxs_train = np.setdiff1d(np.arange(sample_num), idxs_valid, assume_unique=True)
            if task=='CIFAR10':
                dataset_train = Subset(torchvision.datasets.CIFAR10(
                    datasets_dir, train=True, transform=t_train,
                    ), idxs_train)
                dataset_valid = Subset(torchvision.datasets.CIFAR10(
                    datasets_dir, train=True, transform=t_test,
                    ), idxs_valid)
            if task=='CIFAR100':
                dataset_train = Subset(torchvision.datasets.CIFAR100(
                    datasets_dir, train=True, transform=t_train,
                    ), idxs_train)
                dataset_valid = Subset(torchvision.datasets.CIFAR100(
                    datasets_dir, train=True, transform=t_test,
                    ), idxs_valid)

            weight = None
            return dataset_train, dataset_valid, dataset_test, None

    if task=='16ImageNet':
        t_test = transforms.ToTensor()
        dataset_test = torchvision.datasets.ImageFolder(
            f'{datasets_dir}/16imagenet_split/test',
            transform=t_test,
            )

        if valid_num is None:
            return dataset_test
        else:
            t_train = transforms.Compose([
                transforms.RandomCrop(256, padding=32, padding_mode='reflect'),
                transforms.RandomHorizontalFlip(),
                ]+([transforms.ToTensor()] if train_tensor else []))
            class_names = os.listdir(f'{datasets_dir}/16imagenet_split/train')
            sample_nums = [len(os.listdir(f'{datasets_dir}/16imagenet_split/train/{c_name}')) \
                           for c_name in class_names]
            idxs_valid = [random.sample(range(sample_num), valid_num) \
                          for sample_num in sample_nums]
            idxs_valid = np.concatenate([
                np.array(idxs_valid[i])+sum(sample_nums[:i]) for i in range(len(class_names))
                ])
            idxs_train = np.setdiff1d(np.arange(sum(sample_nums)), idxs_valid, assume_unique=True)
            dataset_train = Subset(torchvision.datasets.ImageFolder(
                f'{datasets_dir}/16imagenet_split/train',
                transform=t_train,
                ), idxs_train)
            dataset_valid = Subset(torchvision.datasets.ImageFolder(
                f'{datasets_dir}/16imagenet_split/train',
                transform=t_test,
                ), idxs_valid)

            weight = 1/torch.tensor(
                np.array(sample_nums)-valid_num, dtype=torch.float
                )
            return dataset_train, dataset_valid, dataset_test, weight

    if task=='ImageNet':
        t_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            ])
        dataset_test = torchvision.datasets.ImageFolder(
            f'{datasets_dir}/ILSVRC2012/val', transform=t_test,
            )

        if valid_num is None:
            return dataset_test
        else:
            t_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                ]+([transforms.ToTensor()] if train_tensor else []))
            class_names = [c for c in os.listdir(f'{datasets_dir}/ILSVRC2012/train') if c.startswith('n')]
            sample_nums = [len(os.listdir(f'{datasets_dir}/ILSVRC2012/train/{c_name}')) \
                           for c_name in class_names]
            idxs_valid = [random.sample(range(sample_num), valid_num) \
                          for sample_num in sample_nums]
            idxs_valid = np.concatenate([
                np.array(idxs_valid[i])+sum(sample_nums[:i]) for i in range(len(class_names))
                ])
            idxs_train = np.setdiff1d(np.arange(sum(sample_nums)), idxs_valid, assume_unique=True)
            dataset_train = Subset(torchvision.datasets.ImageFolder(
                f'{datasets_dir}/ILSVRC2012/train', transform=t_train,
                ), idxs_train)
            dataset_valid = Subset(torchvision.datasets.ImageFolder(
                f'{datasets_dir}/ILSVRC2012/train', transform=t_test,
                ), idxs_valid)

            weight = None
            return dataset_train, dataset_valid, dataset_test, None


def prepare_model(task, arch):
    r"""Prepares model.

    Args
    ----
    task: str
        The name of the dataset, e.g. ``'CIFAR10'``.
    arch: str
        The name of model architecture, e.g. ``ResNet18``.

    """
    if task=='CIFAR10':
        class_num = 10
    if task=='CIFAR100':
        class_num = 100
    if task=='16ImageNet':
        class_num = 16
    if task=='ImageNet':
        class_num = 1000

    model = MODELS[arch](class_num=class_num)
    return model


def evaluate(model, dataset, device, batch_size, worker_num):
    r"""Evaluates the task performance of the model.

    Args
    ----
    model: nn.Module
        The model to be evaluated.
    dataset: Dataset
        The dataset to evaluate the model on.
    device: str
        The device used for evaluation.
    batch_size: int
        The batch size of the data loader.
    worker_num: int
        The number of workers of the data loader.

    Returns
    -------
    loss: float
        The cross-entropy loss averaged over the dataset.
    acc: float
        The classification accuracy averaged over the dataset.

    """
    if device=='cuda' and not torch.cuda.is_available():
        device = 'cpu'
    model.eval().to(device)
    criterion = torch.nn.CrossEntropyLoss(reduction='sum').to(device)

    loader = DataLoader(dataset, batch_size=batch_size, num_workers=worker_num)
    loss, count = 0., 0.
    for images, labels in loader:
        with torch.no_grad():
            logits = model(images.to(device))
            loss += criterion(logits, labels.to(device)).item()
            _, predicts = logits.max(dim=1)
            count += (predicts.cpu()==labels).to(torch.float).sum().item()
    loss = loss/len(dataset)
    acc = count/len(dataset)
    return loss, acc
