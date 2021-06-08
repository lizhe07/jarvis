# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 21:59:48 2020

@author: Zhe
"""

import os, random, pickle, torch, torchvision
from importlib import resources
import numpy as np
from torch.utils.data import Subset, DataLoader
from torchvision import transforms

from .models import lenet, resnet

MODELS = {
    'LeNet': lenet.lenet,
    'ResNet18': resnet.resnet18,
    'ResNet34': resnet.resnet34,
    'ResNet50': resnet.resnet50,
    'ResNet101': resnet.resnet101,
    'ResNet152': resnet.resnet152,
    }

DEFAULT_IMAGE_AUG = lambda size: [
    transforms.Pad(4, padding_mode='reflect'),
    transforms.RandomRotation(2),
    transforms.RandomCrop(size),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(.25, .25, .25),
    ]
DEFAULT_DIGIT_AUG = lambda size: [
    transforms.Pad(4, padding_mode='reflect'),
    transforms.RandomRotation(2),
    transforms.RandomCrop(size),
    ]

IMAGENET_TRAIN = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(
        brightness=0.1, contrast=0.1, saturation=0.1,
        ),
    transforms.ToTensor(),
    ])
IMAGENET_TEST = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    ])


def imagenet_dataset(datasets_dir, train=False, transform=None):
    r"""Returns dataset for ImageNet dataset.

    Args
    ----
    datasets_dir: str
        The directory of the datasets, must includes 'ILSVRC2012'.
    train: bool
        Whether returns training set or testing set.
    transform: callable
        The input transformation that takes an PIL image as input.

    Returns
    -------
    dataset:
        An ImageFolder dataset.

    """
    if train:
        dataset = torchvision.datasets.ImageFolder(
            os.path.join(datasets_dir, 'ILSVRC2012', 'train'), transform,
            )
    else:
        dataset = torchvision.datasets.ImageFolder(
            os.path.join(datasets_dir, 'ILSVRC2012', 'val'), transform,
            )
    return dataset

# (dataset, t_aug, in_channels, class_num, sample_num) for different datasets
DATASETS_META = {
    'MNIST': (torchvision.datasets.MNIST, DEFAULT_DIGIT_AUG(28), 1, 10, 60000),
    'FashionMNIST': (torchvision.datasets.FashionMNIST, DEFAULT_IMAGE_AUG(28), 1, 10, 60000),
    'CIFAR10': (torchvision.datasets.CIFAR10, DEFAULT_IMAGE_AUG(32), 3, 10, 50000),
    'CIFAR100': (torchvision.datasets.CIFAR100, DEFAULT_IMAGE_AUG(32), 3, 100, 50000),
    'ImageNet': (imagenet_dataset, None, 3, 1000, 14197122),
    }


def prepare_datasets(task, datasets_dir, split_ratio=None, t_train=None, t_test=None):
    r"""Prepares vision datasets.

    Args
    ----
    task: str
        The name of the datasets, e.g. ``'CIFAR10'``.
    datasets_dir: str
        The directory of the datasets. Automatic download is disabled.
    split_ratio: float
        The ratio of training set, usually `0.9` or `0.8`. When `split_ratio`
        is ``None``, only testing set will be returned.
    augment: bool
        Whether to apply data augmentation on training set.
    grayscale: bool
        Whether to use grayscale images.

    Returns
    -------
    dataset_train, dataset_valid, dataset_test: Dataset
        The training/validation/testing datasets for training models. Each
        dataset item is a tuple `(image, label)`. `image` is a float tensor of
        shape `(3, H, W)` or `(1, H, W)` in [0, 1], and `label` is an integer.

    Examples
    --------
    >>> dataset_test = prepare_datasets(task, datasets_dir)

    >>> dataset_train, dataset_valid, dataset_test = prepare_datasets(
            task, datasets_dir, split_ratio
            )

    """
    if t_test is None: # load default testing transform
        if task=='ImageNet':
            t_test = IMAGENET_TEST
        else:
            t_test = transforms.ToTensor()
    dataset, augs, *_, sample_num = DATASETS_META[task]

    dataset_test = dataset(datasets_dir, train=False, transform=t_test)
    if task=='ImageNet':
        # load a random fixed shuffle of testing images
        idxs = pickle.loads(resources.read_binary('jarvis.resources', 'imagenet_test_idxs'))
        dataset_test.samples = [dataset_test.samples[idx] for idx in idxs]
        dataset_test.targets = [s[1] for s in dataset_test.samples]
        dataset_test.imgs = dataset_test.samples
        # load class names from https://github.com/anishathalye/imagenet-simple-labels
        dataset_test.class_names = pickle.loads(
            resources.read_binary('jarvis.resources', 'imagenet_class_names')
            )
    else:
        dataset_test.class_names = dataset_test.classes
    if split_ratio is None:
        return dataset_test

    assert split_ratio>0 and split_ratio<1
    if t_train is None:
        if task=='ImageNet':
            t_train = IMAGENET_TRAIN
        else:
            t_train = transforms.Compose(augs+[transforms.ToTensor()])
    idxs_train = np.array(random.sample(range(sample_num), int(sample_num*split_ratio)))
    idxs_valid = np.setdiff1d(np.arange(sample_num), idxs_train, assume_unique=True)
    dataset_train = Subset(dataset(datasets_dir, train=True, transform=t_train), idxs_train)
    dataset_valid = Subset(dataset(datasets_dir, train=True, transform=t_test), idxs_valid)
    return dataset_train, dataset_valid, dataset_test


def prepare_model(task, arch, in_channels=None, **kwargs):
    r"""Prepares model.

    Args
    ----
    task: str
        The name of the dataset, e.g. ``'CIFAR10'``.
    arch: str or callable
        The name of model architecture, e.g. ``ResNet18``, or a function that
        returns a model object.
    in_channels: int
        The number of input channels. Default values are used if `in_channels`
        is ``None``.

    Returns
    -------
    model: nn.Module
        A pytorch model that can be called by `logits = model(images)`.

    """
    _, _, _in_channels, class_num, _ = DATASETS_META[task]
    if in_channels is None:
        in_channels = _in_channels
    if isinstance(arch, str):
        arch = MODELS[arch]
    model = arch(class_num=class_num, in_channels=in_channels, **kwargs)
    return model


def evaluate(model, dataset, batch_size=100, device='cuda', worker_num=2):
    r"""Evaluates the task performance of the model.

    Args
    ----
    model: nn.Module
        The model to be evaluated.
    dataset: Dataset
        The dataset to evaluate the model on.
    batch_size: int
        The batch size of the data loader.
    device: str
        The device used for evaluation.
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

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=worker_num)
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
