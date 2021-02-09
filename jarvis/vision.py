# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 21:59:48 2020

@author: Zhe
"""

import random, torch, torchvision
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

# (dataset, t_aug, class_num, sample_num) for different datasets
DATASETS_META = {
    'MNIST': (torchvision.datasets.MNIST, DEFAULT_DIGIT_AUG(28), 1, 10, 60000),
    'CIFAR10': (torchvision.datasets.CIFAR10, DEFAULT_IMAGE_AUG(32), 3, 10, 50000),
    'CIFAR100': (torchvision.datasets.CIFAR100, DEFAULT_IMAGE_AUG(32), 3, 100, 50000),
    }


def prepare_datasets(task, datasets_dir, split_ratio=None,
                     augment=True, grayscale=False):
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
    if grayscale:
        t_test = [transforms.Grayscale(), transforms.ToTensor()]
    else:
        t_test = [transforms.ToTensor()]
    dataset, t_aug, *_, sample_num = DATASETS_META[task]
    if augment:
        t_train = t_aug+t_test
    else:
        t_train = t_test
    t_test = transforms.Compose(t_test)
    t_train = transforms.Compose(t_train)

    dataset_test = dataset(datasets_dir, train=False, transform=t_test)
    if split_ratio is None:
        return dataset_test

    assert split_ratio>0 and split_ratio<1
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
        model = MODELS[arch](class_num=class_num, in_channels=in_channels, **kwargs)
    else:
        model = arch(class_num=class_num, in_channels=in_channels, **kwargs)
    return model


def sgd_optimizer(model, lr, momentum, weight_decay):
    r"""Returns a SGD optimizer.

    Only parameters whose name ends with ``'weight'`` will be trained with
    weight decay.

    Args
    ----
    model: nn.Module
        The pytorch model.
    lr: float
        Learning rate for all parameters.
    momentum: float
        The momentum parameter for SGD.
    weight_decay: float
        The weight decay parameter for layer weights but not biases.

    Returns
    -------
    optimizer: optimizer
        The SGD optimizer.

    """
    params = []
    params.append({
        'params': [param for name, param in model.named_parameters() if name.endswith('weight')],
        'weight_decay': weight_decay,
        })
    params.append({
        'params': [param for name, param in model.named_parameters() if not name.endswith('weight')],
        })
    optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum)
    return optimizer


def cyclic_scheduler(optimizer, epoch_num, cycle_num, phase_num, gamma):
    r"""Returns a simple cyclic scheduler.

    The full training is divided into several cycles, within each learning
    rates are adjusted as in StepLR scheduler. At the beginning of each cycle,
    the learning rate is reset to the initial value.

    Args
    ----
    optimizer: optimizer
        The pytorch optimizer.
    epoch_num: int
        The number of epochs.
    cycle_num: int
        The number of cycles.
    phase_num: int
        The number of phasese within each cycle. Learning rate decays by a
        fixed factor between phases.
    gamma: float
        The decay factor between phases, must be in `(0, 1]`.

    Returns
    -------
    scheduler: scheduler
        The cyclic scheculer.

    """
    cycle_len = -(-epoch_num//cycle_num)
    phase_len = -(-cycle_len//phase_num)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: gamma**((epoch%cycle_len)//phase_len)
        )
    return scheduler


def evaluate(model, dataset, batch_size, device, worker_num):
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
