# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 14:04:58 2019

@author: zhe
"""

import argparse, random, torch
import numpy as np


def time_str(t_elapse, progress=1.):
    r"""Returns a formatted string for a duration.

    Args
    ----
    t_elapse: float
        The elapsed time in seconds.
    progress: float
        The estimated progress, used for estimating field width.

    """
    field_width = int(np.log10(max(t_elapse, 1e-6)/60/progress))+1
    return '{{:{}d}}m{{:05.2f}}s'.format(field_width).format(int(t_elapse//60), t_elapse%60)


def progress_str(i, total, show_percent=False):
    r"""Returns a formatted string for progress.

    Args
    ----
    i： int
        The current iteration index.
    total: int
        The total iteration number.
    show_percent: bool
        Whether to show percentage or not.

    """
    field_width = int(np.log10(total))+1
    disp_str = '{{:{}d}}/{{:{}d}}'.format(field_width, field_width).format(i, total)
    if show_percent:
        disp_str += ', ({:6.1%})'.format(i/total)
    return disp_str


def get_seed(seed=None, max_seed=1000):
    r"""Returns a random seed.

    """
    if seed is None:
        return random.randrange(max_seed)
    else:
        return seed%max_seed


def set_seed(seed):
    r"""Sets random seed for random, numpy and torch.

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def flatten(nested_dict):
    r"""Flattens a nested dictionary.

    A nested dictionary like `{'A': {'B', val}}` will be converted to
    `{'A::B', val}`.

    Args
    ----
    nested_dict: dict
        A nested dictionary possibly contains dictionaries as values.

    Returns
    -------
    flat_dict: dict
        A flat dictionary with keys containing ``'::'`` for hierarchy.

    """
    flat_dict = {}
    for key, val in nested_dict.items():
        if isinstance(val, dict) and val and isinstance(next(iter(val)), str):
            flat_dict.update(dict(
                (key+'::'+subkey, subval) for subkey, subval in flatten(val).items()
                ))
        else:
            flat_dict[key] = val
    return flat_dict


def nest(flat_dict):
    r"""Nests a flat dictionary.

    A flat dictionary like `{'A::B', val}` will be converted to
    `{'A': {'B', val}}`.

    Args
    ----
    flat_dict: dict
        A flat dictionary with keys containing ``'::'`` for hierarchy.

    Returns
    -------
    nested_dict: dict
        A nested dictionary possibly contains dictionaries as values.

    """
    keys = set([k.split('::')[0] if isinstance(k, str) and '::' in k else k for k in flat_dict])
    nested_dict = {}
    for key in keys:
        if key in flat_dict:
            nested_dict[key] = flat_dict[key]
        else:
            subdict = dict(
                (key_full.replace(key+'::', '', 1), val) for key_full, val in flat_dict.items()
                if key_full.startswith(key)
                )
            nested_dict[key] = nest(subdict)
    return nested_dict


def match_cond(config, cond=None):
    r"""Checks if a configuration matched condition.

    """
    if cond is None:
        cond = {}
    flat_config, flat_cond = flatten(config), flatten(cond)
    for key in flat_cond:
        if flat_cond[key]!={} and (key not in flat_config or flat_cond[key]!=flat_config[key]):
            return False
    return True


def grouping(configs, nuisances=None):
    r"""Organizes configs into groups.

    Configuration dictionaries are flatten and grouped based on the values that
    is not constant. Nuisance keys such as random seeds will be ignored.

    Args
    ----
    configs: list
        A list of dictionaries with same structure.
    nuisances: set
        The set of nuisance specifications of flat configs. Each element could
        be a flat config key, or the beginning part of one.

    Returns
    -------
    groups: dict
        A dictionary with hashable dictionaries as keys, each corresponding to
        the varying part of `configs`. Each value is a list of the
        corresponding subset of `configs`.

    """
    if nuisances is None:
        nuisances = set()

    # get union of all flat keys
    f_configs = [HashableDict(flatten(c)) for c in configs]
    f_keys = set().union(*[c.keys() for c in f_configs])

    # remove nuisance keys
    n_keys = set()
    for f_key in f_keys:
        if f_key.endswith('seed'):
            n_keys.add(f_key)
        else:
            for nuisance in nuisances:
                if f_key.startswith(nuisance):
                    n_keys.add(f_key)
    f_keys -= n_keys

    # remove keys with unique values
    u_keys = set()
    for f_key in f_keys:
        in_all, vals = True, set()
        for f_config in f_configs:
            if f_key in f_config:
                vals.add(to_hashable(f_config[f_key]))
            else:
                in_all = False
        if in_all and len(vals)==1:
            u_keys.add(f_key)
    f_keys -= u_keys

    # group configs based on changing values
    groups = {}
    for config, f_config in zip(configs, f_configs):
        g_key = HashableDict(nest(dict(
            (f_key, f_config[f_key]) for f_key in f_keys if f_key in f_config
            )))
        if g_key in groups:
            groups[g_key].append(config)
        else:
            groups[g_key] = [config]
    return groups


def to_hashable(val):
    r"""Converts to hashable data type.

    """
    if isinstance(val, list):
        return HashableList(val)
    if isinstance(val, tuple):
        return HashableTuple(val)
    if isinstance(val, set):
        return HashableSet(val)
    if isinstance(val, dict):
        return HashableDict(val)

    try:
        hash(val)
    except:
        raise TypeError('hashable type is not implemented')
    else:
        return val


def _is_custom_hashable(val):
    r"""Returns whether the input is a custom hashable type.

    All custom hashable class implements `native` method.

    """
    return (
        isinstance(val, HashableList)
        or isinstance(val, HashableTuple)
        or isinstance(val, HashableSet)
        or isinstance(val, HashableDict)
        )


class HashableList(list):
    r"""Hashable list class.

    """

    def __init__(self, vals):
        super(HashableList, self).__init__([to_hashable(val) for val in vals])

    def __hash__(self):
        return hash(tuple(self))

    def native(self):
        converted = []
        for val in self:
            converted.append(val.native() if _is_custom_hashable(val) else val)
        return converted


class HashableTuple(tuple):
    r"""Hashable tuple class.

    """

    def __new__(cls, vals):
        return super(HashableTuple, cls).__new__(cls, tuple(to_hashable(val) for val in vals))

    def native(self):
        converted = []
        for val in self:
            converted.append(val.native() if _is_custom_hashable(val) else val)
        return tuple(converted)


class HashableSet(set):
    r"""Hashable set class.

    """

    def __init__(self, vals):
        super(HashableSet, self).__init__([to_hashable(val) for val in vals])

    def __hash__(self):
        return hash(frozenset(self))

    def native(self):
        converted = []
        for val in self:
            converted.append(val.native() if _is_custom_hashable(val) else val)
        return set(converted)


class HashableDict(dict):
    r"""Hashable dictionary class.

    """

    def __init__(self, vals):
        super(HashableDict, self).__init__((key, to_hashable(val)) for key, val in vals.items())

    def __hash__(self):
        return hash(frozenset(self.items()))

    def native(self):
        converted = {}
        for key, val in self.items():
            converted[key] = val.native() if _is_custom_hashable(val) else val
        return converted


def numpy_dict(model_state):
    r"""Converts state dict to numpy arrays.

    """
    return dict((name, param.data.cpu().clone().numpy()) \
                for name, param in model_state.items())


def tensor_dict(model_state):
    r"""Converts state dict to pytorch tensors.

    """
    return dict((name, torch.tensor(param, dtype=torch.float)) \
                for name, param in model_state.items())


def job_parser():
    r"""Returns a base parser for job processing.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec_pth')
    parser.add_argument('--process_num', default=1, type=int,
                        help='number of works to process')
    parser.add_argument('--max_wait', default=1, type=float,
                        help='seconds of wait before each job')
    parser.add_argument('--tolerance', default=float('inf'), type=float,
                        help='hours since start')
    return parser


def sgd_optimizer(model, lr, momentum=0.9, weight_decay=0.):
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


def cyclic_scheduler(optimizer, epoch_num, cycle_num=2, phase_num=3, gamma=0.1):
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
