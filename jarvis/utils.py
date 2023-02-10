import random, torch
import numpy as np
from typing import Optional
from torch.utils.data import (
    TensorDataset, DataLoader, WeightedRandomSampler,
)

from .alias import Tensor, Module, Optimizer, Scheduler


def time_str(t_elapse: float, progress: float = 1.) -> str:
    r"""Returns a formatted string for a duration.

    Args
    ----
    t_elapse:
        The elapsed time in seconds.
    progress:
        The estimated progress, used for estimating field width.

    """
    field_width = int(np.log10(max(t_elapse, 1e-6)/60/progress))+1
    return '{{:{}d}}m{{:05.2f}}s'.format(field_width).format(int(t_elapse//60), t_elapse%60)


def progress_str(i: int, n: int, show_percent: bool = False) -> str:
    r"""Returns a formatted string for progress.

    Args
    ----
    i：
        The current iteration index.
    n:
        The total iteration number.
    show_percent: bool
        Whether to show percentage or not.

    """
    field_width = int(np.log10(n))+1
    disp_str = '{{:{}d}}/{{:{}d}}'.format(field_width, field_width).format(i, n)
    if show_percent:
        disp_str += ', ({:6.1%})'.format(i/n)
    return disp_str


def get_seed(seed=None, max_seed=1000):
    r"""Returns a random seed."""
    if seed is None:
        return random.randrange(max_seed)
    else:
        return seed%max_seed


def set_seed(seed, strict=False):
    r"""Sets random seed for random, numpy and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if strict:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def flatten(nested_dict: dict) -> dict:
    r"""Flattens a nested dictionary.

    A nested dictionary like `{'A': {'B', val}}` will be converted to
    `{('B', '@', 'A'), val}`.

    Args
    ----
    nested_dict: dict
        A nested dictionary possibly contains dictionaries as values.

    Returns
    -------
    flat_dict: dict
        A flat dictionary with tuple keys for hierarchy.

    """
    flat_dict = {}
    for key, val in nested_dict.items():
        if isinstance(val, dict) and len(val)>0:
            for subkey, subval in flatten(val).items():
                flat_dict[(subkey, '@', key)] = subval
        else:
            flat_dict[key] = val
    return flat_dict


def nest(flat_dict: dict) -> dict:
    r"""Nests a flat dictionary.

    A flat dictionary like `{('B', '@', 'A'), val}` will be converted to
    `{'A': {'B', val}}`.

    Args
    ----
    flat_dict: dict
        A flat dictionary with tuple keys for hierarchy.

    Returns
    -------
    nested_dict: dict
        A nested dictionary possibly contains dictionaries as values.

    """
    nested_dict = {}
    for key, val in flat_dict.items():
        if isinstance(key, tuple) and len(key)==3 and key[1]=='@':
            subkey, _, parkey = key
            if parkey in nested_dict:
                assert isinstance(nested_dict[parkey], dict)
            else:
                nested_dict[parkey] = {}
            nested_dict[parkey][subkey] = val
        else:
            if key in nested_dict:
                assert isinstance(nested_dict[key], dict) and isinstance(val, dict)
                nested_dict[key].update(val)
            else:
                nested_dict[key] = val
    for key, val in nested_dict.items():
        if isinstance(val, dict):
            nested_dict[key] = nest(val)
    return nested_dict


def numpy_dict(state: dict) -> dict:
    r"""Returns a state dictionary with tensors replaced by numpy arrays.

    Each tensor is converted to a tuple containing the numpy array and tensor
    dtype.

    Args
    ----
    state:
        State dictionary potentially containing tensors, returned by torch
        module, optimizer or scheduler.

    Returns
    -------
    A dictionary with same structure, with tensors converted to numpy arrays.

    """
    f_state = flatten(state)
    for key, val in f_state.items():
        if isinstance(val, Tensor):
            f_state[key] = (val.data.cpu().clone().numpy(), val.dtype)
    return nest(f_state)


def tensor_dict(state: dict, device='cpu') -> dict:
    r"""Returns a state dictionary with numpy arrays replaced by tensors.

    This is the inverted function of `numpy_dict`.

    Args
    ----
    state:
        The state dictionary converted by `numpy_dict`.
    device:
        Tensor device of the converted state dictionary.

    """
    f_state = flatten(state)
    for key, val in f_state.items():
        if isinstance(val, tuple) and len(val)==2 and isinstance(val[0], np.ndarray) and isinstance(val[1], torch.dtype):
            f_state[key] = torch.tensor(val[0], dtype=val[1], device=device)
    return nest(f_state)


def sgd_optimizer(
    model: Module,
    lr: float,
    momentum: float = 0.9,
    weight_decay: float = 0.,
) -> Optimizer:
    r"""Returns a SGD optimizer.

    Only parameters whose name ends with 'weight' will be trained with weight
    decay.

    Args
    ----
    model:
        The pytorch model.
    lr:
        Learning rate for all parameters.
    momentum:
        The momentum parameter for SGD.
    weight_decay:
        The weight decay parameter for layer weights but not biases.

    Returns
    -------
    optimizer:
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


def cyclic_scheduler(
    optimizer: Optimizer,
    phase_len: int = 4,
    num_phases: int = 3,
    gamma: float = 0.3,
) -> Scheduler:
    r"""Returns a simple cyclic scheduler.

    Learning rate is scheduled to follow cycles, each of which contains a fixed
    number of phases. At the beginning of each cycle the learning rate is reset
    to the initial value.

    Args
    ----
    optimizer:
        The pytorch optimizer.
    phase_len:
        The length of each phase, during which the learning rate is fixed.
    num_phases:
        The number of phasese within each cycle. Learning rate decays by a
        fixed factor between phases.
    gamma:
        The decay factor between phases, must be in `(0, 1]`.

    Returns
    -------
    scheduler:
        The cyclic scheculer.

    """
    cycle_len = phase_len*num_phases
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: gamma**((epoch%cycle_len)//phase_len),
    )
    return scheduler


def create_mlp_layers(
    in_features: int, out_features: int,
    num_features: Optional[list[int]] = None,
    nonlinearity: str = 'ReLU',
    last_linear: bool = True,
) -> torch.nn.ModuleList:
    nonlinearity = getattr(torch.nn, nonlinearity)
    layers = torch.nn.ModuleList()
    for l_idx in range(len(num_features)+1):
        if l_idx==0:
            _in_features = in_features
        else:
            _in_features = num_features[l_idx-1]
        if l_idx==len(num_features):
            _out_features = out_features
        else:
            _out_features = num_features[l_idx]
        layers.append(torch.nn.Sequential(
            torch.nn.Linear(_in_features, _out_features),
            nonlinearity() if l_idx<len(num_features) or not last_linear else torch.nn.Identity(),
        ))
    return layers


def train_one_epoch(
    dset: TensorDataset, criterion, optimizer: Optimizer,
    batch_size: int, weights = None, device: str = 'cuda',
):
    if weights is None:
        loader = DataLoader(dset, batch_size=batch_size, shuffle=True, drop_last=True)
    else:
        sampler = WeightedRandomSampler(weights, len(dset))
        loader = DataLoader(dset, batch_size=batch_size, sampler=sampler, drop_last=True)

    gamma, loss_train = 0.5**(10/len(loader)), None
    for xs in loader:
        xs = (t.to(device) for t in xs)
        loss: Tensor = criterion(*xs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_train = loss.item() if loss_train is None else gamma*loss_train+(1-gamma)*loss.item()
    return loss_train
