import random, torch
import numpy as np
import inspect
from collections.abc import Callable

from .alias import Tensor, Module, Optimizer, Scheduler

from tqdm.auto import tqdm as auto_tqdm
from tqdm.asyncio import tqdm as asyncio_tqdm
class dummy_tqdm:
    def __init__(self, iterable=None, *args, **kwargs):
        self.iterable = iterable
    def __enter__(self):
        return self
    def __exit__(self, *args):
        ...
    def update(self, *args, **kwargs):
        ...
    def set_description(self, *args, **kwargs):
        ...
    def __iter__(self):
        for obj in self.iterable:
            yield obj

if auto_tqdm==asyncio_tqdm:
    tqdm_kwargs = {'ncols': 100, 'ascii': True}
else:
    tqdm_kwargs = {'ncols': 720}
def tqdm(*args, **kwargs):
    for key, val in tqdm_kwargs.items():
        if key not in kwargs:
            kwargs[key] = val
    if kwargs.get('disable', False):
        return dummy_tqdm(*args, **kwargs)
    if 'smoothing' not in kwargs:
        total = kwargs.get('total', None)
        if total is None:
            try:
                total = len(args[0])
            except:
                total = None
        if total is not None:
            kwargs['smoothing'] = 50/max(total, 150)
    return auto_tqdm(*args, **kwargs)


def time_str(t_elapse: float, progress: float|None = None) -> str:
    r"""Returns a formatted string for a duration.

    Args
    ----
    t_elapse:
        The elapsed time in seconds.
    progress:
        The estimated progress, used for estimating field width.

    """
    t_str = ''
    if t_elapse>50 or progress is not None:
        field_width = int(np.log10(max(t_elapse, 1e-6)/60/(progress or 1)))+1
        t_str += '{{:{}d}}m'.format(field_width).format(int(t_elapse//60))
    t_str += '{:05.2f}s'.format(t_elapse%60)
    return t_str


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
        disp_str += ' ({:6.1%})'.format(i/n)
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


def sgd_optimizer(
    model: Module,
    lr: float,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
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


def tensor2array(tensors):
    r"""Converts pytorch tensors to numpy arrays.

    To save tensors with 'pickle' without warning, tensors or containers with
    them need to be converted. Each tensor will be replaced by a tuple like
    `('_T', vals, dtype)`.

    """
    if isinstance(tensors, Tensor):
        return ('_T', tensors.data.cpu().clone().numpy(), tensors.dtype)
    elif isinstance(tensors, dict):
        return {k: tensor2array(v) for k, v in tensors.items()}
    elif isinstance(tensors, (list, tuple, set)):
        arrays = [tensor2array(vals) for vals in tensors]
        if isinstance(tensors, list):
            return arrays
        elif isinstance(tensors, tuple):
            return tuple(arrays)
        elif isinstance(tensors, set):
            return set(arrays)
    else:
        return tensors

def array2tensor(arrays, device='cpu'):
    r"""Reverts numpy arrays back to tensors.

    To convert the arrays returned by `tensor2array` back to tensors, with given
    tensor device.

    """
    if isinstance(arrays, tuple) and len(arrays)==3 and arrays[0]=='_T':
        return torch.tensor(arrays[1], dtype=arrays[2], device=device)
    elif isinstance(arrays, dict):
        return {k: array2tensor(v) for k, v in arrays.items()}
    elif isinstance(arrays, (list, tuple, set)):
        tensors = [array2tensor(vals) for vals in arrays]
        if isinstance(arrays, list):
            return tensors
        elif isinstance(arrays, tuple):
            return tuple(tensors)
        elif isinstance(arrays, set):
            return set(tensors)
    else:
        return arrays


def create_mlp_layers(
    in_features: int, out_features: int,
    num_features: list[int]|None = None,
    nonlinearity: str = 'ReLU',
    last_linear: bool = True,
) -> torch.nn.ModuleList:
    if num_features is None:
        num_features = [4*int((in_features*out_features)**0.5)]
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


def get_defaults(func: Callable, keys: list[str]|None = None) -> dict:
    r"""Returns default arguments of a callable object."""
    sig = inspect.signature(func)
    defaults = {}
    for param in sig.parameters.values():
        if param.default is not inspect.Parameter.empty:
            if keys is None or param.name in keys:
                defaults[param.name] = param.default
    return defaults


def cls_name(x) -> str:
    r"""Returns class name of an object."""
    return '{}.{}'.format(x.__class__.__module__, x.__class__.__qualname__)
