import argparse, random, torch
import numpy as np

Array = np.ndarray


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


def numpy_dict(model_state):
    r"""Converts state dict to numpy arrays."""
    return dict((name, param.data.cpu().clone().numpy()) for name, param in model_state.items())


def tensor_dict(model_state):
    r"""Converts state dict to pytorch tensors."""
    return dict((name, torch.tensor(param, dtype=torch.float)) for name, param in model_state.items())


def job_parser():
    r"""Returns a base parser for job processing."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec-path')
    parser.add_argument('--num-works', default=0, type=int,
                        help='number of works to process')
    parser.add_argument('--max-wait', default=1, type=float,
                        help='seconds of wait before each job')
    parser.add_argument('--patience', default=float('inf'), type=float,
                        help='hours since last modification')
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


def cyclic_scheduler(optimizer, num_epochs, num_cycles=2, num_phases=3, gamma=0.1):
    r"""Returns a simple cyclic scheduler.

    The full training is divided into several cycles, within each learning
    rates are adjusted as in StepLR scheduler. At the beginning of each cycle,
    the learning rate is reset to the initial value.

    Args
    ----
    optimizer: optimizer
        The pytorch optimizer.
    num_epochs: int
        The number of epochs.
    num_cycles: int
        The number of cycles.
    num_phases: int
        The number of phasese within each cycle. Learning rate decays by a
        fixed factor between phases.
    gamma: float
        The decay factor between phases, must be in `(0, 1]`.

    Returns
    -------
    scheduler: scheduler
        The cyclic scheculer.

    """
    cycle_len = -(-num_epochs//num_cycles)
    phase_len = -(-cycle_len//num_phases)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: gamma**((epoch%cycle_len)//phase_len)
        )
    return scheduler
