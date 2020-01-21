# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 14:04:58 2019

@author: zhe
"""

import random, torch
import numpy as np

def time_str(t_elapse, progress=1.):
    r"""Returns a formatted string for a duration.
    
    Args:
        t_elapse (float): elapsed time in seconds.
        progress (float): estimated progress, used for estimating field width.
    
    """
    field_width = int(np.log10(t_elapse/60/progress))+1
    return '{{:{}d}}m{{:05.2f}}s'.format(field_width).format(int(t_elapse//60), t_elapse%60)

def progress_str(i, total, show_percent=False):
    r"""Returns a formatted string for progress.
    
    Args:
        i (int): current iteration index.
        total (int): total iteration number.
        show_percent (bool): whether to show percentage or not.
    
    """
    field_width = int(np.log10(total))+1
    disp_str = '{{:{}d}}/{{:{}d}}'.format(field_width, field_width).format(i, total)
    if show_percent:
        disp_str += ', ({:5.1f}%)'.format(100.*i/total)
    return disp_str

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
    
    A nested dictionary like `{'A': {'B', val}}` will be converted to `{'A::B', val}`.
    
    Args:
        nested_dict (dict): a nested dictionary possibly contains dictionaries as values.
    
    Returns:
        flat_dict (dict): a flat dictionary with keys containing '::' for hierarchy.
    
    """
    flat_dict = {}
    for key, val in nested_dict.items():
        if isinstance(val, dict) and val and isinstance(next(iter(val)), str):
            flat_dict.update(dict((
                    key+'::'+subkey,
                    flatten(subval) if isinstance(subval, dict) else subval
                    ) for subkey, subval in val.items()))
        else:
            flat_dict[key] = val
    return flat_dict

def nest(flat_dict):
    r"""Nests a flat dictionary.
    
    A flat dictionary like `{'A::B', val}` will be converted to `{'A': {'B', val}}`.
    
    Args:
        flat_dict (dict): a flat dictionary with keys containing '::' for hierarchy.
    
    Returns:
        nested_dict (dict): a nested dictionary possibly contains dictionaries as values.
    
    """
    keys = set([k.split('::')[0] if isinstance(k, str) and '::' in k else k for k in flat_dict])
    nested_dict = {}
    for key in keys:
        if key in flat_dict:
            nested_dict[key] = flat_dict[key]
        else:
            subdict = dict((key_full.replace(key+'::', '', 1), val) for key_full, val in flat_dict.items() \
                           if key_full.startswith(key))
            nested_dict[key] = nest(subdict)
    return nested_dict
