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
    
    A nested dictionary like `{'A': {'B', val}}` will be converted to `{'A::B', val}`.
    
    Args:
        nested_dict (dict): a nested dictionary possibly contains dictionaries as values.
    
    Returns:
        flat_dict (dict): a flat dictionary with keys containing '::' for hierarchy.
    
    """
    flat_dict = {}
    for key, val in nested_dict.items():
        if isinstance(val, dict) and val and isinstance(next(iter(val)), str):
            flat_dict.update(dict((key+'::'+subkey, subval) for subkey, subval in flatten(val).items()))
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

def match_cond(config, cond):
    r"""Checks if a configuration matched condition.
    
    """
    flat_config, flat_cond = flatten(config), flatten(cond)
    for key in flat_cond:
        if key not in flat_config or flat_cond[key]!=flat_config[key]:
            return False
    return True

def grouping(configs, nuisance=None):
    r"""Organizes configs into groups.
    
    Configuration dictionaries are flatten and grouped based on the values that is not
    constant. Nuisance keys such as random seeds will be ignored.
    
    Args:
        configs (list): list of dictionaries with same structure.
        nuisance (set): set of keys of flat configs.
    
    Returns:
        groups (dict): a dictionary with hashable dictionaries as keys, each corresponding
            to the varying part of `configs`. Each value is a list of the corresponding
            subset of `configs`.
    
    """
    if nuisance is None:
        nuisance = set()
    
    flat_configs = [HashableDict(**flatten(c)) for c in configs]
    flat_keys = None
    for c in flat_configs:
        if flat_keys is None:
            flat_keys = c.keys()
        else:
            assert flat_keys==c.keys()
    assert nuisance.issubset(flat_keys)
    
    for key in flat_keys:
        if key.endswith('seed'):
            nuisance.add(key)
    val_nums = {}
    for key in flat_keys:
        if key not in nuisance:
            vals = set() # values may be lists (unhashable), use list instead of set here
            for c in flat_configs:
                vals.add(c[key])
            val_nums[key] = len(vals)
    varying_keys = [key for key in val_nums if val_nums[key]>1]
    varying_configs = list(set([HashableDict(**nest(dict((key, flat_config[key]) for key in varying_keys)))\
                                for flat_config in flat_configs]))
    groups = dict((v, [c for c in configs if match_cond(c, v)]) for v in varying_configs)
    return groups

class HashableList(list):
    def __init__(self, vals):
        converted = []
        for val in vals:
            if isinstance(val, list):
                converted.append(HashableList(val))
            elif isinstance(val, dict):
                converted.append(HashableDict(**val))
            elif isinstance(val, set):
                converted.append(frozenset(val))
            else:
                converted.append(val)
        super(HashableList, self).__init__(converted)
    
    def __hash__(self):
        return hash(tuple(self))
    
    def __eq__(self, other):
        return self.__hash__()==other.__hash__()
    
    def native(self):
        r"""Returns the native version of the list.
        
        """
        converted = []
        for val in self:
            if isinstance(val, HashableList) or isinstance(val, HashableDict):
                converted.append(val.native())
            elif isinstance(val, frozenset):
                converted.append(set(val))
            else:
                converted.append(val)
        return converted

class HashableDict(dict):
    def __init__(self, **kwargs):
        converted = {}
        for key, val in kwargs.items():
            if isinstance(val, list):
                converted[key] = HashableList(val)
            elif isinstance(val, dict):
                converted[key] = HashableDict(**val)
            elif isinstance(val, set):
                converted[key] = frozenset(val)
            else:
                converted[key] = val
        super(HashableDict, self).__init__(**converted)
    
    def __hash__(self):
        return hash(frozenset(self.items()))
    
    def __eq__(self, other):
        return self.__hash__()==other.__hash__()
    
    def native(self):
        r"""Returns the native version of the dictionary.
        
        """
        converted = {}
        for key, val in self.items():
            if isinstance(val, HashableList) or isinstance(val, HashableDict):
                converted[key] = val.native()
            elif isinstance(val, frozenset):
                converted[key] = set(val)
            else:
                converted[key] = val
        return converted
