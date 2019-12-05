# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 14:04:58 2019

@author: zhe
"""

import numpy as np

# string for a duration
def time_str(t_elapse, progress=1.):
    field_width = int(np.log10(t_elapse/60/progress))+1
    return '{{:{}d}}m{{:05.2f}}s'.format(field_width).format(int(t_elapse//60), t_elapse%60)

# string for progress
def progress_str(i, total, show_percent=False):
    field_width = int(np.log10(total))+1
    disp_str = '{{:{}d}}/{{:{}d}}'.format(field_width, field_width).format(i, total)
    if show_percent:
        disp_str += ', ({:5.1f}%)'.format(100.*i/total)
    return disp_str

# flatten a nested dictionary
def flatten(nested_dict):
    flat_dict = {}
    for key, val in nested_dict.items():
        if isinstance(val, dict) and isinstance(next(iter(val)), str):
            flat_dict.update(dict((
                    key+'::'+subkey,
                    flatten(subval) if isinstance(subval, dict) else subval
                    ) for subkey, subval in val.items()))
        else:
            flat_dict[key] = val
    return flat_dict

# nest a flat dictionary
def nest(flat_dict):
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
