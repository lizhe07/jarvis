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
