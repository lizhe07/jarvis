# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 22:57:30 2019

@author: Zhe
"""

import itertools
import numpy as np
from .utils import progress_str

class Job:
    r"""Data structure for managing works.
    
    A Job object creates a list of task IDs from the specified search space,
    and associates with the corresponding configuration, status and checkpoint
    archives. When being processed, a work is selected and executed by the
    function provided.
    
    """
    def __init__(self, search_spec, get_config, work_func, configs, stats, ckpts,
                 custom_converter=None, disp_num=20):
        self.search_spec = search_spec
        self.get_config, self.work_func = get_config, work_func
        self.configs, self.stats, self.ckpts = configs, stats, ckpts
        if custom_converter is None:
            self.custom_converter = {}
        else:
            assert isinstance(custom_converter, dict)
            self.custom_converter = custom_converter
        self.disp_num = disp_num
        
        self.works = set()
    
    def create(self, custom_convert=None):
        r"""Creates the list of task IDs from search specification.
        
        """
        arg_keys = list(self.search_spec.keys())
        arg_lists = [self.search_spec[key] for key in arg_keys]
        total_num = np.prod([len(l) for l in arg_lists])
        search_space = itertools.product(*arg_lists)
        
        for idx, arg_vals in enumerate(search_space, 1):
            arg_strs = []
            for arg_key, arg_val in zip(arg_keys, arg_vals):
                if arg_key in self.custom_converter:
                    arg_strs += self.custom_converter[arg_key](arg_val)
                elif isinstance(arg_val, bool):
                    if arg_val:
                        arg_strs += ['--'+arg_key]
                elif isinstance(arg_val, list):
                    if arg_val:
                        arg_strs += ['--'+arg_key]+[str(v) for v in arg_val]
                else:
                    arg_strs += ['--'+arg_key, str(arg_val)]
            work_config = self.get_config(arg_strs)
            w_id = self.configs.fetch_id(work_config)
            if w_id is None:
                w_id = self.configs.add(work_config)
            self.works.add(w_id)
            
            if idx%(-(-total_num//self.disp_num))==0 or idx==total_num:
                print('{}, ({:5.1f}%)'.format(progress_str(idx, total_num), 100.*idx/total_num))
