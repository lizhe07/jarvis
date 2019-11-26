# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 22:57:30 2019

@author: Zhe
"""

import itertools, random, time
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
                 custom_converter=None):
        self.search_spec = search_spec
        self.get_config, self.work_func = get_config, work_func
        self.configs, self.stats, self.ckpts = configs, stats, ckpts
        if custom_converter is None:
            self.custom_converter = {}
        else:
            assert isinstance(custom_converter, dict)
            self.custom_converter = custom_converter
        
        self.work_ids = set()
    
    def __repr__(self):
        repr_str = 'Job object defined over'
        for key in self.search_spec:
            repr_str += '\n'+key+': '+str(self.search_spec[key])
        repr_str += '\nconfigs: '+str(self.configs)
        repr_str += '\nstats: '+str(self.stats)
        repr_str += '\nckpts: '+str(self.ckpts)
        return repr_str
    
    def __str__(self):
        return 'Job object defined over {} search specs'.format(len(self.search_spec))
    
    def init(self, disp_num=20):
        r"""Initiates the list of task IDs from search specification.
        
        """
        assert not self.work_ids, 'work set already exists'
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
            self.work_ids.add(w_id)
            
            if idx%(-(-total_num//disp_num))==0 or idx==total_num:
                print('{}, ({:5.1f}%)'.format(progress_str(idx, total_num), 100.*idx/total_num))
    
    def process(self, process_num=0, max_wait=60., tolerance=float('inf'), **kwargs):
        r"""Processes works in random order.
        
        """
        def to_run(w_id, tolerance):
            if not self.stats.has_id(w_id):
                return True
            stat = self.stats.fetch_record(w_id)
            if not stat['finished'] and time.time()-stat['tic']>tolerance:
                return True
        
        random_wait = random.random()*max_wait
        print('random wait {:.1f}s'.format(random_wait))
        time.sleep(random_wait)
        
        work_ids = list(self.work_ids)
        random.shuffle(work_ids)
        work_iterator = (w_id for w_id in work_ids if to_run(w_id, tolerance))
        
        count = 0
        while process_num==0 or (process_num>0 and count<process_num):
            w_id = next(work_iterator, None)
            if w_id is None:
                print('all trainings finished or running')
                break
            
            work_config = self.configs.fetch_record(w_id)
            self.work_func(**work_config, **kwargs)
            count += 1
