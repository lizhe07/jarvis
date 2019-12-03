# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 22:57:30 2019

@author: Zhe
"""

import itertools, random, time
import numpy as np
from .utils import progress_str, time_str

class Job:
    r"""Data structure for managing works.
    
    A Job object creates a list of task IDs from the specified search space, and is
    associated with the corresponding configuration, status and checkpoint archives.
    
    Args:
        search_spec (dict): specification of search space. Each combination of specified
            values corresponds to a list of argument strings.
        configs (Archive): archive of work configurations.
        stats (Archive): archive of work statuses. Each record within must have a boolean
            'completed' and two float 'tic' and 'toc' as dictionary values.
        ckpts (Archive): archive of work chekpoints.
    
    """
    def __init__(self, search_spec, configs, stats, ckpts):
        for key, val in search_spec.items():
            assert isinstance(val, list), '{} in search_spec should be a list'.format(key)
        self.search_spec = search_spec
        self.configs, self.stats, self.ckpts = configs, stats, ckpts
        
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
    
    def init(self, get_config, custom_converter=None, disp_num=20, **kwargs):
        r"""Initiates the list of task IDs from search specification.
        
        Args:
            get_config (function): a function that takes a list of argument strings as
                the first input with optional keyword arguments. This function returns a
                dictionary of configuration.
            custom_converter (dict): a dictionary of custom converters. The keys of this
                dictionary are a subset of search_spec. The values of it are functions
                that convert search_spec value to a list of argument strings.
            disp_num (int): number of displays during job initialization.
            kwargs: additional keyword arguments for get_config.
        
        """
        assert not self.work_ids, 'work set already exists'
        
        if custom_converter is None:
            custom_converter = {}
        else:
            assert isinstance(custom_converter, dict)
        
        arg_keys = list(self.search_spec.keys())
        arg_lists = [self.search_spec[key] for key in arg_keys]
        total_num = np.prod([len(l) for l in arg_lists])
        search_space = itertools.product(*arg_lists)
        
        for idx, arg_vals in enumerate(search_space, 1):
            arg_strs = []
            for arg_key, arg_val in zip(arg_keys, arg_vals):
                if arg_key in custom_converter:
                    # use custom_converter if provided
                    arg_strs += custom_converter[arg_key](arg_val)
                elif isinstance(arg_val, bool):
                    # all boolean arguments are assumed to use 'store_true'
                    if arg_val:
                        arg_strs += ['--'+arg_key]
                elif isinstance(arg_val, list):
                    # a list of values corresponds to arguments with nargs='+'
                    if arg_val:
                        arg_strs += ['--'+arg_key]+[str(v) for v in arg_val]
                elif arg_val is not None:
                    arg_strs += ['--'+arg_key, str(arg_val)]
            work_config = get_config(arg_strs, **kwargs)
            w_id = self.configs.fetch_id(work_config)
            if w_id is None:
                w_id = self.configs.add(work_config)
            self.work_ids.add(w_id)
            
            if idx%(-(-total_num//disp_num))==0 or idx==total_num:
                print('{}, ({:5.1f}%)'.format(progress_str(idx, total_num), 100.*idx/total_num))
    
    def process(self, work_func, process_num=0, max_wait=60., tolerance=float('inf'), **kwargs):
        r"""Processes works in random order.
        
        Args:
            work_func (function): the main function to execute. It receives work
                configuration as keyword arguments.
            process_num (int): number of works to process. When process_num=0, the method
                returns when no work is pending.
            max_wait (float): maximum wait time before start. The random wait is designed
                to avoid conflict when deployed to servers.
            tolerance (float): maximum allowed running time. Any work started earlier than
                the threshold will be restarted.
            kwargs: additional keyword arguments for work_func.
        
        """
        def to_run(w_id, tolerance):
            if not self.stats.has_id(w_id):
                return True
            stat = self.stats.fetch_record(w_id)
            if not stat['completed'] and (time.time()-stat['tic'])/3600>tolerance:
                return True
            else:
                return False
        
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
                print('all works completed or running')
                break
            
            work_config = self.configs.fetch_record(w_id)
            work_func(**work_config, **kwargs)
            count += 1
    
    def overview(self):
        r"""Prints overview of job progress.
        
        """
        durations = []
        r_files = self.stats._r_files()
        for r_file in r_files:
            records = self.stats._safe_read(r_file)
            for w_id in records:
                if records[w_id]['completed'] and (w_id in self.work_ids):
                    durations.append(records[w_id]['toc']-records[w_id]['tic'])
        if durations:
            print('{} works completed, average duration is {}'.format(
                    progress_str(len(durations), len(self.work_ids)),
                    time_str(np.mean(durations))
                    ))
        else:
            print('0 work completed')
