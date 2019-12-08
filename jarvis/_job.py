# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 22:57:30 2019

@author: Zhe
"""

import itertools, random, time
import numpy as np
from .utils import progress_str, time_str, flatten, nest

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
            kwargs (dict): additional keyword arguments for get_config.
        
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
            kwargs (dict): additional keyword arguments for work_func.
        
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

def grouping(work_ids, configs, stats, cond_dict=None, get_score=None, min_group=1,
             return_nested=False, **kwargs):
    r"""Groups completed works and sorts the resulting groups.
    
    All completed works in a given set with the given conditioned config values are
    fetched. The variable part except random seeds are idendified, and all works are
    grouped with respect to each unique configuration. Groups are sorted according to the
    mean score of each member work, in an ascending order.
    
    Args:
        work_ids (set): the set of work IDs to organize.
        configs (Archive): archive of work configurations.
        stats (Archive): archive of work statuses.
        cond_dict (dict): the config values to be conditioned on, should be part of a
            valid config.
        get_score (function): a function that takes a stat as input and returns a scalar
            as output. Usually this returns the test loss of a completed training.
        min_group (int): minimum size of a group. Group with works fewer than the value
            will not be counted.
        return_nested (bool): whether to return nested config dict or flat one.
        kwargs (dict): additional keyword arguments for get_score.
    
    Returns:
        constant_config (dict): the config dict shared by all completed works
        unique_configs (list): a list of unique config dict that is not constant or
            nuisance (ends with 'seed').
        grouped_ids (list): a list of grouped work IDs. Each element is a list of strings.
        grouped_scores (list): a list of grouped scores, with the same structure of
            grouped_ids.

    """
    def get_test_loss(stat):
        return stat['losses']['test'][stat['best_idx']]
    
    if cond_dict is None:
        cond_dict = {}
    else:
        assert isinstance(cond_dict, dict)
    if get_score is None:
        get_score = get_test_loss
    
    # gather completed works
    for w_id in work_ids:
        assert configs.has_id(w_id), '{} does not exist in configs'.format(w_id)
    completed_ids, flat_configs, scores = [], [], []
    for w_id in work_ids:
        if stats.has_id(w_id) and stats.fetch_record(w_id)['completed']:
            completed_ids.append(w_id)
            flat_configs.append(flatten(configs.fetch_record(w_id)))
            scores.append(get_score(stats.fetch_record(w_id), **kwargs))
    
    # verify all configs are consistent
    full_keys = None
    for config in flat_configs:
        if full_keys is None:
            full_keys = config.keys()
        else:
            assert full_keys==config.keys(), 'config keys inconsistent'
    
    # filter out works that do not match conditioned config
    flat_cond = flatten(cond_dict)
    assert set(flat_cond).issubset(full_keys), 'keys of conditioning dict is incompatible'
    def match_cond(flat_config, flat_cond):
        for key in flat_cond:
            if flat_cond[key]!=flat_config[key]:
                return False
        return True
    idxs = [i for i, flat_config in enumerate(flat_configs) if match_cond(flat_config, flat_cond)]
    if len(idxs)==0:
        print('no configs matching the conditions found')
        return None, None, [], []
    matched_ids = [completed_ids[i] for i in idxs]
    flat_configs = [flat_configs[i] for i in idxs]
    scores = [scores[i] for i in idxs]
    
    # identify constant config and varying config, all keys end with 'seed' are considered nuisance key
    val_nums = {}
    for key in full_keys:
        if not key.endswith('seed'):
            vals = [] # values may be unhashable, use list instead of set here
            for config in flat_configs:
                if config[key] not in vals:
                    vals.append(config[key])
            val_nums[key] = len(vals)
    constant_keys, varying_keys = [], []
    for key in val_nums:
        if val_nums[key]==1:
            constant_keys.append(key)
        else:
            varying_keys.append(key)
    print('{} constant config keys, {} varying config keys'.format(len(constant_keys), len(varying_keys)))
    constant_config = dict((key, flat_configs[0][key]) for key in constant_keys)
    varying_configs = [dict((key, flat_config[key]) for key in varying_keys) for flat_config in flat_configs]
    
    # group all works into unique configs
    unique_configs, grouped_ids, grouped_scores = [], [], []
    for w_id, config, score in zip(matched_ids, varying_configs, scores):
        if config in unique_configs:
            idx = unique_configs.index(config)
            grouped_ids[idx].append(w_id)
            grouped_scores[idx].append(score)
        else:
            unique_configs.append(config)
            grouped_ids.append([w_id])
            grouped_scores.append([score])
    print('{} unique configs found'.format(len(unique_configs)))
    
    # remove groups that are smaller than required size
    idxs = [i for i, g in enumerate(grouped_ids) if len(g)>=min_group]
    unique_configs = [unique_configs[i] for i in idxs]
    grouped_ids = [grouped_ids[i] for i in idxs]
    grouped_scores = [grouped_scores[i] for i in idxs]
    print('{} configs have at least {} completed works'.format(len(unique_configs), min_group))
    
    # sort the group according to mean score
    idxs = np.argsort([np.mean(s) for s in grouped_scores])
    unique_configs = [unique_configs[i] for i in idxs]
    grouped_ids = [grouped_ids[i] for i in idxs]
    grouped_scores = [grouped_scores[i] for i in idxs]
    print('results sorted in ascending order of scores')
    
    # nest config dicts if specified
    if return_nested:
        constant_config = nest(constant_config)
        unique_configs = [nest(config) for config in unique_configs]
    return constant_config, unique_configs, grouped_ids, grouped_scores
