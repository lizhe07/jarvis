# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 22:57:30 2019

@author: Zhe
"""

import random, time
import numpy as np
from .utils import time_str, flatten, nest

def process(search_spec, configs, stats, get_config, work_func,
            custom_converter=None, c_kwargs=None,
            process_num=0, tolerance=float('inf'), w_kwargs=None):
    r"""Processes works in random order.
    
    Args:
        search_spec (dict): specification of search space. Each combination of specified
            values corresponds to a list of argument strings.
        configs (Archive): work configuration archive.
        stats (Archive): work status archive.
        get_config (function): a function that takes a list of argument strings as the
            first input with optional keyword arguments. This function returns a
            work configuration as dictionary.
        work_func (function): the main function to execute. It receives work configuration
            as keyword arguments.
        custom_converter (dict): a dictionary of custom converters. The keys of this
            dictionary are a subset of search_spec. The values of it are functions that
            convert search_spec value to a list of argument strings.
        c_kwargs (dict): additional keyword arguments for get_config.
        process_num (int): number of works to process. `process_num=0` means to process
            all pending works.
        tolerance (float): maximum allowed running time. Any work started earlier than
            the threshold will be restarted.
        w_kwargs (dict): additional keyword arguments for work_func.
    
    """
    if custom_converter is None:
        custom_converter = {}
    else:
        assert isinstance(custom_converter, dict)
    if c_kwargs is None:
        c_kwargs = {}
    else:
        assert isinstance(c_kwargs, dict)
    if w_kwargs is None:
        w_kwargs = {}
    else:
        assert isinstance(w_kwargs, dict)
    
    arg_keys = list(search_spec.keys())
    arg_lists = [search_spec[key] for key in arg_keys]
    space_dim = [len(l) for l in arg_lists]
    total_num = np.prod(space_dim)
    
    def idx2args(idx):
        sub_idxs = np.unravel_index(idx, space_dim)
        arg_vals = [arg_list[sub_idx] for sub_idx, arg_list in zip(sub_idxs, arg_lists)]
        return arg_vals
    def to_run(work_config):
        w_id = configs.add(work_config)
        if not stats.has_id(w_id):
            return True
        stat = stats.fetch_record(w_id)
        if not stat['completed'] and (time.time()-stat['tic'])/3600>tolerance:
            return True
        else:
            return False
    
    count = 0
    for idx in random.sample(range(total_num), total_num):
        arg_vals = idx2args(idx)
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
        work_config = get_config(arg_strs, **c_kwargs)
        if to_run(work_config):
            work_func(**work_config, **w_kwargs)
            count += 1
        
        if process_num>0 and count==process_num:
            break

def grouping(work_ids, configs, stats, cond_dict=None, nuisance=None,
             get_score=None, min_group=1, return_nested=False, **kwargs):
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
        nuisance (set): the set of config keys that should be considered as nuisance. Keys
            are for the flat config dict.
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
    
    def match_cond(flat_config, flat_cond):
        for key in flat_cond:
            if flat_cond[key]!=flat_config[key]:
                return False
        return True
    
    if cond_dict is None:
        cond_dict = {}
    else:
        assert isinstance(cond_dict, dict)
    if nuisance is None:
        nuisance = set()
    else:
        assert isinstance(nuisance, set)
    if get_score is None:
        get_score = get_test_loss
    
    # gather completed works
    print('gathering completed works that matches conditioning...')
    tic = time.time()
    for w_id in work_ids:
        assert configs.has_id(w_id), '{} does not exist in configs'.format(w_id)
    
    matched_ids, flat_configs, scores = [], [], []
    flat_cond, flat_keys = flatten(cond_dict), None
    for w_id in work_ids:
        if stats.has_id(w_id) and stats.fetch_record(w_id)['completed']:
            flat_config = flatten(configs.fetch_record(w_id))
            # verify all configs are consistent
            if flat_keys is None:
                flat_keys = flat_config.keys()
                assert set(flat_cond).issubset(flat_keys)
            else:
                assert flat_keys==flat_config.keys(), 'keys of conditioning dict are incompatible'
            
            if match_cond(flat_config, flat_cond):
                matched_ids.append(w_id)
                flat_configs.append(flat_config)
                scores.append(get_score(stats.fetch_record(w_id), **kwargs))
    toc = time.time()
    print('{} elapsed'.format(time_str(toc-tic)))
    if matched_ids:
        print('{} matched configs found'.format(len(matched_ids)))
    else:
        print('no matched configs found')
        return None, None, [], []
    
    # identify constant config and varying config, all keys end with 'seed' are considered nuisance key
    assert nuisance.issubset(flat_keys), 'nuisance flat keys are incompatible'
    for key in flat_keys:
        if key.endswith('seed'):
            nuisance.add(key)
    val_nums = {}
    for key in flat_keys:
        if key not in nuisance:
            vals = [] # values may be lists (unhashable), use list instead of set here
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
