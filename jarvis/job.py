# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 22:57:30 2019

@author: Zhe
"""

import os, random, time
import numpy as np
from .utils import time_str, progress_str, match_cond, grouping
from .archive import Archive


class BaseJob:
    r"""Base class for batch processing.

    The job is associated with different directories storing configurations,
    status, results and previews of all works. Method `main` need to be
    implemented by child class. Method `get_config` need to be implemented by
    child class in order to process works in batch.

    Args
    ----
    store_dir: str
        The directory for storing data.

    """

    def __init__(self, store_dir=None):
        self.store_dir = store_dir
        if self.store_dir is None:
            self.configs = Archive(hashable=True)
            self.stats = Archive()
            self.results = Archive()
            self.previews = Archive()
        else:
            self.configs = Archive(os.path.join(self.store_dir, 'configs'), max_try=60, hashable=True)
            self.stats = Archive(os.path.join(self.store_dir, 'stats'))
            self.results = Archive(os.path.join(self.store_dir, 'results'), pth_len=3, pause=5.)
            self.previews = Archive(os.path.join(self.store_dir, 'previews'), pause=1.)

    def prune(self):
        r"""Removes corrupted files.

        """
        print('pruning configs...')
        self.configs.prune()
        print('pruning stats...')
        self.stats.prune()
        print('pruning results...')
        self.results.prune()
        print('pruning previews...')
        self.previews.prune()

    def main(self, config, verbose):
        r"""Main function of work processing.

        The method needs to be implemented in the child class.

        Args
        ----
        config: dict
            The configuration dictionary to process.
        verbose: bool
            Whether to display information.

        """
        raise NotImplementedError

    def get_config(self, arg_strs):
        r"""Returns a configuratiion dictionary from argument strings.

        The method needs to be implemented in the child class for batch
        processing.

        Args
        ----
        arg_strs: list
            The argument strings as the input of an argument parser.

        Returns
        -------
        config: dict
            The configuration dictionary specified by `arg_strs`.

        """
        raise NotImplementedError

    def is_completed(self, key, strict=False):
        r"""Returns whether a work is completed.

        Args
        ----
        key: str
            The work key.
        strict: bool
            Whether to check `results` and `previews`.

        """
        try:
            stat = self.stats[key]
        except:
            return False
        else:
            if not stat['completed']:
                return False
        if strict and not(key in self.results and key in self.previews):
            return False
        return True

    def process(self, config, policy='preserve', verbose=True):
        r"""Processes one work.

        Args
        ----
        config: dict
            The configuration dictionary to process.
        policy: str
            The process policy regarding the existing record, can be
            ``'preserve'`` or ``'overwrite'``.
        verbose: bool
            Whether to display information.

        Returns
        -------
        result, preview:
            The result and preview of the processed work.

        """
        assert policy in ['overwrite', 'preserve']

        key = self.configs.add(config)
        if self.is_completed(key):
            if policy=='preserve':
                if verbose:
                    print(f"{key} already exists, results and previews will be preserved")
                return self.results[key], self.previews[key]
            if policy=='overwrite':
                if verbose:
                    print(f"{key} already exists, results and previews will be overwritten")

        if verbose:
            print(f'processing {key}...')
        tic = time.time()
        self.stats[key] = {'tic': tic, 'toc': None, 'completed': False}
        result, preview = self.main(config, verbose)
        self.results[key] = result
        self.previews[key] = preview
        toc = time.time()
        self.stats[key] = {'tic': tic, 'toc': toc, 'completed': True}
        return result, preview

    def conjunction_configs(self, search_spec):
        r"""Returns a generator iterates over a search specification randomly.

        Args
        ----
        search_spec: dict
            The work configuration search specification. Dictionary items are
            `(key, vals)`, in which `vals` is a list containing possible
            search values. Each key-val pair will be converted by `converter`.

        Yields
        ------
            A work configuration dictionary in random order.

        """
        def idx2args(idx):
            sub_idxs = np.unravel_index(idx, space_dim)
            arg_vals = [val_list[sub_idx] for sub_idx, val_list in zip(sub_idxs, val_lists)]
            return arg_vals

        def converter(key, val):
            if isinstance(val, bool):
                if val:
                    return ['--'+key]
            elif isinstance(val, list):
                if val:
                    return ['--'+key]+[str(v) for v in val]
            elif val is not None:
                return ['--'+key, str(val)]
            return []

        arg_keys = list(search_spec.keys())
        val_lists = [search_spec[key] for key in arg_keys]
        space_dim = [len(v) for v in val_lists]
        total_num = np.prod(space_dim)

        for idx in random.sample(range(total_num), total_num):
            arg_vals = idx2args(idx)
            arg_strs = []
            for arg_key, arg_val in zip(arg_keys, arg_vals):
                arg_strs += converter(arg_key, arg_val)
            config = self.get_config(arg_strs)
            yield config

    def completed(self, strict=False):
        r"""Returns a generator for completed works.

        Args
        ----
        strict: bool
            Whether to check `results` and `previews`.

        Yields
        ------
        key: str
            The key of a completed work.
        config: dict
            The configuration dictionary.
        stat: dict
            The status dictionary.

        """
        for key, stat in self.stats.items():
            if stat['completed']:
                try:
                    config = self.configs[key]
                except:
                    continue
                if not strict or (key in self.results and key in self.previews):
                    yield key, config, stat

    def matched(self, matcher):
        r"""Returns a generator for completed works matching certain pattern.

        Args
        ----
        matcher: callable
            `matcher(config)` returns ``True`` if `config` matches the pattern,
            ``False`` otherwise.

        Yields
        ------
        key: str
            The key of a completed work that matches `matcher`.
        config: dict
            The configuration dictionary.

        """
        for key, config, _ in self.completed():
            if matcher(config):
                yield key, config

    def conditioned(self, cond):
        r"""Returns a generator for completed works matching a condition.

        Args
        ----
        cond: dict
            A dictionary specifying the condioned values of configurations.

        Yields
        ------
        key: str
            The key of a completed work that is correctly conditioned.
        config: dict
            The configuration dictionary.

        """
        matcher = lambda config: match_cond(config, cond)
        for key, config in self.matched(matcher):
            yield key, config

    def group_and_sort(self, cond, nuisances=None, p_key='acc_test', reverse=None):
        r"""Groups and sorts works based on previews.

        Args
        ----
        cond: dict
            A dictionary specifying the conditioned values.
        nuisances: set
            The nuisance keys, in the flat form (e.g.
            ``'train_config::batch_size'``).
        p_key: str
            The key of preview of interest.
        reverse: bool
            The order of sort. ``True`` indicates descending order.

        Returns
        -------
        g_keys: list
            A list of group keys. Each item is a dictionary containing the
            unshared part of configurations. Sorted by the mean value of each
            `p_vals` item.
        configs: list
            The grouped configurations. Each item is a list of dictionaries
            that match the corresponding group key.
        p_vals: list
            The grouped preview values of `p_key`. Each item is a list of float
            numbers fetched from `self.previews`.

        """
        if reverse is None:
            if p_key.startswith('acc'):
                reverse = True
            else:
                reverse = False
        p_vals =  {}
        for key, config in self.conditioned(cond):
            p_vals[config] = self.previews[key][p_key]
        if not p_vals:
            return [], [], []
        groups = grouping(p_vals.keys(), nuisances)
        g_keys, configs, p_vals = zip(*sorted([
            (g_key, configs, [p_vals[config] for config in configs])
            for g_key, configs in groups.items()
            ], key=lambda x: np.mean(x[-1]), reverse=reverse))
        return g_keys, configs, p_vals

    def overview(self, search_spec=None):
        r"""Displays an overview of the job.

        Progress of the whole job will be displayed, along with the average
        process time.

        Args
        ----
        search_spec: dict
            The work configuration search specification. If ``None``, all
            completed works will be shown.

        """
        if search_spec is not None:
            all_configs = set([c for c in self.conjunction_configs(search_spec)])
        completed_configs, time_costs = [], []
        for key, config, stat in self.completed():
            if search_spec is None or (config in all_configs):
                completed_configs.append(config)
                time_costs.append(stat['toc']-stat['tic'])
        print('{} works completed'.format(
            len(completed_configs) if search_spec is None else
            progress_str(len(completed_configs), len(all_configs))
            ))
        if completed_configs:
            print('average processing time {}'.format(time_str(np.mean(time_costs))))
        return completed_configs

    def random_search(self, search_spec, process_num=0,
                      max_wait=1, tolerance=float('inf'), verbose=True):
        r"""Randomly processes work in the search space.

        Args
        ----
        search_spec: dict
            The work configuration search specification.
        process_num: int
            The number of works to process. `process_num=0` means to process
            all pending works.
        max_wait: float
            Maximum waiting time in the beginning, in seconds.
        tolerance: float
            The maximum allowed time for processing one work, in hours.
        verbose: bool
            Whether to display information.

        """
        def to_run(config):
            r"""Determines whether to run a work.

            """
            key = self.configs.add(config)
            try:
                stat = self.stats[key]
            except:
                return True # stats[key] does not exist
            if not stat['completed'] and (time.time()-stat['tic'])/3600>tolerance:
                return True # running time exceeds tolerance
            else:
                return False

        random_wait = random.random()*max_wait
        if verbose:
            print('random wait {:.1f}s'.format(random_wait))
        time.sleep(random_wait)

        count = 0
        for config in self.conjunction_configs(search_spec):
            if to_run(config):
                self.process(config, verbose=verbose)
                count += 1
            if process_num>0 and count==process_num:
                if verbose:
                    print('\n{} works processed'.format(process_num))
                return
        if verbose:
            print('\nall works processed or being processed')
