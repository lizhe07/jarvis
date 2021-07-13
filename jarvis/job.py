# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 22:57:30 2019

@author: Zhe
"""

import os, random, time
import numpy as np
from .utils import (
    time_str, match_cond, grouping, to_hashable
    )
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

    def __init__(self, store_dir=None, readonly=False,
                 c_pth_len=2, c_pause=0.5, r_pth_len=3, r_pause=5):
        self.store_dir = store_dir
        if self.store_dir is None:
            self.configs = Archive(hashable=True)
            self.stats = Archive()
            self.results = Archive()
            self.previews = Archive()
        else:
            self.configs = Archive(os.path.join(self.store_dir, 'configs'),
                                   pth_len=c_pth_len, max_try=60, pause=c_pause, hashable=True)
            self.stats = Archive(os.path.join(self.store_dir, 'stats'),
                                 pth_len=c_pth_len, pause=c_pause)
            self.results = Archive(os.path.join(self.store_dir, 'results'),
                                   pth_len=r_pth_len, pause=r_pause)
            self.previews = Archive(os.path.join(self.store_dir, 'previews'),
                                    pth_len=c_pth_len, pause=c_pause)

        self.readonly = readonly
        if self.store_dir is not None and readonly:
            for axv in [self.configs, self.stats, self.previews]:
                axv.to_internal()

    def prune(self):
        r"""Removes corrupted files.

        """
        assert not self.readonly, "this is a read-only job"

        print("pruning configs...")
        self.configs.prune()
        removed = []
        print("pruning stats...")
        removed += self.stats.prune()
        print("pruning results...")
        removed += self.results.prune()
        print("pruning previews...")
        removed += self.previews.prune()

        print("clearing records of removed files...")
        to_remove = set()
        for key in self.configs:
            for r in removed:
                if key.startswith(r):
                    to_remove.add(key)
        for key in to_remove:
            for axv in [self.stats, self.results, self.previews]:
                if key in axv:
                    axv.pop(key)

    def pop(self, key):
        r"""Pops out a work by key.

        """
        try:
            config = self.configs.pop(key)
        except:
            config = None
        try:
            stat = self.stats.pop(key)
        except:
            stat = None
        try:
            result = self.results.pop(key)
        except:
            result = None
        try:
            preview = self.previews.pop(key)
        except:
            preview = None
        return config, stat, result, preview

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

    def matched(self, matcher, strict=False):
        r"""Returns a generator for completed works matching certain pattern.

        Args
        ----
        matcher: callable
            `matcher(config)` returns ``True`` if `config` matches the pattern,
            ``False`` otherwise.
        strict: bool
            Whether to check `results` and `previews`.

        Yields
        ------
        key: str
            The key of a completed work that matches `matcher`.
        config: dict
            The configuration dictionary.

        """
        for key, config, _ in self.completed(strict):
            if matcher(config):
                yield key, config

    def conditioned(self, cond, strict=False):
        r"""Returns a generator for completed works matching a condition.

        Args
        ----
        cond: dict
            A dictionary specifying the condioned values of configurations.
        strict: bool
            Whether to check `results` and `previews`.

        Yields
        ------
        key: str
            The key of a completed work that is correctly conditioned.
        config: dict
            The configuration dictionary.

        """
        matcher = lambda config: match_cond(config, cond)
        for key, config in self.matched(matcher, strict):
            yield key, config

    def group_and_sort(self, cond=None, nuisances=None, p_key='acc_test', reverse=None):
        r"""Groups and sorts works based on previews.

        Args
        ----
        cond: dict
            A dictionary specifying the conditioned values.
        nuisances: set
            The nuisance keys, in the flat form (e.g.
            ``'train_config::batch_size'``).
        p_key: str
            The key of work preview to be analyzed.
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
        if cond is None:
            cond = {}
        if reverse is None:
            if p_key.startswith('acc'):
                reverse = True
            else:
                reverse = False
        p_vals = {}
        for key, config in self.conditioned(cond):
            p_vals[config] = self.previews[key][p_key]
        if not p_vals:
            return [], [], []
        groups = grouping(p_vals.keys(), nuisances)
        g_keys, configs, p_vals = zip(*sorted([
            (g_key, configs, [p_vals[config] for config in configs])
            for g_key, configs in groups.items()
            ], key=lambda x: np.mean(x[-1]), reverse=reverse))
        return list(g_keys), list(configs), list(p_vals)

    def remove_duplicates(self, check_val=False):
        r"""Remove duplicate works.

        Args
        ----
        check_val: bool
            Whether to check `result` and `preview` for duplicate works.

        """
        duplicates = self.configs.get_duplicates()
        for config, keys in duplicates.items():
            if check_val:
                raise NotImplementedError
            else:
                random.shuffle(keys)
                for key in keys[1:]:
                    self.pop(key)
        print('all duplicates removed')

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
        assert not self.readonly, "this is a read-only job"

        assert policy in ['overwrite', 'preserve']
        if verbose:
            print("--------")

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
            print(f"processing {key}...")
        tic = time.time()
        self.stats[key] = {'tic': tic, 'toc': None, 'completed': False}
        result, preview = self.main(config, verbose)
        self.results[key] = result
        self.previews[key] = preview
        toc = time.time()
        self.stats[key] = {'tic': tic, 'toc': toc, 'completed': True}
        return result, preview

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

    def random_configs(self, search_spec):
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

        while True:
            idx = random.randint(0, total_num-1)
            arg_vals = idx2args(idx)
            arg_strs = []
            for arg_key, arg_val in zip(arg_keys, arg_vals):
                arg_strs += converter(arg_key, arg_val)
            config = self.get_config(arg_strs)
            yield config

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
            all_configs = set([to_hashable(c) for c in self.random_configs(search_spec)])
        completed_configs, time_costs = [], []
        for key, config, stat in self.completed():
            if search_spec is None or (config in all_configs):
                completed_configs.append(config)
                time_costs.append(stat['toc']-stat['tic'])
        print('{} works completed'.format(
            len(completed_configs) if search_spec is None else
            '{}/{}'.format(len(completed_configs), len(all_configs))
            ))
        if completed_configs:
            print('average processing time {}'.format(time_str(np.mean(time_costs))))

    def random_search(self, search_spec, process_num=0, max_wait=1,
                      tolerance=float('inf'), verbose=True):
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
        for config in self.random_configs(search_spec):
            if to_run(config):
                self.process(config, verbose=verbose)
                count += 1
            if process_num>0 and count==process_num:
                if verbose:
                    print('\n{} works processed'.format(process_num))
                return
        if verbose:
            print('\nall works processed or being processed')
