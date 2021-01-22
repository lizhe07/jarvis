# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 22:57:30 2019

@author: Zhe
"""

import os, random, time
import numpy as np
from .utils import time_str, progress_str
from .archive import Archive


class BaseJob:
    r"""Base class for batch processing.

    The job is associated with different directories storing configurations,
    status, outputs and previews of all works. Methods `get_config` and
    `main` need to be implemented by child class.

    Args
    ----
    save_dir: str
        The directory for saving data.

    """

    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.configs = Archive(os.path.join(self.save_dir, 'configs'), max_try=60, hashable=True)
        self.stats = Archive(os.path.join(self.save_dir, 'stats'))
        self.outputs = Archive(os.path.join(self.save_dir, 'outputs'), pth_len=4, pause=5.)
        self.previews = Archive(os.path.join(self.save_dir, 'previews'), pause=1.)

    def prune(self):
        r"""Removes corrupted files.

        """
        print('pruning configs...')
        self.configs.prune()
        print('pruning stats...')
        self.stats.prune()
        print('pruning outputs...')
        self.outputs.prune()
        print('pruning previews...')
        self.previews.prune()

    def get_config(self, arg_strs):
        r"""Returns a configuratiion dictionary from argument strings.

        The method needs to be implemented in the child class.

        Args
        ----
        arg_strs: list
            The argument strings as the input of an argument parser.

        Returns
        -------
        work_config: dict
            The work configuration dictionary.

        """
        raise NotImplementedError

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

    def is_completed(self, key, strict=False):
        r"""Returns whether a work is completed.

        Args
        ----
        key: str
            The work key.
        strict: bool
            Whether to check `outputs` and `previews`.

        """
        try:
            stat = self.stats[key]
        except:
            return False
        else:
            if not stat[key]['completed']:
                return False
        if strict and not(key in self.outputs and key in self.previews):
            return False
        return True

    def completed_keys(self, strict=False):
        r"""Returns keys of completed works.

        Args
        ----
        strict: bool
            Whether to check `outputs` and `previews`.

        """
        if strict:
            return [key for key, stat in self.stats.items() if (
                stat['completed'] and key in self.outputs and key in self.previews
                )]
        else:
            return [key for key, stat in self.stats.items() if stat['completed']]

    def process(self, config, policy='preserve', verbose=True):
        r"""Processes one work.

        Args
        ----
        config: dict
            The configuration dictionary to process.
        policy: str
            The process policy regarding the existing record, can be
            ``preserve`` or ``overwrite``.
        verbose: bool
            Whether to display information.

        Returns
        -------
        key: str
            The work key.

        """
        assert policy in ['overwrite', 'preserve']

        key = self.configs.add(config, check_duplicate=True)
        if self.is_completed(key):
            if policy=='preserve':
                if verbose:
                    print(f'{key} already exists, outputs and previews will be preserved')
                return key
            if policy=='overwrite':
                if verbose:
                    print(f'{key} already exists, outputs and previews will be overwritten')

        tic = time.time()
        self.stats[key] = {'tic': tic, 'toc': None, 'completed': False}
        output, preview = self.main(config, verbose)
        self.outputs[key] = output
        self.previews[key] = preview
        toc = time.time()
        self.stats[key] = {'tic': tic, 'toc': toc, 'completed': True}
        return key

    def converter(self, key, val):
        r"""Converts the key-val pair to argument strings.

        Args
        ----
        key: str
            The argument key.
        val: bool, float, int, list
            The argument value.

        Returns
        -------
        A list of argument strings.

        """
        if isinstance(val, bool):
            if val:
                return ['--'+key]
        elif isinstance(val, list):
            if val:
                return ['--'+key]+[str(v) for v in val]
        elif val is not None:
            return ['--'+key, str(val)]
        return []

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
        arg_keys = list(search_spec.keys())
        val_lists = [search_spec[key] for key in arg_keys]
        space_dim = [len(v) for v in val_lists]
        total_num = np.prod(space_dim)

        def idx2args(idx):
            sub_idxs = np.unravel_index(idx, space_dim)
            arg_vals = [val_list[sub_idx] for sub_idx, val_list in zip(sub_idxs, val_lists)]
            return arg_vals

        for idx in random.sample(range(total_num), total_num):
            arg_vals = idx2args(idx)
            arg_strs = []
            for arg_key, arg_val in zip(arg_keys, arg_vals):
                arg_strs += self.converter(arg_key, arg_val)
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
            all_configs = set([c for c in self.conjunction_configs(search_spec)])
        completed_configs, time_costs = [], []
        for key in self.completed_keys():
            config = self.configs[key]
            if search_spec is None or (config in all_configs):
                completed_configs.append(config)
                stat = self.stats[key]
                time_costs.append(stat['toc']-stat['tic'])
        print('{} works completed'.format(
            len(completed_configs) if search_spec is None else \
            progress_str(len(completed_configs), len(all_configs))
            ))
        if completed_configs:
            print('average processing time {}'.format(time_str(np.mean(time_costs))))
        return completed_configs

    def to_run(self, config, tolerance):
        r"""Determines whether to run a work.

        Args
        ----
        config: dict
            The configuration dictionary to process.
        tolerance: float
            The maximum allowed running time (hours).

        Returns
        -------
        ``False`` if a work is completed or the running time since its start
        has not exceed `tolerance`, ``True`` otherwise.

        """
        key = self.configs.add(config, check_duplicate=True)
        try:
            stat = self.stats[key]
        except:
            return True # stats[key] does not exist

        if not stat['completed'] and (time.time()-stat['tic'])/3600>tolerance:
            return True # running time exceeds tolerance
        else:
            return False

    def random_search(self, search_spec, process_num=0, tolerance=float('inf'), verbose=True):
        r"""Randomly processes work in the search space.

        Args
        ----
        search_spec: dict
            The work configuration search specification.
        process_num: int
            The number of works to process. `process_num=0` means to process
            all pending works.
        tolerance: float
            The maximum allowed hours.
        verbose: bool
            Whether to display information.

        """
        count = 0
        for config in self.conjunction_configs(search_spec):
            if self.to_run(config, tolerance):
                self.process(config, verbose=verbose)
                count += 1
            if process_num>0 and count==process_num:
                if verbose:
                    print('\n{} works processed'.format(process_num))
                return
        if verbose:
            print('\nall works processed or being processed')
