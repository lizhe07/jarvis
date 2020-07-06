# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 22:57:30 2019

@author: Zhe
"""

import os, random, time
import numpy as np
from .utils import time_str, progress_str, HashableDict
from .archive import Archive


class BaseJob:
    r"""Class of batch job.

    The job is associated with different folders storing configurations,
    status, outputs and previews of all works. Methods `get_work_config` and
    `main` need to be implemented by child class.

    Args
    ----
    save_dir: str
        The directory for saving data.

    """

    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.configs = Archive(os.path.join(self.save_dir, 'configs'), max_try=60, record_hashable=True)
        self.stats = Archive(os.path.join(self.save_dir, 'stats'))
        self.outputs = Archive(os.path.join(self.save_dir, 'outputs'), f_name_len=4, pause=4)
        self.previews = Archive(os.path.join(self.save_dir, 'previews'), pause=1)

    def remove_corrupted(self):
        r"""Removes corrupted files.

        """
        self.configs.remove_corrupted()
        self.stats.remove_corrupted()
        self.outputs.remove_corrupted()
        self.previews.remove_corrupted()

        w_ids = self.configs.all_ids()
        for archive in [self.stats, self.outputs, self.previews]:
            to_remove = [w_id for w_id in archive.all_ids() if w_id not in w_ids]
            for w_id in to_remove:
                archive.remove(w_id)

    def is_completed(self, w_id):
        r"""Returns if a work is completed.

        Args
        ----
        w_id: str
            The work ID.

        """
        return self.stats.has_id(w_id) and self.stats.fetch_record(w_id)['completed'] and self.previews.has_id(w_id)

    def completed_ids(self):
        r"""Returns the completed work IDs.

        """
        c_ids = [w_id for w_id in self.stats.fetch_matched(lambda r: r['completed']) \
                 if self.previews.has_id(w_id)]
        return c_ids

    def process(self, work_config, policy='overwrite', silent_mode=False):
        r"""Processes one work.

        Args
        ----
        work_config: dict
            The work configuration dictionary.
        policy: str
            The process policy regarding the existing work record.
        silent_mode: bool
            No information about job will be printed if `silent_mode` is set to
            ``True``.

        Returns
        -------
        w_id: str
            The work ID.

        """
        assert policy in ['overwrite', 'preserve', 'verify']

        w_id = self.configs.add(work_config)

        if not silent_mode and self.is_completed(w_id):
            info_str = '{} already exists, outputs and previews will be '.format(w_id)
            if policy=='overwrite':
                print(info_str+'overwritten')
            if policy=='preserve':
                print(info_str+'preserved')
            if policy=='verify':
                print(info_str+'verified')
        if self.is_completed(w_id) and policy=='preserve':
            return w_id

        if not silent_mode:
            print('\n{} starts'.format(w_id))
        tic = time.time()
        self.stats.assign(w_id, {
            'tic': tic, 'toc': None,
            'completed': False,
            })

        output, preview = self.main(work_config)
        toc = time.time()

        if policy=='verify':
            raise NotImplementedError('method to verify output and preview is not implemented')
        else:
            self.stats.assign(w_id, {
                'tic': tic, 'toc': toc,
                'completed': True,
                })
            self.outputs.assign(w_id, output)
            self.previews.assign(w_id, preview)
        return w_id

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
            work_config = self.get_work_config(arg_strs)
            yield HashableDict(**work_config)

    def overview(self, search_spec=None):
        r"""Displays an overview of the job.

        Progress of the whole job will be displayed, along with the average
        process time.

        Args
        ----
        search_spec: dict
            The work configuration search specification.

        """
        if search_spec is not None:
            all_configs = set([c for c in self.conjunction_configs(search_spec)])
        completed_configs, time_costs = [], []
        for w_id in self.completed_ids():
            config = self.configs.fetch_record(w_id)
            if search_spec is None or (search_spec is not None and config in all_configs):
                completed_configs.append(config)
                stat = self.stats.fetch_record(w_id)
                time_costs.append(stat['toc']-stat['tic'])
        print('{} works completed'.format(
            len(completed_configs) if search_spec is None else \
            progress_str(len(completed_configs), len(all_configs))
            ))
        if time_costs:
            print('average processing time {}'.format(time_str(np.mean(time_costs))))
        return completed_configs

    def random_search(self, search_spec, process_num=0, tolerance=float('inf')):
        r"""Randomly processes work in the search space.

        Args
        ----
        search_spec: dict
            The work configuration search specification.
        process_num: int
            The number of works to process. `process_num=0` means to process
            all pending works.
        tolerance: float
            The maximum allowed hours. Any unfinished work started earlier than
            the threshold will be restarted.

        """
        def to_run(work_config):
            w_id = self.configs.add(work_config)
            if not self.stats.has_id(w_id):
                return True
            stat = self.stats.fetch_record(w_id)
            if not stat['completed'] and (time.time()-stat['tic'])/3600>tolerance:
                return True
            else:
                return False

        count = 0
        for work_config in self.conjunction_configs(search_spec):
            if to_run(work_config):
                self.process(work_config, 'preserve')
                count += 1

            if process_num>0 and count==process_num:
                print('\n{} works processed'.format(process_num))
                return
        print('\nall works processed or being processed')

    def get_work_config(self, arg_strs):
        r"""Returns work configuratiion dictionary from argument strings.

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

    def main(self, work_config):
        r"""Main function of work processing.

        The method needs to be implemented in the child class.

        Args
        ----
        work_config: dict
            The work configuration dictionary.

        """
        raise NotImplementedError
