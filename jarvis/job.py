import random, time
import numpy as np
from collections.abc import Iterable
from typing import Optional
from .utils import time_str
from .archive import Archive


class BaseJob:
    r"""Base class for batch processing.

    The job is associated with different directories storing configurations,
    status, results and previews of all works. Method `main` need to be
    implemented by child class.

    """

    def __init__(self,
        store_dir: Optional[str] = None,
        read_only: bool = False,
        c_path_len: int = 2, c_pause: float = 0.5,
        r_path_len: int = 3, r_pause: float = 5.,
    ):
        r"""
        Args
        ----
        store_dir:
            Directory for storage. Archives including `configs`, `stats`,
            `results` and `previews` will be saved in separate directories.
        read_only:
            If the job is read-only or not.
        c_pth_len, c_pause:
            Path length and pause time for `configs`, as well as `stats` and
            `previews`.
        r_path_len, r_pause:
            Path length and pause time for `results`.

        """
        self.store_dir = store_dir
        if self.store_dir is None:
            self.configs = Archive(hashable=True)
            self.stats = Archive()
            self.results = Archive()
            self.previews = Archive()
        else:
            self.configs = Archive(
                f'{self.store_dir}/configs', path_len=c_path_len, max_try=60,
                pause=c_pause, hashable=True,
            )
            self.stats = Archive(
                f'{self.store_dir}/stats', path_len=c_path_len, pause=c_pause,
            )
            self.results = Archive(
                f'{self.store_dir}/results', path_len=r_path_len, pause=r_pause,
            )
            self.previews = Archive(
                f'{self.store_dir}/previews', path_len=c_path_len, pause=c_pause,
            )
        self.read_only = read_only
        if self.store_dir is not None and self.read_only:
            for axv in [self.configs, self.stats, self.previews]:
                axv.to_internal()

    def main(self, config: dict, verbose: int = 1):
        r"""Main function that needs implementation by subclasses.

        Args
        ----
        config:
            A configuration dict for the work.
        verbose:
            Level of information display. No message will be printed when
            'verbose' is no greater than 0.

        Returns
        -------
        result, preview:
            Archive records that will be saved in `self.results` and
            `self.previews` respectively.

        """
        raise NotImplementedError

    def is_completed(self, key: str, strict: bool = False):
        r"""Returns whether a work is completed.

        Args
        ----
        key:
            Key of the work.
        strict:
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

    def to_process(self, config: dict, patience: float = float('inf')):
        r"""Returns whether to process a work."""
        key = self.configs.add(config)
        try:
            stat = self.stats[key]
        except:
            return True # stats[key] does not exist
        t_last = stat['tic'] if stat['toc'] is None else stat['toc']
        if not stat['completed'] and (time.time()-t_last)/3600>patience:
            return True # running time exceeds patience
        else:
            return False # work is being processed

    def process(self, config: dict, verbose: int = 1):
        r"""Processes one work."""
        assert not self.read_only, "This is a read-only job."
        if verbose>0:
            print("--------")
        key = self.configs.add(config)
        if self.is_completed(key):
            if verbose>0:
                print(f"{key} already exists.")
            return self.results[key], self.previews[key]
        if verbose>0:
            print(f"Processing {key}...")
        tic = time.time()
        self.stats[key] = {'tic': tic, 'toc': None, 'completed': False}
        result, preview = self.main(config, verbose)
        self.results[key] = result
        self.previews[key] = preview
        toc = time.time()
        self.stats[key] = {'tic': tic, 'toc': toc, 'completed': True}
        if verbose>0:
            print("{} processed ({}).".format(key, time_str(toc-tic)))
            print("--------")
        return result, preview

    def batch(self,
        configs: Iterable,
        num_works: int = 0,
        max_wait: float = 1.,
        patience: float = float('inf'),
        verbose: int = 1,
    ):
        r"""Batch processing.

        Args
        ----
        configs:
            An iterable object containing work configurations. Some are
            potentially processed already.
        num_works:
            The number of works to process. If it is 0, the processing stops
            when no work is left in `configs`.
        max_wait:
            Maximum waiting time in the beginning, in seconds.
        patience:
            Patience time for processing an incomplete work, in hours. The last
            modified time of a work is recorded in `self.stats`.

        """
        random_wait = random.random()*max_wait
        if random_wait>0 and verbose>0:
            print("Random wait {:.1f}s...".format(random_wait))
        time.sleep(random_wait)

        count = 0
        for config in configs:
            if self.to_process(config, patience):
                self.process(config, verbose)
                count += 1
            if num_works>0 and count==num_works:
                if verbose>0:
                    print("{} works processed.".format(num_works))
                return count
        if verbose>0:
            print("All works are processed or being processed.")

    def get_config(self, arg_strs: Optional[list[str]] =None):
        r"""Returns work configuration.

        Args
        ----
        arg_strs:
            A string list that can be parsed to argument parser.

        Returns
        -------
        config: dict
            A dictionary specified by `arg_strs`.

        """
        raise NotImplementedError

    def grid_search(self, search_spec: dict, **kwargs):
        r"""Grid hyper-parameter search.

        Random argument strings are prepared based on search specification, and
        converted to work configuration by `get_config`. The configuration
        generator is passed to batch processing method.

        search_spec:
            The work configuration search specification. Dictionary items are
            `(key, vals)`, in which `vals` is a list containing possible
            search values.

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

        def config_gen():
            for idx in random.sample(range(total_num), total_num):
                arg_vals = idx2args(idx)
                arg_strs = []
                for arg_key, arg_val in zip(arg_keys, arg_vals):
                    arg_strs += converter(arg_key, arg_val)
                config = self.get_config(arg_strs)
                yield config

        self.batch(config_gen(), **kwargs)

    def load_ckpt(self, config: dict, verbose: int = 1):
        r"""Loads checkpoint."""
        try:
            key = self.configs.get_key(config)
            result = self.results[key]
            if verbose>0:
                print(f"Checkpoint ({key}) loaded.")
            return result
        except:
            if verbose>0:
                print("No checkpoints found.")
            return None

    def save_ckpt(self, ckpt, config: dict, verbose: int = 1):
        r"""Saves checkpoint."""
        key = self.configs.get_key(config)
        stat = self.stats[key]
        stat['toc'] = time.time()
        self.stats[key] = stat
        self.results[key] = ckpt
        if verbose>0:
            print(f"Checkpoint ({key}) saved.")
