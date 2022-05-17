import random, time
import numpy as np
from collections.abc import Iterable
from typing import Optional
from .utils import time_str, flatten
from .archive import Archive


class BaseJob:
    r"""Base class for batch processing.

    The job is associated with different directories storing configurations,
    status, ckpts and previews of all works. Method `main` need to be
    implemented by child class.

    """

    def __init__(self,
        store_dir: Optional[str] = None,
        read_only: bool = False,
        s_path_len: int = 2, s_pause: float = 1.,
        l_path_len: int = 3, l_pause: float = 5.,
    ):
        r"""
        Args
        ----
        store_dir:
            Directory for storage. Archives including `configs`, `stats`,
            `ckpts` and `previews` will be saved in separate directories.
        read_only:
            If the job is read-only or not.
        s_path_len, s_pause:
            Short path length and pause time for `configs`, as well as `stats`
            and `previews`.
        l_path_len, l_pause:
            Long path length and pause time for `ckpts`.

        """
        self.store_dir = store_dir
        if self.store_dir is None:
            self.configs = Archive(hashable=True)
            self.stats = Archive()
            self.ckpts = Archive()
            self.previews = Archive()
        else:
            self.configs = Archive(
                f'{self.store_dir}/configs', path_len=s_path_len, pause=s_pause, hashable=True,
            )
            self.stats = Archive(
                f'{self.store_dir}/stats', path_len=s_path_len, pause=s_pause,
            )
            self.ckpts = Archive(
                f'{self.store_dir}/ckpts', path_len=l_path_len, pause=l_pause,
            )
            self.previews = Archive(
                f'{self.store_dir}/previews', path_len=s_path_len, pause=s_pause,
            )
        self.read_only = read_only
        if self.store_dir is not None and self.read_only:
            for axv in [self.configs, self.stats, self.previews]:
                axv.to_internal()

    def main(self, config: dict, num_epochs: int = 1, verbose: int = 1):
        r"""Main function that needs implementation by subclasses.

        Args
        ----
        config:
            A configuration dict for the work.
        num_epochs:
            Number of epochs of the work. If not explicit epochs can be defined,
            use `num_epochs=1` for a single pass.
        verbose:
            Level of information display. No message will be printed when
            'verbose' is no greater than 0.

        Returns
        -------
        ckpt, preview:
            Archive records that will be saved in `self.ckpts` and
            `self.previews` respectively.

        """
        raise NotImplementedError

    def batch(self,
        configs: Iterable,
        num_epochs: int = 1,
        num_works: int = 0,
        patience: float = 168,
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
        patience:
            Patience time for processing an incomplete work, in hours. The last
            modified time of a work is recorded in `self.stats`.

        """
        count = 0
        for config in configs:
            try:
                key = self.configs.add(config)
                try:
                    stat = self.stats[key]
                except:
                    stat = {'epoch': 0, 'toc': -float('inf')}
                assert stat['epoch']<num_epochs and (time.time()-stat['toc'])/3600>patience
                stat['toc'] = time.time()
                self.stats[key] = stat

                if verbose>0:
                    print("------------")
                    print(f"Processing {key}...")
                tic = time.time()
                self.main(config, num_epochs, verbose)
                toc = time.time()
                count += 1
                if verbose>0:
                    print("{} processed. ({})".format(key, time_str(toc-tic)))
                    print("------------")
            except:
                continue
            if num_works>0 and count==num_works:
                break
        if verbose>0:
            print("{} works processed.".format(num_works))
            if num_works==0:
                print("All works are processed or being processed.")

    def strs2config(self, arg_strs: Optional[list[str]] = None):
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
        converted to work configuration by `strs2config`. The configuration
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
                config = self.strs2config(arg_strs)
                yield config

        self.batch(config_gen(), **kwargs)

    def load_ckpt(self, config):
        r"""Loads checkpoint."""
        key = self.configs.add(config)
        epoch = self.stats[key]['epoch']
        ckpt = self.ckpts[key]
        return epoch, ckpt

    def save_ckpt(self, config, epoch, ckpt, preview):
        r"""Saves checkpoint."""
        key = self.configs.add(config)
        self.stats[key] = {'epoch': epoch, 'toc': time.time()}
        self.ckpts[key] = ckpt
        self.previews[key] = preview

    @staticmethod
    def _is_matched(config, cond=None):
        r"""Checks if a configuration matches condition."""
        if cond is None:
            return True
        flat_config, flat_cond = flatten(config), flatten(cond)
        for key in flat_cond:
            if not(key in flat_config and flat_config[key]==flat_cond[key]):
                return False
        return True

    def completed(self, min_epochs=1, cond=None):
        r"""A generator for completed works."""
        for key, stat in self.stats.items():
            if stat['epoch']>=min_epochs:
                try:
                    config = self.configs[key]
                except:
                    continue
                if self._is_matched(config, cond):
                    yield key

    def best_work(self,
        min_epochs: int = 1,
        cond: Optional[dict] = None,
        p_key: str = 'loss_test',
        reverse: Optional[bool] = None,
        verbose: int = 1,
    ):
        r"""Returns the best work given conditions.

        Args
        ----
        min_epochs:
            Minimum number of epochs.
        cond:
            Conditioned value of work configurations. Only completed work with
            matching values will be considered.
        p_key:
            The key of `preview` for comparing works.
        reverse:
            Returns work with the largest value of ``'p_key'`` when `reverse` is
            ``True``, otherwise the smallest.

        Returns
        -------
        best_key: str
            The key of best work.

        """
        assert min_epochs>0
        if reverse is None:
            reverse = p_key.startswith('acc')
        best_val = -float('inf') if reverse else float('inf')
        best_key = None
        count = 0
        for key in self.completed(min_epochs, cond):
            val = self.previews[key][p_key]
            if reverse:
                if val>best_val:
                    best_val = val
                    best_key = key
            else:
                if val<best_val:
                    best_val = val
                    best_key = key
            count += 1
        if verbose>0:
            if min_epochs==1:
                print(f"{count} completed works found.")
            else:
                print(f"{count} works trained with at least {min_epochs} epochs found.")
        return best_key

    def pop(self, key):
        r"""Pops out a work by key."""
        config = self.configs.pop(key)
        stat = self.stats.pop(key)
        ckpt = self.ckpts.pop(key)
        preview = self.previews.pop(key)
        return config, stat, ckpt, preview
