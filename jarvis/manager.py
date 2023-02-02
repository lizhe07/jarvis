import random, time, tarfile, shutil
import numpy as np
from typing import Optional
from collections.abc import Iterable

from .config import Config
from .archive import Archive
from .utils import progress_str, time_str


class Manager:
    r"""Base class for managing model training.

    A manager is associated with different directories storing configurations,
    status, checkpoints and previews of all works. Each individual work is
    specified by a configuration, and assigned with a unique string ID. While
    processing, the manager first sets up itself and then iteratively calls the
    training epoch. Model evaluation and checkpoint saving are done periodically
    during the training.

    Methods `get_config`, `setup`, `init_ckpt`, `load_ckpt` and `save_ckpt`
    usually need to be overridden to function properly, see the doc strings of
    each for more detailed examples. Methods `train` and `eval` must be
    implemented by child class.

    """

    def __init__(self,
        store_dir: Optional[str] = None,
        *,
        read_only: bool = False,
        s_path_len: int = 2, s_pause: float = 1.,
        l_path_len: int = 3, l_pause: float = 5.,
        eval_interval: int = 1, save_interval: int = 1,
        verbose: int = 1,
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
            Long path length and pause time for `ckpts`. Checkpoints usually
            takes larger storage space, therefore they are separated into more
            files and have higher tolerance on I/O failure.
        eval_interval:
            Work is evaluated every `eval_interval` epochs.
        save_interval:
            Checkpoint is saved every `save_interval` epochs.
        verbose:
            Information display level, with '0' referring to quiet mode.

        """
        self.store_dir = store_dir
        if self.store_dir is None:
            self.configs = Archive(is_config=True)
            self.stats = Archive()
            self.ckpts = Archive()
            self.previews = Archive()
        else:
            self.configs = Archive(
                f'{self.store_dir}/configs', path_len=s_path_len, pause=s_pause, is_config=True,
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
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.verbose = verbose
        self.defaults = Config() # can be updated by child class `__init__`

    def get_config(self, config: Optional[dict] = None) -> Config:
        r"""Returns work configuration.

        The method first fills in default values, then performs additional
        steps if necessary. For example, checking compatibility between keys.
        It is possible that the key structure gets changed.

        Overriding
        ----------
        def get_config(self, config):
            config = super().get_config(config)
            # verify and adjust `config` if necessary
            return config

        """
        return Config(config).clone().fill(self.defaults)

    def setup(self, config: Config):
        r"""Sets up manager.

        The method sets `self` properties for future training, for example
        preparing datasets and initializing models. By default, the method
        sets `self.config` for the current work.

        Overriding
        ----------
        def setup(self, config):
            super().setup(config)
            # set up `self` properties

        """
        self.config = config.clone()

    def init_ckpt(self):
        r"""Initializes checkpoint.

        Overriding
        ----------
        def init_ckpt(self):
            super().init_ckpt()
            # update `self.ckpt` with tracked metrics as needed
            # self.ckpt.update({'max_acc': 0.})

        """
        self.epoch = 0
        self._t_train = self._t_eval = None
        self.ckpt = {'eval_records': {}}
        self.preview = {}

    def save_ckpt(self):
        r"""Saves checkpoint.

        Overriding
        ----------
        def save_ckpt(self):
            # update `self.ckpt` and `self.preview`
            super().save_ckpt()

        """
        key = self.configs.add(self.config)
        self.stats[key] = {
            'epoch': self.epoch, 'toc': time.time(),
            't_train': self._t_train, 't_eval': self._t_eval,
        }
        self.ckpts[key] = self.ckpt
        self.previews[key] = self.preview

    def load_ckpt(self):
        r"""Loads checkpoint.

        Overriding
        ----------
        def load_ckpt(self):
            super().load_ckpt()
            # update `self` properties with `self.ckpt`, for example:
            # self.model.load_state_dict(self.ckpt['model_state'])

        """
        key = self.configs.get_key(self.config)
        self.epoch = self.stats[key]['epoch']
        self._t_train = self.stats[key].get('t_train', None)
        self._t_eval = self.stats[key].get('t_eval', None)
        self.ckpt = self.ckpts[key]
        self.preview = self.previews[key]

    def train(self):
        r"""Trains the model for one epoch.

        Learning rate scheduler should be called at the end if it exists.

        """
        raise NotImplementedError

    def eval(self):
        r"""Evaluates the model.

        Typically creates a dictionary of evaluation results and adds it to
        `self.ckpt['eval_records']`.

        >>> eval_record = {'loss': loss, 'acc': acc}
        >>> self.ckpt['eval_records'][self.epoch] = eval_record

        """
        raise NotImplementedError

    def process(self,
        config: Config,
        num_epochs: int = 0,
        resume: bool = True,
    ):
        r"""Processes a work.

        Args
        ----
        config:
            A configuration dictionary of the work.
        num_epochs:
            Number of training epochs for each work. If explicit epochs can not
            be defined, use `num_epochs=0` for an evaluation-only work.
        resume:
            Whether to resume from existing checkpoints. If 'False', process
            each work from scratch.

        """
        self.setup(config)

        try: # load existing checkpoint
            assert resume
            self.load_ckpt()
            assert self.epoch>=0
            if self.verbose>0:
                print("Checkpoint{} loaded.".format(
                    '' if num_epochs==0 else f' (epoch {self.epoch})',
                ))
        except:
            if self.verbose>0:
                print("No checkpoint loaded, initializing from scratch.")
            self.init_ckpt()
            tic = time.time()
            self.eval()
            toc = time.time()
            self._t_eval = toc-tic
            self.save_ckpt()

        while self.epoch<num_epochs:
            if self.verbose>0:
                print(f"\nEpoch: {progress_str(self.epoch+1, num_epochs)}")

            tic = time.time()
            self.train()
            toc = time.time()
            if self._t_train is None:
                self._t_train = toc-tic
            else:
                self._t_train = 0.8*self._t_train+0.2*(toc-tic)

            self.epoch += 1
            if self.epoch%self.eval_interval==0 or self.epoch==num_epochs:
                tic = time.time()
                self.eval()
                toc = time.time()
                if self._t_eval is None:
                    self._t_eval = toc-tic
                else:
                    self._t_eval = 0.8*self._t_eval+0.2*(toc-tic)
            if self.epoch%self.save_interval==0 or self.epoch==num_epochs:
                self.save_ckpt()

    def batch(self,
        configs: Iterable[Config],
        num_epochs: int = 0,
        resume: bool = True,
        count: int = 0,
        patience: float = 4.,
        max_errors: int = 0,
        disp_interval: int = 1,
    ):
        r"""Batch processing.

        Args
        ----
        configs:
            An iterable object containing work configurations. Some are
            potentially processed already.
        num_epochs, resume:
            See `process` for more details.
        count:
            The number of works to process. If it is '0', the processing stops
            when no work is left in `configs`.
        patience:
            Patience time for processing an incomplete work, in hours. The last
            modified time of a work is recorded in `self.stats`.
        max_errors:
            Maximum number of allowed errors. If it is '0', the runtime error
            is immediately raised.
        disp_interval:
            Information display interval, `verbose` will be set to be non-zero
            every `disp_interval` works.

        """
        def to_process(stat, num_epochs, patience):
            return stat['epoch']<num_epochs and (time.time()-stat['toc'])/3600>=patience

        # gather completed works to exclude
        _configs = dict((k, v) for k, v in self.configs.items())
        _stats = dict((k, v) for k, v in self.stats.items())
        completed = set()
        for _key, _stat in _stats.items():
            if _key in _configs and _stat['epoch']>=num_epochs:
                completed.add(_configs[_key])

        w_count = 0 # counter for processed works
        e_count = 0 # counter for runtime errors
        interrupted = False # flag for keyboard interruption
        _verbose = self.verbose
        for config in configs:
            if config in completed:
                continue
            try:
                key = self.configs.add(config)
                try:
                    stat = self.stats[key]
                except:
                    stat = {'epoch': -1, 'toc': -float('inf')}
                if not to_process(stat, num_epochs, patience):
                    continue
                stat['toc'] = time.time() # update modified time
                self.stats[key] = stat

                if w_count%disp_interval==0 or (count>0 and w_count+1==count):
                    self.verbose = _verbose
                else:
                    self.verbose = 0
                if self.verbose>0:
                    print("------------")
                    print("Processing {} ({})...".format(
                        key, progress_str(w_count+1, count) if count>0 else w_count+1,
                    ))
                self.process(config, num_epochs, resume)
                w_count += 1
            except KeyboardInterrupt:
                interrupted = True
                break
            except:
                if max_errors==0:
                    raise
                e_count += 1
                if e_count==max_errors:
                    interrupted = True
                    if _verbose>0:
                        print(f"Max number of errors {max_errors} reached.")
                    break
                else:
                    continue
            if count>0 and w_count==count:
                break
        self.verbose = _verbose
        if self.verbose>0:
            print("{} works processed.".format(w_count))
            if not interrupted and (count==0 or w_count<count):
                print("All works are processed or being processed.")

    def _config_gen(self, choices: dict) -> Config:
        r"""Generator of configurations.

        Work configurations are constructed randomly from `choices`, using
        the method `get_config`.

        Args
        ----
        choices:
            The work configuration search specification, can be nested. It has
            the same key structure as a valid `config` for `get_config` method,
            and values at the leaf level are lists containing possible values.

        """
        choices = Config(choices).flatten()
        keys = list(choices.keys())
        vals, dims = [], []
        for key in keys:
            assert isinstance(choices[key], list)
            vals.append(choices[key])
            dims.append(len(choices[key]))
        total_num = np.prod(dims)

        for idx in random.sample(range(total_num), total_num):
            sub_idxs = np.unravel_index(idx, dims)
            config = Config()
            for i, key in enumerate(keys):
                config[key] = vals[i][sub_idxs[i]]
            config = self.get_config(config.nest())
            yield config

    def sweep(self, choices: dict, order: str = 'random', **kwargs):
        r"""Sweep on a grid of configurations.

        Args
        ----
        choices:
            The configuration value grid, see `_config_gen` for more details.
        order:
            Process order, either 'random' or more advanced 'smart'.

        """
        assert order in ['random', 'smart']
        if order=='smart':
            raise NotImplementedError
        self.batch(self._config_gen(choices), **kwargs)

    def overview(self,
        choices: dict,
        min_epoch: Optional[int] = None,
        p_keys: Optional[list[str]] = None,
    ) -> dict:
        r"""Returns an overview report of the sweep.

        Args
        ----
        choices:
            The configuration value grid, see `_config_gen` for more details.
        min_epoch:
            Minimum number of epochs to be considered as complete, useful when
            printing information about the progress of a sweep.
        p_keys:
            A list of keys in `preview` of each work. Expected to gather float
            numbers at `preview[key]` of all processed works.

        Returns
        -------
        report:
            A dictionary containing different values from all works specified
            by `choices`. Besides the custom keys in `p_keys`, there are default
            keys:
            - 'epoch': Number of epochs of each work.
            - 't_train': Average time of training one epoch, in seconds.
            - 't_eval': Average time of evaluation, in seconds.

        """
        _keys = dict((v, k) for k, v in self.configs.items())
        _stats = dict((k, v) for k, v in self.stats.items())
        _previews = dict((k, v) for k, v in self.previews.items())
        configs = list(set(self._config_gen(choices)))
        report = {
            'epoch': [],
            't_train': [], 't_eval': [],
        }
        if p_keys is None:
            p_keys = []
        for p_key in p_keys:
            report[p_key] = []
        for config in configs:
            try:
                key = _keys[config]
                stat = _stats[key]
                epoch = stat['epoch']
                t_train = stat.get('t_train') or np.nan
                t_eval = stat.get('t_eval') or np.nan
            except:
                epoch = -1
                t_train = t_eval = np.nan
            report['epoch'].append(epoch)
            report['t_train'].append(t_train)
            report['t_eval'].append(t_eval)
            if p_keys:
                try:
                    preview = _previews[key]
                except:
                    preview = {}
                for p_key in p_keys:
                    report[p_key].append(preview.get(p_key) or np.nan)
        for p_key in report:
            report[p_key] = np.array(report[p_key])
        if self.verbose>0:
            if min_epoch is None:
                print("Average number of trained epochs: {:.1f}".format(np.mean(report['epoch'])))
            else:
                p = np.clip(report['epoch']/min_epoch, 0, 1)
                print("Average progress of training {:.1%} ({} epochs as complete).".format(
                    np.mean(p), min_epoch,
                ))
            t_train = np.nanmean(report['t_train'])
            if not np.isnan(t_train):
                print("Approximate training time {} per epoch.".format(time_str(t_train)))
            t_eval = np.nanmean(report['t_eval'])
            if not np.isnan(t_eval):
                print("Approximate evaluation time {}.".format(time_str(t_eval)))
        return report

    @staticmethod
    def _is_matched(config: Config, cond: Config) -> bool:
        r"""Checks if a configuration matches condition."""
        flat_config, flat_cond = config.flatten(), cond.flatten()
        for key in flat_cond:
            if not(key in flat_config and flat_config[key]==flat_cond[key]):
                return False
        return True

    def completed(self, min_epoch: int = 0, cond: Optional[dict] = None) -> str:
        r"""A generator for completed works.

        Args
        ----
        min_epoch:
            Minimum number of trained epochs.
        cond:
            Conditioned value of work configurations.

        """
        cond = Config(cond)
        _configs = dict((k, v) for k, v in self.configs.items())
        _stats = dict((k, v) for k, v in self.stats.items())
        for key, stat in _stats.items():
            if stat['epoch']>=min_epoch:
                try:
                    config = _configs[key]
                except:
                    continue
                if self._is_matched(config, cond):
                    yield key

    def best_work(self,
        min_epoch: int = 0,
        cond: Optional[dict] = None,
        p_key: str = 'loss_test',
        reverse: Optional[bool] = None,
    ) -> str:
        r"""Returns the best work given conditions.

        Only completed work with matching config values will be considered.

        Args
        ----
        min_epoch, cond:
            See `completed` for more details.
        p_key:
            The key of `preview` for comparing works.
        reverse:
            Returns work with the largest `p_key` value if `reverse` is 'True',
            otherwise the smallest.

        Returns
        -------
        best_key:
            The key of best work.

        """
        assert min_epoch>=0
        if reverse is None:
            reverse = p_key.startswith('acc') or p_key.startswith('return')
        best_val = -float('inf') if reverse else float('inf')
        best_key = None
        count = 0
        for key in self.completed(min_epoch, cond):
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
        if self.verbose>0:
            if min_epoch==1:
                print(f"{count} completed works found.")
            else:
                print(f"{count} works trained with at least {min_epoch} epochs found.")
        return best_key

    def pop(self, key: str) -> tuple[Config, dict, dict, dict]:
        r"""Pops out a work by key."""
        config = self.configs.pop(key)
        stat = self.stats.pop(key)
        ckpt = self.ckpts.pop(key)
        preview = self.previews.pop(key)
        return config, stat, ckpt, preview

    def prune(self):
        r"""Removes duplicate works.

        Duplicated works may exist due to accident, remove all records except
        for the most trained one.

        """
        dups = self.configs.get_duplicates()
        if self.verbose>0 and len(dups)>0:
            print(f"{len(dups)} duplicate works found.")
        for keys in dups.values():
            best_key, max_epoch = None, None
            for key in keys:
                if key not in self.stats:
                    continue
                epoch = self.stats[key]['epoch']
                if best_key is None or epoch>max_epoch:
                    best_key = key
                    max_epoch = epoch
            for key in keys:
                if key!=best_key:
                    self.pop(key)

    def _export_dir(self,
        dst_dir: str,
        min_epoch: int = 0,
        cond: Optional[dict] = None,
    ):
        dst_manager = Manager(store_dir=dst_dir)
        keys = set(self.completed(min_epoch, cond))
        self.configs.copy_to(dst_manager.configs.store_dir, keys)
        self.stats.copy_to(dst_manager.stats.store_dir, keys)
        self.ckpts.copy_to(dst_manager.ckpts.store_dir, keys)
        self.previews.copy_to(dst_manager.previews.store_dir, keys)

    def export_tar(self,
        tar_path: str = 'store.tar.gz',
        min_epoch: int = 0,
        cond: Optional[dict] = None,
        compresslevel: int = 1,
    ):
        r"""Exports manager data to a tar file.

        Args
        ----
        tar_path:
            Path of the file to export to.
        min_epoch, cond:
            Minimum number of trained epochs and conditioned values for the
            works to export. See `completed` for more details.
        compressionlevel:
            Compression level of gzip.

        """
        assert self.store_dir is not None
        tic = time.time()

        tmp_dir = '{}/tmp_{}'.format(self.store_dir, self.configs._random_key())
        try:
            self._export_dir(tmp_dir, min_epoch, cond)
            with tarfile.open(tar_path, 'w:gz', compresslevel=compresslevel) as f:
                for axv_name in ['configs', 'stats', 'ckpts', 'previews']:
                    f.add(f'{tmp_dir}/{axv_name}', arcname=axv_name)
        except:
            raise
        finally:
            shutil.rmtree(tmp_dir)

        toc = time.time()
        print(f"Data exported to {tar_path} ({time_str(toc-tic)}).")

    def _load_dir(self, src_dir: str, newer_only: bool = True):
        src_manager = Manager(store_dir=src_dir)
        # divide works into 'cloning' group and 'adding' group
        _old_configs = dict((k, v) for k, v in self.configs.items())
        _old_keys = dict((v, k) for k, v in self.configs.items())
        _old_stats = dict((k, v) for k, v in self.stats.items())
        _new_configs = dict((k, v) for k, v in src_manager.configs.items())
        _new_stats = dict((k, v) for k, v in src_manager.stats.items())
        clone_keys, add_keys = set(), set()
        for new_key, config in _new_configs.items():
            if config in _old_keys:
                old_key = _old_keys[config]
                if newer_only and _new_stats[new_key]['epoch']<=_old_stats[old_key]['epoch']:
                    continue
                if new_key==old_key:
                    clone_keys.add(new_key)
                else:
                    add_keys.add(new_key)
            else:
                if new_key in _old_configs:
                    add_keys.add(new_key)
                else:
                    clone_keys.add(new_key)
        # use 'copy_to' to add 'cloning' group directly
        src_manager.configs.copy_to(self.configs.store_dir, clone_keys)
        src_manager.stats.copy_to(self.stats.store_dir, clone_keys)
        src_manager.ckpts.copy_to(self.ckpts.store_dir, clone_keys)
        src_manager.previews.copy_to(self.previews.store_dir, clone_keys)
        # use 'add' to insert 'adding' group one by one
        for src_key in add_keys:
            dst_key = self.configs.add(_new_configs[src_key])
            self.stats[dst_key] = src_manager.stats[src_key]
            self.ckpts[dst_key] = src_manager.ckpts[src_key]
            self.previews[dst_key] = src_manager.previews[src_key]

    def load_tar(self, tar_path: str, compresslevel: int = 1):
        r"""Loads manager data from a tar file.

        Args
        ----
        tar_path:
            Path of the file to load from.
        compressionlevel:
            Compression level of gzip.

        """
        assert self.store_dir is not None
        tic = time.time()

        tmp_dir = '{}/tmp_{}'.format(self.store_dir, self.configs._random_key())
        try:
            with tarfile.open(tar_path, 'r:gz', compresslevel=compresslevel) as f:
                f.extractall(tmp_dir)
            self._load_dir(tmp_dir)
        except:
            raise
        finally:
            shutil.rmtree(tmp_dir)

        toc = time.time()
        print(f"Data from {tar_path} loaded ({time_str(toc-tic)}).")
