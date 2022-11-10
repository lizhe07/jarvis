import random, time, tarfile, shutil
import numpy as np
from typing import Optional
from collections.abc import Iterable

from .config import Config
from .archive import Archive
from .utils import progress_str


class Manager:
    r"""Base class for managing model training.

    A manager is associated with different directories storing configurations,
    status, checkpoints and previews of all works. Each individual work is
    specified by a configuration, and assigned with a unique ID. While
    processing, the manager first sets up itself and then iteratively calls the
    training epoch. Model evaluation and checkpoint saving are done periodically
    during the training.

    Methods `get_config`, `setup`, `init_ckpt`, `load_ckpt` and `save_ckpt`
    usually need to be overridden to function properly, see the documents of
    each for more detailed examples. Methods `train` and `eval` need to be
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
        self.defaults = Config()

    def get_config(self, config) -> Config:
        r"""Returns work configuration.

        The method first fills in default values, then performs additional
        changes if necessary. For example, checking compatibility between keys.
        It is possible that the key structure gets changed.

        Overriding
        ----------
        def get_config(self, config):
            config = super(ChildManager, self).get_config(config)
            # update `config`
            return config

        """
        return Config(config).clone().fill(self.defaults)

    def setup(self, config: Config):
        r"""Sets up manager.

        The method sets `self` properties for future training, for example
        preparing datasets and initializing models.

        Overriding
        ----------
        def setup(self, config):
            # set up `self` properties
            super(ChildManager, self).setup(config)

        """
        self.config = config

    def init_ckpt(self):
        r"""Initializes checkpoint.

        Overriding
        ----------
        def init_ckpt(self):
            super(ChildManager, self).init_ckpt()
            # update `self.ckpt` with tracked metrics as needed
            # self.ckpt.update({'max_acc': 0.})

        """
        self.epoch = 0
        self.ckpt = {'eval_records': {}}
        self.preview = {}

    def save_ckpt(self):
        r"""Saves checkpoint.

        Overriding
        ----------
        def save_ckpt(self):
            # update `self.ckpt` and `self.preview`
            super(ChildManager, self).save_ckpt()

        """
        key = self.configs.add(self.config)
        self.stats[key] = {'epoch': self.epoch, 'toc': time.time()}
        self.ckpts[key] = self.ckpt
        self.previews[key] = self.preview

    def load_ckpt(self):
        r"""Loads checkpoint.

        Overriding
        ----------
        def load_ckpt(self):
            super(ChildManager, self).load_ckpt()
            # update `self` properties with `self.ckpt`, for example:
            # self.model.load_state_dict(self.ckpt['model_state'])

        """
        key = self.configs.get_key(self.config)
        self.epoch = self.stats[key]['epoch']
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
        num_epochs: int = 1,
        resume: bool = True,
    ):
        r"""Processes a training work.

        Args
        ----
        config:
            A configuration dictionary of the work.
        num_epochs:
            Number of training epochs for each work. If explicit epochs can not
            be defined, use `num_epochs=1` for a single pass.
        resume:
            Whether to resume from existing checkpoints. If 'False', process
            each work from scratch.

        """
        self.setup(config)

        try: # load existing checkpoint
            assert resume
            self.load_ckpt()
            if self.verbose>0:
                print("Checkpoint{} loaded.".format(
                    '' if self.epoch==1 else f' (epoch {self.epoch})',
                ))
        except:
            if self.verbose>0:
                print("No checkpoint loaded.")
            self.init_ckpt()
            self.eval()
            self.save_ckpt()

        while self.epoch<num_epochs:
            if self.verbose>0:
                print(f"\nEpoch: {progress_str(self.epoch+1, num_epochs)}")
            self.train()
            self.epoch += 1
            if self.epoch%self.eval_interval==0 or self.epoch==num_epochs:
                self.eval()
            if self.epoch%self.save_interval==0 or self.epoch==num_epochs:
                self.save_ckpt()

    def batch(self,
        configs: Iterable[Config],
        num_epochs: int = 1,
        resume: bool = True,
        count: int = 0,
        patience: float = 1.,
        max_errors: int = 0,
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

        """
        w_count, e_count, interrupted = 0, 0, False
        for config in configs:
            try:
                key = self.configs.add(config)
                try:
                    stat = self.stats[key]
                except:
                    stat = {'epoch': 0, 'toc': -float('inf')}
                if stat['epoch']>=num_epochs or (time.time()-stat['toc'])/3600<patience:
                    continue
                stat['toc'] = time.time() # update modified time
                self.stats[key] = stat

                if self.verbose>0:
                    print("------------")
                    print(f"Processing {key}...")
                self.process(config, num_epochs, resume)
                w_count += 1
                if self.verbose>0:
                    print("------------")
            except KeyboardInterrupt:
                interrupted = True
                break
            except Exception:
                if max_errors==0:
                    raise
                e_count += 1
                if e_count==max_errors:
                    interrupted = True
                    break
                else:
                    continue
            else:
                if count>0 and w_count==count:
                    break
        if self.verbose>0:
            print("\n{} works processed.".format(w_count))
            if not interrupted and (count==0 or w_count<count):
                print("All works are processed or being processed.")

    def sweep(self, sweep_spec: dict, **kwargs):
        r"""Sweep on a hyper-parameter grid.

        Work configurations are constructed randomly from `sweep_spec`, using
        the method `get_config`. The configuration generator is passed to the
        method `batch`.

        Args
        ----
        sweep_spec:
            The work configuration search specification, can be nested. It has
            the same key structure as a valid `config` for `get_config` method,
            and values at the leaf level are lists containing possible values.

        """
        method = sweep_spec.get('method', 'random')
        assert method in ['random', 'smart']
        if method=='smart':
            raise NotImplementedError

        choices = Config(sweep_spec['choices']).flatten()
        keys = list(choices.keys())
        vals, dims = [], []
        for key in keys:
            assert isinstance(choices[key], list)
            vals.append(choices[key])
            dims.append(len(choices[key]))
        total_num = np.prod(dims)

        def config_gen():
            for idx in random.sample(range(total_num), total_num):
                sub_idxs = np.unravel_index(idx, dims)
                config = Config()
                for i, key in enumerate(keys):
                    config[key] = vals[i][sub_idxs[i]]
                config = self.get_config(config.nest())
                yield config
        self.batch(config_gen(), **kwargs)

    @staticmethod
    def _is_matched(config: Config, cond: Config) -> bool:
        r"""Checks if a configuration matches condition."""
        flat_config, flat_cond = config.flatten(), cond.flatten()
        for key in flat_cond:
            if not(key in flat_config and flat_config[key]==flat_cond[key]):
                return False
        return True

    def completed(self, min_epoch: int = 1, cond: Optional[dict] = None) -> str:
        r"""A generator for completed works."""
        for key, stat in self.stats.items():
            if stat['epoch']>=min_epoch:
                try:
                    config = self.configs[key]
                except:
                    continue
                if cond is None or self._is_matched(config, Config(cond)):
                    yield key

    def best_work(self,
        min_epoch: int = 1,
        cond: Optional[dict] = None,
        p_key: str = 'loss_test',
        reverse: Optional[bool] = None,
    ) -> str:
        r"""Returns the best work given conditions.

        Args
        ----
        min_epoch:
            Minimum number of trained epochs.
        cond:
            Conditioned value of work configurations. Only completed work with
            matching values will be considered.
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
        assert min_epoch>0
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

    def export_tar(self, tar_path: str = 'store.tar.gz'):
        r"""Exports manager data to a tar file."""
        assert self.store_dir is not None
        with tarfile.open(tar_path, 'w:gz') as f:
            for axv_name in ['configs', 'stats', 'ckpts', 'previews']:
                f.add(f'{self.store_dir}/{axv_name}', arcname=axv_name)
        print(f"Data exported to {tar_path}.")

    def load_tar(self, tar_path: str):
        r"""Loads manager data from a tar file."""
        tmp_path = '{}/tmp_{}'.format(
            self.store_dir if self.store_dir is not None else 'store',
            self.configs._random_key(),
        )
        with tarfile.open(tar_path, 'r:gz') as f:
            f.extractall(tmp_path)
        tmp_manager = Manager(store_dir=tmp_path)
        for old_key, config in tmp_manager.configs.items():
            new_key = self.configs.add(config)
            self.stats[new_key] = tmp_manager.stats[old_key]
            self.ckpts[new_key] = tmp_manager.ckpts[old_key]
            self.previews[new_key] = tmp_manager.previews[old_key]
        shutil.rmtree(tmp_path)
        print(f"Data from {tar_path} loaded.")
