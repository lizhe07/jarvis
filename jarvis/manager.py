import os, time, random, shutil, tarfile
from pathlib import Path
from collections import deque
from collections.abc import Callable, Iterator
from typing import Any

from .config import Config
from .archive import Archive, ConfigArchive
from .utils import tqdm

class Manager:
    r"""Base class for managing parallel processing.

    Args
    ----
    store_dir:
        Directory for storing configurations `configs`, status `stats` and
        checkpoints `ckpts`.
    s_pause:
        Short pause time for `configs` and `stats`.
    l_pause:
        Long pause time for `ckpts`. Checkpoints usually take larger storage
        space, therefore they are separated into more files and have higher
        tolerance on I/O failure.
    save_interval:
        Number of epochs between two successive saves.
    patience:
        The time to wait for a running work to finish, in hours.

    """

    # hooks need to be provided by user
    setup: Callable[[Config], int] # sets up workspace and returns max number of epochs
    reset: Callable[[], None] # resets when a work initiates, i.e. epoch=0
    step: Callable[[], None] # runs one epoch
    get_ckpt: Callable[[], Any] # prepares checkpoint data
    load_ckpt: Callable[[Any], int] # loads checkpoint data and returns actual epoch
    pbar_desc: Callable[[Config], str]|None = None # description of a work

    def __init__(self,
        store_dir: Path|str,
        *,
        s_pause: float = 1., l_pause: float = 5.,
        save_interval: int = 1, patience: float = 1.,
    ):
        self.store_dir = Path(store_dir)
        self.configs = ConfigArchive(
            self.store_dir/'configs', path_len=3, pause=s_pause,
        )
        self.stats = Archive(
            self.store_dir/'stats', path_len=3, pause=s_pause,
        )
        self.ckpts = Archive(
            self.store_dir/'ckpts', path_len=4, pause=l_pause,
        )
        self.save_interval = save_interval
        self.patience = patience

    def get_stat(self, key: str) -> dict:
        r"""Returns status of one work."""
        default = {
            'complete': False, 'epoch': -1,
            't_modified': -float('inf'),
        }
        return self.stats.get(key, default)

    def save_ckpt(self, key: str, epoch: int, max_epochs: int):
        r"""Saves checkpoint and updates status."""
        self.ckpts[key] = self.get_ckpt()
        self.stats[key] = {
            'complete': epoch>=max_epochs, 'epoch': epoch,
            't_modified': time.time(),
        }

    def process(self,
        config: dict, num_epochs: int|None = None,
        pbar_kw: dict|None = None,
    ) -> Any:
        r"""Processes a work for given number of epochs.

        Args
        ----
        config:
            Configuration of the work.
        num_epochs:
            Number of epochs of an incremental work. If not provided, will use
            `max_epochs` returned by `self.setup`.
        pbar_kw:
            Keyword argument for progress bar of one work.

        Returns
        -------
        ckpt:
            Latest checkpoint of the processed work.

        """
        config = Config(config)
        key = self.configs.add(config)
        pbar_kw = Config(pbar_kw).fill({
            'desc': key if self.pbar_desc is None else self.pbar_desc(config),
            'unit': 'epoch', 'leave': True,
        })
        max_epochs = self.setup(config)
        if num_epochs is None:
            num_epochs = max_epochs
        else:
            num_epochs = min(num_epochs, max_epochs)
        try:
            ckpt = self.ckpts[key]
            epoch = self.load_ckpt(ckpt)
        except:
            self.reset()
            epoch = 0
            self.save_ckpt(key, epoch, max_epochs)
        if epoch>=num_epochs:
            return self.ckpts[key]
        with tqdm(total=num_epochs, **pbar_kw) as pbar:
            pbar.update(epoch)
            while epoch<num_epochs:
                self.step()
                epoch += 1
                if epoch%self.save_interval==0 or epoch==num_epochs:
                    self.save_ckpt(key, epoch, max_epochs)
                pbar.update()
        return self.ckpts[key]

    def batch(self,
        configs: list[Config],
        num_epochs: int|None = None,
        num_works: int|None = None,
        max_errors: int = 0,
        pbar_kw: dict|None = None,
        process_kw: dict|None = None,
    ):
        r"""Batch processing.

        Args
        ----
        configs:
            An list containing work configurations. Some are potentially
            processed already.
        num_epochs:
            Number of epochs of each work, see `self.process` for more details.
        num_works:
            Number of works to process. If `None`, the batch processing stops
            only when no work is left pending in `configs`, and the progress bar
            will update for complete works in this mode.
        max_errors:
            Maximum number of errors allowed. If `0`, the runtime error is
            immediately raised. `KeyboardInterrupt` error is always raised
            regardless of `max_errors` value.
        pbar_kw:
            Keyword argument for progress bar of the batch.
        process_kw:
            Keyword argument for `self.process`.

        """
        configs = deque(configs)
        if num_works is None:
            total = len(configs)
        else:
            total = min(num_works, len(configs))
        pbar_kw = Config(pbar_kw).fill({'unit': 'work', 'leave': False})
        process_kw = Config(process_kw).fill({'pbar_kw.leave': False})

        w_count = 0 # counter for processed works
        e_count = 0 # counter for runtime errors
        with tqdm(total=total, **pbar_kw) as pbar:
            while len(configs)>0:
                config = configs.popleft()
                key = self.configs.add(config)
                stat = self.get_stat(key)
                if stat['complete'] or (num_epochs is not None and stat['epoch']>=num_epochs):
                    if num_works is None:
                        pbar.update()
                    continue
                if (time.time()-stat['t_modified'])/3600<self.patience:
                    configs.append(config)
                    continue
                stat['t_modified'] = time.time()
                self.stats[key] = stat
                try:
                    self.process(config, num_epochs, **process_kw)
                    w_count += 1
                    pbar.update()
                except KeyboardInterrupt:
                    raise
                except:
                    e_count += 1
                    if e_count>max_errors:
                        print(f"\nMax number of errors {max_errors} reached (current work '{key}')")
                        raise
                if w_count==num_works:
                    break

    def completed(self,
        min_epoch: int = 0,
        recent: float|None = None,
        cond: dict|None = None,
    ) -> Iterator[tuple[str, Config]]:
        r"""A generator for completed works.

        Args
        ----
        min_epoch:
            Minimum number of processed epochs.
        recent:
            How recent works are considered, in unit of hours. For example, if
            only the ones modified within one day is needed, use `recent=24`.
        cond:
            Conditioned value of work configurations, see `ConfigArchive.filter`
            for more details.

        """
        for key, config in self.configs.filter(cond):
            stat = self.get_stat(key)
            if stat['epoch']>=min_epoch and (recent is None or time.time()-stat['t_modified']<=recent):
                yield key, config

    def _export_dir(self,
        dst_dir: Path|str,
        *,
        keys: set|None = None,
        min_epoch: int = 0,
        recent: float|None = None,
        cond: dict|None = None,
    ) -> None:
        r"""Exports manager data to a directory.

        Args
        ----
        dst_dir:
            Destination directory.
        keys:
            Keys of works that will be potentially exported. If ``None``, all
            works satisfying the criterion will be exported.
        min_epoch, recent, cond:
            Filters of the work, see `completed` for more details.

        """
        dst_manager = Manager(dst_dir)
        self.configs.max_try = 1
        self.configs.pause = 0.
        _keys = set(key for key, _ in self.completed(min_epoch, recent, cond))
        if keys is not None:
            _keys.intersection_update(keys)
        self.configs.migrate(dst_manager.configs.store_dir, _keys, pbar_kw={'desc': "Copying 'configs'"})
        self.stats.migrate(dst_manager.stats.store_dir, _keys, pbar_kw={'desc': "Copying 'stats'"})
        self.ckpts.migrate(dst_manager.ckpts.store_dir, _keys, pbar_kw={'desc': "Copying 'ckpts'"})

    def export_tar(self,
        tar_path: str = 'store.tar.gz',
        compresslevel: int = 1,
        **kwargs,
    ) -> None:
        r"""Exports manager data to a tar file.

        Args
        ----
        tar_path:
            Path of the file to export to.
        compressionlevel:
            Compression level of gzip.
        kwargs:
            Keyword arguments for `_export_dir`.

        """
        tmp_dir = self.store_dir/'tmp_{}'.format(self.configs._random_key())
        try:
            self._export_dir(tmp_dir, **kwargs)
            with tarfile.open(tar_path, 'w:gz', compresslevel=compresslevel) as tar:
                full_paths = []
                for root, _, files in os.walk(tmp_dir):
                    for file in files:
                        full_paths.append(Path(root)/file)
                random.shuffle(full_paths)
                for full_path in tqdm(full_paths, desc='Adding files to tar', unit='file'):
                    tar.add(full_path, arcname=os.path.relpath(full_path, tmp_dir))
        except:
            raise
        finally:
            shutil.rmtree(tmp_dir)

    def _load_dir(self,
        src_dir: Path|str,
        *,
        newer_only: bool = True,
        overwrite: bool = False,
    ) -> None:
        r"""Loads manager data from a directory.

        Args
        ----
        src_dir:
            Source directory.
        newer_only:
            Whether to import newer records only, determined by 'epoch' field in
            the work status.
        overwrite:
            Whether to overwrite existing records, see `Archive.migrate` for
            more details.

        """
        src_manager = Manager(src_dir)
        # divide works into 'cloning' group and 'adding' group
        _old_configs = {k: v for k, v in self.configs.items()}
        _old_keys = {self.configs._to_hashable(v): k for k, v in self.configs.items()}
        _old_stats = {k: v for k, v in self.stats.items()}
        _new_configs = {k: v for k, v in src_manager.configs.items()}
        _new_stats = {k: v for k, v in src_manager.stats.items()}
        clone_keys, add_keys = set(), set()
        for new_key, config in _new_configs.items():
            old_key = _old_keys.get(self.configs._to_hashable(config))
            if old_key is None:
                if new_key in _old_configs:
                    add_keys.add(new_key)
                else:
                    clone_keys.add(new_key)
            else:
                if newer_only and (
                    new_key in _new_stats and old_key in _old_stats and
                    _new_stats[new_key]['epoch']<=_old_stats[old_key]['epoch']
                ):
                    continue
                if new_key==old_key:
                    clone_keys.add(new_key)
                else:
                    add_keys.add(new_key)
        # use 'migrate' to add 'cloning' group directly
        src_manager.configs.migrate(self.configs.store_dir, clone_keys, overwrite=True)
        src_manager.stats.migrate(self.stats.store_dir, clone_keys, overwrite=overwrite)
        src_manager.ckpts.migrate(self.ckpts.store_dir, clone_keys, overwrite=overwrite)
        # use 'add' to insert 'adding' group one by one
        for src_key in add_keys:
            dst_key = self.configs.add(_new_configs[src_key])
            self.stats[dst_key] = _new_stats[src_key]
            self.ckpts[dst_key] = src_manager.ckpts[src_key]

    def load_tar(self, tar_path: str, compresslevel: int = 1, **kwargs):
        r"""Loads manager data from a tar file.

        Args
        ----
        tar_path:
            Path of the file to load from.
        compressionlevel:
            Compression level of gzip.
        kwargs:
            Keyword arguments for `_load_dir`.

        """
        tmp_dir = '{}/tmp_{}'.format(self.store_dir, self.configs._random_key())
        try:
            with tarfile.open(tar_path, 'r:gz', compresslevel=compresslevel) as tar:
                members = tar.getmembers()
                total_size = sum(member.size for member in members if member.isfile())
                with tqdm(
                    total=total_size, desc='Extracting files from tar',
                    unit='B', unit_scale=True, unit_divisor=1024,
                ) as pbar:
                    for member in members:
                        tar.extract(member, tmp_dir)
                        pbar.update(member.size if member.isfile() else 0)
            self._load_dir(tmp_dir, **kwargs)
        except:
            raise
        finally:
            shutil.rmtree(tmp_dir)
