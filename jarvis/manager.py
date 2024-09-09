import time
from pathlib import Path
from collections.abc import Callable
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
    load_ckpt: Callable[[Any], None] # loads checkpoint data

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
        config: Config, num_epochs: int|None = None,
        pbar_kw: dict|None = None,
    ) -> None:
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

        """
        pbar_kw = Config(pbar_kw).fill({'unit': 'epoch', 'leave': True})
        key = self.configs.add(config)
        max_epochs = self.setup(config)
        if num_epochs is None:
            num_epochs = max_epochs
        else:
            num_epochs = min(num_epochs, max_epochs)
        stat = self.get_stat(key)
        epoch = stat['epoch']
        if epoch>=num_epochs:
            return
        if epoch>=0: # checkpoint exists
            ckpt = self.ckpts[key]
            self.load_ckpt(ckpt)
        else:
            self.reset()
            epoch = 0
            self.save_ckpt(key, epoch, max_epochs)
        with tqdm(total=num_epochs, **pbar_kw) as pbar:
            pbar.update(epoch)
            while epoch<num_epochs:
                self.step()
                epoch += 1
                if epoch%self.save_interval==0 or epoch==num_epochs:
                    self.save_ckpt(key, epoch, max_epochs)
                pbar.update()

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
            only when no work is left pending in `configs`.
        max_errors:
            Maximum number of errors allowed. If `0`, the runtime error is
            immediately raised. `KeyboardInterrupt` error is always raised
            regardless of `max_errors` value.
        pbar_kw:
            Keyword argument for progress bar of the batch.
        process_kw:
            Keyword argument for `self.process`.

        """
        if num_works is None:
            num_works = len(configs)
        else:
            num_works = min(num_works, len(configs))
        pbar_kw = Config(pbar_kw).fill({'unit': 'work', 'leave': False})
        process_kw = Config(process_kw).fill({'pbar_kw.leave': False})

        w_count = 0 # counter for processed works
        e_count = 0 # counter for runtime errors
        with tqdm(total=num_works, **pbar_kw) as pbar:
            for config in configs:
                key = self.configs.add(config)
                stat = self.get_stat(key)
                if (
                    stat['complete'] or
                    (num_epochs is not None and stat['epoch']>=num_epochs) or
                    (time.time()-stat['t_modified'])/3600<self.patience
                ):
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
                        raise
                if w_count==num_works:
                    break
