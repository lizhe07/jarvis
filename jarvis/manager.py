import time
from pathlib import Path
from collections.abc import Iterable
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
        results `results`.
    s_pause:
        Short pause time for `configs` and `stats`.
    l_pause:
        Long pause time for `results`. Results usually take larger storage
        space, therefore they are separated into more files and have higher
        tolerance on I/O failure.
    patience:
        The time to wait for a running work to finish, in hours.

    """

    def __init__(self,
        store_dir: Path|str,
        *,
        s_pause: float = 1., l_pause: float = 5.,
        patience: float = 0.003,
    ):
        self.store_dir = Path(store_dir)
        self.configs = ConfigArchive(
            self.store_dir/'configs', path_len=3, pause=s_pause,
        )
        self.stats = Archive(
            self.store_dir/'stats', path_len=3, pause=s_pause,
        )
        self.results = Archive(
            self.store_dir/'results', path_len=4, pause=l_pause,
        )
        self.patience = patience

    def main(self, config: Config):
        raise NotImplementedError(
            "Need to be assigned with a function that takes a single argument `config`."
        )

    def _get_stat(self, key: str) -> tuple[bool, float]:
        return self.stats.get(key, {'processed': False, 't_modified': -float('inf')})

    def process(self, config: Config) -> tuple[Any, str]:
        key = self.configs.add(config)
        stat = self._get_stat(key)
        if not stat['processed'] and time.time()-stat['t_modified']<self.patience*3600:
            time.sleep(self.patience)
            stat = self._get_stat(key)
        if not stat['processed']:
            self.stats[key] = {'processed': False, 't_modified': time.time()}
            result = self.main(config)
            self.results[key] = result
            self.stats[key] = {'processed': True, 't_modified': time.time()}
        return result, key

    def batch(self,
        configs: Iterable[Config],
        total: int|None = None,
        max_errors: int = 0,
        pbar_kw: dict|None = None,
    ):
        r"""Batch processing.

        Args
        ----
        configs:
            An iterable object containing work configurations. Some are
            potentially processed already.
        total:
            The total number of works to process. If `None`, the batch
            processing stops only when no work is left pending in `configs`.
        max_errors:
            Maximum number of errors allowed. If `0`, the runtime error is
            immediately raised. `KeyboardInterrupt` error is always raised
            regardless of `max_errors` value.
        pbar_kw:
            Keyword argument for progress bar.

        """
        try:
            _total = len(configs)
            if total is None:
                total = _total
            else:
                total = min(total, _total)
        except:
            pass
        pbar_kw = Config(pbar_kw).fill({'leave': False})

        w_count = 0 # counter for processed works
        e_count = 0 # counter for runtime errors
        with tqdm(total=total, **pbar_kw) as pbar:
            for config in configs:
                if self._get_stat(self.configs.add(config))['processed']:
                    continue
                try:
                    self.process(config)
                    w_count += 1
                except KeyboardInterrupt:
                    raise
                except:
                    e_count += 1
                    if e_count>max_errors:
                        raise
                pbar.update()
                if w_count==total:
                    break
