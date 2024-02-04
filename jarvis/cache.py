import time
from pathlib import Path
import functools
import inspect
from collections.abc import Callable
from typing import Any

from .config import Config
from .archive import Archive, ConfigArchive


class OutOfPatienceError(RuntimeError):
    r"""Raised when too many attempts to open an file fail."""

    def __init__(self, key: str, patience: int):
        self.key, self.patience = key, patience
        msg = f"Max number ({patience}) of wait reached for {key}."
        super().__init__(msg)


class Cache:
    r"""A cache that stores previously computed results."""

    def __init__(self,
        cache_dir: Path|str,
        *,
        path_len: int = 4, s_pause: float = 1., l_pause: float = 30.,
        patience: int = 0, wait: float = 60.,
    ):
        r"""
        Args
        ----
        cache_dir:
            Path to the cache folder, three subfolders 'configs', 'stats',
            'results' will be created if needed.
        path_len:
            External file name length, see `Archive.path_len` for more details.
        s_pause, l_pause:
            Short and long pause time for archives, see `Archive.pause` for more
            details. `configs` and `stats` use `s_pause`, and `results` uses
            `l_pause`.
        patience:
            Maximum number of waits in `process`. During cluster use, a job can
            be running by another node, the cache will wait for a period of time
            before checking the latest stat.
        wait:
            Wait time in seconds before each stat check.

        """
        self.cache_dir = Path(cache_dir)
        self.configs = ConfigArchive(self.cache_dir/'configs', path_len=path_len, pause=s_pause)
        self.stats = Archive(self.cache_dir/'stats', path_len=path_len, pause=s_pause)
        self.results = Archive(self.cache_dir/'results', path_len=path_len, pause=l_pause)
        self.patience, self.wait = patience, wait

    def process(self,
        config: Config,
        func: Callable,
        key_only: bool = False,
    ) -> tuple[str, Any]:
        r"""Process a job from configuration.

        If the job has already been computed, the previous result will be
        fetched. If it is being computed by other machines, the function waits
        until out of patience.

        Args
        ----
        config:
            Configuration of the job, typically contains '_module_' and '_name_'
            that identifies the function, and sometimes '_spec_' to identify an
            object when `func` is a method. Other items are treated as keyword
            arguments for `func`.
        func:
            The function to call. When the job is to call an method, the object
            is embedded in `func`.
        key_only:
            Whether to only return the key in cache, useful for batch processing.

        Returns
        -------
        key, result:
            The key and result of the job.

        """
        key = self.configs.add(config)

        # wait for other nodes in cluster to finish
        count = 0
        while True:
            stat = self.stats.get(key, {'processed': False, 't_modified': -float('inf')})
            if stat['processed'] or time.time()-stat['t_modified']>=self.wait:
                break
            count += 1
            if count>self.patience:
                raise OutOfPatienceError(key, self.patience)
            time.sleep(self.wait)

        if stat['processed']: # fetch from cache
            result = None if key_only else self.results[key]
        else: # compute from scratch
            self.stats[key] = {'processed': False, 't_modified': time.time()}
            result = func(**{
                k: v for k, v in config.items() if k not in ['_module_', '_name_', '_spec_']
            })
            self.results[key] = result
            self.stats[key] = {'processed': True, 't_modified': time.time()}
        return key, result


def cached(cache: Cache):
    r"""Returns decorator that reads from and writes to a cache."""
    def decorator(func: Callable):
        # wrapper for basic function
        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            ba = inspect.signature(func).bind(*args, **kwargs)
            ba.apply_defaults()
            config = Config({
                '_module_': func.__module__,
                '_name_': func.__name__,
            }).fill(ba.arguments)
            _, result = cache.process(config, func)
            return result
        # wrapper for method
        @functools.wraps(func)
        def wrapped_method(obj, *args, **kwargs):
            ba = inspect.signature(func).bind(obj, *args, **kwargs)
            ba.apply_defaults()
            config = Config({
                '_module_': func.__module__,
                '_name_': func.__qualname__,
                '_spec_': obj.spec,
            }).fill(
                {k: v for k, v in ba.arguments.items() if k!='self'}
            )
            _, result = cache.process(config, lambda **kwargs: func(obj, **kwargs))
            return result
        return wrapped_func if func.__name__==func.__qualname__ else wrapped_method
    return decorator
