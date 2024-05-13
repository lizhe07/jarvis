import os, time
import functools
import inspect
from pathlib import Path
import numpy as np
from typing import Any, Union, Optional
from collections.abc import Iterable, Callable

from jarvis.config import Config, _locate
from jarvis.archive import Archive, ConfigArchive
from jarvis.utils import tqdm


class Cache:

    def __init__(self,
        store_dir: Union[str, Path],
        *,
        s_pause: float = 1., l_pause: float = 30.,
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

    def process(self, config: Config, patience: float = 60.) -> str:
        assert '_target_' in config, "Function must be specified by '_target_'"
        _target = _locate(config._target_)
        sig = inspect.signature(_target)
        for k, v in sig.parameters.items():
            if k not in config and v.default is not inspect.Parameter.empty:
                config[k] = v.default
        key = self.configs.add(config)
        stat = self.stats.get(key, {'processed': False, 't_modified': -float('inf')})
        sleep_count = 0
        while not stat['processed'] and time.time()-stat['t_modified']<patience:
            time.sleep(patience)
            stat = self.stats[key]
            sleep_count += 1
            assert sleep_count<60, f"Too many sleeps for {key}"
        if not stat['processed']:
            self.stats[key] = {'processed': False, 't_modified': time.time()}
            result = config.call()
            self.results[key] = result
            self.stats[key] = {'processed': True, 't_modified': time.time()}
        return key

    def batch(self,
        configs: Iterable[Config],
        total: Optional[int] = None,
        patience: float = 60.,
        max_errors: int = 0,
        tqdm_kwargs: Optional[dict] = None,
    ) -> None:
        try:
            _total = len(configs)
            if total is None:
                total = _total
            else:
                total = min(total, _total)
        except:
            pass
        if tqdm_kwargs is None:
            tqdm_kwargs = {'leave': False}
        w_count = 0 # counter for processed works
        e_count = 0 # counter for runtime errors
        with tqdm(total=total, **tqdm_kwargs) as pbar:
            for config in configs:
                try:
                    self.process(config, patience)
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

    def sweep(self,
        func: Callable,
        choices: dict[str, list],
        **kwargs,
    ):
        choices = Config(choices).flatten()
        keys = list(choices.keys())
        vals = [list(choices[key]) for key in keys]
        dims = [len(val) for val in vals]
        total = np.prod(dims)

        def _config_gen():
            rng = np.random.default_rng()
            for idx in rng.permutation(total):
                sub_idxs = np.unravel_index(idx, dims)
                config = Config({
                    '_target_': f'{func.__module__}.{func.__name__}.__wrapped__',
                })
                for i, key in enumerate(keys):
                    config[key] = vals[i][sub_idxs[i]]
                yield config
        self.batch(_config_gen(), total=total, **kwargs)

    def prune(self):
        file_names = list(self.results._store_paths())
        print("Check result files...")
        max_try, pause = self.results.max_try, self.results.pause
        self.results.max_try, self.results.pause = 1, 0.
        r_tags = []
        for file_name in tqdm(file_names, unit='file'):
            try:
                self.results._safe_read(file_name)
            except:
                try:
                    os.remove(file_name)
                except:
                    pass
                parts = file_name.split('/')[-self.results.path_len:]
                parts[-1] = parts[-1][0]
                r_tags.append(''.join(parts))
        self.results.max_try, self.results.pause = max_try, pause
        print(f"{len(r_tags)} broken files found.")
        if r_tags:
            groups = {}
            for r_tag in r_tags:
                c_tag = r_tag[:self.configs.path_len]
                if c_tag in groups:
                    groups[c_tag].append(r_tag)
                else:
                    groups[c_tag] = [r_tag]
            print(f"Clean records...")
            use_buffer = isinstance(self.configs.buffer, dict)
            self.configs.buffer = None
            for axv in [self.configs, self.stats]:
                max_try, pause = axv.max_try, axv.pause
                axv.max_try, axv.pause = 1, 0.
                for c_tag in tqdm(groups, unit='file'):
                    file_name = axv._store_path(c_tag+'0'*(axv.path_len-axv.key_len))
                    records = axv._safe_read(file_name)
                    for key in records:
                        if key[:self.results.path_len] in groups[c_tag]:
                            records.pop(key)
                    axv._safe_write(records, file_name)
                axv.max_try, axv.pause = max_try, pause
            if use_buffer:
                self.configs.buffer = {}


def cached(cache_dir: Union[str, Path]):
    cache_dir = Path(cache_dir)
    def decorator(func):
        @functools.wraps(func)
        def wrapped(session: dict, **kwargs):
            session_id = '{:08d}'.format(int(session['session_start_time']%1e8))
            cache = Cache(cache_dir/session_id)
            config = Config({
                '_target_': f'{func.__module__}.{func.__name__}.__wrapped__',
                'session': session,
            })
            config.update(kwargs)
            e_count = 0
            while e_count<10:
                try:
                    key = cache.process(config)
                    result = cache.results[key]
                    return result
                except KeyError:
                    pass
                e_count += 1
                cache.stats[key] = {'processed': False, 't_modified': -float('inf')}
            raise RuntimeError(f"Too many failed attempt for \n{config}")
        return wrapped
    return decorator
