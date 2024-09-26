import os, pickle, random, time
from pathlib import Path
import numpy as np
from typing import Any
from collections.abc import Iterator

from .config import Config
from .utils import tqdm
from .alias import Array


class MaxTryIOError(RuntimeError):
    r"""Raised when too many attempts to open an file fail."""

    def __init__(self,
        store_path: str, count: int,
    ):
        self.store_path = store_path
        self.count = count
        msg = f"Max number ({count}) of reading tried and failed on {store_path}."
        super().__init__(msg)


class Archive:
    r"""Dictionary-like class that stores data externally.

    Args
    ----
    store_dir:
        The directory for storing data.
    key_len:
        The length of keys.
    path_len:
        The length of external file names, should be no greater than `key_len`.
    max_try:
        The maximum number of trying to read/write external files.
    pause:
        The time (seconds) between two consecutive read/write attempts.
    use_buffer:
        Whether to use a buffer in memory. If ``True``, some methods will be
        attempted on an internal dict first before loading external files.
        Overwriting will be disabled since many Archive objects may be running
        at the same time.

    """
    _alphabet = ['{:X}'.format(i) for i in range(16)]

    def __init__(self,
        store_dir: str,
        key_len: int = 8,
        path_len: int = 4,
        max_try: int = 30,
        pause: float = 0.5,
        use_buffer: bool = False,
    ):
        self.store_dir, self.key_len = Path(store_dir), key_len
        assert key_len>=path_len, "File name length should be no greater than key length."
        os.makedirs(self.store_dir, exist_ok=True)
        self.path_len, self.max_try, self.pause = path_len, max_try, pause
        self.buffer = {} if use_buffer else None

    def __repr__(self) -> str:
        return f"Archive object stored in {self.store_dir}."

    def __setitem__(self, key: str, val: Any):
        store_path = self._store_path(key)
        records = self._safe_read(store_path) if os.path.exists(store_path) else {}
        if self.buffer is not None:
            if key in self.buffer:
                raise RuntimeError(f"Trying to overwrite an existing key {key} in buffer mode.")
            else:
                self.buffer[key] = val
        records[key] = val
        self._safe_write(records, store_path)

    def __getitem__(self, key: str) -> Any:
        store_path = self._store_path(key)
        if self.buffer is None or key not in self.buffer:
            if not os.path.exists(store_path):
                raise KeyError(key)
            records = self._safe_read(store_path)
        return records[key] if self.buffer is None else self.buffer[key]

    def __contains__(self, key: str) -> bool:
        try:
            self[key]
        except:
            return False
        else:
            return True

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(list(self.keys()))

    def _is_valid_key(self, key: str) -> bool:
        r"""Returns if a key is valid."""
        if not (isinstance(key, str) and len(key)==self.key_len):
            return False
        for c in key:
            if c not in self._alphabet:
                return False
        return True

    @staticmethod
    def _file_name(key: str, path_len: int) -> str:
        return '/'.join(key[:path_len])+'.axv'

    def _store_path(self, key: str) -> Path:
        r"""Returns the path of external file associated with a key."""
        if not self._is_valid_key(key):
            raise KeyError(key)
        return self.store_dir/self._file_name(key, self.path_len)

    @staticmethod
    def _file_names(store_dir: Path, depth: int) -> Iterator[str]:
        r"""A generator for names of existing records file."""
        if depth==1:
            file_names = os.listdir(store_dir)
            random.shuffle(file_names)
            for file_name in file_names:
                if len(file_name)==5 and file_name[0] in Archive._alphabet and file_name.endswith('.axv'):
                    yield file_name
        else:
            subdir_names = [f for f in os.listdir(store_dir) if f in Archive._alphabet]
            random.shuffle(subdir_names)
            for subdir_name in subdir_names:
                for file_name in Archive._file_names(store_dir/subdir_name, depth-1):
                    yield f'{subdir_name}/{file_name}'

    def _store_paths(self) -> Iterator[Path]:
        r"""Returns all valid external files in the directory."""
        for file_name in self._file_names(self.store_dir, self.path_len):
            yield self.store_dir/file_name

    def _sleep(self):
        r"""Waits for a random period of time."""
        time.sleep(self.pause*(0.8+random.random()*0.4))

    def _safe_read(self, store_path: Path) -> dict:
        r"""Safely reads a file.

        This is designed for cluster use, if too many tries have failed, try to
        delete the file. If `buffer` is used, it will be updated gradually every
        time a file is read.

        """
        count = 0
        while count<self.max_try:
            try:
                with open(store_path, 'rb') as f:
                    records = pickle.load(f)
            except KeyboardInterrupt as e:
                raise e
            except:
                count += 1
                self._sleep()
            else:
                break
        if count==self.max_try:
            if os.path.exists(store_path):
                os.remove(store_path)
            records = {}
        if self.buffer is not None:
            parts = list(store_path.parts)[-self.path_len:]
            parts[-1] = parts[-1][0]
            head = ''.join(parts)
            for key in self.buffer:
                if key.startswith(head) and key not in records:
                    # remove values do not exist any more
                    self.buffer.pop(key)
            self.buffer.update(records)
        return records

    def _safe_write(self, records: dict, store_path: Path):
        r"""Safely writes a file."""
        count = 0
        while count<self.max_try:
            try:
                os.makedirs(store_path.parent, exist_ok=True)
                with open(store_path, 'wb') as f:
                    pickle.dump(records, f)
            except KeyboardInterrupt as e:
                raise e
            except:
                count += 1
                self._sleep()
            else:
                break
        if count==self.max_try:
            raise MaxTryIOError(store_path, count)

    def _random_key(self) -> str:
        r"""Returns a random key."""
        return ''.join(random.choices(self._alphabet, k=self.key_len))

    def _new_key(self) -> str:
        r"""Returns a new key."""
        while True:
            key = self._random_key()
            if key not in self:
                break
        return key

    def _clear(self):
        r"""Removes all data."""
        for store_path in self._store_paths():
            os.remove(store_path)

    def _prune(self) -> None:
        r"""Removes corrupted files."""
        max_try, pause = self.max_try, self.pause
        self.max_try, self.pause = 1, 0.
        if self.buffer is not None:
            self.buffer = {}
        for store_path in tqdm(
            list(self._store_paths()), desc='Pruning', unit='file',
        ):
            self._safe_read(store_path) # corrupted files are removed in `_safe_read`
        self.max_try, self.pause = max_try, pause

    def keys(self) -> Iterator[str]:
        r"""A generator for keys."""
        for store_path in self._store_paths():
            records = self._safe_read(store_path)
            for key in records.keys():
                yield key

    def values(self) -> Any:
        r"""A generator for values."""
        for store_path in self._store_paths():
            records = self._safe_read(store_path)
            for val in records.values():
                yield val

    def items(self) -> Iterator[tuple[str, Any]]:
        r"""A generator for items."""
        for store_path in self._store_paths():
            records = self._safe_read(store_path)
            for key, val in records.items():
                yield key, val

    def get(self, key: str, val: Any = None) -> Any:
        try:
            return self[key]
        except:
            return val

    def pop(self, key: str) -> Any:
        r"""Removes a specified key and return its value."""
        if self.buffer is not None:
            raise RuntimeError("Attempting to pop values in buffer mode.")
        store_path = self._store_path(key)
        if not os.path.exists(store_path):
            return None
        records = self._safe_read(store_path)
        val = records.pop(key, None)
        if records:
            self._safe_write(records, store_path)
        elif os.path.exists(store_path):
            os.remove(store_path) # remove empty external file
        return val

    def delete(self, keys: list[str]) -> None:
        r"""Removes keys in batch processing."""
        if self.buffer is not None:
            raise RuntimeError("Attempting to delete keys in buffer mode.")
        to_remove = {}
        for key in keys:
            store_path = self._store_path(key)
            if store_path in to_remove:
                to_remove[store_path].append(key)
            else:
                to_remove[store_path] = [key]
        for store_path in tqdm(to_remove, unit='file', leave=False):
            if not os.path.exists(store_path):
                continue
            records = self._safe_read(store_path)
            modified = False
            for key in to_remove[store_path]:
                if key in records:
                    records.pop(key)
                    modified = True
            if modified:
                if records:
                    self._safe_write(records, store_path)
                elif os.path.exists(store_path):
                    os.remove(store_path)

    def migrate(self,
        dst_dir: str,
        keys: set[str]|None = None,
        overwrite: bool = False,
    ):
        r"""Clones the archive to a new directory.

        Args
        ----
        dst_dir:
            Path to the new directory.
        keys:
            Keys of the records to be cloned. Clone everything if is ``None``.
        overwrite:
            Whether to overwrite existing keys.

        """
        os.makedirs(dst_dir, exist_ok=True)
        # check key consistency
        path_len = []
        for depth in range(1, self.key_len+1):
            file_name =  next(iter(self._file_names(dst_dir, depth)), None)
            if file_name is not None:
                path_len.append(depth)
        if len(path_len)>1:
            raise RuntimeError(f"Multiple hierarchies detected in {dst_dir}")
        elif len(path_len)==1:
            dst_path_len = path_len[0]
            file_name =  next(iter(self._file_names(dst_dir, dst_path_len)), None)
            records = self._safe_read(f'{dst_dir}/{file_name}')
            key = next(iter(records.keys()))
            assert self._is_valid_key(key), f"Invalid key detected in {dst_dir} ({key})"
        else:
            dst_path_len = self.path_len
        # prepare generator of source file paths
        if keys is None:
            src_paths = list(self._store_paths())
        else:
            src_paths = set()
            for key in keys:
                src_paths.add(self._store_path(key))
            src_paths = list(src_paths)
        # copy records to destination directory
        if dst_path_len>=self.path_len:
            random.shuffle(src_paths)
            for src_path in tqdm(src_paths, unit='file', leave=False):
                src_records = self._safe_read(src_path)
                src_keys = [k for k in src_records if keys is None or k in keys]
                key_dicts = {} # keys grouped by files in dst_dir
                for key in src_keys:
                    _key = key[:dst_path_len]
                    if _key in key_dicts:
                        key_dicts[_key].append(key)
                    else:
                        key_dicts[_key] = [key]
                for dst_keys in key_dicts.values():
                    dst_path = f'{dst_dir}/{self._file_name(dst_keys[0], dst_path_len)}'
                    if os.path.exists(dst_path):
                        dst_records = self._safe_read(dst_path)
                        modified = False
                    else:
                        dst_records = {}
                        modified = True
                    for key in dst_keys:
                        if key not in dst_records or overwrite:
                            dst_records[key] = src_records[key]
                            modified = True
                    if modified:
                        self._safe_write(dst_records, dst_path)
        else:
            raise NotImplementedError("Files from src_dir needs to be merged.")


class HashableRecordArchive(Archive):
    r"""Archive class for hashable record."""

    def __init__(self, *args, use_buffer: bool = True, **kwargs):
        super().__init__(*args, use_buffer=use_buffer, **kwargs)

    @classmethod
    def _to_hashable(cls, n_val):
        r"""Converts an object to a hashable record."""
        if isinstance(n_val, dict):
            h_val = ('_D', frozenset((k, cls._to_hashable(v)) for k, v in n_val.items()))
        elif isinstance(n_val, list):
            h_val = ('_L', tuple(cls._to_hashable(v) for v in n_val))
        elif isinstance(n_val, tuple):
            h_val = ('_T', tuple(cls._to_hashable(v) for v in n_val))
        elif isinstance(n_val, set):
            h_val = ('_S', frozenset(cls._to_hashable(v) for v in n_val))
        elif isinstance(n_val, Array):
            h_val = ('_A', (tuple(n_val.reshape(-1)), n_val.dtype, n_val.shape))
        else:
            h_val = n_val
        hash(h_val)
        return h_val

    def _to_native(self, h_val):
        r"""Converts a hashable record to a native object."""
        if isinstance(h_val, tuple) and len(h_val)==2 and h_val[0] in ['_D', '_L', '_T', '_S', '_A']:
            if h_val[0]=='_D':
                n_val = {k: self._to_native(v) for k, v in h_val[1]}
            if h_val[0]=='_L':
                n_val = [self._to_native(v) for v in h_val[1]]
            if h_val[0]=='_T':
                n_val = tuple(self._to_native(v) for v in h_val[1])
            if h_val[0]=='_S':
                n_val = set(self._to_native(v) for v in h_val[1])
            if h_val[0]=='_A':
                n_val = np.array(h_val[1][0], dtype=h_val[1][1]).reshape(h_val[1][2])
        else:
            n_val = h_val
        return n_val

    @classmethod
    def equals(cls, n_val_0, n_val_1) -> bool:
        r"""Checks if two values are equal."""
        return cls._to_hashable(n_val_0)==cls._to_hashable(n_val_1)

    def __setitem__(self, key: str, n_val: Any):
        h_val = self._to_hashable(n_val)
        super().__setitem__(key, h_val)

    def __getitem__(self, key: str) -> Any:
        return self._to_native(super().__getitem__(key))

    def values(self):
        for val in super().values():
            yield self._to_native(val)

    def items(self):
        for key, val in super().items():
            yield key, self._to_native(val)

    def get_key(self, n_val) -> str|None:
        r"""Returns the key of a record."""
        h_val = self._to_hashable(n_val)
        if self.buffer is not None: # search in buffer first
            for key, val in self.buffer.items():
                if val==h_val:
                    return key
        for key, val in super().items():
            if val==h_val:
                return key
        return None

    def add(self, val) -> str:
        r"""Adds a new item if it has not already existed."""
        key = self.get_key(val)
        if key is None:
            key = self._new_key()
            self[key] = val
        return key

    def get_duplicates(self) -> list[tuple[Any, list[str]]]:
        r"""Returns all duplicate records.

        Returns
        -------
        duplicates:
            A list of tuples containing the duplicate record and the associated
            keys.

        """
        inv_dict = {} # dictionary for inverse archive
        for key, h_val in super().items():
            if h_val in inv_dict:
                inv_dict[h_val].append(key)
            else:
                inv_dict[h_val] = [key]
        duplicates = [(self._to_native(h_val), keys) for h_val, keys in inv_dict.items() if len(keys)>1]
        return duplicates


class ConfigArchive(HashableRecordArchive):
    r"""Archive class for Config objects."""

    def __init__(self, *args, max_try: int = 300, **kwargs):
        super().__init__(*args, max_try=max_try, **kwargs)

    def _to_native(self, h_val):
        n_val = super()._to_native(h_val)
        if isinstance(n_val, dict):
            n_val = Config(n_val)
        return n_val

    def __setitem__(self, key: str, val: dict):
        return super().__setitem__(key, Config(val))

    def get_key(self, val: dict) -> str|None:
        return super().get_key(Config(val))

    @classmethod
    def compare(cls, configs: list[Config]) -> tuple[Config, list[Config]]:
        r"""Returns the different parts of multiple configs."""
        f_dicts = [config.flatten() for config in configs]
        shared_keys = None
        for f_dict in f_dicts:
            if shared_keys is None:
                shared_keys = set(f_dict.keys())
            else:
                shared_keys &= set(f_dict.keys())
        keys_with_diff_vals = set()
        for key in shared_keys:
            vals = set()
            for f_dict in f_dicts:
                vals.add(HashableRecordArchive._to_hashable(f_dict[key]))
            if len(vals)>1:
                keys_with_diff_vals.add(key)
        shared_keys -= keys_with_diff_vals
        shared = Config({k: f_dicts[0][k] for k in shared_keys})
        diffs = [
            Config({k: v for k, v in f_dict.items() if k not in shared_keys})
            for f_dict in f_dicts
        ]
        return shared, diffs

    def filter(self, cond: dict) -> Iterator[str]:
        r"""Generator of matching record key.

        Args
        ----
        cond:
            Filter conditions, specifying parts of config values. Each value can
            be an exact match or a callable function that defines a criterion.

        """
        f_cond = Config(cond).flatten()
        for key, config in self.items():
            f_config = config.flatten()
            matched = True
            for k, v in f_cond.items():
                if not (
                    k in f_config and (
                        (callable(v) and v(f_config[k])==True) or f_config[k]==v
                    )
                ):
                    matched = False
                    break
            if matched:
                yield key
