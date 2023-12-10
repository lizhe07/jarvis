import os, pickle, random, time
from pathlib import Path
import numpy as np
from typing import Any, Optional

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
    r"""Dictionary-like class that stores data externally."""
    _alphabet = ['{:X}'.format(i) for i in range(16)]

    def __init__(self,
        store_dir: str,
        key_len: int = 8,
        path_len: int = 4,
        max_try: int = 30,
        pause: float = 0.5,
        use_buffer: bool = False,
    ):
        r"""
        Args
        ----
        store_dir:
            The directory for storing data.
        key_len:
            The length of keys.
        path_len:
            The length of external file names, should be no greater than
            `key_len`.
        max_try:
            The maximum number of trying to read/write external files.
        pause:
            The time (seconds) between two consecutive read/write attempts.
        use_buffer:
            Whether to use a buffer in memory. If ``True``, some methods will be
            attempted on an internal dict first before loading external files.
            Overwriting will be disabled since many Archive objects may be
            running at the same time.

        """
        self.store_dir, self.key_len = store_dir, key_len
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

    def _store_path(self, key: str) -> str:
        r"""Returns the path of external file associated with a key."""
        if not self._is_valid_key(key):
            raise KeyError(key)
        return f"{self.store_dir}/{'/'.join(key[:self.path_len])}.axv"

    def _file_names(self, folder_name: Optional[str] = None, depth: Optional[int] = None) -> str:
        r"""A generator for names of existing records file."""
        if folder_name is None:
            folder_name = self.store_dir
        if depth is None:
            depth = self.path_len
        if depth==1:
            file_names = os.listdir(folder_name)
            random.shuffle(file_names)
            for file_name in file_names:
                if len(file_name)==5 and file_name[0] in self._alphabet and file_name.endswith('.axv'):
                    yield file_name
        else:
            subfolder_names = [f for f in os.listdir(folder_name) if f in self._alphabet]
            random.shuffle(subfolder_names)
            for subfolder_name in subfolder_names:
                for file_name in self._file_names(f'{folder_name}/{subfolder_name}', depth-1):
                    yield f'{subfolder_name}/{file_name}'

    def _store_paths(self) -> list[str]:
        r"""Returns all valid external files in the directory."""
        store_paths = [f'{self.store_dir}/{f}' for f in self._file_names()]
        return store_paths

    def _sleep(self):
        r"""Waits for a random period of time."""
        time.sleep(self.pause*(0.8+random.random()*0.4))

    def _safe_read(self, store_path: str) -> dict:
        r"""Safely reads a file.

        This is designed for cluster use. A MaxTryIOError is raised when too
        many attempts have failed.

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
            raise MaxTryIOError(store_path, count)
        if self.buffer is not None:
            self.buffer.update(records)
        return records

    def _safe_write(self, records: dict, store_path: str):
        r"""Safely writes a file."""
        count = 0
        while count<self.max_try:
            try:
                os.makedirs(Path(store_path).parent, exist_ok=True)
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
        to_remove = []
        for store_path in self._store_paths():
            try:
                self._safe_read(store_path)
            except:
                to_remove.append(store_path)
        for store_path in tqdm(to_remove, unit='file', leave=False):
            os.remove(store_path)
        if to_remove:
            print(f"{len(to_remove)} corrupted files removed.")
        self.max_try, self.pause = max_try, pause

    def keys(self) -> str:
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

    def items(self) -> tuple[str, Any]:
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
        else:
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
                else:
                    os.remove(store_path)

    def copy_to(self,
        dst_dir: str,
        keys: Optional[set[str]] = None,
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
        raise NotImplementedError
        os.makedirs(dst_dir, exist_ok=True)
        # check key consistency of existing axv files
        for file_name in os.listdir(dst_dir):
            if file_name.endswith('.axv'):
                assert len(file_name)==(self.path_len+4), (
                    f"Data in the destination '{dst_dir}' have inconsistent file name length "
                    f"{self.path_len}."
                )
                _records = self._safe_read(f'{dst_dir}/{file_name}')
                _key = next(iter(_records.keys()))
                assert len(_key)==self.key_len, (
                    f"Data in the destination '{dst_dir}' have inconsistent key length "
                    f"{self.key_len}."
                )
        # merge records
        for file_name in self._file_names():
            try:
                src_path = f'{self.store_dir}/{file_name}'
                dst_path = f'{dst_dir}/{file_name}'

                src_records = self._safe_read(src_path)
                src_records = {k: v for k, v in src_records.items() if keys is None or k in keys}
                dst_records = self._safe_read(dst_path) if os.path.exists(dst_path) else {}
                if not overwrite:
                    for key in src_records:
                        assert key not in dst_records, (
                            f"Existing key '{key}' detected in the destination '{dst_dir}', cloning "
                            "operation aborted."
                        )
                dst_records.update(src_records)
                if dst_records:
                    self._safe_write(dst_records, dst_path)
            except MaxTryIOError as e:
                print(str(e))


class HashableRecordArchive(Archive):

    def __init__(self, *args, path_len: int = 2, use_buffer: bool = True, **kwargs):
        super().__init__(*args, path_len=path_len, use_buffer=use_buffer, **kwargs)

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

    def get_key(self, n_val) -> Optional[str]:
        r"""Returns the key of a record."""
        h_val = self._to_hashable(n_val)
        if self.buffer is not None:
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

    def __init__(self, *args, max_try: int = 300, **kwargs):
        super().__init__(*args, max_try=max_try, **kwargs)

    def _to_native(self, h_val):
        n_val = super()._to_native(h_val)
        if isinstance(n_val, dict):
            n_val = Config(n_val)
        return n_val

    def __setitem__(self, key: str, val: dict):
        return super().__setitem__(key, Config(val))

    def get_key(self, val: dict) -> Optional[str]:
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
