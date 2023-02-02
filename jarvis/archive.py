import os, pickle, random, time
from typing import Any, Optional

from .config import Config
from .utils import progress_str, time_str


class MaxTryIOError(RuntimeError):
    r"""Raised when too many attempts to open an file fail."""

    def __init__(self,
        store_path: str, count: int,
    ):
        self.file_path = store_path
        self.count = count
        msg = f"Max number ({count}) of reading tried and failed on {store_path}."
        super().__init__(msg)


class Archive:
    r"""Dictionary-like class that stores data externally."""
    _alphabet = ['{:X}'.format(i) for i in range(16)]

    def __init__(self,
        store_dir: Optional[str] = None,
        key_len: int = 8,
        path_len: int = 2,
        max_try: int = 30,
        pause: float = 0.5,
        is_config: bool = False,
    ):
        r"""
        Args
        ----
        store_dir:
            The directory for storing data. If `store_dir` is ``None``, an
            internal dictionary `__store__` is used.
        key_len:
            The length of keys.
        path_len:
            The length of external file names, should be no greater than
            `key_len`.
        max_try:
            The maximum number of trying to read/write external files.
        pause:
            The time (seconds) between two consecutive read/write attempts.
        is_config:
            Whether the records are configurations.

        """
        self.store_dir, self.key_len = store_dir, key_len
        if self.store_dir is None:
            self.__store__ = {}
        else:
            assert key_len>=path_len, "File name length should be no greater than key length."
            os.makedirs(self.store_dir, exist_ok=True)
            self.path_len, self.max_try, self.pause = path_len, max_try, pause
        self.is_config = is_config

    def __repr__(self) -> str:
        if self.store_dir is None:
            return "Archive object with no external storage."
        else:
            return f"Archive object stored in {self.store_dir}."

    def __setitem__(self, key: str, val: Any):
        val = Config(val) if self.is_config else val
        if self.store_dir is None:
            self.__store__[key] = val
        else:
            store_path = self._store_path(key)
            records = self._safe_read(store_path) if os.path.exists(store_path) else {}
            records[key] = val
            self._safe_write(records, store_path)

    def __getitem__(self, key: str) -> Any:
        try:
            if self.store_dir is None:
                return self.__store__[key]
            else:
                store_path = self._store_path(key)
                assert os.path.exists(store_path)
                records = self._safe_read(store_path)
                return records[key]
        except:
            raise KeyError(key)

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
        assert self._is_valid_key(key), f"Invalid key '{key}' encountered."
        return f'{self.store_dir}/{key[:self.path_len]}.axv'

    def _store_paths(self) -> list[str]:
        r"""Returns all valid external files in the directory."""
        return [
            f'{self.store_dir}/{f}' for f in os.listdir(self.store_dir)
            if f.endswith('.axv') and len(f)==(self.path_len+4)
        ]

    def _sleep(self):
        time.sleep(self.pause*(0.8+random.random()*0.4))

    def _safe_read(self, store_path: str) -> dict:
        r"""Safely reads a file.

        This is designed for cluster use. An error is raised when too many
        attempts have failed.

        """
        count = 0
        while count<self.max_try:
            try:
                with open(store_path, 'rb') as f:
                    records = pickle.load(f)
            except KeyboardInterrupt:
                raise
            except:
                count += 1
                self._sleep()
            else:
                break
        if count==self.max_try:
            raise MaxTryIOError(store_path, count)
        return records

    def _safe_write(self, records: dict, store_path: str):
        r"""Safely writes a file."""
        count = 0
        while count<self.max_try:
            try:
                with open(store_path, 'wb') as f:
                    pickle.dump(records, f)
            except KeyboardInterrupt:
                raise
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

    def keys(self) -> str:
        r"""A generator for keys."""
        if self.store_dir is None:
            for key in self.__store__.keys():
                yield key
        else:
            store_paths = self._store_paths()
            random.shuffle(store_paths) # randomization to avoid cluster conflict
            for store_path in store_paths:
                records = self._safe_read(store_path)
                for key in records.keys():
                    yield key

    def values(self) -> Any:
        r"""A generator for values."""
        if self.store_dir is None:
            for val in self.__store__.values():
                yield val
        else:
            store_paths = self._store_paths()
            random.shuffle(store_paths)
            for store_path in store_paths:
                records = self._safe_read(store_path)
                for val in records.values():
                    yield val

    def items(self) -> tuple[str, Any]:
        r"""A generator for items."""
        if self.store_dir is None:
            for key, val in self.__store__.items():
                yield key, val
        else:
            store_paths = self._store_paths()
            random.shuffle(store_paths)
            for store_path in store_paths:
                records = self._safe_read(store_path)
                for key, val in records.items():
                    yield key, val

    def get_key(self, val: dict) -> Optional[str]:
        r"""Returns the key of a record.

        Returns
        -------
        key:
            The key of record being searched for. ``None`` if not found.

        """
        assert self.is_config, "Key search is implemented for config records only."
        h_val = Config(val)
        for key, val in self.items():
            if val==h_val:
                return key
        return None

    def add(self, val: dict) -> str:
        r"""Adds a new item if it has not already existed.

        Returns
        -------
        key:
            The key of added record.

        """
        assert self.is_config, "Item adding is implemented for config records only."
        key = self.get_key(val)
        if key is None:
            key = self._new_key()
            self[key] = val
        return key

    def pop(self, key: str) -> Any:
        r"""Pops out an item by key."""
        if self.store_dir is None:
            return self.__store__.pop(key, None)
        else:
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

    def prune(self) -> list[str]:
        r"""Removes corrupted files.

        Returns
        -------
        removed:
            The name of removed files.

        """
        removed = []
        if self.store_dir is None:
            print("No external store detected.")
        else:
            store_paths = self._store_paths()
            verbose, tic = None, time.time()
            for i, store_path in enumerate(store_paths, 1):
                try:
                    self._safe_read(store_path)
                except:
                    print("{} corrupted, will be removed".format(store_path))
                    os.remove(store_path)
                    removed.append(store_path[(-4-self.path_len):-4])
                if i%(-(-len(store_paths)//10))==0 or i==len(store_paths):
                    toc = time.time()
                    if verbose is None:
                        # display progress if estimated time is longer than 20 mins
                        verbose = (toc-tic)/i*len(store_paths)>1200
                    if verbose:
                        print("{} ({})".format(
                            progress_str(i, len(store_paths)),
                            time_str(toc-tic, progress=i/len(store_paths)),
                        ))
            if removed:
                print(f"{len(removed)} corrupted files removed.")
            else:
                print("No corrupted files detected.")
        return removed

    def get_duplicates(self) -> dict[Config, list[str]]:
        r"""Returns all duplicate records.

        Returns
        -------
        duplicates:
            A dictionary of which the key is the duplicate record value, while
            the value is the list of keys associated with it.

        """
        assert self.is_config, "Duplicate detection is implemented for config records only."
        inv_dict = {} # dictionary for inverse archive
        for key, val in self.items():
            if val in inv_dict:
                inv_dict[val].append(key)
            else:
                inv_dict[val] = [key]
        duplicates = dict((val, keys) for val, keys in inv_dict.items() if len(keys)>1)
        return duplicates

    def copy_to(self, dst_dir: str, keys: Optional[set[str]] = None):
        r"""Clones the archive to a new folder.

        Args
        ----
        target_dir:
            Path to the target directory.
        keys:
            Keys of the records to be cloned. Clone everything if is ``None``.

        """
        if keys is None:
            keys = set()
        os.makedirs(dst_dir, exist_ok=True)
        for file_name in os.listdir(dst_dir):
            _records = None
            if file_name.endswith('.axv'):
                assert len(file_name)==(self.path_len+4), (
                    f"Data in the target directory {dst_dir} have inconsistent file name length "
                    f"than {self.path_len}."
                )
                if _records is None or len(_records)==0:
                    _records = self._safe_read(f'{dst_dir}/{file_name}')
            if len(_records)>0:
                _key = next(iter(_records.keys()))
                assert len(_key)==self.key_len, (
                    f"Data in the target directory {dst_dir} have inconsistent key length than "
                    f"{self.key_len}."
                )

        file_names = [
            f for f in os.listdir(self.store_dir) if f.endswith('.axv') and len(f)==(self.path_len+4)
        ]
        random.shuffle(file_names)
        for file_name in file_names:
            src_path = f'{self.store_dir}/{file_name}'
            dst_path = f'{dst_dir}/{file_name}'

            src_records = self._safe_read(src_path)
            src_records = dict((k, v) for k, v in src_records.items() if len(keys)==0 or k in keys)

            if os.path.exists(dst_path):
                dst_records = self._safe_read(dst_path)
            else:
                dst_records = {}
            for key in src_records:
                assert key not in dst_records, (
                    f"Existing key '{key}' detected in the target directory {dst_dir}, cloning "
                    "operation aborted."
                )
            dst_records.update(src_records)
            self._safe_write(dst_records, dst_path)

    def to_internal(self):
        r"""Moves external storage to internal."""
        if self.store_dir is not None:
            self.__store__ =  dict((key, val) for key, val in self.items())
            self.store_dir = None
