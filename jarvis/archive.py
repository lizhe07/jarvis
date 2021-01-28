# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 23:03:10 2019

@author: Zhe
"""

import os, pickle, random, time
from .utils import to_hashable


class Archive():
    r"""Dictionary-like class that stores data externally.

    Args
    ----
    store_dir: str
        The directory for storing data. If `store_dir` is ``None``, an internal
        dictionary `__store__` is used.
    key_len: int
        The length of keys.
    pth_len: int
        The length of external file names, should be no greater than `key_len`.
    max_try: int
        The maximum number of trying to read/write external files.
    pause: float
        The time (seconds) between two consecutive read/write attempts.
    hashable: bool
        Whether the record is hashable.

    """
    _alphabet = ['{:X}'.format(i) for i in range(16)]

    def __init__(self, store_dir=None, key_len=8, pth_len=2, max_try=30, pause=0.5, hashable=False):
        self.store_dir, self.key_len = store_dir, key_len
        if self.store_dir is None:
            self.__store__ = {}
        else:
            if not os.path.exists(self.store_dir):
                os.makedirs(self.store_dir)
            self.pth_len, self.max_try, self.pause = pth_len, max_try, pause
            assert key_len>=pth_len, "file name length should be no greater than key length"
        self.hashable = hashable

    def __repr__(self):
        if self.store_dir is None:
            return 'Archive object with no external storage'
        else:
            return 'Archive object stored in {}'.format(self.store_dir)

    def __setitem__(self, key, val):
        if self.store_dir is None:
            self.__store__[key] = to_hashable(val) if self.hashable else val
        else:
            store_pth = self._store_pth(key)
            records = self._safe_read(store_pth) if os.path.exists(store_pth) else {}
            records[key] = to_hashable(val) if self.hashable else val
            self._safe_write(records, store_pth)

    def __getitem__(self, key):
        if self.store_dir is None:
            assert key in self.__store__, f"{key} does not exist"
            return self.__store__[key]
        else:
            store_pth = self._store_pth(key)
            assert os.path.exists(store_pth), f"{key} does not exist"
            records = self._safe_read(store_pth)
            assert key in records, f"{key} does not exist"
            return records[key]

    def __contains__(self, key):
        if self.store_dir is None:
            return key in self.__store__
        else:
            try:
                store_pth = self._store_pth(key)
            except:
                return False # invalid key
            if not os.path.exists(store_pth):
                return False
            records = self._safe_read(store_pth)
            return key in records

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(list(self.keys()))

    def _is_valid_key(self, key):
        r"""Returns if a key is valid.

        """
        if len(key)!=self.key_len:
            return False
        for c in key:
            if c not in Archive._alphabet:
                return False
        return True

    def _store_pth(self, key):
        r"""Returns the path of external file associated with a key.

        """
        assert self._is_valid_key(key), f"invalid key '{key}' encountered"
        return os.path.join(self.store_dir, key[:self.pth_len]+'.axv')

    def _store_pths(self):
        r"""Returns all valid external files in the directory.

        """
        return [os.path.join(self.store_dir, f) for f in os.listdir(self.store_dir)
                if f.endswith('.axv') and len(f)==(self.pth_len+4)]

    def _safe_read(self, store_pth):
        r"""Safely reads a file.

        This is designed for cluster use. An error is raised when too many
        attempts have failed.

        """
        count = 0
        while count<self.max_try:
            try:
                with open(store_pth, 'rb') as f:
                    records = pickle.load(f)
            except:
                count += 1
                time.sleep(self.pause)
            else:
                break
        if count==self.max_try:
            raise RuntimeError('max number ({}) of reading tried and failed'.format(count))
        return records

    def _safe_write(self, records, store_pth):
        r"""Safely writes a file.

        """
        count = 0
        while count<self.max_try:
            try:
                with open(store_pth, 'wb') as f:
                    pickle.dump(records, f)
            except:
                count += 1
                time.sleep(self.pause)
            else:
                break
        if count==self.max_try:
            raise RuntimeError('max number ({}) of writing tried and failed'.format(count))

    def _random_key(self):
        r"""Returns a random key.

        """
        return ''.join(random.choices(Archive._alphabet, k=self.key_len))

    def _new_key(self):
        r"""Returns a new key.

        """
        while True:
            key = self._random_key()
            if key not in self:
                break
        return key

    def keys(self):
        r"""Returns a generator for keys.

        """
        if self.store_dir is None:
            for key in self.__store__.keys():
                yield key
        else:
            for store_pth in sorted(self._store_pths()):
                records = self._safe_read(store_pth)
                for key in records.keys():
                    yield key

    def values(self):
        r"""Returns a generator for values.

        """
        if self.store_dir is None:
            for val in self.__store__.values():
                yield val
        else:
            for store_pth in sorted(self._store_pths()):
                records = self._safe_read(store_pth)
                for val in records.values():
                    yield val

    def items(self):
        r"""Returns a generator for items.

        """
        if self.store_dir is None:
            for key, val in self.__store__.items():
                yield key, val
        else:
            for store_pth in sorted(self._store_pths()):
                records = self._safe_read(store_pth)
                for key, val in records.items():
                    yield key, val

    def get_key(self, val):
        r"""Returns the key of a record.

        Implemented for hashable records only. ``None`` is returned if the item
        is not found.

        """
        assert self.hashable, "key search is implemented for hashable records only"
        h_val = to_hashable(val)
        for key, val in self.items():
            if val==h_val:
                return key
        return None

    def add(self, val):
        r"""Adds a new item if it has not already existed.

        """
        key = self.get_key(val)
        if key is None:
            key = self._new_key()
            self[key] = val
        return key

    def pop(self, key):
        r"""Pops out an item.

        """
        assert key in self, f"{key} does not exist"
        if self.store_dir is None:
            return self.__store__.pop(key)
        else:
            store_pth = self._store_pth(key)
            records = self._safe_read(store_pth)
            val = records.pop(key)
            if records:
                self._safe_write(records, store_pth)
            else:
                os.remove(store_pth) # remove empty external file
            return val

    def prune(self):
        r"""Removes corrupted files.

        """
        if self.store_dir is None:
            print('no external store detected')
        else:
            count = 0
            for store_pth in self._store_pths():
                try:
                    self._safe_read(store_pth)
                except:
                    print('{} corrupted, will be removed'.format(store_pth))
                    os.remove(store_pth)
                    count += 1
            if count==0:
                print('no corrupted files detected')
            else:
                print('{} corrupted files removed'.format(count))
