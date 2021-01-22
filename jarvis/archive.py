# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 23:03:10 2019

@author: Zhe
"""

import os, pickle, random, time
from .utils import to_hashable


class Archive():
    r"""Custom dictionary that stores data externally.

    Args
    ----
    save_dir: str
        The directory for saving data.
    key_len: int
        The length of keys.
    pth_len: int
        The length of external file names, should be no greater than `key_len`.
    max_try: int
        The maximum number of trying to read/write.
    pause: float
        The time (seconds) between two consecutive tries.
    hashable: bool
        Whether the record is hashable.

    """
    _alphabet = ['{:X}'.format(i) for i in range(16)]

    def __init__(self, save_dir, key_len=8, pth_len=2, max_try=30, pause=0.5, hashable=False):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.save_dir = save_dir
        self.key_len, self.pth_len = key_len, pth_len
        assert self.key_len>=self.pth_len, "file name length should be no greater than key length"
        self.max_try, self.pause = max_try, pause

    def __repr__(self):
        return 'Archive object saved in {}'.format(self.save_dir)

    def __setitem__(self, key, val):
        save_pth = self._save_pth(key)
        records = self._safe_read(save_pth) if os.path.exists(save_pth) else {}
        records[key] = to_hashable(val) if self.hashable else val
        self._safe_write(records, save_pth)

    def __getitem__(self, key):
        save_pth = self._save_pth(key)
        assert os.path.exists(save_pth), f"{key} does not exist"
        records = self._safe_read(save_pth)
        assert key in records, f"{key} does not exist"
        return records[key]

    def __contains__(self, key):
        if not self._is_valid_key(key):
            return False
        save_pth = self._save_pth(key)
        if not os.path.exists(save_pth):
            return False
        records = self._safe_read(save_pth)
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

    def _save_pth(self, key):
        r"""Returns the path of external file associated with a key.

        """
        assert self._is_valid_key(key), f"invalid key '{key}' encountered"
        return os.path.join(self.save_dir, key[:self.pth_len]+'.axv')

    def _save_pths(self):
        r"""Returns all valid external files in the directory.

        """
        return [os.path.join(self.save_dir, f) for f in os.listdir(self.save_dir) \
               if f.endswith('.axv') and len(f)==(self.pth_len+4)]

    def _safe_read(self, save_pth):
        r"""Safely reads a file.

        This is designed for cluster use. An error is raised when too many
        attempts have failed.

        """
        count = 0
        while count<self.max_try:
            try:
                with open(save_pth, 'rb') as f:
                    records = pickle.load(f)
            except:
                count += 1
                time.sleep(self.pause)
            else:
                break
        if count==self.max_try:
            raise RuntimeError('max number ({}) of reading tried and failed'.format(count))
        return records

    def _safe_write(self, records, save_pth):
        r"""Safely writes a file.

        """
        count = 0
        while count<self.max_try:
            try:
                with open(save_pth, 'wb') as f:
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
        for save_pth in sorted(self._save_pths()):
            records = self._safe_read(save_pth)
            for key in records.keys():
                yield key

    def values(self):
        r"""Returns a generator for values.

        """
        for save_pth in sorted(self._save_pths()):
            records = self._safe_read(save_pth)
            for val in records.values():
                yield val

    def items(self):
        r"""Returns a generator for items.

        """
        for save_pth in sorted(self._save_pths()):
            records = self._safe_read(save_pth)
            for key, val in records.items():
                yield key, val

    def add(self, val, check_duplicate=False):
        r"""Adds a new item.

        Args
        ----
        val:
            The value to add.
        check_duplicate: bool
            Whether to check duplicates. Only implemented for hashable records.

        """
        if check_duplicate:
            assert self.hashable, "'check_duplicate' implemented for hashable records only"
            for key, _val in self.items():
                if _val==to_hashable(val):
                    return key
        key = self._new_key()
        self[key] = val
        return key

    def pop(self, key):
        r"""Pops out an item.

        """
        save_pth = self._save_pth(key)
        assert os.path.exists(save_pth), f"{key} does not exist"
        records = self._safe_read(save_pth)
        val = records.pop(key)
        if records:
            self._safe_write(records, save_pth)
        else:
            os.remove(save_pth)
        return val

    def prune(self):
        r"""Removes corrupted files.

        """
        count = 0
        for save_pth in self._save_pths():
            try:
                self._safe_read(save_pth)
            except:
                print('{} corrupted, will be removed'.format(save_pth))
                os.remove(save_pth)
                count += 1
        if count==0:
            print('no corrupted files detected')
        else:
            print('{} corrupted files removed'.format(count))
