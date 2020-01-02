# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 23:03:10 2019

@author: Zhe
"""

import os, pickle, random, time

class Archive:
    def __init__(self, save_dir, r_id_len=8, f_name_len=2, max_try=10, pause=0.1,
                 record_hashable=False):
        r"""Data structure for storing records.
        
        An Archive object stores records in a dictionary manner, but using multiple files
        to save data separately according to the record ID prefix.
        
        Args:
            save_dir (str): directory of where the files are stored.
            r_id_len (int): record ID length. ID for each record is a fixed length string
                composed of '0'-'9' and 'A'-'F'.
            f_name_len (int): file name length. Each external file is named as 'XX.axv' in
                which 'XX' is a string of length `f_name_len`. The file contains a
                dictionary of records whose ID startswith the file name.
            max_try (int): maximum number of tries to read or write file via pickle.
            pause (float): pause time between each try of read or write.
            record_hashable (bool): whether record is hashable.
        
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.save_dir = save_dir
        self.r_id_len, self.f_name_len = r_id_len, f_name_len
        assert self.f_name_len<=self.r_id_len, 'file name length should be no greater than record ID length'
        self.max_try, self.pause = max_try, pause
        self.record_hashable = record_hashable
        if self.record_hashable:
            self.init_hash()
    
    def __repr__(self):
        return 'Archive object saved in {}\nfile name length is {}'.format(self.save_dir, self.f_name_len)
    
    def __str__(self):
        return 'Archive object saved in {}'.format(self.save_dir)
    
    def __len__(self):
        return self.record_num()
    
    def _safe_read(self, r_file):
        r"""Safely writes a file.
        
        Args:
            r_file (str): archive file path.
        
        Returns:
            records (dict): the dictionary saved in the archive file.
        
        """
        count = 0
        while count<self.max_try:
            try:
                with open(r_file, 'rb') as f:
                    records = pickle.load(f)
            except:
                count += 1
                time.sleep(self.pause)
            else:
                break
        if count==self.max_try:
            raise RuntimeError('max number ({}) of reading tried and failed'.format(count))
        return records
    
    def _safe_write(self, records, r_file):
        r"""Safely writes a file.
        
        Args:
            r_file (str): archive file path.
        
        """
        count = 0
        while count<self.max_try:
            try:
                with open(r_file, 'wb') as f:
                    pickle.dump(records, f)
            except:
                count += 1
                time.sleep(self.pause)
            else:
                break
        if count==self.max_try:
            raise RuntimeError('max number ({}) of writing tried and failed'.format(count))
    
    def _random_id(self):
        r"""Generate a random ID.
        
        Returns:
            a string composed of '0'-'9' and 'A'-'F'.
        
        """
        return ''.join(['{:X}'.format(random.randrange(16)) for _ in range(self.r_id_len)])
    
    def _new_id(self):
        r"""Generate a new ID.
        
        Returns:
            a string that hasn't been used as record ID yet.
        
        """
        r_id = None
        while True:
            r_id = self._random_id()
            if not self.has_id(r_id):
                break
        return r_id
    
    def _r_file(self, r_id):
        r"""Returns the file path of corresponding record ID.
        
        Args:
            r_id (str): record ID.
        
        """
        return os.path.join(self.save_dir, r_id[:self.f_name_len]+'.axv')
    
    def _r_file_names(self):
        r"""Returns the names of all files.
        
        Returns:
            a list of strings, containing names of all files.
        
        """
        return [f for f in os.listdir(self.save_dir) \
                if f.endswith('.axv') and len(f)==(self.f_name_len+4)]
    
    def _r_files(self):
        r"""Returns the paths of all files.
        
        Returns:
            a list of strings, containing paths of all files.
        
        """
        return [os.path.join(self.save_dir, f) for f in self._r_file_names()]
    
    def record_num(self):
        r"""Returns number of records.
        
        """
        count = 0
        for r_file in self._r_files():
            records = self._safe_read(r_file)
            count += len(records)
        return count
    
    def file_num(self):
        r"""Returns number of files.
        
        """
        return len(self._r_files())
    
    def clear(self):
        r"""Removes all files.
        
        """
        r_files = self._r_files()
        for r_file in r_files:
            os.remove(r_file)
        if self.record_hashable:
            os.remove(self._hash_file())
        os.removedirs(self.save_dir)
    
    def update_files(self, new_f_name_len):
        r"""Updates files with new name length.
        
        """
        old_heads = [f[:self.f_name_len] for f in self._r_file_names()]
        if new_f_name_len<self.f_name_len:
            new_heads = list(set([h[:new_f_name_len] for h in old_heads]))
            for new_h in new_heads:
                records = {}
                r_files = [os.path.join(self.save_dir, old_h+'.axv') for old_h in old_heads \
                           if old_h.startswith(new_h)]
                for r_file in r_files:
                    records.update(self._safe_read(r_file))
                    os.remove(r_file)
                self._safe_write(records, os.path.join(self.save_dir, new_h+'.axv'))
        if new_f_name_len>self.f_name_len:
            for old_h in old_heads:
                r_file = os.path.join(self.save_dir, old_h+'.axv')
                records = self._safe_read(r_file)
                os.remove(r_file)
                new_heads = list(set([key[:new_f_name_len] for key in records]))
                for new_h in new_heads:
                    self._safe_write(
                            dict((key, records[key]) for key in records if key.startswith(new_h)),
                            os.path.join(self.save_dir, new_h+'.axv')
                            )
        self.f_name_len = new_f_name_len
    
    def remove_corrupted(self):
        r"""Removes corrupted files.
        
        """
        count = 0
        for r_file in self._r_files():
            try:
                self._safe_read(r_file)
            except:
                print('{} corrupted, will be removed'.format(r_file))
                os.remove(r_file)
                if self.record_hashable:
                    raise NotImplementedError
                count += 1
        if count==0:
            print('no corrupted files detected.')
        else:
            print('{} corrupted files removed.'.format(count))
    
    def has_id(self, r_id):
        r"""Checks if certain record ID exists.
        
        """
        r_file = self._r_file(r_id)
        if not os.path.exists(r_file):
            return False
        records = self._safe_read(r_file)
        return r_id in records
    
    def fetch_matched(self, matcher=None, mode='all'):
        r"""Fetches matched record(s).
        
        Args:
            matcher (function): a function takes a record as inputs and returns boolean.
                If a matcher is not specified, all records will be returned.
            mode (str): fetching mode, can be 'all' or 'random'.
        
        Returns:
            If mode is 'all', a list of record IDs (possibly empty) that fits the matcher
            is returned.
            If mode is 'random', one random record ID that fits the matcher is returned.
            If no such record exists, a `None` is returned.
        
        """
        def _matched_ids(records, matcher):
            if matcher is None:
                return list(records.keys())
            else:
                return [r_id for r_id, record in records.items() if matcher(record)]
        r_files = self._r_files()
        random.shuffle(r_files)
        
        if mode=='all':
            matched_ids = []
            for r_file in r_files:
                records = self._safe_read(r_file)
                matched_ids += _matched_ids(records, matcher)
            return matched_ids
        if mode=='random':
            for r_file in r_files:
                records = self._safe_read(r_file)
                matched_ids = _matched_ids(records, matcher)
                if matched_ids:
                    return random.choice(matched_ids)
            return None
    
    def all_ids(self):
        r"""Returns all record IDs.
        
        """
        return self.fetch_matched()
    
    def fetch_one(self):
        r"""Fetches one random record.
        
        """
        return self.fetch_record(self.fetch_matched(mode='random'))
    
    def fetch_id(self, record):
        r"""Fetches the ID of a record.
        
        If records are hashable, record ID is looked up from hash file. Otherwise, record
        ID is searched for by direct value comparison, and a random one is returned if
        multiple matches exist.
        
        """
        if self.record_hashable:
            r_hash = Archive.record_hash(record)
            if r_hash in self.hash_dict:
                return self.hash_dict[r_hash]
            else:
                return None
        else:
            return self.fetch_matched(lambda r: r==record, 'random')
    
    def fetch_record(self, r_id):
        r"""Fetches the record.
        
        """
        r_file = self._r_file(r_id)
        if not os.path.exists(r_file):
            raise RuntimeError('{} does not exist'.format(r_id))
        records = self._safe_read(r_file)
        if r_id in records:
            return records[r_id]
        else:
            raise RuntimeError('{} does not exist'.format(r_id))
    
    def find_duplicates(self):
        r"""Finds duplicate records.
        
        This method is performed by direct comparison, whether the records are hashable or
        not.
        
        Returns:
            a list of duplicate records.
        
        """
        all_records, duplicates = [], []
        for r_file in self._r_files():
            records = self._safe_read(r_file)
            for r in records.values():
                if r in all_records:
                    duplicates.append(r)
                else:
                    all_records.append(r)
        return duplicates
    
    @staticmethod
    def record_hash(r):
        r"""Returns hash of a record.
        
        Definition of 'hashable' is different than python default. If the record is one of
        the basic form, e.g. int, float and str, default hash is returned. If the record
        is one container such as list, tuple, set or dict, a new container containing
        hashes of its elements are created and converted to tuple or frozenset to get the
        hash.
        
        Args:
            r: a record in the archive.
        
        Returns:
            hash of the record.
        
        """
        if r is None or isinstance(r, bool) or isinstance(r, int) or isinstance(r, float) or isinstance(r, str):
            return hash(r)
        if isinstance(r, list):
            hashes = [Archive.record_hash(x) for x in r]
            return hash(('list', tuple(hashes)))
        if isinstance(r, tuple):
            hashes = [Archive.record_hash(x) for x in r]
            return hash(('tuple', tuple(hashes)))
        if isinstance(r, set):
            hashes = [Archive.record_hash(x) for x in r]
            return hash(frozenset(hashes))
        if isinstance(r, dict):
            hashes = [Archive.record_hash(x) for x in r.items()]
            return hash(frozenset(hashes))
    
    def init_hash(self):
        r"""Initializes the record hash dictionary.
        
        A new hash file is created from the existing archive records.
        
        """
        if not self.record_hashable:
            raise RuntimeError('this archive is not for hashable record')
        self.hash_dict = {}
        for r_file in self._r_files():
            records = self._safe_read(r_file)
            for r_id, r in records.items():
                r_hash = Archive.record_hash(r)
                if r_hash in self.hash_dict:
                    print('hash conflict found, {} will overwrite {}'.format(r_id, self.hash_dict[r_hash]))
                self.hash_dict[r_hash] = r_id
        print('record hash dictionary has been initialized')
    
    def assign(self, r_id, record):
        r"""Assigns a record to an ID.
        
        If `r_id` already exists, old record will be replaced. If `r_id` does not exist, a
        new record will be added.
        
        Args:
            r_id (str): record ID.
            record: the record to be assigned to `r_id`.
        
        """
        r_file = self._r_file(r_id)
        if os.path.exists(r_file):
            records = self._safe_read(r_file)
        else:
            records = {}
        records[r_id] = record
        self._safe_write(records, r_file)
        if self.record_hashable:
            r_hash = Archive.record_hash(record)
            self.hash_dict[r_hash] = r_id
    
    def add(self, record, check_duplicate=True):
        r"""Adds a new record.
        
        Args:
            record: the record to add.
            check_duplicate (bool): whether to check duplicate before adding with a new
                record ID.
        
        Returns:
            r_id (str): record ID of the record. If `check_duplicate` is `True` and at
                least one same record does exist, the record ID of any one duplicate is
                returned.
        
        """
        if check_duplicate:
            r_id = self.fetch_id(record)
            if r_id is not None:
                return r_id
        r_id = self._new_id()
        self.assign(r_id, record)
        return r_id
    
    def remove(self, r_id):
        r"""Removes a record by ID.
        
        """
        if self.has_id(r_id):
            r_file = self._r_file(r_id)
            records = self._safe_read(r_file)
            records.pop(r_id)
            if records:
                self._safe_write(records, r_file)
            else:
                os.remove(r_file)
