import random, time
from collections.abc import Iterable
from .utils import time_str
from .archive import Archive


class BaseJob:
    r"""Base class for batch processing.

    The job is associated with different directories storing configurations,
    status, results and previews of all works. Method `main` need to be
    implemented by child class.

    """

    def __init__(self,
        store_dir: Optional[str] = None,
        read_only: bool = False,
        c_path_len: int = 2, c_pause: float = 0.5,
        r_path_len: int = 3, r_pause: float = 5.,
    ):
        r"""
        Args
        ----
        store_dir:
            Directory for storage. Archives including `configs`, `stats`,
            `results` and `previews` will be saved in separate directories.
        read_only:
            If the job is read-only or not.
        c_pth_len, c_pause:
            Path length and pause time for `configs`, as well as `stats` and
            `previews`.
        r_path_len, r_pause:
            Path length and pause time for `results`.

        """
        self.store_dir = store_dir
        if self.store_dir is None:
            self.configs = Archive(hashable=True)
            self.stats = Archive()
            self.results = Archive()
            self.previews = Archive()
        else:
            self.configs = Archive(
                f'{self.store_dir}/configs', path_len=c_path_len, max_try=60,
                pause=c_pause, hashable=True,
            )
            self.stats = Archive(
                f'{self.store_dir}/stats', path_len=c_path_len, pause=c_pause,
            )
            self.results = Archive(
                f'{self.store_dir}/results', path_len=r_path_len, pause=r_pause,
            )
            self.previews = Archive(
                f'{self.store_dir}/previews', path_len=c_path_len, pause=c_pause,
            )
        self.read_only = read_only
        if self.store_dir is not None and self.read_only:
            for axv in [self.configs, self.stats, self.previews]:
                axv.to_internal()

    def main(self, config, verbose: int = 1):
        r"""Main function that needs implementation by subclasses.

        Args
        ----
        config:
            A configuration dict for the work.
        verbose:
            Level of information display. No message will be printed when
            'verbose' is no greater than 0.

        Returns
        -------
        result, preview:
            Archive records that will be saved in `self.results` and
            `self.previews` respectively.

        """
        raise NotImplementedError

    def is_completed(self, key, strict=False):
        r"""Returns whether a work is completed.

        Args
        ----
        key: str
            Key of the work.
        strict: bool
            Whether to check `results` and `previews`.

        """
        try:
            stat = self.stats[key]
        except:
            return False
        else:
            if not stat['completed']:
                return False
        if strict and not(key in self.results and key in self.previews):
            return False
        return True

    def process(self, config, verbose=1):
        r"""Processes one work."""
        assert not self.read_only, "This is a read-only job."
        if verbose:
            print("--------")
        key = self.configs.add(config)
        if self.is_completed(key):
            if verbose:
                print(f"{key} already exists.")
            return self.results[key], self.previews[key]
        if verbose:
            print(f"Processing {key}...")
        tic = time.time()
        self.stats[key] = {'tic': tic, 'toc': None, 'completed': False}
        result, preview = self.main(config, verbose)
        self.results[key] = result
        self.previews[key] = preview
        toc = time.time()
        self.stats[key] = {'tic': tic, 'toc': toc, 'completed': True}
        if verbose:
            print("{} processed ({}).".format(key, time_str(toc-tic)))
            print("--------")
        return result, preview

    def to_process(self, config, patience=float('inf')):
        r"""Returns whether to process a work."""
        key = self.configs.add(config)
        try:
            stat = self.stats[key]
        except:
            return True # stats[key] does not exist
        t_last = stat['tic'] if stat['toc'] is None else stat['toc']
        if not stat['completed'] and (time.time()-t_last)/3600>patience:
            return True # running time exceeds patience
        else:
            return False # work is being processed

    def batch(self,
        configs: Iterable,
        num_works: int = 0,
        max_wait: float = 1.,
        patience: float = float('inf'),
        verbose: int = 1,
    ):
        r"""Batch processing.

        Args
        ----
        configs:
            An iterable object containing work configurations. Some are
            potentially processed already.
        num_works:
            The number of works to process. If it is 0, the processing stops
            when no work is left in `configs`.
        max_wait:
            Maximum waiting time in the beginning, in seconds.
        patience:
            Patience time for processing an incomplete work, in hours. The last
            modified time of a work is recorded in `self.stats`.

        """
        random_wait = random.random()*max_wait
        if random_wait>0 and verbose:
            print("Random wait {:.1f}s...".format(random_wait))
        time.sleep(random_wait)

        count = 0
        for config in configs:
            if self.to_process(config, patience):
                self.process(config, verbose)
                count += 1
            if num_works>0 and count==num_works:
                if verbose:
                    print("{} works processed".format(num_works))
                return count
        if verbose:
            print("All works are processed or being processed.")
