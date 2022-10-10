import sys, yaml
from typing import Optional

from .hashable import HashableDict
from .utils import flatten, nest

class Config(HashableDict):

    def __init__(self, config: Optional[dict] = None):
        super(Config, self).__init__(config or {})
        for key, val in self.items():
            if isinstance(val, HashableDict):
                self[key] = Config(val)

    def __getattr__(self, key):
        try:
            return self[key]
        except:
            return super(Config, self).__getattr__(key)

    def __setattr__(self, key, val):
        try:
            super(Config, self).__setattr__(key, val)
        except:
            self[key] = val

    def flatten(self):
        r"""Returns a flat configuration."""
        return Config(flatten(self))

    def nest(self):
        r"""Returns a nested configuration."""
        return Config(nest(self))

    def update(self, config: dict, ignore_unknown: bool = False):
        r"""Returns an updated configuration.

        Args
        ----
        config:
            The new configuration, can be partially defined.
        ignore_unknown:
            Whether unknown fields should be ignored.

        """
        f_config = self.flatten()
        for key, val in Config(config).flatten().items():
            if key in f_config or not ignore_unknown:
                f_config[key] = val
        super(Config, self).update(f_config.nest())

    def fill(self, defaults: dict):
        r"""Fills default values."""
        f_config = self.flatten()
        for key, val in Config(defaults).flatten().items():
            if key not in f_config:
                f_config[key] = val
        return f_config.nest()

def from_cli(argv: Optional[list[str]] = None):
    if argv is None:
        argv = sys.argv[1:]
    config = Config()

    def create(keys: list[str], val):
        if len(keys)==1:
            return {keys[0]: val}
        else:
            return {keys[0]: create(keys[1:], val)}

    for _argv in argv:
        assert _argv.count('=')==1, f"Expects one '=' in '{_argv}'."
        keys, val = _argv.split('=')
        keys = keys.split('.')
        val = yaml.safe_load(val)
        config.update(create(keys, val), ignore_unknown=False)
    return config
