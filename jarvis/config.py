import sys, yaml, time, random
from copy import deepcopy
from importlib import import_module
from typing import Optional, Callable

from .hashable import HashableList, HashableSet, HashableDict, HashableArray
from .utils import flatten, nest

def _convert(spec):
    r"""Converts an object to Config value.

    If `spec` is a container, use the hashable container class for nesting.

    """
    if isinstance(spec, (list, tuple)) and not isinstance(spec, HashableArray):
        # replace tuple with list since Config usually deals with yaml file
        return HashableList([_convert(v) for v in spec])
    if isinstance(spec, set):
        return HashableSet([_convert(v) for v in spec])
    if isinstance(spec, dict):
        return Config(spec)
    return spec


class Config(HashableDict):
    r"""Configuration class.

    Compared to `HashableDict`, a few new features are implemented. Items can be
    accessed via dot expression. An object can be instantiated if a callable
    `_target_` field exists.

    """

    def __init__(self, config: Optional[dict] = None):
        super(Config, self).__init__(config or {})
        for key, val in self.items():
            self[key] = _convert(val)

        # deal with existing nested keys
        keys = list(self.keys())
        for key in keys:
            if isinstance(key, str) and '.' in key:
                self.update(Config._create(key.split('.'), self.pop(key)))

    @classmethod
    def _create(cls, keys: list[str], val) -> dict:
        r"""Creates nested dictionary.

        Args
        ----
        keys:
            A list of strings specifying the nested key.
        val:
            Dictionary value.

        """
        if len(keys)==1:
            return {keys[0]: val}
        else:
            return {keys[0]: cls._create(keys[1:], val)}

    def __getattr__(self, key):
        r"""Returns configuration value."""
        try:
            return self[key]
        except:
            return getattr(super(Config, self), key)

    def __setitem__(self, key, val):
        super(Config, self).__setitem__(key, _convert(val))

    def __setattr__(self, key, val):
        r"""Sets configuration value."""
        try:
            self[key] = val
        except:
            setattr(super(Config, self), key, val)

    def get(self, key):
        # TODO implement for nested key as dot string
        return super(Config, self).get(key)

    def clone(self):
        r"""Returns a clone of the configuration."""
        return deepcopy(self)

    def flatten(self):
        r"""Returns a flat configuration."""
        return Config(flatten(self))

    def nest(self):
        r"""Returns a nested configuration."""
        return Config(nest(self))

    def update(self, config: dict, ignore_unknown: bool = False):
        r"""Updates the configuration.

        Args
        ----
        config:
            The new configuration, can be partially defined.
        ignore_unknown:
            Whether unknown fields should be ignored.

        """
        f_config = self.flatten()
        for key, val in Config(config).clone().flatten().items():
            if key in f_config or not ignore_unknown:
                f_config[key] = val
        super(Config, self).update(f_config.nest())

    def fill(self, defaults: dict):
        r"""Returns a configuration with filled default values."""
        f_config = self.flatten()
        for key, val in Config(defaults).clone().flatten().items():
            if key not in f_config:
                f_config[key] = val
        return f_config.nest()

    def instantiate(self, *args, **kwargs): # one-level instantiation
        r"""Instantiates an object using the configuration.

        When a callable object is specified by a string in '_target_', this
        method will use other key-values as arguments to instantiate a new
        object. '_target_' should be a string that can be imported from the
        working directory, it can be a class or a function.

        """
        assert '_target_' in self, "A callable needs to be specified as '_target_'."
        try:
            _target = _locate(self._target_)
            assert callable(_target)
        except:
            raise RuntimeError(f"The '_target_' ({_target}) is not callable.")
        _kwargs = {k: v for k, v in self.items() if k!='_target_'}
        _kwargs.update(kwargs)
        return _target(*args, **_kwargs)


def from_cli(argv: Optional[list[str]] = None):
    r"""Constructs a configuration from command line."""
    if argv is None:
        argv = sys.argv[1:]
    config = Config()
    for _argv in argv:
        assert _argv.count('=')==1, f"Expects one '=' in '{_argv}'."
        keys, val = _argv.split('=')
        keys = keys.split('.')
        val = yaml.safe_load(val)
        config.update(Config._create(keys, val), ignore_unknown=False)

    # implement random wait to avoid file reading conflicts in cluster usage
    if 'max_wait' in config:
        wait = config.pop('max_wait')*random.random()
        if wait>0:
            print("Wait for {:.1f} secs before execution.".format(wait))
            time.sleep(wait)
    return config


def _locate(path: str) -> Callable:
    r"""Returns a callable object from string.

    This is a simplied version of hydra._internal.utils._locate.

    """
    parts = path.split('.')

    part = parts[0]
    try:
        obj = import_module(part)
    except:
        raise
    for m in range(1, len(parts)):
        part = parts[m]
        try:
            obj = getattr(obj, part)
        except:
            try:
                obj = import_module('.'.join(parts[:(m+1)]))
            except:
                raise
    return obj
