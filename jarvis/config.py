import sys, yaml
from copy import deepcopy
from importlib import import_module
from typing import Optional, Callable

from .hashable import HashableList, HashableSet, HashableDict, HashableArray
from .utils import flatten, nest

def _convert(spec):
    if isinstance(spec, (list, tuple)) and not isinstance(spec, HashableArray):
        return HashableList([_convert(v) for v in spec])
    if isinstance(spec, set):
        return HashableSet([_convert(v) for v in spec])
    if isinstance(spec, dict):
        return Config(spec)
    return spec


class Config(HashableDict):

    def __init__(self, config: Optional[dict] = None):
        super(Config, self).__init__(config or {})
        for key, val in self.items():
            self[key] = _convert(val)

    def __getattr__(self, key):
        try:
            return self[key]
        except:
            return super(Config, self).__getattr__(key)

    def __setattr__(self, key, val):
        try:
            self[key] = _convert(val)
        except:
            super(Config, self).__setattr__(key, val)

    def clone(self):
        return deepcopy(self)

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

    def instantiate(self, *args, **kwargs): # one-level instantiation
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


def instantiate(spec, *args, **kwargs):
    if isinstance(spec, HashableList) and not isinstance(spec, HashableArray):
        return [instantiate(v) for v in spec]
    if isinstance(spec, HashableSet):
        return set([instantiate(v) for v in spec])
    if isinstance(spec, Config) and  '_target_' in spec:
        try:
            _target = _locate(spec['_target_'])
            assert callable(_target)
        except:
            raise RuntimeError(f"The '_target_' ({spec['_target_']}) is not callable.")
        _kwargs = {}
        for k, v in spec.items():
            if k=='_target_':
                continue
            _kwargs[k] = instantiate(v)
        _kwargs.update(kwargs)
        return _target(*args, **_kwargs)
    return spec
