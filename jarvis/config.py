import sys, yaml, time, random
from pathlib import Path
from importlib import import_module
from typing import Optional, Union
from collections.abc import Callable

def _format(val):
    r"""Formats an object to configuration style.

    Dicts are converted to Config objects and tuples are converted to lists.

    """
    if isinstance(val, dict):
        return Config({k: _format(v) for k, v in val.items()})
    elif isinstance(val, (list, tuple)):
        return [_format(v) for v in val]
    elif isinstance(val, set):
        return set(_format(v) for v in val)
    else:
        return val


class Config(dict):
    r"""Configuration class.

    Dot notation is supported. When a callable '_target_' is specified, an
    object can be instantiated according to the configuration.

    """

    def __init__(self, config: Optional[dict] = None):
        super().__init__()
        for key, val in (config or {}).items():
            self[key] = val

    def _is_valid_key(self, key):
        return isinstance(key, str) and not (
            key in super().__dir__()
            or len(key)==0 or key[0]=='.' or key[-1]=='.'
            or '..' in key
        )

    def __setitem__(self, key, val):
        assert self._is_valid_key(key)
        val = _format(val)
        dot_pos = key.find('.')
        if dot_pos==-1:
            super().__setitem__(key, val)
        else:
            key_head, key_tail = key[:dot_pos], key[dot_pos+1:]
            if key_head in self:
                assert isinstance(self[key_head], Config)
                self[key_head][key_tail] = val
            else:
                self[key_head] = Config({key_tail: val})

    def __getitem__(self, key):
        dot_pos = key.find('.')
        if dot_pos==-1:
            return super().__getitem__(key)
        else:
            key_head, key_tail = key[:dot_pos], key[dot_pos+1:]
            assert key_head in self and isinstance(self[key_head], Config)
            return self[key_head][key_tail]

    def __setattr__(self, key, val):
        if self._is_valid_key(key):
            self[key] = val
        else:
            super().__setattr__(key, val)

    def __getattr__(self, key):
        if self._is_valid_key(key):
            return self[key]
        else:
            return super().__getattr__(key)

    def update(self, config):
        for key, val in config.items():
            self[key] = val

    def flatten(self) -> dict:
        r"""Returns a flattened dictionary."""
        f_dict = {}
        for p_key, p_val in self.items(): # parent key and value
            if isinstance(p_val, Config):
                for c_key, c_val in p_val.flatten().items(): # child key and value
                    f_dict[f'{p_key}.{c_key}'] = c_val
            else:
                f_dict[p_key] = p_val
        return f_dict

    def fill(self,
        config: Union[dict, Path, str, None] = None,
        defaults: Union[dict, Path, str, None] = None,
    ):
        r"""Fills in values from another configuration accompanied with default
        values.

        This method first copies new values from the provided config dictionary
        to itself, then looks up in a dictionary for default values of relevant
        '_target_' values recursively.

        Args
        ----
        config:
            The base configuration to take values from, can be a dictionary or a
            path to a yaml file. Existing keys will NOT be overwritten.
        defaults:
            The default values for relevant classes/functions mentioned in the
            current config object.

        """
        if isinstance(config, (Path, str)):
            with open(config, 'r') as f:
                config = yaml.safe_load(f)
        config = Config(config)
        for key, val in config.flatten().items():
            try: # 'in' and 'setdefault' do not work
                self[key]
            except:
                self[key] = val

        if defaults is None:
            return
        elif isinstance(defaults, (Path, str)):
            with open(defaults, 'r') as f:
                defaults = yaml.safe_load(f)
        assert isinstance(defaults, dict)
        for key, val in self.items():
            if isinstance(val, Config):
                if '_target_' in val and val._target_ in defaults:
                    _config = defaults[val._target_]
                else:
                    _config = {}
                val.fill(_config, defaults)

    def clone(self):
        config = Config()
        for key, val in self.flatten().items():
            config[key] = val
        return config

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
        key, val = _argv.split('=')
        config[key] = yaml.safe_load(val)
    # add random wait to avoid conflicts in the following I/O operations on cluster
    if 'max_wait' in config:
        wait = config.pop('max_wait')*random.random()
        if wait>0:
            print("Wait for {:.1f} secs before execution.".format(wait))
            time.sleep(wait)
    return config


def _locate(path: str) -> Callable:
    r"""Returns a callable object from string.

    This is a simplified version of 'hydra._internal.utils._locate' (==1.3.1).

    """
    parts = path.split('.')
    part = parts[0]
    obj = import_module(part)
    for m in range(1, len(parts)):
        part = parts[m]
        try:
            obj = getattr(obj, part)
        except:
            obj = import_module('.'.join(parts[:(m+1)]))
    return obj
