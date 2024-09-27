import sys, yaml, time, random
import numpy as np
from pathlib import Path
from importlib import import_module
from collections.abc import Callable

def _format(val):
    r"""Formats an object to configuration style.

    Dicts are converted to Config objects.

    """
    if isinstance(val, dict):
        return Config({k: _format(v) for k, v in val.items()})
    elif isinstance(val, list):
        return [_format(v) for v in val]
    elif isinstance(val, tuple):
        return tuple(_format(v) for v in val)
    elif isinstance(val, set):
        return set(_format(v) for v in val)
    else:
        return val


class Config(dict):
    r"""Configuration class.

    Dot notation is supported. When a callable '_target_' is specified, an
    object can be instantiated according to the configuration.

    """

    def __init__(self, config: dict|None = None):
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
        assert self._is_valid_key(key), f"'{key}' is not a valid key."
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
        try:
            return self[key]
        except:
            return super().__getattr__(key)

    def get(self, key, val=None):
        try:
            val = self[key]
        except:
            pass
        if isinstance(val, dict):
            return Config(val)
        else:
            return val

    def pop(self, key):
        dot_pos = key.find('.')
        if dot_pos==-1:
            return super().pop(key)
        else:
            key_head, key_tail = key[:dot_pos], key[dot_pos+1:]
            val = self[key_head].pop(key_tail)
            if len(self[key_head])==0:
                super().pop(key_head)
            return val

    def flatten(self) -> dict:
        r"""Returns a flattened dictionary."""
        f_dict = {}
        for p_key, p_val in self.items(): # parent key and value
            if isinstance(p_val, Config) and len(p_val)>0:
                for c_key, c_val in p_val.flatten().items(): # child key and value
                    f_dict[f'{p_key}.{c_key}'] = c_val
            else:
                f_dict[p_key] = p_val
        return f_dict

    def asdict(self) -> dict:
        r"""Returns basic dict version."""
        # TODO deal with list and set
        n_dict = {}
        for key, val in self.items():
            if isinstance(val, Config):
                n_dict[key] = val.asdict()
            else:
                n_dict[key] = val
        return n_dict

    def update(self, config: dict|Path|str|None):
        r"""Overwrites from a new config."""
        config = _load_dict(config)
        for key, val in config.flatten().items():
            self[key] = val

    def fill(self, config: dict|Path|str|None):
        r"""Fills value from a new config."""
        config = _load_dict(config)
        if not('_target_' in self and '_target_' in config and self._target_!=config._target_):
            for key, val in config.items():
                if key in self:
                    if isinstance(self[key], Config) and isinstance(val, Config):
                        self[key].fill(val)
                else:
                    self[key] = val
        return self

    def clone(self):
        r"""Returns a clone of the configuration."""
        return Config(self.flatten())

    def instantiate(self, **kwargs):
        return instantiate(dict(
            **{k: self[k] for k in self if k not in kwargs}, **kwargs,
        ))

    # aliasing 'instantiate' for function calling
    call = instantiate


def from_cli(argv: list[str]|None = None) -> Config:
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


def _load_dict(config: dict|Path|str|None) -> Config:
    if isinstance(config, (Path, str)):
        with open(config, 'r') as f:
            config = yaml.safe_load(f)
    config = Config(config)
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


def instantiate(spec):
    r"""Instantiates an object."""
    if isinstance(spec, dict):
        if '_target_' in spec:
            _target = _locate(spec['_target_'])
            return _target(**{k: v for k, v in spec.items() if k!='_target_'})
        else:
            return {k: instantiate(v) for k, v in spec.items()}
    elif isinstance(spec, list):
        return [instantiate(val) for val in spec]
    elif isinstance(spec, tuple):
        return tuple(instantiate(val) for val in spec)
    else:
        return spec


def choices2configs(choices: dict|Path|str, num_configs: int|None = None) -> list[Config]:
    r"""Generates configs on a mesh grid.

    Args
    ----
    choices:
        A config-like object, except the leaf nodes are list of value choices.
        The list of random configs will be constructed as the outer product. If
        a file path is provided, it is expected to be a yaml file.
    num_configs:
        The number of configs to return. If not provided, will return all
        possible combinations of choices.

    Returns
    -------
    configs:
        The list of configs randomly selected from the mesh grid.

    """
    if isinstance(choices, (Path, str)):
        with open(choices, 'r') as f:
            choices = yaml.safe_load(f)
    choices = Config(choices).flatten()
    keys = list(choices.keys())
    vals = [list(choices[key]) for key in keys]
    dims = [len(val) for val in vals]
    total_num = np.prod(dims)
    num_configs = total_num if num_configs is None else min(total_num, num_configs)

    configs = []
    for idx in random.sample(range(total_num), num_configs):
        sub_idxs = np.unravel_index(idx, dims)
        config = Config()
        for i, key in enumerate(keys):
            config[key] = vals[i][sub_idxs[i]]
        configs.append(config)
    return configs
