from typing import Optional

from .hashable import HashableDict

class Config(HashableDict):

    def __init__(self, config: Optional[dict] = None):
        super(Config, self).__init__(config or {})
        for key, val in self.items():
            if isinstance(val, HashableDict):
                self[key] = Config(val)

    def flatten(self):
        r"""Returns a flat configuration.

        A nested configuration like `{'A': {'B', val}}` will be converted to
        `{('B', '@', 'A'), val}`.

        Returns
        -------
        flat_config:
            A flat configuration with tuple keys for hierarchy.

        """
        flat_config = {}
        for key, val in self.items():
            if isinstance(val, Config) and len(val)>0:
                for subkey, subval in val.flatten().items():
                    flat_config[(subkey, '@', key)] = subval
            else:
                flat_config[key] = val
        return Config(flat_config)

    def nest(self):
        r"""Returns a nested configuration.

        A flat configuration like `{('B', '@', 'A'), val}` will be converted to
        `{'A': {'B', val}}`.

        Returns
        -------
        nested_config:
            A nested configuration possibly contains dictionaries as values.

        """
        nested_config = {}
        for key, val in self.items():
            if isinstance(key, tuple) and len(key)==3 and key[1]=='@':
                subkey, _, parkey = key
                if parkey not in nested_config:
                    nested_config[parkey] = {}
                nested_config[parkey][subkey] = val
            else:
                if key in nested_config:
                    if isinstance(val, dict):
                        nested_config[key].update(val)
                    else:
                        raise ValueError(f"Conflicting values found for {key}.")
                else:
                    nested_config[key] = val
        for key, val in nested_config.items():
            if isinstance(val, dict):
                nested_config[key] = Config(val).nest()
        return Config(nested_config)

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
        f_config = self.flatten()
        for key, val in Config(defaults).flatten().items():
            if key not in f_config:
                f_config[key] = val
        return f_config.nest()
