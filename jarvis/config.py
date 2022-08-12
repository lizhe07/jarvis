from typing import Optional

from .hashable import HashableDict
from .utils import flatten, nest

class Config(HashableDict):

    def __init__(self, config: Optional[dict] = None):
        super(Config, self).__init__(config or {})
        for key, val in self.items():
            if isinstance(val, HashableDict):
                self[key] = Config(val)

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
