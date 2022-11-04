with open('version.txt', 'r') as f:
    __version__ = f.readline().split('"')[1]

from .config import Config
from .archive import Archive
from .manager import Manager
