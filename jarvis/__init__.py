from pathlib import Path
with open(Path(__file__).parent/'VERSION.txt', 'r') as f:
    __version__ = f.readline().split('"')[1]

from .config import Config
from .archive import Archive
from .manager import Manager
