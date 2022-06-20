from jarvis import __version__
from setuptools import setup, find_packages

setup(
    name="jarvis",
    version=__version__,
    author='Zhe Li',
    python_requires='>=3.9',
    packages=find_packages(),
    install_requires=['numpy', 'torch'],
)
