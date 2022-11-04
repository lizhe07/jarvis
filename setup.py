from setuptools import setup, find_packages

with open('jarvis/version.txt', 'r') as f:
    __version__ = f.readline().split('"')[1]

setup(
    name="jarvis",
    version=__version__,
    author='Zhe Li',
    python_requires='>=3.9',
    packages=find_packages(),
    install_requires=['torch', 'pyyaml'],
)
