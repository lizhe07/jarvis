from setuptools import setup, find_packages

with open('jarvis/VERSION.txt', 'r') as f:
    VERSION = f.readline().split('"')[1]

setup(
    name="jarvis",
    version=VERSION,
    author='Zhe Li',
    python_requires='>=3.9',
    packages=find_packages(),
    package_data={'jarvis': ['VERSION.txt']},
    install_requires=['numpy', 'torch', 'pyyaml'],
)
