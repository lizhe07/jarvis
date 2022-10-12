from setuptools import setup, find_packages

setup(
    name="jarvis",
    version="0.5.1",
    author='Zhe Li',
    python_requires='>=3.9',
    packages=find_packages(),
    install_requires=['torch', 'pyyaml'],
)
