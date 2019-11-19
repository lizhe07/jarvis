# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 20:51:54 2019

@author: Zhe
"""

from jarvis import __version__

from setuptools import setup, find_packages
setup(
    name="jarvis",
    version=__version__,
    author='Zhe Li',
    python_requires='>=3',
    packages=find_packages(),
)
