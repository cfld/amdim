#!/usr/bin/env/python

import exline
from setuptools import setup, find_packages

VERSION = exline.__version__

setup(
    name='amdim',
    author='Ben Johnson',
    author_email='ben@canfield.io',
    classifiers=[],
    description='amdim',
    keywords=['amdim'],
    license='MIT',
    packages=find_packages(),
    version=VERSION
)