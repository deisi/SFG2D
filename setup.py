#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
from setuptools import setup
from setuptools.extension import Extension
from glob import glob


with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    #'Click>=6.0',
    #'numpy',
    #'matplotlib',
    #'scipy',
    #'pandas',
    #'ipython',
]

test_requirements = [
    # TODO: put package test requirements here
]

setup(
    name='sfg2d',
    version='0.6.0',
    description="Python Toolkit for Analsys of 2d-sfg spectra",
    long_description=readme + '\n\n' + history,
    author="Malte Deiseroth",
    author_email='deiseroth@mpip-mainz.mpgs.de',
    url='https://github.com/deisi/sfg2d',
    packages=[
        'sfg2d',
        'sfg2d.data',
        'sfg2d.data.calib',
        'sfg2d.io',
        'sfg2d.utils',
    ],
    package_dir={'sfg2d':
                 'sfg2d'},
    entry_points={
        'console_scripts': [
            'sfg2d=sfg2d.cli:main'
        ]
    },
    package_data={
        'sfg2d': ['data/calib/params_Ne_670.npy']
    },
    include_package_data=True,
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords='sfg2d',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Scientists',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.5',
    ],
    test_suite='tests',
    tests_require=test_requirements,
)
