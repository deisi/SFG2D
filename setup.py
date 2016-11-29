#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup
from Cython.Build import cythonize


with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'Click>=6.0',
    'numpy',
    'matplotlib',
    'scipy',
    'pandas',
    'cython',
    #'notebook', # must be installed in the system virtualenv is not enouth
    #'bqplot', # system ...
    #'widgetsnbextension',
    #'ipywidgets',
    #'jupyter',
]

test_requirements = [
    # TODO: put package test requirements here
]

setup(
    name='sfg2d',
    version='0.5.0',
    description="Python Toolkit for Analsys of 2d-sfg spectra",
    long_description=readme + '\n\n' + history,
    author="Malte Deiseroth",
    author_email='deiseroth@mpip-mainz.mpgs.de',
    url='https://github.com/deisi/sfg2d',
    packages=[
        'sfg2d',
    ],
    package_dir={'sfg2d':
                 'sfg2d'},
    entry_points={
        'console_scripts': [
            'sfg2d=sfg2d.cli:main'
        ]
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
    ext_modules=cythonize('./sfg2d/utils/static_c.pyx')
)
