===============================
SFG2D
===============================


.. image:: https://img.shields.io/pypi/v/sfg2d.svg
        :target: https://pypi.python.org/pypi/sfg2d

.. image:: https://img.shields.io/travis/deisi/sfg2d.svg
        :target: https://travis-ci.org/deisi/sfg2d

.. image:: https://readthedocs.org/projects/sfg2d/badge/?version=latest
        :target: https://sfg2d.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://pyup.io/repos/github/deisi/sfg2d/shield.svg
     :target: https://pyup.io/repos/github/deisi/sfg2d/
     :alt: Updates


Python Toolkit for Analsys if 2d-sfg spectra


* Free software: MIT license
* Documentation: https://sfg2d.readthedocs.io.

Installation
------------
Download package from github:
`git clone git@github.com:deisi/SFG2D.git`

Install package with:
`pip install sfg2d`

If you want to use the dashboards run:
`jupyter nbextension enable --py --sys-prefix bqplot`
and
`jupyter dashboards quick-setup --sys-prefix`

Dexcription
-----------
This is a toolkit to analyse mostly sfg2d data with python3 using jupyter
notebooks. It is not really generic, but rather specific to the problems
and tasks I have to encounter here at the MPIP. It is nowhere near stable
and things might change drastically at any point in time. If you want to use
this I encurage you to create you own fork and work with your own version.
At the time of wrtiting, there is also almost no documentation available.
I think this will cahnge in time when things become more stable but up to now.
Its not worth documenting much since it might be different next time anyway.


Features
--------
- Import data from Veronica, Viktor and .spe (version 2 and 3) files.
- Datastructure based on pandas DataFrames to organize ans structure data.
- A dashboard for the viktor lab.


Credits
---------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage

The .spe file importer is based on the code of James Battat, Kasey Russell
and

For the strucuture of the module I was inspired by 
