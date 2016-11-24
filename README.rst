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

Manual install dependencues
---------------------------
Some dependencies must be installed manually due to some bugs
`pip install numpy cython`
and you will need pyqt4 but it doesn't install from pip so:
`pacman -S python-pyqt4`

Installation
------------
Download package from github::

    git clone git@github.com:deisi/SFG2D.git

Install package with::

  pip install sfg2d

If you want to use the dashboards run::

  jupyter nbextension enable --py --sys-prefix bqplot

and::

  jupyter dashboards quick-setup --sys-prefix

Installation on Windows
-----------------------

Install iminuit by hand.
    Download from http://www.lfd.uci.edu/~gohlke/pythonlibs/#iminuit
    Install with::

          pip install iminuit-*.whl

To build probfit on windows one needs visual studio
    Download from http://landinghub.visualstudio.com/visual-cpp-build-tools
Install probfit by hand.
    Download from https://github.com/iminuit/probfit
    Install with::

        pip install .
    


Virtual Env
-----------
Install python-virtualenv::

  pacman -S python-virtualenv

Install virtualenvwrapper::

  pacman -S python-virtualenvwrapper

Setup virtualenvwrapper put::

  source virtualenvwrapper.sh

in ``~/.bashrc`` or ``~/.profile``

Setup a virutal env::

 mkvirtualenv --system-site-packages -a ~/SFG2D -p python3 sfg2d

Install Dependencies as described above.
Install sfg2d in editable mode::

 pip install -e .

note: change the path at ``~/SFG2D`` to the location of sfg2d package

enter into virtualenv with::

  workon sfg2d

Install sfg2d in editable mode::

  pip install -e .

To run jupyter/ipython kernel in the virtualenv I adopted the info from
https://help.pythonanywhere.com/pages/IPythonNotebookVirtualenvs/

First create a new kernel with::

  ipython3 kernelspec install-self --user

Now edit this kernel to use the sfg2d virtualenv by first moving it with::

  mv ~/.local/share/jupyter/kernels/python3 ~/.local/share/jupyter/kernels/sfg2d

And then edit the ``~/.local/share/jupyter/kernels/sfg2d/kernel.json``
and adjust the content to be simiar to::

    json
    {
     "argv": [
      "/home/malte/.virtualenvs/sfg2d/bin/python3",
      "-m",
      "ipykernel",
      "-f",
      "{connection_file}"
     ],
     "display_name": "sfg2d",
     "language": "python"
    }

The value of the ``display_name`` field is what jupyter will know the kernel by. The important line is the first arguemtn of the ``argv``. This must be the full path to the python3 binary within the virutalenv.


Test setup by running a notebook server::

    jupyter notebook

Create a New Notebook and choose the sfg2d kernel from the Dropdown menu and try to run::

  import sfg2d

If there is trouble with missing PyQt, install it system wide and then link PyQt4
with the virtalenv. PyQt4 cant be installed via pip.
e.g.::

  ln -s /usr/lib/python3.5/site-packages/PyQt4 ~/.virtualenv/sfg2d/lib/python3.5/site-packages/


Description
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
- A minimal fit gui
- Import ``.spe`` spectra files
- Import ``.ntb`` surface tension files



Credits
---------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _audreyr/cookiecutter-pypackage: https://github.com/audreyr/cookiecutter-pypackage

The .spe file importer is based on the code of James Battat, Kasey Russell
and

For the strucuture of the module I was inspired by 
