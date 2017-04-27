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


Python Toolkit to analyze 2d-sfg spectra and more.


* Free software: MIT license
* Documentation: https://sfg2d.readthedocs.io.

Pre Install
-----------
The following is mostly a collection of commands and steps one might need to do. If you run
all of them brainlessly, this will most likely not work.

Easiest is, to just install anaconda on you platform. You can get it from:

Get Anaconda https://www.continuum.io/downloads#windows
tested with Anaconda 4.2.0

Otherwise you have to setup your own python environment with a command prompt and **pip** as package manager. `Cygwin` might be a way to go here, but I haven't tried.

Install git for Windows from: https://git-scm.com/download/win

The following commands can all be executed from within the **Anaconda Promt**.

Install
-------
Download source with::

    git clone git@github.com:deisi/SFG2D.git

or download manually and extract the folder somewhere.

From within the source folder, run::

    pip install .

This installs the minimal user version of sfg2d. It currently supports:

- Windows 32 and 64 bit python 3.5
- Linux 64 bit and python 3.5

To install on other platforms, see the following section about installing the
development version.

If you actually want to do something with the package, like really use it, **install the additional requirements**. From within the package folder::

    pip install -r requirements.txt

If you want to use the dashboards run::

    jupyter nbextension enable --py --sys-prefix bqplot

and::

    jupyter dashboards quick-setup --sys-prefix


Install Development Version from Source
----------------------------------------

Some dependencies must be pre installed manually::

    pip install numpy cython

Download package from github::

    git clone git@github.com:deisi/SFG2D.git

From within the package folder::

  pip install -r requirements.txt
  pip install -r requirements_dev.txt
  pip install -r requirements_fit.txt
  pip install -e .

If the requirements_fit.txt installation fails. You must install **iminuit** and **probfit** manually.

If you want to use the dashboards run::

  jupyter nbextension enable --py --sys-prefix bqplot

and::

  jupyter dashboards quick-setup --sys-prefix

Installation on Windows
-----------------------
Get Anaconda as described in the **Pre Install** section.

Iminuit
  Download from http://www.lfd.uci.edu/~gohlke/pythonlibs/#iminuit
  Install with::
  
        cd Downloads
        pip install iminuit-*.whl

Probfit
  Currently probfit must be compiled manually. To do so, we need visualstudio.
  Download from http://landinghub.visualstudio.com/visual-cpp-build-tools
  Start installation and grab a cup of coffee. For me this took +1 hour to complete....
  
  Install probfit by hand.
  Download from https://github.com/iminuit/probfit
  Install with::
  
      cd probfit
      pip install .
    
Install sfg2d with::

    pip install -r requirements.txt
    pip install -r requirements_dev.txt
    pip install -e .

Virtualenv
-----------
On Arch Linux
    Install python-virtualenv::
    
      pacman -S python-virtualenv
    
    Install virtualenvwrapper::
    
      pacman -S python-virtualenvwrapper
    
    Setup virtualenvwrapper put::
    
      source virtualenvwrapper.sh
    
    in ``~/.bashrc`` or ``~/.profile``
    
    Setup a virutalenv on linux with::
    
        mkvirtualenv -a ~/SFG2D/ -p python3.5 sfg2d

    Enter virtualenv with::

        workon sfg2d
    
    note: change the path at ``~/SFG2D`` to the location of sfg2d package.
    matplotlib needs a plotting backend to work from within a virtualenv, thus::
    
        toggleglobalsitepackages pyqt
    
    or use `--system-site-packages` during the setup of the virtualenv. See
    http://matplotlib.org/faq/virtualenv_faq.html for insights.

    And follow the already described installation procedure.

On Windows with Conda
    Run::

      conda create -n sfg2d python+3.5 anaconda

    Enter virtual env with::

      activate sfg2d

    And follow the above described installation procedure.

Jupuyter notebooks and virtualenvs
    Up to now I have only done this under linux.
    To run jupyter/ipython kernel in the virtualenv I adopted the info from
    https://help.pythonanywhere.com/pages/IPythonNotebookVirtualenvs/
    
    First create a new kernel with::
    
      ipython3 kernelspec install-self --user
    
    Now edit this kernel to use the sfg2d virtualenv by first moving it with::
    
      mv ~/.local/share/jupyter/kernels/python3 ~/.local/share/jupyter/kernels/sfg2d
    
    And then edit the ``~/.local/share/jupyter/kernels/sfg2d/kernel.json``
    and adjust the content to be similar to::
    
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
    
    The value of the ``display_name`` field is what jupyter will know the kernel by. The important line is the first argument of the ``argv``. This must be the full path to the python3 binary within the virutalenv.
    
    
    Test setup by running a notebook server::
    
        jupyter notebook
    
    Create a New Notebook and choose the sfg2d kernel from the drop-down menu and try to run::
    
      import sfg2d
    
    If there is trouble with missing PyQt, install it system wide and then link PyQt4
    with the virtalenv. PyQt4 cant be installed via pip.
    e.g.::
    
      ln -s /usr/lib/python3.5/site-packages/PyQt4 ~/.virtualenv/sfg2d/lib/python3.5/site-packages/

Officesetup
-----------
Requirements are installed into the home folder with pip3 using::

    pip3 install --user --upgrade -r requirements.txt
    pip3 install --user --upgrade -r requirements_dev.txt

Then created a virtual env with::

    mkvirtualenv --system-site-packages --python=/usr/bin/python3.5 -a ~/sfg2d -r ~/sfg2d/requirements.txt sfg2d

And installed sfg2d from within the virtualenv::

    pip install -e .

Because requirements are installed into the user folder outside of the virtualenv,
we need to install the javascript nbextensions with::

    jupyter nbextension enable --py --user widgetsnbextension

Description
-----------
This is a toolkit to analyze mostly sfg2d data with python3 using jupyter
notebooks. It is not really generic, but rather specific to the problems
and tasks I have to encounter here at the MPIP. It is nowhere near stable
and things might change drastically at any point in time. If you want to use
this I encourage you to create you own fork and work with your own version.
At the time of writing, there is also almost no documentation available.
I think this will change in time when things become more stable but up to now.
Its not worth documenting much since it might be different next time anyway.


Features
--------
- Import data from Veronica, Viktor and .spe (version 2 and 3) files.
- Data-structure based on pandas DataFrames to organize ans structure data.
- A dashboard for the Viktor lab.
- A minimal fit gui (dashboards/fit_starter/fit_starter.ipynb)
- Import ``.spe`` spectra files
- Import ``.ntb`` surface tension files



Credits
---------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _audreyr/cookiecutter-pypackage: https://github.com/audreyr/cookiecutter-pypackage

The .spe file importer is based on the code of James Battat, Kasey Russell
and

For the structure of the module I was inspired by the Scikit packages.
