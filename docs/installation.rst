Installation
============


Anaconda (recommended)
----------------------

ESPEI does not require any special compiler, but several dependencies do.
Therefore it is suggested to install ESPEI from conda-forge.

.. code-block:: bash

    conda install -c pycalphad -c msys2 -c conda-forge --yes espei

After installation, you must turn off dask's work stealing.
Change the work stealing setting to ``distributed.scheduler.work-stealing: False`` in dask's configuration.
See configuration_ below for more details.

PyPI
----

Before you install ESPEI via PyPI, be aware that pycalphad and
emcee must be compiled and pycalphad requires an external
dependency of `Ipopt <https://projects.coin-or.org/Ipopt>`_.

.. code-block:: bash

    pip install espei

After installation, you must turn off dask's work stealing.
Change the work stealing setting to ``distributed.scheduler.work-stealing: False`` in dask's configuration.
See configuration_ below for more details.

Development versions
--------------------

You may install ESPEI however you like, but here we suggest using
Anaconda to download all of the required dependencies. This
method installs ESPEI with Anaconda, removes specifically the
ESPEI package, and replaces it with the package from GitHub.

.. code-block:: bash

    git clone https://github.com/phasesresearchlab/espei.git
    cd espei
    conda install espei
    conda remove --force espei
    pip install -e .

Upgrading ESPEI later requires you to run ``git pull`` in this directory.

After installation, you must turn off dask's work stealing.
Change the work stealing setting to ``distributed.scheduler.work-stealing: False`` in dask's configuration.
See configuration_ below for more details.

Testing
~~~~~~~

If you will be developing in ESPEI, it is likely that you'll want to run the
test suite or build the documentation. The tests require the addition of the
pytest and mock packages, while building the docs requires sphinx and
sphinx_rtd_theme. These can be installed by running

.. code-block:: bash

   conda install mock pytest sphinx sphinx_rtd_theme
   
The tests can be run from the root directory of the cloned repository:

.. code-block:: bash

   pytest tests/

Documentation
~~~~~~~~~~~~~

The docs can be built by running the docs/Makefile (or docs/make.bat on
Windows). Then Python can be used to serve the html files in the _build
directory and you can visit ``http://localhost:8000`` in your broswer to
see the built documentation.

For Unix systems:

.. code-block:: bash

   cd docs
   make html
   cd _build/html
   python -m http.server

Windows:

.. code-block:: bash

   cd docs
   make.bat html
   cd _build\html
   python -m http.server
        

.. _configuration:

Configuration
-------------

ESPEI uses dask-distributed to parallelize ESPEI.

After installation, you must turn off dask's work stealing!
Change the your dask configuration file to look something like:


.. code-block:: YAML

   distributed:
     version: 2
     scheduler:
       work-stealing: False


The configuration file paths can be found by running ``from espei.utils import get_dask_config_paths; get_dask_config_paths()`` in a Python interpreter.
If multiple configurations are found, the latter configurations take precendence over the former, so check them from last to first.
See the `dask-distributed documentation <https://distributed.readthedocs.io/en/latest/configuration.html>`_ for more.
