Installation
============


Anaconda
--------

Installing ESPEI from PyPI (by ``pip install espei``) is **not** supported. Please install ESPEI using Anaconda.

.. code-block:: bash

    conda install -c pycalphad -c conda-forge --yes espei

After installation, you must turn off dask's work stealing.
Change the work stealing setting to ``distributed.scheduler.work-stealing: False`` in dask's configuration.
See configuration_ below for more details.


Development versions
--------------------

To make changes to the ESPEI source code, the development version must be
installed. If you'll need to make changes to pycalphad simultaneously, follow
the `instructions to install the development version of pycalphad <https://pycalphad.org/docs/latest/INSTALLING.html#development-versions-advanced-users>`_ first.

To install the latest development version of ESPEI, use Anaconda to download
ESPEI and all of the required dependencies, then remove the installed release
version of ESPEI and replace it with the latest version from GitHub:

.. code-block:: bash

    git clone https://github.com/phasesresearchlab/espei.git
    cd espei
    conda install -c pycalphad -c conda-forge espei
    conda remove --force espei
    pip install --no-deps -e .

Upgrading ESPEI later requires you to run ``git pull`` in this directory.

After installation, you must turn off dask's work stealing.
Change the work stealing setting to ``distributed.scheduler.work-stealing: False`` in dask's configuration.
See configuration_ below for more details.


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
