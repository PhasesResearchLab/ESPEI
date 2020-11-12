Installation
============


Anaconda
--------

Installing ESPEI from PyPI (by ``pip install espei``) is **not** supported. Please install ESPEI using Anaconda.

.. code-block:: bash

    conda install -c pycalphad -c conda-forge --yes espei


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