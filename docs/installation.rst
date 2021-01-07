Installation
============


Anaconda
--------

Installing ESPEI from PyPI (by ``pip install espei``) is **not** supported. Please install ESPEI using Anaconda.

.. code-block:: bash

    conda install -c conda-forge espei


Development versions
--------------------

To make changes to the ESPEI source code, the development version must be
installed. If you'll need to make changes to pycalphad simultaneously, follow
the `instructions to install the development version of pycalphad <https://pycalphad.org/docs/latest/INSTALLING.html#development-versions-advanced-users>`_ first.

To install the latest development version of ESPEI, use Anaconda to install the
required dependencies using the ``environment-dev.yml`` file found in ESPEI's
GitHub repository, then install ESPEI using ``pip``.:

.. code-block:: bash

    git clone https://github.com/phasesresearchlab/espei.git
    cd espei
    conda env update --file environment-dev.yml
    pip install --no-deps -e .

With the development version installed, you can run the automated tests by
running ``pytest``.

To upgrade your development version to the latest version, run ``git pull``
from the top level ESPEI directory (the directory containing the ``setup.py``
file).