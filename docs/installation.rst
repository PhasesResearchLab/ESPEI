Installation
============


Anaconda
--------

Installing ESPEI from PyPI (by ``pip install espei``) is **not** supported.
Please install ESPEI using Anaconda package manager.
If you do not have Anaconda installed, we recommend you download and install `Miniconda3 <https://docs.conda.io/en/latest/miniconda.html>`_.
Optionally, you may also be want to install `JupyterLab <https://jupyter.org>`_ at the same time.

.. code-block:: bash

    conda install -c conda-forge espei jupyterlab


.. _installing-development-versions:

Development versions
--------------------

To make changes to the ESPEI source code, the development version must be
installed. If you'll need to make changes to pycalphad simultaneously, follow
the `instructions to install the development version of pycalphad <https://pycalphad.org/docs/latest/INSTALLING.html#development-versions-advanced-users>`_ first.

We recommend that users interested in developing packages work in a
`Conda virtual environment <https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html>`_.
The steps below will create an environment called ``espei-dev``, which can be
entered using ``conda activate espei-dev``.
The environment name is arbitrary - you can use whatever name you prefer.
If you already have an environment, use ``conda env update ...`` instead of
``conda env create ...`` when following the steps below.

ESPEI uses `Git <https://git-scm.com/book/en/v2>`_ and
`GitHub <https://github.com/PhasesResearchLab/ESPEI>`_ for version control.
Windows users: if you do not have a working version of Git,
`download it here <https://git-scm.com/downloads>`_ first.

To install the latest development version of ESPEI, use Anaconda to install the
required dependencies using the ``environment-dev.yml`` file found in ESPEI's
repository, then install ESPEI as editable using ``pip``.:

.. code-block:: bash

    git clone https://github.com/phasesresearchlab/espei.git
    cd espei
    conda env create -n espei-dev --file environment-dev.yml
    conda activate espei-dev
    pip install --no-deps -e .

Optionally, you may also be want to install `JupyterLab <https://jupyter.org>`_.
Each environment needs its own copy of JupyterLab, so you will need to install
it even if it is already installed in your ``base`` environment.
You can install it to your ``espei-dev`` environment by running


.. code-block:: bash

    conda install -n espei-dev jupyterlab

Every time you open a new terminal window, you start in the ``base``
environment. You can activate your ``espei-dev`` environment by running

.. code-block:: bash

    conda activate espei-dev

and return to your ``base`` environment by running

.. code-block:: bash

    conda deactivate espei-dev

With the development version installed and your environment activated,
you can run the automated tests by running

.. code-block:: bash

   pytest

If the test suite passes, you are ready to start using the development version
or making changes yourself! See the
:ref:`guide for contributing to ESPEI <Contributing guide>` to learn more.
If any tests fail, please report the failure to the
`ESPEI issue tracker on GitHub <https://github.com/phasesresearchlab/espei/issues>`_.

To upgrade your development version to the latest version, run ``git pull``
from the top level ESPEI directory (the directory containing the ``setup.py``
file).