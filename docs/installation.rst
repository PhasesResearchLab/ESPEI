Installation
============


pip (recommended)
-----------------

To install ESPEI from PyPI using pip:

.. code-block:: bash

   pip install -U pip
   pip install -U espei

A recommended best practice is to install Python packages into a virtual environment.
To create an environment and install ESPEI on Linux and macOS/OSX:

.. code-block:: bash

   python -m venv calphad-env
   source calphad-env/bin/activate
   pip install -U pip
   pip install -U pycalphad

On Windows:

.. code-block:: batch

   python -m venv calphad-env
   calphad-env\Scripts\activate
   pip install -U pip
   pip install -U pycalphad


Anaconda
--------

If you prefer using Anaconda, ESPEI is distributed on conda-forge.
If you do not have Anaconda installed, we recommend you download and install `Miniconda3 <https://docs.conda.io/en/latest/miniconda.html>`_.
ESPEI can be installed with the conda package manager by:

.. code-block:: bash

    conda install -c conda-forge espei


.. _installing-development-versions:

Development versions
--------------------

To make changes to the ESPEI source code, the development version must be
installed. If you'll need to make changes to pycalphad simultaneously, follow
the `instructions to install the development version of pycalphad <https://pycalphad.org/docs/latest/INSTALLING.html#development-versions-advanced-users>`_ first.

We strongly recommend that users interested in developing packages work in a virtual environment.
The steps below will create an environment called ``calphad-dev``, which can be
entered using ``source calphad-dev/bin/activate`` on Linux or macOS/OSX or
``activate calphad-dev\bin\activate`` on Windows.
The environment name is arbitrary - you can use whatever name you prefer.

ESPEI uses `Git <https://git-scm.com/book/en/v2>`_ and
`GitHub <https://github.com/PhasesResearchLab/ESPEI>`_ for version control.
Windows users: if you do not have a working version of Git,
`download it here <https://git-scm.com/downloads>`_ first.

To install the latest development version of ESPEI using pip:

.. code-block:: bash

   python -m venv calphad-env
   source calphad-env/bin/activate
   git clone https://github.com/phasesresearchlab/espei.git
   cd espei
   pip install -U pip
   pip install -U --editable .[dev]

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