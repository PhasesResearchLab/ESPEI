Quickstart
==========

ESPEI has two different fitting modes: single-phase and multi-phase fitting.
You can run either of these modes or both of them sequentially.

To run either of the modes, you need to have a phase models file that describes the phases in the system using the standard CALPHAD approach within the compound energy formalism.
You also need to describe the data that ESPEI should fit to.
You will need single-phase and multi-phase data for a full run.
Fit settings and all datasets are stored as JSON files and described in detail at the :ref:`Input data` page.
All of your input datasets should be validated by running ``espei --check-datasets my-input-datasets``, where ``my-input-datasets`` is a folder of all your JSON files.

The main output result is going to be a database (defaults to ``out.tdb``), an array of the steps in the MCMC trace (defaults to ``trace.npy``), and the an array of the log-probabilities for each iteration and chain (defaults to ``lnprob.npy``).

Single-phase only
-----------------

If you have only heat capacity, entropy and enthalpy data and mixing data (e.g. from first-principles),
you may want to see the starting point for your MCMC calculation.

Create an input file called ``espei-in.yaml``.

.. code-block:: yaml

    system:
      phase_models: my-phases.json
      datasets: my-input-datasets
    generate_parameters:
      excess_model: linear
      ref_state: SGTE91

Then ESPEI can be run by running

.. code-block:: bash

    espei --input espei-in.yaml


Multi-phase only
----------------

If you have a database already and just want to do a multi-phase fitting, you can specify a starting TDB file (named ``my-tdb.tdb``) with

.. code-block:: YAML

    system:
      phase_models: my-phases.json
      datasets: my-input-data
    mcmc:
      iterations: 1000
      input_db: my-tdb.tdb

The TDB file you input must have all of the degrees of freedom you want as FUNCTIONs with names beginning with ``VV``.

Restart from previous run-phase only
------------------------------------

If you've run an MCMC fitting already in ESPEI and have a trace file called ``my-previous-trace.npy`` , then you can resume the calculation with the following input file

.. code-block:: yaml

    system:
      phase_models: my-phases.json
      datasets: my-input-data
    mcmc:
      iterations: 1000
      input_db: my-tdb.tdb
      restart_trace: my-previous-trace.npy


Full run
--------

A minimal full run of ESPEI with single phase fitting and MCMC fitting is done by the following

.. code-block:: yaml

    system:
      phase_models: my-phases.json
      datasets: my-input-data
    generate_parameters:
      excess_model: linear
      ref_state: SGTE91
    mcmc:
      mcmc_steps: 1000


Input Customization
-------------------

ESPEI lets you control many aspects of your calculations with the input files shown above.
See :ref:`Writing input files` for a full description of all possible inputs.


FAQ
---

Q: There is an error in my JSON files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A: Common mistakes are using single quotes instead of the double quotes required by JSON files.
Another common source of errors is misaligned open/closing brackets.

Many mistakes are found with ESPEI's ``check-datasets`` utility.
Run ``espei check-datasets my-input-datasets`` on your directory ``my-input-datasets``.

Q: How do I analyze my results?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A: By default, ESPEI will create ``trace.npy`` and ``lnprob.npy`` for the MCMC chain at the specified save interval and according to the save interval (defaults to ever iteration).
These are created from arrays via ``numpy.save()`` and can thus be loaded with ``numpy.load()``.
Note that the arrays are preallocated with zeros.
These filenames and settings can be changed using in the input file.
You can then use these chains and corresponding log-probabilities to make corner plots, calculate autocorrelations, find optimal parameters for databases, etc..
Finally, you can use py:mod:`espei.plot` functions such as ``multiplot`` to plot phase diagrams with your input equilibria data and ``plot_parameters`` to compare single-phase data (e.g. formation and mixing data) with the properties calculated with your database.

Q: Can I run ESPEI on a supercomputer supporting MPI?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A: Yes! ESPEI has MPI support. See the :ref:`MPI` page for more details.

Q: How is the log probability reported by ESPEI calculated?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MCMC simulation requires determining the probability of the data given a set of parameters, :math:`p(D|\theta)`.
In MCMC, the log probability is often used to avoid floating point errors that arise from multiplying many small floating point numbers.
For each type of data the *error*, often interpreted as the absolute difference between the expected and calculated value, is determined.
For the types of data and how the error is calculated, refer to the ESPEI paper [1]_.

The error is assumed to be normally distributed around the experimental data point that the prediction of a set of parameters is being compared against.
The log probability of each data type is calculated by the log probability density function of the error in this normal distribution with a mean of zero and the standard deviation as given by the data type and the adjustable weights (see ``data_weights`` in :ref:`Writing input files`).
The total log probability is the sum of all log probabilities.

Note that any probability density function always returns a positive value between 0 and 1, so the log probability density function should return negative numbers and the log probability reported by ESPEI should be negative.

:ref:`Writing input files`

References
==========

.. ﻿[1] B. Bocklund, R. Otis, A. Egorov, A. Obaied, I. Roslyakova, Z.-K. Liu, ESPEI for efficient thermodynamic database development, modification, and uncertainty quantification: application to Cu-Mg, (2019). http://arxiv.org/abs/1902.01269.

