.. _Writing input files:

===================
Writing ESPEI input
===================

This page aims to completely describe the ESPEI input file in the YAML format.
Possibly useful links are the `YAML refcard <http://www.yaml.org/refcard.html>`_ and the (possibly less useful) `Full YAML specification <http://www.yaml.org/spec/>`_.
These are all key value pairs in the format

.. code-block:: YAML

   key: value

They are nested for purely organizational purposes.

.. code-block:: YAML

   top_level_key:
     key: value

As long as keys are nested under the correct heading, they have no required order.
All of the possible keys are

.. code-block:: YAML

   system:
    phase_models
    datasets

   output:
     verbosity
     output_db
     tracefile
     probfile

   generate_parameters:
     excess_model
     ref_state

   mcmc:
     mcmc_steps
     mcmc_save_interval
     cores
     scheduler
     input_db
     restart_chain
     chains_per_parameter
     chain_std_deviation


The next sections describe each of the keys individually.
If a setting has a default of ``required`` it must be set explicitly.

system
======

The ``system`` key is intended to describe the specific system you are fitting, including the components, phases, and the data to fit to.

phase_models
------------

:type: string
:default: required

The JSON file describing the CALPHAD models for each phase.
See :ref:`input_phase_descriptions` for an example of how to write this file.

datasets
--------

:type: string
:default: required

The path to a directory containing JSON files of input datasets.
The file extension to each of the datasets must be named as ``.json``, but they can otherwise be named freely.

For an examples of writing these input JSON files, see :ref:`Input data`.

output
======

verbosity
---------

:type: int
:default: 0

Controls the logging level.

=====  =========
Value  Log Level
=====  =========
0      Warning
1      Info
2      Debug
=====  =========

output_db
---------

:type: string
:default: out.tdb

The database to write out.
Can be any file format that can be written by a pycalphad `Database <https://pycalphad.org/docs/latest/api/pycalphad.io.html?highlight=database#pycalphad.io.database.Database>`_.

tracefile
---------

:type: string
:default: chain.npy

Name of the file that the MCMC trace is written to.
The array has shape ``(number of chains, iterations, number of parameters)``.

The array is preallocated and padded with zeros, so if you selected to take 2000 MCMC steps, but only got through 1500, the last 500 values would be all 0.

You must choose a unique file name.
An error will be raised if file specified by ``tracefile`` already exists.

probfile
--------

:type: string
:default: lnprob.npy

Name of the file that the MCMC ln probabilities are written to.
The array has shape ``(number of chains, iterations)``.

The array is preallocated and padded with zeros, so if you selected to take 2000 MCMC steps, but only got through 1500, the last 500 values would be all 0.

You must choose a unique file name.
An error will be raised if file specified by ``probfile`` already exists.


generate_parameters
===================

The options in ``generate_parameters`` are used to control parameter selection and fitting to single phase data.
This should be used if you have input thermochemical data, such as heat capacities and mixing energies.

Generate parameters will use the `Akaike information criterion <https://en.wikipedia.org/wiki/Akaike_information_criterion>`_ to select model parameters and fit them, creating a database.


excess_model
------------

:type: string
:default: required
:options: linear

Which type of model to use for excess mixing parameters.
Currently only `linear` is supported.

The `exponential` model is planned, as well as support for custom models.

ref_state
---------

:type: string
:default: required
:options: SGTE91 | SR2016

The reference state to use for the pure elements and lattice stabilities.
Currently only `SGTE91` and `SR2016` (for certain elements) is supported.

There are plans to extend to support custom reference states.


mcmc
====

The options in ``mcmc`` control how Markov Chain Monte Carlo is performed using the emcee package.

In order to run an MCMC fitting, you need to specify one and only one source of parameters somewhere in the input file.
The parameters can come from including a ``generate_parameters`` step, or by specifying the ``mcmc.input_db`` key with a file to load as pycalphad Database.

If you choose to use the parameters from a database, you can then further control settings based on whether it is the first MCMC run for a system (you are starting fresh) or whether you are continuing from a previous run (a 'restart').

mcmc_steps
----------

:type: int
:default: required

Number of iterations to perform in emcee.
Each iteration consists of accepting one step for each chain in the ensemble.


mcmc_save_interval
------------------

:type: int
:default: 20

Controls the interval for saving the MCMC chain and probability files.

cores
-----
:type: int
:min: 1

How many cores from available cores to use during parallelization with dask or emcee.
If the chosen number of cores is larger than available, then this value is ignored and espei defaults to using the number available.

Cores does not take affect for MPIPool scheduler option. MPIPool requires the number of processors be set directly with MPI.

scheduler
---------

:type: string
:default: emcee
:options: emcee | MPIPool | dask | None

Which scheduler to use for parallelization.
You can choose from either `dask`, `emcee`, `MPIPool` or `None`.

Choosing dask or emcee allows for the choice of cores used through the cores key.

Choosing MPIPool will allow you to set the number of cores directly using MPI.

Choosing None will result in no parallel scheduler being used. This is useful for debugging.

It is recommended to use MPIPool if you will be running jobs on supercomputing clusters.

input_db
--------

:type: string
:default: null

A file path that can be read as a pycalphad `Database <https://pycalphad.org/docs/latest/api/pycalphad.io.html?highlight=database#pycalphad.io.database.Database>`_.
The parameters to fit will be taken from this database.

For a parameter to be fit, it must be a symbol where the name starts with ``VV``, e.g. ``VV0001``.
For a ``TDB`` formatted database, this means that the free parameters must be functions of a single value that are used in your parameters.
For example, the following is a valid symbol to fit:

.. code-block:: none

   FUNCTION VV0000  298.15  10000; 6000 N !

restart_chain
-------------

:type: string
:default: null

If you have run a previous MCMC calculation, then you will have a trace file that describes the position and history of all of the chains from the run.
You can use these chains to start the emcee run and pick up from where you left off in the MCMC run by passing the trace file (e.g. ``chain.npy``) to this key.

If you are restarting from a previous calculation, you must also specify the same database file (with ``input_db``) as you used to run that calculation.

chains_per_parameter
--------------------

:type: int
:default: 2

This controls the number of chains to run in the MCMC calculation as an integer multiple of the number of parameters.

This parameter can only be used when initializing the first MCMC run.
If you are restarting a calculation, the number of chains per parameter is fixed by the number you chose previously.

Ensemble samplers require at least ``2*p`` chains for ``p`` fitting parameters to be able to make proposals.
If ``chains_per_parameter = 2``, then the number of chains if there are 10 parameters to fit is 20.

The value of ``chains_per_parameter`` must be an **EVEN integer**.


chain_std_deviation
-------------------

:type: float
:default: 0.1

The standard deviation to use when initializing chains in a Gaussian distribution from a set of parameters as a fraction of the parameter.

A value of 0.1 means that for parameters with values ``(-1.5, 2000, 50000)`` the chains will be initialized using those values as the mean and ``(0.15, 200, 5000)`` as standard deviations for each parameter, respectively.

This parameter can only be used when initializing the first MCMC run.
If you are restarting a calculation, the standard deviation for your chains are fixed by the value you chose previously.

You may technically set this to any positive value, you would like.
Be warned that too small of a standard deviation may cause convergence to a local minimum in parameter space and slow convergence, while a standard deviation that is too large may cause convergence to meaningless thermodynamic descriptions. 

