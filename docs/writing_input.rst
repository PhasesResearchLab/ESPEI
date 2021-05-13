.. raw:: latex

   \chapter{YAML input files}

.. _Writing input files:

======================
ESPEI YAML input files
======================

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
     tags

   output:
     verbosity
     logfile
     output_db
     tracefile
     probfile

   generate_parameters:
     excess_model
     ref_state
     ridge_alpha
     aicc_penalty_factor
     input_db

   mcmc:
     iterations
     prior
     save_interval
     cores
     scheduler
     input_db
     restart_trace
     chains_per_parameter
     chain_std_deviation
     deterministic
     approximate_equilibrium
     data_weights
     symbols


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

tags
----

:type: dict
:default: required

Mapping of keys to values to add to datasets with matching tags.
These can be used to dynamically drive values in datasets without adjusting the datasets themselves.
Useful for adjusting weights or other values in datasets in bulk.
For an examples of using tags in input JSON files, see :ref:`Datasets Tags`.


output
======

verbosity
---------

:type: int
:default: 0

Controls the logging level. Most users will probably want to use ``Info`` or ``Trace``.

``Warning`` logs should almost never occur and this log level will be
relatively quiet. ``Debug`` is a fire hose of information, but may be useful in
fixing calculation errors or adjusting weights.

=====  =========
Value  Log Level
=====  =========
0      Warning
1      Info
2      Trace
3      Debug
=====  =========

logfile
-------

:type: string
:default: null

Name of the file that the logs (controlled by ``verbosity``) will be output to.
The default is ``None`` (in Python, ``null`` in YAML), meaning the logging will
be output to stdout and stderr.

output_db
---------

:type: string
:default: out.tdb

The database to write out.
Can be any file format that can be written by a pycalphad `Database <https://pycalphad.org/docs/latest/api/pycalphad.io.html?highlight=database#pycalphad.io.database.Database>`_.

tracefile
---------

:type: string
:default: trace.npy

Name of the file that the MCMC trace is written to.
The array has shape ``(number of chains, iterations, number of parameters)``.

The array is preallocated and padded with zeros, so if you selected to take 2000 MCMC iterations, but only got through 1500, the last 500 values would be all 0.

You must choose a unique file name.
An error will be raised if file specified by ``tracefile`` already exists.
If you don't want a file to be output (e.g. for debugging), you can enter ``null``.

probfile
--------

:type: string
:default: lnprob.npy

Name of the file that the MCMC ln probabilities are written to.
The array has shape ``(number of chains, iterations)``.

The array is preallocated and padded with zeros, so if you selected to take 2000 MCMC iterations, but only got through 1500, the last 500 values would be all 0.

You must choose a unique file name.
An error will be raised if file specified by ``probfile`` already exists.
If you don't want a file to be output (e.g. for debugging), you can enter ``null``.


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

.. _input_yaml_generate_ref_state:

ref_state
---------

:type: string
:default: required
:options: SGTE91 | SR2016

The reference state to use for the pure elements and lattice stabilities.
Currently only `SGTE91` and `SR2016` (for certain elements) is supported.

There are plans to extend to support custom reference states.

ridge_alpha
-----------

:type: float
:default: 1.0e-100

Controls the ridge regression hyperparameter, :math:`\alpha`, as given in the following equation for the ridge regression problem

.. math::

   \min_w || Xw - y||^2_2 + \alpha ||w||^2_2


``ridge_alpha`` should be a positive floating point number which scales the relative contribution of parameter magnitudes to the residuals.

If an exponential form is used, the floating point value must have a decimal place before the ``e``,
that is ``1e-4`` is invalid while ``1.e-4`` is valid. More generally, the floating point must match the following
regular expression per the `YAML 1.1 spec <http://yaml.org/type/float.html>`_: ``[-+]?([0-9][0-9_]*)?\.[0-9.]*([eE][-+][0-9]+)?``.


aicc_penalty_factor
-------------------

:type: dict
:default: null


This parameter is a mapping from a phase name and property to a penalty factor to apply to the AICc number of parameters. The default is ``null``, which means that all the penalty factors are one (1) for all phases, which means no bias for more or fewer parameters compared to the textbook definition of AICc. If phases or data are not included, the penalty factors remain one.

Increasing the penalty factor will increase the penalty for more parameters, so it will bias parameter selection to choose fewer parameters. This can be especially useful when there is not many data points and an exact fit is possible (e.g. 4 points and 4 parameters), but modeling intutition would suggest that fewer parameters are required. A negative penalty factor will bias ESPEI's parameter selection to choose more parameters, which can be useful for generating more degrees of freedom for MCMC.

.. code-block:: yaml

     aicc_penalty_factor:
       BCC_A2:
         HM: 5.0
         SM: 5.0
       LIQUID:
         HM: 2.0
         SM: 2.0


input_db
--------

:type: string
:default: null

A file path that can be read as a pycalphad
`Database <https://pycalphad.org/docs/latest/api/pycalphad.io.html?highlight=database#pycalphad.io.database.Database>`_,
which can provide existing parameters to add as a starting point for parameter
generation, for example magnetic parameters.

If you have single phase data, ESPEI will try to fit parameters to that data
regardless of whether or not parameters were passed in for that phase. You must
be careful to only add initial parameters that do not have data that ESPEI will
try to fit. For example, do not include liquid enthalpy of mixing data for
ESPEI to fit if you are providing an initial set of parameters.


mcmc
====

The options in ``mcmc`` control how Markov Chain Monte Carlo is performed using the emcee package.

In order to run an MCMC fitting, you need to specify one and only one source of parameters somewhere in the input file.
The parameters can come from including a ``generate_parameters`` step, or by specifying the ``mcmc.input_db`` key with a file to load as pycalphad Database.

If you choose to use the parameters from a database, you can then further control settings based on whether it is the first MCMC run for a system (you are starting fresh) or whether you are continuing from a previous run (a 'restart').

iterations
----------

:type: int
:default: required

Number of iterations to perform in emcee.
Each iteration consists of accepting one step for each chain in the ensemble.

prior
-----

:type: list or dict
:default: {'name': 'zero'}

Either a single prior dictionary or a list of prior dictionaries corresponding to
the number of parameters. See :ref:`Specifying Priors` for examples and details
on writing priors.


save_interval
-------------

:type: int
:default: 1

Controls the interval (in number of iterations) for saving the MCMC chain and probability files.
By default, new files will be written out every iteration. For large files (many mcmc iterations and chains per parameter),
these might become expensive to write out to disk.

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
:default: dask
:options: dask | null | JSON file

Which scheduler to use for parallelization.
You can choose from either ``dask``, ``null``, or pass the path to a JSON scheduler file created by dask-distributed.

Choosing ``dask`` allows for the choice of cores used through the cores key.

Choosing ``null`` will result in no parallel scheduler being used. This is useful for debugging.

Passing the path to a JSON scheduler file will use the resources set up by the scheduler.
JSON file schedulers are most useful because schedulers can be started on MPI clusters using ``dask-mpi`` command.
See :ref:`MPI` for more information.

input_db
--------

:type: string
:default: null

A file path that can be read as a pycalphad `Database <https://pycalphad.org/docs/latest/api/pycalphad.io.html?highlight=database#pycalphad.io.database.Database>`_.
The parameters to fit will be taken from this database.

For a parameter to be fit, it must be a symbol where the name starts with ``VV``, e.g. ``VV0001``.
For a ``TDB`` formatted database, this means that the free parameters must be functions of a single value that are used in your parameters.
For example, the following is a valid symbol to fit:

::

   FUNCTION VV0000  298.15  10000; 6000 N !

restart_trace
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


approximate_equilibrium
-----------------------

:type: bool
:default: False

If True, an approximate version of pycalphad's ``equilibrium()`` function will
be used to calculate the driving force for phase boundary data. It uses
pycalphad's ``starting_point`` to construct a approximate equilibrium
hyperplanes of the lowest energy solution from a numerical sampling of each
active phases's internal degrees of freedom. This can give speedups of up to
10x for calculating the ZPF likelihood, but may miss low energy solutions that
are not sampled well numerically, especially for phases with many sublattices,
which have low energy solutions far from the endmembers.


deterministic
-------------

:type: bool
:default: True

Toggles whether ESPEI runs are deterministic. If this is True, running
ESPEI with the same Database and initial settings (either the same
``chains_per_parameter`` and ``chain_std_deviation`` or the same
``restart_trace``) will result in exactly the same results.

Starting two runs with the same TDB or with parameter generation
(which is deterministic) will result in the chains being at exactly
the same position after 100 iterations. If these are both restarted after
100 iterations for another 50 iterations, then the final chain after 150 iterations
will be the same.

It is important to note that this is only explictly True when
*starting* at the same point. If Run 1 and Run 2 are started with the
same initial parameters and Run 1 proceeds 50 iterations while Run 2
proceeds 100 iterations, restarting Run 1 for 100 iterations and Run 2 for 50
iterations (so they are both at 150 total iterations) will **NOT** give the same
result.

data_weights
------------

:type: dict
:default: {'ZPF': 1.0, 'ACR': 1.0, 'HM': 1.0, 'SM': 1.0, 'CPM': 1.0}

Each type of data can be weighted: zero phase fraction (``ZPF``), activity
(``ACR``) and the different types of thermochemical error. These weights are
used to modify the initial standard deviation of each data type by

.. math::

   \sigma = \frac{\sigma_{\mathrm{initial}}} {w}



.. _input_mcmc_symbols:

symbols
-------

:type: list[str]
:default: null

By default, any symbol in the database following the naming pattern `VV####`
where `####` is any number is optimized by ESPEI. If this option is set, this
can be used to manually fit a subset of the degrees of freedom in the system,
or fit degrees of freedom that do not folow the naming convention of 'VV####'::

   symbols: ['VV0000', 'FF0000', ...]
