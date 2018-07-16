.. raw:: latex

   \chapter{Software design}

Software design
===============

This is the place to start as a new ESPEI contributor.

The following sections elaborate on the design principles on the software side.
The goal is to make it clear how different modules in ESPEI fit together and where to find specific functionality to override or improve.

ESPEI provides tools to

1. Parameterize CALPHAD models by optimizing the compromise between model accuracy and complexity. We typically call this parameter generation or model selection.
2. Refine parameterized CALPHAD models to multi-phase or other custom data with uncertainty quantification via Markov chain Monte Carlo

API
---

ESPEI has two levels of API that users should expect to interact with:

1. Input from YAML files on the command line (via ``espei --input <input_file>`` or by Python via the ``espei.espei_script.run_espei`` function
2. Work directly with the Python functions for parameter selection ``espei.paramselect.generate_parameters`` and MCMC ``espei.mcmc.mcmc_fit``

YAML files are the recommended way to use ESPEI and should have a way to express most if not all of the options that
the Python functions support. The schema for YAML files is located in the root of the ESPEI directory as ``input-schema.yaml``
and is validated in the ``espei_script.py`` module by the `Cerberus <http://docs.python-cerberus.org/en/stable/>`_ package.

Module Hierarchy
----------------

* ``espei_script.py`` is the main entry point for the YAML input API.
* ``paramselect.py`` is where parameter generation happens.
* ``mcmc.py`` creates the likelihood function and runs MCMC.
* ``error_functions`` is a package with modules for each type of likelihood function.
* ``parameter_selection`` is a package with core pieces of parameter selection.
* ``utils.py`` are utilities with reuse potential across several parts of ESPEI.
* ``plot.py`` holds plotting functions.
* ``datasets.py`` manages validating and loading datasets into a TinyDB in memory database.
* ``core_utils.py`` are legacy utility functions that should be refactored out to be closer to individual modules and packages where they are used.

Parameter selection
-------------------

Parameter selection goes through the ``generate_parameters`` function in the ``espei.paramselect`` module.
The goal of parameter selection is go through each phase (one at a time) and fit a CALPHAD model to the data.

For each phase, the endmembers are fit first, followed by binary and ternary interactions.
For each individual endmember or interaction to fit, a series of candidate models are generated that have increasing
complexity in both temperature and interaction order (an L0 excess parameter, L0 and L1, ...).

Each model is then fit by ``espei.parameter_selection.selection.fit_model``, which currently uses a simple
pseudo-inverse linear model from scikit-learn. Then the tradeoff between the goodness of fit and the model complexity
is scored by the AICc (see :ref:`Theory`) in ``espei.parameter_selection.selection.score_model``.
The optimal scoring model is accepted as the model with the fit model parameters set as degrees of freedom for the MCMC step.

The main principle is that ESPEI transforms the data and candidate models to vectors and matricies that fit a typical machine learning type problem of :math:`Ax = b`.
Extending ESPEI to use different or custom models in the current scheme basically comes down to formulating candidate models in terms of this type of problem.
The main ways to improve on the fitting or scoring methods used in parameter selection is to override the fit and score functions.

Currently the capabilities for providing custom models or contributions (e.g. magnetic data) in the form of generic pycalphad Model objects are limited.
This is also true for custom types of data that one would use in fitting a custom model.

MCMC optimization and uncertainty quantification
------------------------------------------------

Most of the Markov chain Monte Carlo optimization and uncertainty quantification happen in the ``espei.mcmc`` module through the ``espei.mcmc.mcmc_fit`` function.

The main reason ESPEI's parameter selection and MCMC routines are split up is that custom Models or existing TDB files can be provided and fit.
In other words, if you are using a model that doesn't need parameter selection or is for a property that is not Gibbs energy, MCMC can fit it with uncertainty quantification.

The general process is

1. Take a database with degrees of freedom as database symbols named ``VV####``, where ``####`` is a number, e.g. ``0001``.
   The symbols correspond to ``FUNCTION`` in the TDB files.
2. Initialize those degrees of freedom to a starting distribution for ensemble MCMC (see :ref:`Theory`).
   The starting distribution is controlled by the ``espei.mcmc.generate_parameter_distribution`` function, which currently
   supports initializing the parameters to a Gaussian ball.
3. Use the `emcee <http://dfm.io/emcee/current/>`_ package to run ensemble MCMC

ESPEI's MCMC is quite flexible for customization. To fit a custom model, it just needs to be read by pycalphad and
have correctly named degrees of freedom (``VV####``).

To fit an existing or custom model to new types of data, just write a function that takes in datasets and the parameters
that are required to calculate the values (e.g. pycalphad Database, components, phases, ...) and returns the error.
Then override the ``espei.mcmc.lnprob`` function to include your custom error contribution.
There are examples of these functions ``espei.error_functions`` that ESPEI uses by default.

In addition, custom starting distributions can be obtained by overriding ``espei.mcmc.generate_parameter_distribution``.