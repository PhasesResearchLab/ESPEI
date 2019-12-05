==========
What's New
==========

0.7.3 (2019-12-02)
==================

This change includes several new features and performance improvements.

* Drop Python 2 support (Python 2 is no longer supported on January 1, 2020).
* Update dask and distributed support to versions >=2. (`@bocklund`_)
* Users can tweak the AICc penalty factor for each phase to nudge parameter selection towards adding more or fewer parameters based on user modeling intitution. (`@bocklund`_)
* Allow for tracefile and probfile to be set to None. (`@jwsiegel2510`_)
* Weighting individual datasets in single phase fitting is now implemented via scikit-learn.  (`@bocklund`_)
* Performance improvements by reducing overhead. (`@bocklund`_)
* Increased solver accuracy by using pycalphad's exact Hessian solver. (`@bocklund`_)
* Support writing SER reference state information to the `ELEMENT` keyword in TDBs based on the SGTE unary 5 database.  (`@bocklund`_)
* MCMC now calculates the likelihood of the initial parameter set so the starting point can be reasonably compared.  (`@bocklund`_)
* Fixed a bug where mis-aligned configurations and site occupancies in single phase datasets passed the dataset checker  (`@bocklund`_)

0.7.2 (2019-06-12)
==================

This is a small bugfix release that fixes the inability to provide the EmceeOptimizer a ``restart_trace``.


0.7.1 (2019-06-03)
==================

This is a significant update reflecting many internal improvements, new features, and bugfixes. Users using the YAML input or the ``run_espei`` Python API should see entirely backwards compatible changes with ESPEI 0.6.2.

pycalphad 0.8, which introduced many `key features <https://pycalphad.org/docs/latest/CHANGES.html>`_ for these changes is now required.
This should almost completely eliminate the time to build phases due to the symengine backend (phases will likely build in less time than to call the MCMC objective function).
Users can expect a slight performance improvement for MCMC fitting.

Improvements
------------
* Priors can now be specified and are documented online.
* Weights for different datasets are added and are supported by a ``"weight"`` key at the top level of any dataset.
* Weights for different types of data are added. These are controlled via the input YAML and are documented there.
* A new internal API is introduced for generic fitting of parameters to datasets in the ``OptimizerBase`` class. The MCMC optimizer in emcee was migrated to this API (the ``mcmc_fit`` function is now deprecated, but still works until the next major version of ESPEI). A simple SciPy-based optimizer was implemented using this API.
* Parameter selection can now be passed initial databases with parameters (e.g. for adding magnetic or other parameters manually).
* pycalphad's reference state support can now be used to properly reference out different types of model contributions (ideal mixing, magnetic, etc.). This is especially useful for DFT thermochemical data which does not include model contributions from ideal mixing or magnetic heat capacity. Useful for experimental data which does include ideal mixing (previously ESPEI assumed all data
* Datasets and input YAML files now have a tag system where tags that are specified in the input YAML can override any keys/values in the JSON datasets at runtime. This is useful for tagging data with different weights/model contribution exclusions (e.g. DFT tags may get lower weights and can be set to exclude model contributions). If no tags are applied, removing ideal mixing from all thermochemical data is applied automatically for backwards compatibility. This backwards compatibility feature will be removed in the next major version of ESPEI (all model contributions will be included by default and exclusions must be specified manually).

Bug fixes
---------
* Bug fixed where asymmetric ternary parameters were not properly replaced in SymPy
* Fixed error where ZPF error was considering the chemical potentials of stoichiometric phases in the target hyperplane (they are meaningless)
* Report the actual file paths when dask's work-stealing is set to false.
* Errors in the ZPF error function are no longer swallowed with -np.inf error. Any errors should be reported as bugs.
* Fix bug where subsets of symbols to fit are not built properly for thermochemical data

Other
-----
* Documentation recipe added for `plot_parameters`
* [Developer] ZPF and thermochemical datasets now have an function to get all the data up front in a dictionary that can be used in the functions for separation of concerns and calculation efficiency by not recalculating the same thing every iteration.
* [Developer] a function to generate the a context dict to pass to lnprob now exists. It gets the datasets automatically using the above.
* [Developer] transition to pycalphad's new build_callables function, taking care of the ``v.N`` state variable.
* [Developer] Load all YAML safely, silencing warnings.

0.6.2 (2018-11-27)
==================

This backwards-compatible release includes several bug fixes and improvements.

* Updated branding to include the new ESPEI logo. See the logo in the ``docs/_static`` directory.
* Add support for fitting excess heat capacity.
* Bug fix for broken potassium unary.
* Documentation improvements for recipes
* pycalphad 0.7.1 fixes for dask, sympy, and gmpy2 should mean that ESPEI should not require package upgrade or downgrades. Please report any installations issues in `ESPEI's Gitter Channel <https://gitter.im/PhasesResearchLab/ESPEI>`.
* [Developers] ESPEI's ``eq_callables_dict`` is now ``pycalphad.codegen.callables.build_callables``.
* [Developers] matplotlib plotting tests are removed because nose is no longer supported.


0.6.1 (2018-08-28)
==================

This a major release with several important features and bug fixes.

* Enable use of ridge regression alpha for parameter selection via the ``parameter_generation.ridge_alpha`` input parameter.
* Add ternary parameter selection. Works by default, just add data.
* Set memory limit to zero to avoid dask killing workers near the dask memory limits.
* Remove ideal mixing from plotting models so that ``plot_parameters`` gives the correct entropy values.
* Add `recipes documentation <https://github.com/PhasesResearchLab/ESPEI/blob/master/docs/recipes.rst>`_ that contains some Python code for common utility operations.
* Add documentation for running custom distributed schedulers in ESPEI


0.6 (2018-07-02)
================

This is a update including *breaking changes to the input files* and several minor improvements.

* Update input file schema and Python API to be more consistent so that the ``trace`` always refers to the collection of chains and ``chain`` refers to individual chains. Additionally removed some redundancy in the parameters nested under the ``mcmc`` heading, e.g. ``mcmc_steps`` is now ``iterations`` and ``mcmc_save_interval`` is now ``save_interval`` in the input file and Python API. See `Writing Input <http://espei.org/en/latest/writing_input.html>`_ documentation for all of the updates.
* The default save interval is now 1, which is more reasonable for most MCMC systems with significant numbers of phase equilibria.
* Bug fixes for plotting and some better plotting defaults for plotting input data
* Dataset parsing and cleaning improvements.
* Documentation improvements (see the `PDF <http://readthedocs.org/projects/espei/downloads/pdf/latest/>`_!)

0.5.2 (2018-04-28)
==================

This is a major bugfix release for MCMC multi-phase fitting runs for single phase data.

* Fixed a major issue where single phase thermochemical data was always compared to Gibbs energy, giving incorrect errors in MCMC runs.
* Single phase errors in ESPEI incorrectly compared values with ideal mixing contributions to data, which is excess only.
* Fixed a bug where single phase thermochemical data with that are dependent on composition and pressure and/or temperature were not fit correctly.
* Added utilities for analyzing ESPEI results and add them to the Cu-Mg example docs.

0.5.1 (2018-04-17)
==================

This is a minor bugfix release.

* Parameter generation for phases with vacancies would produce incorrect parameters because the vacancy site fractions were not being correctly removed from the contributions due to their treatment as ``Species`` objects in ``pycalphad >=0.7``.

0.5 (2018-04-03)
================

* Parameter selection now uses the corrected AIC, which further prevents overparameterization where there is sparse training data.
* Activity and single phase thermochemical data can now be included in MCMC fitting runs. Including single phase data can help anchor metastable phases to DFT data when they are not on the stable phase diagram. See the `Gathering input data <http://espei.org/en/latest/input_data.html>`_ documentation for information on how to input activity data.
* Dataset checking has been improved. Now there are checks to make sure sublattice interactions are properly sorted and mole fractions sum to less than 1.0 in ZPF data.
* Support for fitting phases with arbitrary pycalphad Models in MCMC, including (charged and neutral) species and ionic liquids. There are several consequences of this:

  - ESPEI requires support on ``pycalphad >=0.7``
  - ESPEI now uses pycalphad ``Model`` objects directly. Using the JIT compiled Models has shown up to a *50% performance improvement* in MCMC runs.
  - Using JIT compiled ``Model`` objects required the use of ``cloudpickle`` everywhere. Due to challenges in overriding ``pickle`` for upstream packages, we now rely solely on ``dask`` for scheduler tasks, including ``mpi`` via ``dask-mpi``. Note that users must turn off ``work-stealing`` in their ``~/.dask/config.yaml`` file.

* [Developers] Each method for calculating error in MCMC has been moved into a module for that method in an ``error_functions`` subpackage. One top level function from each module should be imported into the ``mcmc.py`` and used in ``lnprob``. Developers should then just customize ``lnprob``.
* [Developers] Significant internal docs improvements: all non-trivial functions have complete docstrings.

0.4.1 (2018-02-05)
==================

* Enable plotting of isothermal sections with data using ``dataplot`` (and ``multiplot``, etc.)
* Tielines are now plotted in ``dataplot`` for isothermal sections and T-x phase diagrams
* Add a useful ``ravel_conditions`` method to unpack conditions from datasets

0.4 (2017-12-29)
================

* MCMC is now deterministic by default (can be toggled off with the ``mcmc.deterministic`` setting).
* Added support for having no scheduler (running with no parallelism) with the ``mcmc.scheduler`` option set to ``None``. This may be useful for debugging.
* Logging improvements

  - Extraneous warnings that may be confusing for users and dirty the log are silenced.
  - A warning is added for when there are no datasets found.
  - Fixed a bug where logging was silenced with the dask scheduler

* Add ``optimal_parameters`` utility function as a helper to get optimal parameter sets for analysis
* Several improvements to plotting

  - Users can now plot phase diagram data alone with ``dataplot``, useful for checking datasets visually. This changes the API for ``dataplot`` to no longer infer the conditions from an equilibrium ``Dataset`` (from pycalphad). That functionality is preserved in ``eqdataplot``.
  - Experimental data points are now plotted with unique symbols depending on the reference key in the dataset. This is for both phase diagram and single phase parameter plots.
  - Options to control plotting parameters (e.g. symbol size) and take user supplied Axes and Figures in the plotting functions. The symbol size is now smaller by default.

* Documentation improvements for API and separation of theory from the Cu-Mg example
* Fixes a bug where elements with single character names would not find the correct reference state (which are typically named GHSERCC for the example of C).
* [Developer] All MCMC code is moved from the ``paramselect`` module to the ``mcmc`` module to separate these tasks
* [Developer] Support for arbitrary user reference states (so long as the reference state is in the ``refdata`` module and follows the same format as SGTE91)

0.3.1.post2 (2017-10-31)
========================

* Propagate the new entry point to setup.py

0.3.1.post1 (2017-10-31)
========================

* Fix for module name/function conflict in entry point

0.3.1 (2017-10-31)
==================

* ESPEI is much easier to run interactively in Python and in Jupyter Notebooks
* Reference data is now included in ESPEI instead of in pycalphad
* Several reference data fixes including support for single character elements ('V', 'B', 'C', ...)
* Support for using multiprocessing to parallelize MCMC runs, used by default (@olivia-higgins)
* Improved documentation for installing and developing ESPEI

0.3.post2 (2017-09-20)
======================

* Add input-schema.yaml file to installer

0.3.post1 (2017-09-20)
======================

* Add LICENSE to manifest

0.3 (2017-09-20)
================

* **ESPEI input is now described by a file.** This change is breaking. Old command line arguments are not supported. See `Writing input files <http://espei.org/en/latest/writing_input.html>`_ for a full description of all the inputs.
* New input options are supported, including modifying the number of chains and standard deviation from the mean
* ESPEI is now available on conda-forge
* TinyDB 2 support is dropped in favor of TinyDB 3 for conda-forge deployment
* Allow for restarting previous mcmc calculations with a trace file
* Add Cu-Mg example to documentation

0.2.1 (2017-08-17)
==================

Fixes to the 0.2 release plotting interface

* ``multiplot`` is renamed from ``multi_plot``, as in docs.
* Fixed an issue where phases in datasets, but not in equilibrium were not plotted by dataplot and raised an error.

0.2 (2017-08-15)
==================

* New ``multiplot`` interface for convenient plotting of phase diagrams + data. ``dataplot`` function underlies key data plotting features and can be used with ``eqplot``. See their API docs for examples. Will break existing code using multiplot.
* MPI support for local/HPC runs. Only single node runs are explicitly supported currently. Use ``--scheduler='MPIPool'`` command line option. Requires ``mpi4py``.
* Default debug reporting of acceptance ratios
* Option (and default) to output the log probability array matching the trace. Use ``--probfile`` option to control.
* Optimal parameters are now chosen based on lowest error in chain.
* Bug fixes including

   - py2/3 compatibility
   - Unicode datasets
   - handling of singular matrix errors from pycalphad's ``equilibrium``
   - reporting of failed conditions

0.1.5 (2017-08-02)
==================

* Significant error checking of JSON inputs.
* Add new ``--check-datasets`` option to check the datasets at path. It should be run before you run ESPEI fittings. All errors must be resolved before you run.
* Move the espei script module from ``fit.py`` to ``run_espei.py``.
* Better docs building with mocking
* Google docstrings are now NumPy docstrings

0.1.4 (2017-07-24)
==================

* Documentation improvements for usage and API docs
* Fail fast on JSON errors

0.1.3 (2017-06-23)
==================

* Fix bad version pinning in setup.py
* Explicitly support Python 2.7

0.1.2 (2017-06-23)
==================

* Fix dask incompatibility due to new API usage

0.1.1 (2017-06-23)
==================

* Fix a bug that caused logging to raise if bokeh isn't installed

0.1 (2017-06-23)
==================

ESPEI is now a package! New features include

* Fork https://github.com/richardotis/pycalphad-fitting
* Use emcee for MCMC fitting rather than pymc
* Support single-phase only fitting
* More control options for running ESPEI from the command line
* Better support for incremental saving of the chain
* Control over output with logging over printing
* Significant code cleanup
* Better usage documentation

.. _`@bocklund`: https://github.com/bocklund
.. _`@jwsiegel2510`: https://github.com/jwsiegel2510