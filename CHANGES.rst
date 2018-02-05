==========
What's New
==========

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

* Propogate the new entry point to setup.py

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

* New ``multiplot`` interface for convienent plotting of phase diagrams + data. ``dataplot`` function underlies key data plotting features and can be used with ``eqplot``. See their API docs for examples. Will break existing code using multiplot.
* MPI support for local/HPC runs. Only single node runs are explictly supported currently. Use ``--scheduler='MPIPool'`` command line option. Requires ``mpi4py``.
* Default debug reporting of acceptance ratios
* Option (and default) to output the log probability array matching the trace. Use ``--probfile`` option to control.
* Optimal parameters are now chosen based on lowest error in chain.
* Bug fixes including
   
   - py2/3 compatibiltiy
   - unicode datasets
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

* Fix dask incompatibilty due to new API usage

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

