==========
What's New
==========

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

