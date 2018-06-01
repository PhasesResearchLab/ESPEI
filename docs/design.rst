.. raw:: latex

   \chapter{Software design}

Software design
===============

Module Hierarchy
----------------

* ``espei_script.py`` is the main entry point
* ``paramselect.py`` is where parameter generation happens.
* ``mcmc.py`` creates the likelihood function and runs MCMC.
* ``error_functions`` is a package with modules for each type of likelihood function.
* ``core_utils.py`` contains specialized utilities for ESPEI.
* ``utils.py`` are utilities with reuse potential outside of ESPEI.
* ``plot.py`` holds plotting functions
