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
* ``parameter_selection`` is a package with core pieces of parameter selection.
* ``core_utils.py`` are legacy utility functions that should be refactored out to be closer to individual modules and packages where they are used
* ``utils.py`` are utilities with reuse potential across several parts of ESPEI.
* ``plot.py`` holds plotting functions


Parameter selection
-------------------

Parameter selection goes through the ``generate_parameters`` function in the ``espei.paramselect`` module.
The goal of parameter selection is go through each phase (one at a time) and fit a CALPHAD model to the data.

For each phase, the endmembers are fit first, followed by binary and ternary interactions.
For each individual endmember or interaction to fit, a series of candidate models are generated that have increasing
complexity in both temperature and interaction order (an L0 excess parameter, L0 and L1, ...).

Each model is then fit by ``espei.parameter_selection.selection.fit_model``, which currently uses a simple
pseudo-inverse linear model from scikit-learn. Then the tradeoff between the goodness of fit and the model complexity
is scored by the AICc (see :ref:`Theory`). The optimal scoring model is accepted as the model with the fit model
parameters set as degrees of freedom for the MCMC step.
