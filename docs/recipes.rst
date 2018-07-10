.. raw:: latex

   \chapter{Recipes}

=======
Recipes
=======

Here you can find some useful snippets of code to make using ESPEI easier.

Optimal parameter TDBs
======================

Creating TDBs of optimal parameters from a tracefile and probfile:

.. code-block:: python

   """
   This script updates an input TDB file with the optimal parameters from an ESPEI run.

   Change the capitalized variables to your desired input and output filenames.
   """

   INPUT_TDB_FILENAME = 'CU-MG_param_gen.tdb'
   OUTPUT_TDB_FILENAME = 'CU-MG_opt_params.tdb'
   TRACE_FILENAME = 'trace.npy'
   LNPROB_FILENAME = 'lnprob.npy'

   import numpy as np
   from pycalphad import Database
   from espei.analysis import truncate_arrays
   from espei.utils import database_symbols_to_fit, optimal_parameters

   trace = np.load(TRACE_FILENAME)
   lnprob = np.load(LNPROB_FILENAME)
   trace, lnprob = truncate_arrays(trace, lnprob)

   dbf = Database(INPUT_TDB_FILENAME)
   opt_params = dict(zip(database_symbols_to_fit(dbf), optimal_parameters(trace, lnprob)))
   dbf.symbols.update(opt_params)
   dbf.to_file(OUTPUT_TDB_FILENAME)


