.. _MPI:

===
MPI
===

Overview
========

ESPEI uses dask-distributed for MPI parallelization.
To use this, you need to start a dask scheduler outside of ESPEI rather than depending on ESPEI to start one for you.
dask provides a ``dask-mpi`` command that sets this up for you and creates a scheduler file to pass to ESPEI.
The scheduler information will be serialized as a JSON file that you set in your ESPEI input file.

After starting the scheduler on the cluster, you run ESPEI like normal.


ESPEI Input
===========

For the most part, this ESPEI input file is the same as you use locally, except the ``scheduler`` parameter is set to the name of your scheduler file.

Here is an example for multiphase fitting starting from a generated TDB with a scheduler file named ``my-scheduler.json``:

.. code-block:: YAML

    system:
      phase_models: my-phases.json
      datasets: my-input-data
    mcmc:
      mcmc_steps: 1000
      input_db: my-tdb.tdb
      scheduler: my-scheduler.json


Example Batch Script
====================

To start this scheduler and create the scheduler file, you run ``dask-mpi`` via ``mpirun`` (or other MPI executable).
Since many MPI jobs are run through batch schedulers, an example script for a PBS job looks like:

.. code-block:: bash

    #!/bin/bash

    #PBS -l nodes=1:ppn=20
    #PBS -l walltime=48:00:00
    #PBS -A open
    #PBS -N espei-mpi
    #PBS -o espei-mpi.out
    #PBS -e espei-mpi.error

    # starts the scheduler on MPI and creates the scheduler file called 'my_scheduler.json'
    mpirun dask-mpi --scheduler-file my_scheduler.json --nthreads 1 &

    # runs ESPEI as normal
    espei --in espei-mpi-input.yaml

References
==========

See http://distributed.readthedocs.io/en/latest/setup.html?highlight=dask-mpi#using-mpi for more details.
