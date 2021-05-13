
.. raw:: latex

   \chapter{Advanced Schedulers}

.. _MPI:

===================
Advanced Schedulers
===================

ESPEI uses dask-distributed for parallelization and provides an easy way to deploy clusters locally via TCP with the ``mcmc.scheduler: dask`` setting.

Sometimes ESPEI's dask scheduling options are not sufficiently flexible for different environments.

As an alternative to setting the cores with the ``mcmc.scheduler: dask`` setting,
you can provide ESPEI with a scheduler file from dask that has information about
how to connect to a dask parallel scheduler.

This is generally a two step process of

1. Starting a scheduler with workers and writing a scheduler file
2. Running ESPEI and connecting to the existing scheduler

In order to let the system manage the memory and prevent dask from pausing or killing workers, the memory limit should be set to zero.


Starting a scheduler
====================

MPI-based dask scheduler
------------------------

Dask provides a ``dask-mpi`` package that sets this up for you and creates a scheduler file to pass to ESPEI.
The scheduler information will be serialized as a JSON file that you set in your ESPEI input file.

The dask-mpi package (version 2.0.0 or greater) must be installed before you can use it:

.. code-block:: bash

    conda install -c conda-forge --yes "dask-mpi>=2"

Note that you may also need a particular MPI implementation, conda-forge provides packages for OpenMPI or MPICH. You can pick a particular one by installing dask-mpi using either:

.. code-block:: bash

    conda install -c conda-forge --yes "dask-mpi>=2" "mpi=*=openmpi"

or

.. code-block:: bash

    conda install -c conda-forge --yes "dask-mpi>=2" "mpi=*=mpich"

or let conda pick one for you by not including any.

To start the scheduler and workers in the background, you can run the ``dask-mpi`` command (use ``dask-mpi --help`` to check the arguments).
The following command will start a scheduler on the main MPI task, then a worker for each remaining MPI task that ``mpirun`` sees.

.. code-block:: shell

   mpirun dask-mpi --scheduler-file my_scheduler.json --nthreads 1 --memory-limit 0 &


Generic scheduler
-----------------

If you need further customization of dask schedulers, you can start a distributed Client any way you like, then write out the scheduler file for ESPEI to use.

For example, if you name the following file ``start_scheduler.py``, you can run this Python script in the background, which will contain the scheduler and workers, then ESPEI will connect to it.

.. code-block:: python

   # start_scheduler.py
   from distributed import Client, LocalCluster
   from tornado.ioloop import IOLoop

   if __name__ == '__main__':
       loop = IOLoop()
       cluster = LocalCluster(n_workers=4, threads_per_worker=1, memory_limit=0)
       client = Client(cluster)
       client.write_scheduler_file('my-scheduler.json')
       loop.start()  # keeps the scheduler running
       loop.close()


Running ``start_scheduler.py &``, will run this process in the background with 4 processes.

ESPEI Input
===========

After starting the scheduler on the cluster, you run ESPEI like normal.

For the most part, this ESPEI input file is the same as you use locally, except the ``scheduler`` parameter is set to the name of your scheduler file.

Here is an example for Bayesian parameter estimation using MCMC starting from a generated TDB with a scheduler file named ``my-scheduler.json``:

.. code-block:: YAML

    system:
      phase_models: my-phases.json
      datasets: my-input-data
    mcmc:
      iterations: 1000
      input_db: my-tdb.tdb
      scheduler: my-scheduler.json



Example Queue Script - MPI
==========================

To run on through a queueing system, you'll often use queue scripts that start batch jobs.


This example will create an MPI scheduler using ``dask-mpi`` via ``mpirun`` (or other MPI executable).
Since many MPI jobs are run through batch schedulers, an example script for a PBS job looks like:

.. code-block:: shell

    #!/bin/bash

    #PBS -l nodes=1:ppn=20
    #PBS -l walltime=48:00:00
    #PBS -A open
    #PBS -N espei-mpi
    #PBS -o espei-mpi.out
    #PBS -e espei-mpi.error

    # starts the scheduler on MPI and creates the scheduler file called 'my_scheduler.json'
    # you can replace this line with any script that starts a scheduler
    # e.g. a `start_scheduler.py` file
    # make sure it ends with `&` to run the process in the background
    mpirun dask-mpi --scheduler-file my_scheduler.json --nthreads 1 --memory-limit 0 &

    # runs ESPEI as normal
    espei --in espei-mpi-input.yaml

.. admonition:: See also
   :class: seealso

   See https://docs.dask.org/en/latest/setup/hpc.html for more details on using dask on HPC machines.
