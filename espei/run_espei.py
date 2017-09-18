"""
Automated fitting script.

A minimal run must specify an input.json and a datasets folder containing input files.
"""

from __future__ import print_function

import os
import argparse
import logging
import multiprocessing
import sys

import numpy as np
from pycalphad import Database

from espei import fit, schema
from espei.utils import ImmediateClient
from espei.datasets import DatasetError, load_datasets, recursive_glob

parser = argparse.ArgumentParser(description=__doc__)

parser.add_argument(
    "--input", "--in",
    default=None,
    help="Input file for the run"
    )

parser.add_argument(
    "--check-datasets",
    metavar="PATH",
    default=None,
    help="Check input datasets at the path. Does not run ESPEI.")

def get_run_settings(input_dict):
    """
    Validate settings from a dict of possible input.

    Performs the following actions:
    1. Normalize (apply defaults)
    2. Validate against the schema

    Parameters
    ----------
    input_dict : dict
        Dictionary of input settings

    Returns
    -------
    dict
        Validated run settings

    Raises
    ------
    ValueError
    """
    run_settings = schema.normalized(input_dict)
    # can't have chain_std_deviation and chains_per_parameters defaults with restart_chain
    if run_settings.get('mcmc') is not None:
            if run_settings['mcmc'].get('restart_chain') is None:
                run_settings['mcmc']['chains_per_parameter'] = 2
                run_settings['mcmc']['chain_std_deviation'] = 0.1
    if not schema.validate(run_settings):
        raise ValueError(schema.errors)
    return run_settings

def main():
    args = parser.parse_args(sys.argv[1:])

    # if desired, check datasets and return
    if args.check_datasets:
        dataset_filenames = sorted(recursive_glob(args.check_datasets, '*.json'))
        errors = []
        for dataset in dataset_filenames:
            try:
                load_datasets([dataset])
            except (ValueError, DatasetError) as e:
                errors.append(e)
        if len(errors) > 0:
            print(*errors, sep='\n')
            return 1
        else:
            return 0

    # if we aren't checking datasets, then we will check

    # handle verbosity
    verbosity = {0: logging.WARNING,
                 1: logging.INFO,
                 2: logging.DEBUG}
    user_verbosity = args.verbose if args.verbose < 2 else 2
    logging.basicConfig(level=verbosity[user_verbosity])

    # run ESPEI fitting
    # create the scheduler if not passed
    if args.scheduler == 'MPIPool':
        from emcee.utils import MPIPool
        # code recommended by emcee: if not master, wait for instructions then exit
        client = MPIPool()
        if not client.is_master():
            logging.warning('MPIPool is not master. Waiting for instructions...')
            client.wait()
            sys.exit(0)
        logging.info("Using MPIPool on {} MPI ranks".format(client.size))
    elif not args.scheduler:
        from distributed import LocalCluster
        args.scheduler = LocalCluster(n_workers=int(multiprocessing.cpu_count()/2), threads_per_worker=1, processes=True)
        client = ImmediateClient(args.scheduler)
        logging.info("Running with dask scheduler: %s [%s cores]" % (args.scheduler, sum(client.ncores().values())))
        try:
            logging.info("bokeh server for dask scheduler at localhost:{}".format(client.scheduler_info()['services']['bokeh']))
        except KeyError:
            logging.info("Install bokeh to use the dask bokeh server.")
    else:
        raise ValueError('Custom schedulers not supported. Use \'MPIPool\' or accept the default Dask LocalCluster.')
    # load datasets and handle i/o
    datasets = load_datasets(sorted(recursive_glob(args.datasets, '*.json')))
    tracefile = args.tracefile if args.tracefile else None
    probfile = args.probfile if args.probfile else None
    run_mcmc = not args.no_mcmc
    # check that the MCMC output files do not already exist
    # only matters if we are actually running MCMC
    if run_mcmc:
        if os.path.exists(tracefile):
            raise OSError('Tracefile "{}" exists and would be overwritten by a new run. Use --tracefile to set a different name.'.format(tracefile))
        if os.path.exists(probfile):
            raise OSError('Probfile "{}" exists and would be overwritten by a new run. Use --probfile to set a different name.'.format(tracefile))
    if args.input_tdb:
        resume_tdb = Database(args.input_tdb)
    else:
        resume_tdb = None
    if args.restart:
        if not resume_tdb:
            raise ValueError('The --input-tdb option must be specified in order to start a run from a previous chain.')
        restart_chain = np.load(args.restart)
    else:
        restart_chain = None

    dbf, sampler, parameters = fit(args.fit_settings, datasets, scheduler=client,
                                   tracefile=tracefile, probfile=probfile,
                                   resume=resume_tdb, run_mcmc=run_mcmc,
                                   mcmc_steps=args.mcmc_steps, restart_chain=restart_chain,
                                   save_interval=args.save_interval)
    dbf.to_file(args.output_tdb, if_exists='overwrite')

if __name__ == '__main__':
    main()
