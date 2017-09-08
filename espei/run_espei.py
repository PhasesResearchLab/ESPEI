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

from espei import fit
from espei.utils import ImmediateClient
from espei.datasets import DatasetError, load_datasets, recursive_glob

parser = argparse.ArgumentParser(description=__doc__)

parser.add_argument(
    "-v", "--verbose",
    action="count",
    default=0,
    help="Set the logging verbosity. Stacks up to two times (-vv)."
    )

parser.add_argument(
    "--scheduler",
    metavar="HOST:PORT",
    help="Host and port of dask distributed scheduler or 'MPIPool' to use MPI.")

parser.add_argument(
    "--tracefile",
    metavar="FILE",
    default='chain.npy',
    help="Output file for recording MCMC trace array. Defaults to chain.npy")

parser.add_argument(
    "--probfile",
    metavar="FILE",
    default='lnprob.npy',
    help="Output file for recording MCMC log probability array corresponding to chain. Defaults to lnprob.npy")

parser.add_argument(
    "--fit-settings",
    metavar="FILE",
    default="input.json",
    help="Input JSON file with settings for fit")

parser.add_argument(
    "--datasets",
    metavar="PATH",
    default=os.getcwd(),
    help="Path containing input datasets as JSON files. Datasets can be organized into sub-directories.")

parser.add_argument(
    "--input-tdb",
    metavar="FILE",
    default=None,
    help="Input TDB file, with desired degrees of freedom to fit specified as FUNCTIONs starting with 'VV'")

parser.add_argument(
    "--output-tdb",
    metavar="FILE",
    default="out.tdb",
    help="Output TDB file")

parser.add_argument(
    "--mcmc-steps",
    default=1000,
    type=int,
    metavar="",
    help="Number of MCMC steps. Total chain steps is (mcmc steps * DOF).")

parser.add_argument(
    "--save-interval",
    default=100,
    type=int,
    metavar="",
    help="Controls the interval for saving the MCMC chain")

parser.add_argument(
    "--no-mcmc",
    action="store_true",
    help="Turns off MCMC calculation. Useful for first-principles only run.")

parser.add_argument(
    "--check-datasets",
    metavar="PATH",
    default=None,
    help="Check input datasets at the path. Does not run a fit.")

parser.add_argument(
    "--restart",
    metavar="FILE",
    default=None,
    help="Restart a run with the specified chain trace. Requires input-tdb to be specified.")

def main():
    args = parser.parse_args(sys.argv[1:])

    # handle verbosity
    verbosity = {0: logging.WARNING,
                 1: logging.INFO,
                 2: logging.DEBUG}
    user_verbosity = args.verbose if args.verbose < 2 else 2
    logging.basicConfig(level=verbosity[user_verbosity])

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
