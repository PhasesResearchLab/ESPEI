"""
Automated fitting script.

A minimal run must specify an input.json and a datasets folder containing input files.
"""

import os
import argparse
import logging
import multiprocessing
import sys

from distributed import LocalCluster
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
    "--dask-scheduler",
    metavar="HOST:PORT",
    help="Host and port of dask distributed scheduler")

parser.add_argument(
    "--tracefile",
    metavar="FILE",
    default='chain.txt',
    help="Output file for recording MCMC trace (txt). Defaults to chain.txt")

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
    if not args.dask_scheduler:
        args.dask_scheduler = LocalCluster(n_workers=int(multiprocessing.cpu_count()/2), threads_per_worker=1, processes=True)
    client = ImmediateClient(args.dask_scheduler)
    logging.info("Running with dask scheduler: %s [%s cores]" % (args.dask_scheduler, sum(client.ncores().values())))
    try:
        logging.info("bokeh server for dask scheduler at localhost:{}".format(client.scheduler_info()['services']['bokeh']))
    except KeyError:
        logging.info("Install bokeh to use the dask bokeh server.")
    # load datasets and handle i/o
    datasets = load_datasets(sorted(recursive_glob(args.datasets, '*.json')))
    tracefile = args.tracefile if args.tracefile else None
    run_mcmc = not args.no_mcmc
    if args.input_tdb:
        resume = Database(args.input_tdb)
    else:
        resume = None
    dbf, sampler, parameters = fit(args.fit_settings, datasets, scheduler=client,
                                   tracefile=tracefile, resume=resume, run_mcmc=run_mcmc,
                                   mcmc_steps=args.mcmc_steps, save_interval=args.save_interval)
    dbf.to_file(args.output_tdb, if_exists='overwrite')

if __name__ == '__main__':
    main()
