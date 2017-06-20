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
from espei.utils import ImmediateClient, load_datasets, recursive_glob

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
    "--iter-record",
    metavar="FILE",
    help="Output file for recording iterations (CSV)")

parser.add_argument(
    "--tracefile",
    metavar="FILE",
    help="Output file for recording MCMC trace (HDF5)")

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


def main():
    args = parser.parse_args(sys.argv[1:])
    # handle verbosity
    verbosity = {0: logging.WARNING,
                 1: logging.INFO,
                 2: logging.DEBUG}
    user_verbosity = args.verbose if args.verbose < 2 else 2
    logging.basicConfig(level=verbosity[user_verbosity])
    # create the scheduler if not passed
    if not args.dask_scheduler:
        pass
        args.dask_scheduler = LocalCluster(n_workers=int(multiprocessing.cpu_count()/2), threads_per_worker=1, processes=True)
    client = ImmediateClient(args.dask_scheduler)
    logging.info("Running with dask scheduler: %s [%s cores]" % (args.dask_scheduler, sum(client.ncores().values())))
    logging.info("bokeh server for dask scheduler at localhost:{}".format(client.scheduler_info()['services']['bokeh']))
    # load datasets and handle i/o
    datasets = load_datasets(sorted(recursive_glob(args.datasets, '*.json')))
    recfile = open(args.iter_record, 'a') if args.iter_record else None
    tracefile = args.tracefile if args.tracefile else None
    if args.input_tdb:
        resume = Database(args.input_tdb)
    else:
        resume = None
    # fitting
    dbf, sampler, parameters = fit(args.fit_settings, datasets, scheduler=client, recfile=recfile, tracefile=tracefile, resume=resume)
    dbf.to_file(args.output_tdb, if_exists='overwrite')

if __name__ == '__main__':
    main()
