"""
Automated fitting script.
"""
import os
import sys
import fnmatch
import argparse
import logging
import multiprocessing
from paramselect import fit, load_datasets
from distributed import Client, LocalCluster

parser = argparse.ArgumentParser(description=__doc__)

parser.add_argument(
    "--dask-scheduler",
    metavar="HOST:PORT",
    help="Host and port of dask distributed scheduler")

def recursive_glob(start, pattern):
    matches = []
    for root, dirnames, filenames in os.walk(start):
        for filename in fnmatch.filter(filenames, pattern):
            matches.append(os.path.join(root, filename))
    return matches

if __name__ == '__main__':
    args = parser.parse_args(sys.argv[1:])
    if not args.dask_scheduler:
        args.dask_scheduler = LocalCluster(n_workers=int(multiprocessing.cpu_count() / 2), threads_per_worker=1, nanny=True)
    client = Client(args.dask_scheduler)
    logging.info(
        "Running with dask scheduler: %s [%s cores]" % (
            args.dask_scheduler,
            sum(client.ncores().values())))
    datasets = load_datasets(sorted(recursive_glob('Al-Ni', '*.json')))
    dbf, mdl, model_dof = fit('input.json', datasets, saveall=True, scheduler=client)
