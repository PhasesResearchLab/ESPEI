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

parser.add_argument(
    "--iter-record",
    metavar="FILE",
    help="Output file for recording iterations (CSV)")

parser.add_argument(
    "--fit-settings",
    metavar="FILE",
    default="input.json",
    help="Input JSON file with settings for fit")

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
    recfile = open(args.iter_record, 'a') if args.iter_record else None
    try:
        dbf, mdl, model_dof = fit(args.fit_settings, datasets, scheduler=client, recfile=recfile)
    finally:
        if recfile:
            recfile.close()
        dbf.to_file(args.output_tdb, if_exists='overwrite')



