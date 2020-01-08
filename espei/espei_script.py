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
import json

import numpy as np
import yaml
import dask
import distributed
import sympy
import symengine
import emcee
import pycalphad
from pycalphad import Database

import espei
from espei.validation import schema
from espei import generate_parameters
from espei.utils import ImmediateClient, get_dask_config_paths, database_symbols_to_fit
from espei.datasets import DatasetError, load_datasets, recursive_glob, apply_tags, add_ideal_exclusions
from espei.optimizers.opt_mcmc import EmceeOptimizer

TRACE = 15  # TRACE logging level


parser = argparse.ArgumentParser(description=__doc__)

parser.add_argument(
    "--input", "--in",
    default=None,
    help="Input file for the run. Should be either a `YAML` or `JSON` file."
    )

parser.add_argument(
    "--check-datasets",
    metavar="PATH",
    default=None,
    help="Check input datasets at the path. Does not run ESPEI.")

parser.add_argument("--version", "-v", action='version',
                    version='%(prog)s version '+str(espei.__version__))


def log_version_info():
    """Print version info to the log"""
    logging.info('espei version       ' + str(espei.__version__))
    logging.debug('pycalphad version   ' + str(pycalphad.__version__))
    logging.debug('dask version        ' + str(dask.__version__))
    logging.debug('distributed version ' + str(distributed.__version__))
    logging.debug('sympy version       ' + str(sympy.__version__))
    logging.debug('symengine version   ' + str(symengine.__version__))
    logging.debug('emcee version       ' + str(emcee.__version__))
    logging.info("If you use ESPEI for work presented in a publication, we ask that you cite the following paper:\n    {}".format(espei.__citation__))

def get_dask_config_paths():
    candidates = dask.config.paths
    file_paths = []
    for path in candidates:
        if os.path.exists(path):
            if os.path.isdir(path):
                file_paths.extend(sorted([
                    os.path.join(path, p)
                    for p in os.listdir(path)
                    if os.path.splitext(p)[1].lower() in ('.json', '.yaml', '.yml')
                ]))
            else:
                file_paths.append(path)
    return file_paths

def _raise_dask_work_stealing():
    """
    Raise if work stealing is turn on in dask

    Raises
    -------
    ValueError

    """
    import distributed
    has_work_stealing = distributed.config['distributed']['scheduler']['work-stealing']
    if has_work_stealing:
        raise ValueError("The parameter 'work-stealing' is on in dask. Enabling this parameter causes some instability. "
            "Set 'distributed.scheduler.work-stealing: False' in your dask configuration. "
            "Configuration files on this machine are:\n{} (latter files have priority).\n"
            "See the example at http://espei.org/en/latest/installation.html#configuration for more.".format(get_dask_config_paths()))


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
    # can't have chain_std_deviation and chains_per_parameter defaults with restart_trace
    if run_settings.get('mcmc') is not None:
            if run_settings['mcmc'].get('restart_trace') is None:
                run_settings['mcmc']['chains_per_parameter'] = run_settings['mcmc'].get('chains_per_parameter', 2)
                run_settings['mcmc']['chain_std_deviation'] = run_settings['mcmc'].get('chain_std_deviation', 0.1)
    if not schema.validate(run_settings):
        raise ValueError(schema.errors)
    return run_settings


def run_espei(run_settings):
    """Wrapper around the ESPEI fitting procedure, taking only a settings dictionary.

    Parameters
    ----------
    run_settings : dict
        Dictionary of input settings

    Returns
    -------
    Either a Database (for generate parameters only) or a tuple of (Database, sampler)
    """
    run_settings = get_run_settings(run_settings)
    system_settings = run_settings['system']
    output_settings = run_settings['output']
    generate_parameters_settings = run_settings.get('generate_parameters')
    mcmc_settings = run_settings.get('mcmc')

    # handle verbosity
    verbosity = {
        0: logging.WARNING,
        1: logging.INFO,
        2: TRACE,
        3: logging.DEBUG
    }
    logging.basicConfig(level=verbosity[output_settings['verbosity']], filename=output_settings['logfile'])

    log_version_info()

    # load datasets and handle i/o
    logging.log(TRACE, 'Loading and checking datasets.')
    dataset_path = system_settings['datasets']
    datasets = load_datasets(sorted(recursive_glob(dataset_path, '*.json')))
    if len(datasets.all()) == 0:
        logging.warning('No datasets were found in the path {}. This should be a directory containing dataset files ending in `.json`.'.format(dataset_path))
    apply_tags(datasets, system_settings.get('tags', dict()))
    add_ideal_exclusions(datasets)
    logging.log(TRACE, 'Finished checking datasets')

    with open(system_settings['phase_models']) as fp:
        phase_models = json.load(fp)

    if generate_parameters_settings is not None:
        refdata = generate_parameters_settings['ref_state']
        excess_model = generate_parameters_settings['excess_model']
        ridge_alpha = generate_parameters_settings['ridge_alpha']
        aicc_penalty = generate_parameters_settings['aicc_penalty_factor']
        input_dbf = generate_parameters_settings.get('input_db', None)
        if input_dbf is not None:
            input_dbf = Database(input_dbf)
        dbf = generate_parameters(phase_models, datasets, refdata, excess_model,
                                  ridge_alpha=ridge_alpha, dbf=input_dbf,
                                  aicc_penalty_factor=aicc_penalty,)
        dbf.to_file(output_settings['output_db'], if_exists='overwrite')

    if mcmc_settings is not None:
        tracefile = output_settings['tracefile']
        probfile = output_settings['probfile']
        # Set trace and prob files to None if specified by the user.
        if tracefile == 'None':
          tracefile = None
        if probfile == 'None':
          probfile = None
        # check that the MCMC output files do not already exist
        # only matters if we are actually running MCMC
        if tracefile is not None and os.path.exists(tracefile):
            raise OSError('Tracefile "{}" exists and would be overwritten by a new run. Use the ``output.tracefile`` setting to set a different name.'.format(tracefile))
        if probfile is not None and os.path.exists(probfile):
            raise OSError('Probfile "{}" exists and would be overwritten by a new run. Use the ``output.probfile`` setting to set a different name.'.format(probfile))

        # scheduler setup
        # Turn off numpy's automatic parallelization since it can interfere with our parallelization.
        os.environ["OMP_NUM_THREADS"] = "1"
        if mcmc_settings['scheduler'] == 'dask':
            _raise_dask_work_stealing()  # check for work-stealing
            from distributed import LocalCluster
            cores = mcmc_settings.get('cores', multiprocessing.cpu_count())
            if (cores > multiprocessing.cpu_count()):
                cores = multiprocessing.cpu_count()
                logging.warning("The number of cores chosen is larger than available. "
                                "Defaulting to run on the {} available cores.".format(cores))
            # TODO: make dask-scheduler-verbosity a YAML input so that users can debug. Should have the same log levels as verbosity
            scheduler = LocalCluster(n_workers=cores, threads_per_worker=1, processes=True, memory_limit=0)
            client = ImmediateClient(scheduler)
            client.run(logging.basicConfig, level=verbosity[output_settings['verbosity']], filename=output_settings['logfile'])
            logging.info("Running with dask scheduler: %s [%s cores]" % (scheduler, sum(client.ncores().values())))
            try:
                bokeh_server_info = client.scheduler_info()['services']['bokeh']
                logging.info("bokeh server for dask scheduler at localhost:{}".format(bokeh_server_info))
            except KeyError:
                logging.info("Install bokeh to use the dask bokeh server.")
        elif mcmc_settings['scheduler'] == 'None':
            client = None
            logging.info("Not using a parallel scheduler. ESPEI is running MCMC on a single core.")
        elif mcmc_settings['scheduler'] == 'multiprocessing':
            logging.info("Using multiprocessing to parallelize")
            cores = mcmc_settings.get('cores', multiprocessing.cpu_count())
            client = multiprocessing.Pool(processes=cores)
        else: # we were passed a scheduler file name
            _raise_dask_work_stealing()  # check for work-stealing
            client = ImmediateClient(scheduler_file=mcmc_settings['scheduler'])
            client.run(logging.basicConfig, level=verbosity[output_settings['verbosity']], filename=output_settings['logfile'])
            logging.info("Running with dask scheduler: %s [%s cores]" % (client.scheduler, sum(client.ncores().values())))

        # get a Database
        if mcmc_settings.get('input_db'):
            dbf = Database(mcmc_settings.get('input_db'))

        # load the restart trace if needed
        if mcmc_settings.get('restart_trace'):
            restart_trace = np.load(mcmc_settings.get('restart_trace'))
        else:
            restart_trace = None

        # load the remaining mcmc fitting parameters
        iterations = mcmc_settings.get('iterations')
        save_interval = mcmc_settings.get('save_interval')
        chains_per_parameter = mcmc_settings.get('chains_per_parameter')
        chain_std_deviation = mcmc_settings.get('chain_std_deviation')
        deterministic = mcmc_settings.get('deterministic')
        prior = mcmc_settings.get('prior')
        data_weights = mcmc_settings.get('data_weights')
        syms = mcmc_settings.get('symbols')

        # set up and run the EmceeOptimizer
        optimizer = EmceeOptimizer(dbf, scheduler=client)
        optimizer.save_interval = save_interval
        all_symbols = syms if syms is not None else database_symbols_to_fit(dbf)
        optimizer.fit(all_symbols, datasets, prior=prior, iterations=iterations,
                      chains_per_parameter=chains_per_parameter,
                      chain_std_deviation=chain_std_deviation,
                      deterministic=deterministic, restart_trace=restart_trace,
                      tracefile=tracefile, probfile=probfile,
                      mcmc_data_weights=data_weights)
        optimizer.commit()

        optimizer.dbf.to_file(output_settings['output_db'], if_exists='overwrite')
        # close the scheduler, if possible
        if hasattr(client, 'close'):
                client.close()
        return optimizer.dbf, optimizer.sampler
    return dbf


def main():
    """
    Handle starting ESPEI from the command line.
    Parse command line arguments and input file.
    """
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

    # if we aren't checking datasets, then we will check that the input file exists
    input_file = args.input
    if input_file is None:
        raise ValueError('To run ESPEI, provide an input file with the `--input` option.')

    # continue with setup
    # load the settings
    ext = os.path.splitext(input_file)[-1]
    if ext == '.yml' or ext == '.yaml':
        with open(input_file) as f:
            input_settings = yaml.safe_load(f)
    elif ext == '.json':
        with open(input_file) as f:
            input_settings = json.load(f)
    else:
        raise ValueError('Unknown file type {} for input file {}. YAML and JSON are supported'.format(ext, input_file))

    run_espei(input_settings)


if __name__ == '__main__':
    main()
