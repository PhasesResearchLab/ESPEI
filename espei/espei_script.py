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
import warnings

import numpy as np
import yaml
import dask
import distributed
import sympy
import symengine
from tinydb import where
import emcee
import pycalphad
from pycalphad import Database

import espei
from espei.validation import schema
from espei import generate_parameters
from espei.utils import ImmediateClient, database_symbols_to_fit
from espei.datasets import DatasetError, load_datasets, recursive_glob, apply_tags
from espei.optimizers.opt_mcmc import EmceeOptimizer

_log = logging.getLogger(__name__)

# Force distributed's work-stealing to be False
dask.config.set({'distributed.scheduler.work-stealing': False})

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
    _log.info('espei version       %s', espei.__version__)
    _log.debug('pycalphad version   %s', pycalphad.__version__)
    _log.debug('dask version        %s', dask.__version__)
    _log.debug('distributed version %s', distributed.__version__)
    _log.debug('sympy version       %s', sympy.__version__)
    _log.debug('symengine version   %s', symengine.__version__)
    _log.debug('emcee version       %s', emcee.__version__)
    _log.info("If you use ESPEI for work presented in a publication, we ask that you cite the following paper:\n    %s", espei.__citation__)


def _raise_dask_work_stealing():
    """
    Raise if work stealing is turned on in dask

    Raises
    -------
    ValueError

    Examples
    --------
    >>> _raise_dask_work_stealing()  # should not raise if dask is set correctly

    """
    import dask, distributed
    has_work_stealing = dask.config.get('distributed.scheduler.work_stealing')
    if has_work_stealing:
        raise ValueError("The parameter 'distributed.scheduler.work-stealing' is on in dask. "
                         "This parameter causes some instability for long-running processes. "
                         "As of ESPEI v0.7.9, 'work-stealing' should be disabled automatically. "
                         "If you are seeing this error, please contact a developer.")


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
        if run_settings['mcmc']['scheduler'] == 'None':
            warnings.warn(
                "Setting scheduler to the string 'None' will be deprecated in ESPEI "
                "0.9. Use `null` in YAML or `None` in Python.", FutureWarning
            )
            run_settings['mcmc']['scheduler'] = None
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

    # Configure logger
    log_verbosity = output_settings['verbosity']
    log_filename = output_settings['logfile']
    espei.logger.config_logger(verbosity=log_verbosity, filename=log_filename)

    log_version_info()

    # load datasets and handle i/o
    _log.trace('Loading and checking datasets.')
    dataset_path = system_settings['datasets']
    datasets = load_datasets(sorted(recursive_glob(dataset_path, '*.json')))
    apply_tags(datasets, system_settings.get('tags', dict()))
    removed_ids = datasets.remove(where('disabled') == True)
    if len(removed_ids) > 0:
        _log.debug('Removed %d disabled datasets', len(removed_ids))
    if len(datasets.all()) == 0:
        _log.warning('No datasets were found in the path %s. This should be a directory containing dataset files ending in `.json`.', dataset_path)
    _log.trace('Finished checking datasets')

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
            raise OSError('Tracefile "%s" exists and would be overwritten by a new run. Use the ``output.tracefile`` setting to set a different name.', tracefile)
        if probfile is not None and os.path.exists(probfile):
            raise OSError('Probfile "%s" exists and would be overwritten by a new run. Use the ``output.probfile`` setting to set a different name.', probfile)

        # scheduler setup
        if mcmc_settings['scheduler'] is not None:
            _raise_dask_work_stealing()  # check for work-stealing
            if mcmc_settings['scheduler'] == 'dask':
                _raise_dask_work_stealing()  # check for work-stealing
                from distributed import LocalCluster
                cores = mcmc_settings.get('cores', multiprocessing.cpu_count())
                if (cores > multiprocessing.cpu_count()):
                    cores = multiprocessing.cpu_count()
                    _log.warning("The number of cores chosen is larger than available. "
                                 "Defaulting to run on the %s available cores.", cores)
                # TODO: make dask-scheduler-verbosity a YAML input so that users can debug. Should have the same log levels as verbosity
                scheduler = LocalCluster(n_workers=cores, threads_per_worker=1, processes=True, memory_limit=0)
                client = ImmediateClient(scheduler)
                try:
                    bokeh_server_info = client.scheduler_info()['services']['bokeh']
                    _log.info("bokeh server for dask scheduler at localhost:%s", bokeh_server_info)
                except KeyError:
                    _log.info("Install bokeh to use the dask bokeh server.")
            else: # we were passed a scheduler file name
                client = ImmediateClient(scheduler_file=mcmc_settings['scheduler'])
            client.run(espei.logger.config_logger, verbosity=log_verbosity, filename=log_filename)
            client.run(np.set_printoptions, linewidth=sys.maxsize)
            _log.info("Running with dask scheduler: %s [%s cores]" % (client.scheduler, sum(client.ncores().values())))
        else:
            client = None
            _log.info("Not using a parallel scheduler. ESPEI is running MCMC on a single core.")

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
        approximate_equilibrium = mcmc_settings.get('approximate_equilibrium')

        # set up and run the EmceeOptimizer
        optimizer = EmceeOptimizer(dbf, scheduler=client)
        optimizer.save_interval = save_interval
        all_symbols = syms if syms is not None else database_symbols_to_fit(dbf)
        optimizer.fit(all_symbols, datasets, prior=prior, iterations=iterations,
                      chains_per_parameter=chains_per_parameter,
                      chain_std_deviation=chain_std_deviation,
                      deterministic=deterministic, restart_trace=restart_trace,
                      tracefile=tracefile, probfile=probfile,
                      mcmc_data_weights=data_weights,
                      approximate_equilibrium=approximate_equilibrium,
                      )
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
        raise ValueError('Unknown file type %s for input file %s. YAML and JSON are supported', ext, input_file)

    run_espei(input_settings)


if __name__ == '__main__':
    main()
