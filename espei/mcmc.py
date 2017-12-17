"""Module for running MCMC in ESPEI"""

import textwrap, time, sys, operator, logging
from collections import OrderedDict, defaultdict

import dask
import sympy
import numpy as np
import tinydb
from numpy.linalg import LinAlgError
from pycalphad import calculate, equilibrium, CompiledModel, variables as v
import emcee

from espei.utils import database_symbols_to_fit, optimal_parameters


def estimate_hyperplane(dbf, comps, phases, current_statevars, comp_dicts, phase_models, parameters):
    region_chemical_potentials = []
    parameters = OrderedDict(sorted(parameters.items(), key=str))
    for cond_dict, phase_flag in comp_dicts:
        # We are now considering a particular tie vertex
        for key, val in cond_dict.items():
            if val is None:
                cond_dict[key] = np.nan
        cond_dict.update(current_statevars)
        if np.any(np.isnan(list(cond_dict.values()))):
            # This composition is unknown -- it doesn't contribute to hyperplane estimation
            pass
        else:
            # Extract chemical potential hyperplane from multi-phase calculation
            # Note that we consider all phases in the system, not just ones in this tie region
            multi_eqdata = equilibrium(dbf, comps, phases, cond_dict, verbose=False,
                                       model=phase_models, scheduler=dask.local.get_sync, parameters=parameters)
            # Does there exist only a single phase in the result with zero internal degrees of freedom?
            # We should exclude those chemical potentials from the average because they are meaningless.
            num_phases = len(np.squeeze(multi_eqdata['Phase'].values != ''))
            zero_dof = np.all((multi_eqdata['Y'].values == 1.) | np.isnan(multi_eqdata['Y'].values))
            if (num_phases == 1) and zero_dof:
                region_chemical_potentials.append(np.full_like(np.squeeze(multi_eqdata['MU'].values), np.nan))
            else:
                region_chemical_potentials.append(np.squeeze(multi_eqdata['MU'].values))
    region_chemical_potentials = np.nanmean(region_chemical_potentials, axis=0, dtype=np.float)
    return region_chemical_potentials


def tieline_error(dbf, comps, current_phase, cond_dict, region_chemical_potentials, phase_flag,
                  phase_models, parameters, debug_mode=False):
    if np.any(np.isnan(list(cond_dict.values()))):
        # We don't actually know the phase composition here, so we estimate it
        single_eqdata = calculate(dbf, comps, [current_phase],
                                  T=cond_dict[v.T], P=cond_dict[v.P],
                                  model=phase_models, parameters=parameters, pdens=100)
        driving_force = np.multiply(region_chemical_potentials,
                                    single_eqdata['X'].values).sum(axis=-1) - single_eqdata['GM'].values
        error = float(driving_force.max())
    elif phase_flag == 'disordered':
        # Construct disordered sublattice configuration from composition dict
        # Compute energy
        # Compute residual driving force
        # TODO: Check that it actually makes sense to declare this phase 'disordered'
        num_dof = sum([len(set(c).intersection(comps)) for c in dbf.phases[current_phase].constituents])
        desired_sitefracs = np.ones(num_dof, dtype=np.float)
        dof_idx = 0
        for c in dbf.phases[current_phase].constituents:
            dof = sorted(set(c).intersection(comps))
            if (len(dof) == 1) and (dof[0] == 'VA'):
                return 0
            # If it's disordered config of BCC_B2 with VA, disordered config is tiny vacancy count
            sitefracs_to_add = np.array([cond_dict.get(v.X(d)) for d in dof],
                                        dtype=np.float)
            # Fix composition of dependent component
            sitefracs_to_add[np.isnan(sitefracs_to_add)] = 1 - np.nansum(sitefracs_to_add)
            desired_sitefracs[dof_idx:dof_idx + len(dof)] = sitefracs_to_add
            dof_idx += len(dof)
        single_eqdata = calculate(dbf, comps, [current_phase],
                                  T=cond_dict[v.T], P=cond_dict[v.P], points=desired_sitefracs,
                                  model=phase_models, parameters=parameters)
        driving_force = np.multiply(region_chemical_potentials,
                                    single_eqdata['X'].values).sum(axis=-1) - single_eqdata['GM'].values
        error = float(np.squeeze(driving_force))
    else:
        # Extract energies from single-phase calculations
        single_eqdata = equilibrium(dbf, comps, [current_phase], cond_dict, verbose=False,
                                    model=phase_models,
                                    scheduler=dask.local.get_sync, parameters=parameters)
        if np.all(np.isnan(single_eqdata['NP'].values)):
            error_time = time.time()
            template_error = """
            from pycalphad import Database, equilibrium
            from pycalphad.variables import T, P, X
            import dask
            dbf_string = \"\"\"
            {0}
            \"\"\"
            dbf = Database(dbf_string)
            comps = {1}
            phases = {2}
            cond_dict = {3}
            parameters = {4}
            equilibrium(dbf, comps, phases, cond_dict, scheduler=dask.local.get_sync, parameters=parameters)
            """
            template_error = textwrap.dedent(template_error)
            if debug_mode:
                logging.warning('Dumping', 'error-'+str(error_time)+'.py')
                with open('error-'+str(error_time)+'.py', 'w') as f:
                    f.write(template_error.format(dbf.to_string(fmt='tdb'), comps, [current_phase], cond_dict, {key: float(x) for key, x in parameters.items()}))
        # Sometimes we can get a miscibility gap in our "single-phase" calculation
        # Choose the weighted mixture of site fractions
            logging.debug('Calculation failure: all NaN phases with conditions: {}'.format(cond_dict))
            return 0
        select_energy = float(single_eqdata['GM'].values)
        region_comps = []
        for comp in [c for c in sorted(comps) if c != 'VA']:
            region_comps.append(cond_dict.get(v.X(comp), np.nan))
        region_comps[region_comps.index(np.nan)] = 1 - np.nansum(region_comps)
        error = np.multiply(region_chemical_potentials, region_comps).sum() - select_energy
        error = float(error)
    return error


def multi_phase_fit(dbf, comps, phases, datasets, phase_models, parameters=None, scheduler=None):
    scheduler = scheduler or dask.local
    # TODO: support distributed schedulers for multi_phase_fit.
    # This can be done if the scheduler passed is a distributed.worker_client
    if scheduler is not dask.local:
        raise NotImplementedError('Schedulers other than dask.local are not currently supported for multiphase fitting.')
    desired_data = datasets.search((tinydb.where('output') == 'ZPF') &
                                   (tinydb.where('components').test(lambda x: set(x).issubset(comps))) &
                                   (tinydb.where('phases').test(lambda x: len(set(phases).intersection(x)) > 0)))

    def safe_get(itms, idxx):
        try:
            return itms[idxx]
        except IndexError:
            return None

    fit_jobs = []
    for data in desired_data:
        payload = data['values']
        conditions = data['conditions']
        data_comps = list(set(data['components']).union({'VA'}))
        phase_regions = defaultdict(lambda: list())
        # TODO: Fix to only include equilibria listed in 'phases'
        for idx, p in enumerate(payload):
            phase_key = tuple(sorted(rp[0] for rp in p))
            if len(phase_key) < 2:
                # Skip single-phase regions for fitting purposes
                continue
            # Need to sort 'p' here so we have the sorted ordering used in 'phase_key'
            # rp[3] optionally contains additional flags, e.g., "disordered", to help the solver
            comp_dicts = [(dict(zip([v.X(x.upper()) for x in rp[1]], rp[2])), safe_get(rp, 3))
                          for rp in sorted(p, key=operator.itemgetter(0))]
            cur_conds = {}
            for key, value in conditions.items():
                value = np.atleast_1d(np.asarray(value))
                if len(value) > 1:
                    value = value[idx]
                cur_conds[getattr(v, key)] = float(value)
            phase_regions[phase_key].append((cur_conds, comp_dicts))
        for region, region_eq in phase_regions.items():
            for req in region_eq:
                # We are now considering a particular tie region
                current_statevars, comp_dicts = req
                region_chemical_potentials = \
                    dask.delayed(estimate_hyperplane)(dbf, data_comps, phases, current_statevars, comp_dicts,
                                                      phase_models, parameters)
                # Now perform the equilibrium calculation for the isolated phases and add the result to the error record
                for current_phase, cond_dict in zip(region, comp_dicts):
                    # TODO: Messy unpacking
                    cond_dict, phase_flag = cond_dict
                    # We are now considering a particular tie vertex
                    for key, val in cond_dict.items():
                        if val is None:
                            cond_dict[key] = np.nan
                    cond_dict.update(current_statevars)
                    error = dask.delayed(tieline_error)(dbf, data_comps, current_phase, cond_dict, region_chemical_potentials, phase_flag,
                                                        phase_models, parameters)
                    fit_jobs.append(error)
    errors = dask.compute(*fit_jobs, get=scheduler.get_sync)
    return errors


def lnprob(params, comps=None, dbf=None, phases=None, datasets=None,
           symbols_to_fit=None, phase_models=None, scheduler=None):
    """
    Returns the error from multiphase fitting as a log probability.
    """
    parameters = {param_name: param for param_name, param in zip(symbols_to_fit, params)}
    try:
        iter_error = multi_phase_fit(dbf, comps, phases, datasets, phase_models,
                                     parameters=parameters, scheduler=scheduler)
    except (ValueError, LinAlgError) as e:
        iter_error = [np.inf]
    iter_error = [np.inf if np.isnan(x) else x ** 2 for x in iter_error]
    iter_error = -np.sum(iter_error)
    return np.array(iter_error, dtype=np.float64)


def generate_parameter_distribution(parameters, num_samples, std_deviation, deterministic=True):
    """
    Return an array of num_samples from a Gaussian distribution about each parameter.

    Parameters
    ----------
    parameters : ndarray
        1D array of initial parameters that will be the mean of the distribution.
    num_samples : int
        Number of chains to initialize.
    std_deviation : float
        Fractional standard deviation of the parameters to use for initialization.
    deterministic : bool
        True if the parameters should be generated deterministically.

    Returns
    -------
    ndarray
    """
    if deterministic:
        rng = np.random.RandomState(1769)
    else:
        rng = np.random.RandomState()
    # apply a Gaussian random to each parameter with std dev of std_deviation*parameter
    tiled_parameters = np.tile(parameters, (num_samples, 1))
    return rng.normal(tiled_parameters, np.abs(tiled_parameters * std_deviation))


def mcmc_fit(dbf, datasets, mcmc_steps=1000, save_interval=100, chains_per_parameter=2,
             chain_std_deviation=0.1, scheduler=None, tracefile=None, probfile=None,
             restart_chain=None, deterministic=True,):
    """Run Markov Chain Monte Carlo on the Database given datasets

    Parameters
    ----------
    dbf : Database
        A pycalphad Database to fit with symbols to fit prefixed with `VV`
        followed by a number, e.g. `VV0001`
    datasets : PickleableTinyDB
        A database of single- and multi-phase to fit
    mcmc_steps : int
        Number of chain steps to calculate in MCMC. Note the flattened chain will
        have (mcmc_steps*DOF) values. Default is 1000 steps.
    save_interval :int
        interval of steps to save the chain to the tracefile and probfile
    chains_per_parameter : int
        number of chains for each parameter. Must be an even integer greater or
        equal to 2. Defaults to 2.
    chain_std_deviation : float
        standard deviation of normal for parameter initialization as a fraction
        of each parameter. Must be greater than 0. Default is 0.1, which is 10%.
    scheduler : callable
        Scheduler to use with emcee. Must implement a map method.
    tracefile : str
        filename to store the flattened chain with NumPy.save. Array has shape
        (nwalkers, iterations, nparams)
    probfile : str
        filename to store the flattened ln probability with NumPy.save
    restart_chain : np.ndarray
        ndarray of the previous chain. Should have shape (nwalkers, iterations, nparams)
    deterministic : bool
        If True, the emcee sampler will be seeded to give deterministic sampling
        draws. This will ensure that the runs with the exact same database,
        chains_per_parameter, and chain_std_deviation (or restart_chain) will
        produce exactly the same results.

    Returns
    -------
    dbf : Database
        Resulting pycalphad database of optimized parameters
    sampler : EnsembleSampler, ndarray)
        emcee sampler for further data wrangling
    """
    comps = sorted([sp for sp in dbf.elements])
    symbols_to_fit = database_symbols_to_fit(dbf)

    if len(symbols_to_fit) == 0:
        raise ValueError('No degrees of freedom. Database must contain symbols starting with \'V\' or \'VV\', followed by a number.')
    else:
        logging.info('Fitting {} degrees of freedom.'.format(len(symbols_to_fit)))

    for x in symbols_to_fit:
        if isinstance(dbf.symbols[x], sympy.Piecewise):
            logging.debug('Replacing {} in database'.format(x))
            dbf.symbols[x] = dbf.symbols[x].args[0].expr

    # get initial parameters and remove these from the database
    # we'll replace them with SymPy symbols initialized to 0 in the phase models
    initial_parameters = np.array([np.array(float(dbf.symbols[x])) for x in symbols_to_fit])
    for x in symbols_to_fit:
        del dbf.symbols[x]

    # construct the models for each phase, substituting in the SymPy symbol to fit.
    phase_models = dict()
    logging.debug('Building phase models')
    # 0 is placeholder value
    phases = sorted(dbf.phases.keys())
    for phase_name in phases:
        mod = CompiledModel(dbf, comps, phase_name, parameters=OrderedDict([(sympy.Symbol(s), 0) for s in symbols_to_fit]))
        phase_models[phase_name] = mod
    logging.debug('Finished building phase models')
    dbf = dask.delayed(dbf, pure=True)
    phase_models = dask.delayed(phase_models, pure=True)

    # contect for the log probability function
    error_context = {'comps': comps, 'dbf': dbf,
                     'phases': phases, 'phase_models': phase_models,
                     'datasets': datasets, 'symbols_to_fit': symbols_to_fit,
                     }

    def save_sampler_state(sampler):
        if tracefile:
            logging.debug('Writing chain to {}'.format(tracefile))
            np.save(tracefile, sampler.chain)
        if probfile:
            logging.debug('Writing lnprob to {}'.format(probfile))
            np.save(probfile, sampler.lnprobability)


    # initialize the walkers either fresh or from the restart
    if restart_chain is not None:
        walkers = restart_chain[np.nonzero(restart_chain)].reshape(
            (restart_chain.shape[0], -1, restart_chain.shape[2]))[:, -1, :]
        nwalkers = walkers.shape[0]
        ndim = walkers.shape[1]
        initial_parameters = walkers.mean(axis=0)
        logging.info('Restarting from previous calculation with {} chains ({} per parameter).'.format(nwalkers, nwalkers / ndim))
        logging.debug('Means of restarting parameters are {}'.format(initial_parameters))
        logging.debug('Standard deviations of restarting parameters are {}'.format(walkers.std(axis=0)))
    else:
        logging.debug('Initial parameters: {}'.format(initial_parameters))
        ndim = initial_parameters.size
        nwalkers = ndim * chains_per_parameter
        logging.info('Initializing {} chains with {} chains per parameter.'.format(nwalkers, chains_per_parameter))
        walkers = generate_parameter_distribution(initial_parameters, nwalkers, chain_std_deviation, deterministic=deterministic)

    # the pool must implement a map function
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, kwargs=error_context, pool=scheduler)
    if deterministic:
        from espei.rstate import numpy_rstate
        sampler.random_state = numpy_rstate
        logging.info('Using a deterministic ensemble sampler.')
    progbar_width = 30
    logging.info('Running MCMC with {} steps.'.format(mcmc_steps))
    try:
        for i, result in enumerate(sampler.sample(walkers, iterations=mcmc_steps)):
            # progress bar
            if (i + 1) % save_interval == 0:
                save_sampler_state(sampler)
                logging.debug('Acceptance ratios for parameters: {}'.format(sampler.acceptance_fraction))
            n = int((progbar_width + 1) * float(i) / mcmc_steps)
            sys.stdout.write("\r[{0}{1}] ({2} of {3})\n".format('#'*n, ' '*(progbar_width - n), i + 1, mcmc_steps))
        n = int((progbar_width + 1) * float(i + 1) / mcmc_steps)
        sys.stdout.write("\r[{0}{1}] ({2} of {3})\n".format('#'*n, ' '*(progbar_width - n), i + 1, mcmc_steps))
    except KeyboardInterrupt:
        pass
    # final processing
    save_sampler_state(sampler)
    optimal_params = optimal_parameters(sampler.chain, sampler.lnprobability)
    logging.debug('Intial parameters: {}'.format(initial_parameters))
    logging.debug('Optimal parameters: {}'.format(optimal_params))
    logging.debug('Change in parameters: {}'.format(np.abs(initial_parameters - optimal_params) / initial_parameters))
    dbf = dbf.compute()
    for param_name, value in zip(symbols_to_fit, optimal_params):
        dbf.symbols[param_name] = value
    logging.info('MCMC complete.')
    return dbf, sampler
