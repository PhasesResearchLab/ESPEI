"""Module for running MCMC in ESPEI"""

import textwrap, time, sys, operator, logging, itertools
from collections import OrderedDict, defaultdict

import dask
import sympy
import numpy as np
import tinydb
from numpy.linalg import LinAlgError
from pycalphad import calculate, equilibrium, Model, variables as v
import emcee

from espei.utils import database_symbols_to_fit, optimal_parameters, eq_callables_dict
from espei.core_utils import ravel_conditions


def estimate_hyperplane(dbf, comps, phases, current_statevars, comp_dicts, phase_models, parameters,
                        massfuncs=None, massgradfuncs=None,
                        callables=None, grad_callables=None,
                        hess_callables=None,
                        ):
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
                                       model=phase_models, scheduler=dask.local.get_sync, parameters=parameters,
                                       massfuncs=massfuncs,
                                       massgradfuncs=massgradfuncs,
                                       callables=callables,
                                       grad_callables=grad_callables,
                                       hess_callables=hess_callables,
                                       )
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
                  phase_models, parameters, debug_mode=False,
                  massfuncs=None, massgradfuncs=None,
                  callables=None, grad_callables=None, hess_callables=None,
                  ):
    if np.any(np.isnan(list(cond_dict.values()))):
        # We don't actually know the phase composition here, so we estimate it
        single_eqdata = calculate(dbf, comps, [current_phase],
                                  T=cond_dict[v.T], P=cond_dict[v.P],
                                  model=phase_models, parameters=parameters, pdens=100,
                                  massfuncs=massfuncs, callables=callables,
                                  )
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
                                  model=phase_models, parameters=parameters, massfuncs=massfuncs,
                                  callables=callables,)
        driving_force = np.multiply(region_chemical_potentials,
                                    single_eqdata['X'].values).sum(axis=-1) - single_eqdata['GM'].values
        error = float(np.squeeze(driving_force))
    else:
        # Extract energies from single-phase calculations
        single_eqdata = equilibrium(dbf, comps, [current_phase], cond_dict, verbose=False,
                                    model=phase_models,
                                    scheduler=dask.local.get_sync, parameters=parameters,
                                    massfuncs=massfuncs,
                                    massgradfuncs=massgradfuncs,
                                    callables=callables,
                                    grad_callables=grad_callables,
                                    hess_callables=hess_callables,
                                    )
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


def multi_phase_fit(dbf, comps, phases, datasets, phase_models, parameters=None, scheduler=None,
                    massfuncs=None, massgradfuncs=None,
                    callables=None, grad_callables=None, hess_callables=None,
                    ):
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
                                                      phase_models, parameters, massfuncs=massfuncs, massgradfuncs=massgradfuncs,
                                                      callables=callables, grad_callables=grad_callables, hess_callables=hess_callables,)
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
                                                        phase_models, parameters, massfuncs=massfuncs, massgradfuncs=massgradfuncs,
                                                        callables=callables, grad_callables=grad_callables, hess_callables=hess_callables,)
                    fit_jobs.append(error)
    errors = dask.compute(*fit_jobs, get=scheduler.get_sync)
    return errors


def calculate_points_array(phase_constituents, configuration, occupancies=None):
    """
    Calculate the points array to use in pycalphad calculate calls.

    Converts the configuration data (and occupancies for mixing data) into the
    points array by looking up the indices in the active phase constituents.

    Parameters
    ----------
    phase_constituents : list
        List of active constituents in a phase
    configuration : list
        List of the sublattice configuration
    occupancies : list
        List of sublattice occupancies. Required for mixing sublattices, otherwise takes no effect.

    Returns
    -------
    np.ndarray

    Notes
    -----
    Errors will be raised if components in the configuration are not in the
    corresponding phase constituents sublattice.
    """
    # pad the occupancies for zipping if none were passed (the case for non-mixing)
    if occupancies is None:
        occupancies = [0] * len(configuration)

    # construct the points array from zeros
    points = np.zeros(sum(len(subl) for subl in phase_constituents))
    current_subl_idx = 0  # index that marks the beginning of the sublattice
    for phase_subl, config_subl, subl_occupancies in zip(phase_constituents,
                                                         configuration,
                                                         occupancies):
        phase_subl = list(phase_subl)
        if isinstance(config_subl, (tuple, list)):
            # we have mixing on the sublattice
            for comp, comp_occupancy in zip(config_subl, subl_occupancies):
                points[
                    current_subl_idx + phase_subl.index(comp)] = comp_occupancy
        else:
            points[current_subl_idx + phase_subl.index(config_subl)] = 1
        current_subl_idx += len(phase_subl)
    return points


def get_prop_data(comps, phase_name, prop, datasets):
    """
    Return datasets that match the components, phase and property

    Parameters
    ----------
    comps : list
        List of components to get data for
    phase_name : str
        Name of the phase to get data for
    prop : str
        Property to get data for
    datasets : espei.utils.PickleableTinyDB
        Datasets to search for data

    Returns
    -------
    list
        List of dictionary datasets that match the criteria

    """
    # TODO: we should only search and get phases that have the same sublattice_site_ratios as the phase in the database
    desired_data = datasets.search(
        (tinydb.where('output').test(lambda x: x in prop)) &
        (tinydb.where('components').test(lambda x: set(x).issubset(comps))) &
        (tinydb.where('phases') == [phase_name])
    )
    return desired_data


def get_prop_samples(dbf, comps, phase_name, desired_data):
    """
    Return data values and the conditions to calculate them by pycalphad
    calculate from the datasets

    Parameters
    ----------
    dbf : pycalphad.Database
        Database to consider
    comps : list
        List of active component names
    phase_name : str
        Name of the phase to consider from the Database
    desired_data : list
        List of dictionary datasets that contain the values to sample

    Returns
    -------
    dict
        Dictionary of condition kwargs for pycalphad's calculate and the expected values

    """
    # TODO: assumes T, P as conditions
    phase_constituents = dbf.phases[phase_name].constituents
    # phase constituents must be filtered to only active:
    phase_constituents = [sorted(subl_constituents.intersection(set(comps))) for subl_constituents in phase_constituents]

    # calculate needs points, state variable lists, and values to compare to
    calculate_dict = {
        'P': np.array([]),
        'T': np.array([]),
        'points': np.atleast_2d([[]]).reshape(-1, sum([len(subl) for subl in phase_constituents])),
        'values': np.array([]),
    }

    for datum in desired_data:
        # extract the data we care about
        datum_T = datum['conditions']['T']
        datum_P = datum['conditions']['P']
        configurations = datum['solver']['sublattice_configurations']
        occupancies = datum['solver'].get('sublattice_occupancies')
        values = np.array(datum['values'])

        # broadcast and flatten the conditions arrays
        P, T = ravel_conditions(values, datum_P, datum_T)
        if occupancies is None:
            occupancies = [None] * len(configurations)

        # calculate the points arrays, should be 2d array of points arrays
        points = np.array([calculate_points_array(phase_constituents, config, occup) for config, occup in zip(configurations, occupancies)])

        # add everything to the calculate_dict
        calculate_dict['P'] = np.concatenate([calculate_dict['P'], P])
        calculate_dict['T'] = np.concatenate([calculate_dict['T'], T])
        calculate_dict['points'] = np.concatenate([calculate_dict['points'], points], axis=0)
        calculate_dict['values'] = np.concatenate([calculate_dict['values'], values.flatten()])

    return calculate_dict


def calculate_single_phase_error(dbf, comps, phases, datasets, parameters=None, phase_models=None):
    """
    Calculate the weighted single phase error in the Database

    Parameters
    ----------
    dbf : pycalphad.Database
        Database to consider
    comps : list
        List of active component names
    phases : list
        List of phases to consider
    datasets : espei.utils.PickleableTinyDB
        Datasets that contain single phase data
    parameters : dict
        Dictionary of symbols that will be overridden in pycalphad.calculate
    phase_models : dict
        Phase models to pass to pycalphad calculations

    Returns
    -------
    float
        A single float of the residual sum of square errors

    Notes
    -----
    There are different single phase values, HM_MIX, SM_FORM, CP_FORM, etc.
    Each of these have different units and the error cannot be compared directly.
    To normalize all of the errors, a normalization factor must be used.
    Equation 2.59 and 2.60 in Lukas, Fries, and Sundman "Computational Thermodynamics" shows how this can be considered.
    Each type of error will be weighted by the reciprocal of the estimated uncertainty in the measured value and conditions.
    The weighting factor is calculated by
    $p_i = (\Delta L_i)^{-1}$
    where $\Delta L_i$ is the uncertainty in the measurement.
    We will neglect the uncertainty for quantities such as temperature, assuming they are small.

    """
    if parameters is None:
        parameters = {}

    # property weights factors as fractions of the parameters
    # for now they are all set to 5%
    property_prefix_weight_factor = {
        'HM': 0.05,
        'SM': 0.05,
        'CPM': 0.05,
    }
    propery_suffixes = ('_FORM', '_MIX')
    # the kinds of properties, e.g. 'HM'+suffix =>, 'HM_FORM', 'HM_MIX'
    # we could also include the bare property ('' => 'HM'), but these are rarely used in ESPEI
    properties = [''.join(prop) for prop in itertools.product(property_prefix_weight_factor.keys(), propery_suffixes)]

    sum_square_error = 0
    for phase_name in phases:
        for prop in properties:
            desired_data = get_prop_data(comps, phase_name, prop, datasets)
            if len(desired_data) == 0:
                #logging.debug('Skipping {} in phase {} because no data was found.'.format(prop, phase_name))
                continue
            calculate_dict = get_prop_samples(dbf, comps, phase_name, desired_data)
            if prop.endswith('_FORM'):
                calculate_dict['output'] = ''.join(prop.split('_')[:-1])
                params = parameters.copy()
                params.update({'GHSER' + (c.upper() * 2)[:2]: 0 for c in comps})
            else:
                calculate_dict['output'] = prop
                params = parameters
            sample_values = calculate_dict.pop('values')
            results = calculate(dbf, comps, phase_name, broadcast=False, parameters=params, model=phase_models, **calculate_dict)[calculate_dict['output']].values
            weight = (property_prefix_weight_factor[prop.split('_')[0]]*np.abs(np.mean(sample_values)))**(-1.0)
            error = np.sum((results-sample_values)**2) * weight
            #logging.debug('Weighted sum of square error for property {} of phase {}: {}'.format(prop, phase_name, error))
            sum_square_error += error
    return -sum_square_error


def lnprob(params, comps=None, dbf=None, phases=None, datasets=None,
           symbols_to_fit=None, phase_models=None, scheduler=None,
           massfuncs=None, massgradfuncs=None,
           callables=None, grad_callables=None, hess_callables=None,
           ):
    """
    Returns the error from multiphase fitting as a log probability.
    """
    parameters = {param_name: param for param_name, param in zip(symbols_to_fit, params)}
    try:
        multi_phase_error = multi_phase_fit(dbf, comps, phases, datasets, phase_models,
                                     parameters=parameters, scheduler=scheduler,
                                     massfuncs=massfuncs, massgradfuncs=massgradfuncs,
                                     callables=callables, grad_callables=grad_callables, hess_callables=hess_callables,
                                     )
    except (ValueError, LinAlgError) as e:
        multi_phase_error = [np.inf]
    multi_phase_error = [np.inf if np.isnan(x) else x ** 2 for x in multi_phase_error]
    multi_phase_error = -np.sum(multi_phase_error)

    single_phase_error = calculate_single_phase_error(dbf, comps, phases, datasets, parameters, phase_models=phase_models)
    total_error = multi_phase_error + single_phase_error
    logging.debug('Single phase error: {:0.2f}. Multi phase error: {:0.2f}. Total error: {:0.2f}'.format(single_phase_error, multi_phase_error, total_error))
    return np.array(total_error, dtype=np.float64)


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
    logging.debug('Building phase models (this may take some time)')
    # 0 is placeholder value
    phases = sorted(dbf.phases.keys())
    eq_callables = eq_callables_dict(dbf, comps, phases, model=Model, param_symbols=sorted([sympy.Symbol(sym) for sym in symbols_to_fit], key=str))
    # because error_context expencts 'phase_models' key, change it
    eq_callables['phase_models'] = eq_callables.pop('model')
    logging.debug('Finished building phase models')
    #dbf = dask.delayed(dbf, pure=True)
    #phase_models = dask.delayed(phase_models, pure=True)

    # context for the log probability function
    error_context = {'comps': comps, 'dbf': dbf,
                     'phases': phases, 'phase_models': eq_callables['phase_models'],
                     'datasets': datasets, 'symbols_to_fit': symbols_to_fit,
                     }

    error_context.update(**eq_callables)

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
    for param_name, value in zip(symbols_to_fit, optimal_params):
        dbf.symbols[param_name] = value
    logging.info('MCMC complete.')
    return dbf, sampler
