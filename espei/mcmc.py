"""
Module for running MCMC in ESPEI

MCMC uses an EnsembleSampler based on Goodman and Weare, Ensemble Samplers with
Affine Invariance. Commun. Appl. Math. Comput. Sci. 5, 65-80 (2010).

"""

import logging, time, sys

import sympy
import numpy as np
from numpy.linalg import LinAlgError
import emcee

from pycalphad import Model

from espei.utils import database_symbols_to_fit, optimal_parameters, eq_callables_dict
from espei.error_functions import calculate_activity_error, calculate_thermochemical_error, calculate_zpf_error


def lnprob(params, comps=None, dbf=None, phases=None, datasets=None,
           symbols_to_fit=None, phase_models=None, scheduler=None,
           massfuncs=None, massgradfuncs=None,
           callables=None, grad_callables=None, hess_callables=None,
           thermochemical_callables=None,
           ):
    """
    Returns the error from multiphase fitting as a log probability.
    """
    starttime = time.time()
    parameters = {param_name: param for param_name, param in zip(symbols_to_fit, params)}
    try:
        multi_phase_error = calculate_zpf_error(dbf, comps, phases, datasets, phase_models,
                                     parameters=parameters, scheduler=scheduler,
                                     massfuncs=massfuncs, massgradfuncs=massgradfuncs,
                                     callables=callables, grad_callables=grad_callables, hess_callables=hess_callables,
                                     )
    except (ValueError, LinAlgError) as e:
        multi_phase_error = [np.inf]
    multi_phase_error = [np.inf if np.isnan(x) else x ** 2 for x in multi_phase_error]
    multi_phase_error = -np.sum(multi_phase_error)
    single_phase_error = calculate_thermochemical_error(dbf, comps, phases, datasets, parameters, phase_models=phase_models, callables=thermochemical_callables)
    actvity_error = calculate_activity_error(dbf, comps, phases, datasets, parameters=parameters, phase_models=phase_models, callables=callables, grad_callables=grad_callables, hess_callables=hess_callables, massfuncs=massfuncs, massgradfuncs=massgradfuncs)
    total_error = multi_phase_error + single_phase_error + actvity_error
    logging.debug('Single phase error: {:0.2f}. Multi phase error: {:0.2f}. Activity Error: {:0.2f}. Total error: {:0.2f}'.format(single_phase_error, multi_phase_error, actvity_error, total_error))
    logging.debug('lnprob time: {}'.format(time.time() - starttime))
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


def mcmc_fit(dbf, datasets, iterations=1000, save_interval=100, chains_per_parameter=2,
             chain_std_deviation=0.1, scheduler=None, tracefile=None, probfile=None,
             restart_trace=None, deterministic=True, ):
    """
    Run Markov Chain Monte Carlo on the Database given datasets

    Parameters
    ----------
    dbf : Database
        A pycalphad Database to fit with symbols to fit prefixed with `VV`
        followed by a number, e.g. `VV0001`
    datasets : PickleableTinyDB
        A database of single- and multi-phase to fit
    iterations : int
        Number of trace iterations to calculate in MCMC. Default is 1000 iterations.
    save_interval :int
        interval of iterations to save the tracefile and probfile
    chains_per_parameter : int
        number of chains for each parameter. Must be an even integer greater or
        equal to 2. Defaults to 2.
    chain_std_deviation : float
        standard deviation of normal for parameter initialization as a fraction
        of each parameter. Must be greater than 0. Default is 0.1, which is 10%.
    scheduler : callable
        Scheduler to use with emcee. Must implement a map method.
    tracefile : str
        filename to store the trace with NumPy.save. Array has shape
        (chains, iterations, parameters)
    probfile : str
        filename to store the log probability with NumPy.save. Has shape (chains, iterations)
    restart_trace : np.ndarray
        ndarray of the previous trace. Should have shape (chains, iterations, parameters)
    deterministic : bool
        If True, the emcee sampler will be seeded to give deterministic sampling
        draws. This will ensure that the runs with the exact same database,
        chains_per_parameter, and chain_std_deviation (or restart_trace) will
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
    # we also need to build models that have no ideal mixing for thermochemical error and to build them for each property we might calculate
    # TODO: potential optimization to only calculate for phase/property combos that we have in the datasets
    # first construct dict of models without ideal mixing
    mods_no_idmix = {}
    for phase_name in phases:
        mods_no_idmix[phase_name] = Model(dbf, comps, phase_name)
        mods_no_idmix[phase_name].models['idmix'] = 0
    # now construct callables for each possible property that can be calculated
    thermochemical_callables = {}  # will be dict of {output_property: eq_callables_dict}
    whitelist_properties = ['HM', 'SM', 'CPM']
    whitelist_properties = whitelist_properties + [prop+'_MIX' for prop in whitelist_properties]
    for prop in whitelist_properties:
        thermochemical_callables[prop] = eq_callables_dict(dbf, comps, phases, model=mods_no_idmix, output=prop, param_symbols=sorted([sympy.Symbol(sym) for sym in symbols_to_fit], key=str), build_gradients=False)
        # pop off the callables not used in properties because we don't want them around (they should be None, anyways)
        thermochemical_callables[prop].pop('grad_callables')
        thermochemical_callables[prop].pop('hess_callables')
        thermochemical_callables[prop].pop('massgradfuncs')
    logging.debug('Finished building phase models')

    # context for the log probability function
    error_context = {'comps': comps, 'dbf': dbf,
                     'phases': phases, 'phase_models': eq_callables['phase_models'],
                     'datasets': datasets, 'symbols_to_fit': symbols_to_fit,
                     'thermochemical_callables': thermochemical_callables,
                     }

    error_context.update(**eq_callables)

    def save_sampler_state(sampler):
        if tracefile:
            logging.debug('Writing trace to {}'.format(tracefile))
            np.save(tracefile, sampler.chain)
        if probfile:
            logging.debug('Writing lnprob to {}'.format(probfile))
            np.save(probfile, sampler.lnprobability)


    # initialize the walkers either fresh or from the restart
    if restart_trace is not None:
        walkers = restart_trace[np.nonzero(restart_trace)].reshape(
            (restart_trace.shape[0], -1, restart_trace.shape[2]))[:, -1, :]
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
    logging.info('Running MCMC for {} iterations.'.format(iterations))
    try:
        for i, result in enumerate(sampler.sample(walkers, iterations=iterations)):
            # progress bar
            if (i + 1) % save_interval == 0:
                save_sampler_state(sampler)
                logging.debug('Acceptance ratios for parameters: {}'.format(sampler.acceptance_fraction))
            n = int((progbar_width + 1) * float(i) / iterations)
            sys.stdout.write("\r[{0}{1}] ({2} of {3})\n".format('#' * n, ' ' * (progbar_width - n), i + 1, iterations))
        n = int((progbar_width + 1) * float(i + 1) / iterations)
        sys.stdout.write("\r[{0}{1}] ({2} of {3})\n".format('#' * n, ' ' * (progbar_width - n), i + 1, iterations))
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
