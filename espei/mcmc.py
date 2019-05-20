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

from pycalphad import variables as v
from pycalphad.codegen.callables import build_callables
from pycalphad.core.utils import instantiate_models

from espei.utils import database_symbols_to_fit, optimal_parameters
from espei.priors import build_prior_specs, PriorSpec
from espei.error_functions import calculate_activity_error, calculate_thermochemical_error, calculate_zpf_error, get_zpf_data, get_thermochemical_data


TRACE = 15  # TRACE logging level


def lnlikelihood(params, symbols_to_fit, zpf_kwargs, activity_kwargs, thermochemical_kwargs):
    """Calculate the likelihood, $$ \ln p(\theta|y) $$ """
    starttime = time.time()
    parameters = {param_name: param for param_name, param in zip(symbols_to_fit, params)}
    if zpf_kwargs is not None:
        try:
            multi_phase_error = calculate_zpf_error(parameters=parameters, **zpf_kwargs)
        except (ValueError, LinAlgError) as e:
            print(e)
            multi_phase_error = -np.inf
    else:
        multi_phase_error = 0
    if activity_kwargs is not None:
        actvity_error = calculate_activity_error(parameters=parameters, **activity_kwargs)
    else:
        actvity_error = 0
    if thermochemical_kwargs is not None:
        single_phase_error = calculate_thermochemical_error(parameters=parameters, **thermochemical_kwargs)
    else:
        single_phase_error = 0
    total_error = multi_phase_error + single_phase_error + actvity_error
    logging.log(TRACE, 'Likelihood - {:0.2f}s - Thermochemical: {:0.3f}. ZPF: {:0.3f}. Activity: {:0.3f}. Total: {:0.3f}.'.format(time.time() - starttime, single_phase_error, multi_phase_error, actvity_error, total_error))
    return np.array(total_error, dtype=np.float64)


def lnprior(params, priors):
    """
    Calculate the log prior given the parameters and prior distributions

    Parameters
    ----------
    params : array_like
        Array of parameters to fit.
    priors : list of scipy.stats.rv_continuous-like
        List of priors for each parameter in that obey the rv_contnuous type
        interface. Specifically, each element in the list must have a ``logpdf``
        method. Must correspond (same shape and order) to ``params``.

    Returns
    -------
    float

    """
    # multivariate prior is the sum of log univariate priors
    lnprior_multivariate = [rv.logpdf(theta) for rv, theta in zip(priors, params)]
    logging.debug('Priors: {}'.format(lnprior_multivariate))
    return np.sum(lnprior_multivariate)


def lnprob(params, prior_rvs=None, symbols_to_fit=None,
           zpf_kwargs=None, activity_kwargs=None, thermochemical_kwargs=None,
           ):
    """
    Returns the log probability of a set of parameters

    $$ \ln p(y|\theta) \propto \ln p(\theta) + \ln p(\theta|y) $$

    Parameters
    ----------
    params : array_like
        Array of parameters to fit.
    prior_rvs : list of scipy.stats.rv_continuous-like
        List of priors for each parameter in that obey the rv_contnuous type
        interface. Specifically, each element in the list must have a ``logpdf``
        method. Must correspond (same shape and order) to ``params``.
    symbols_to_fit : list
        List of names of parameter symbols to replace. Must correspond (same
        shape and order) to ``params``.
    zpf_kwargs : dict
        Keyword arguments for `calculate_zpf_error`
    activity_kwargs : list
        Keyword arguments for `calculate_activity_error`
    thermochemical_kwargs : list
        Keyword arguments for `calculate_thermochemical_error`

    Returns
    -------
    float

    """
    logging.debug('Parameters - {}'.format(params))
    logprior = lnprior(params, prior_rvs)
    if np.isneginf(logprior):
        # It doesn't matter what the likelihood is. We can skip calculating it to save time.
        logging.log(TRACE, 'Proposal - lnprior: {:0.4f}, lnlike: {}, lnprob: {:0.4f}'.format(logprior, np.nan, logprior))
        return logprior

    loglike = lnlikelihood(params, symbols_to_fit=symbols_to_fit, zpf_kwargs=zpf_kwargs,
                           activity_kwargs=activity_kwargs,
                           thermochemical_kwargs=thermochemical_kwargs,)

    logprob = logprior + loglike
    logging.log(TRACE, 'Proposal - lnprior: {:0.4f}, lnlike: {:0.4f}, lnprob: {:0.4f}'.format(logprior, loglike, logprob))
    return logprob


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
             restart_trace=None, deterministic=True, prior=None, mcmc_data_weights=None):
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
    prior : str
        Prior to use to generate priors. Defaults to 'zero', which keeps
        backwards compatibility. Can currently choose 'normal', 'uniform',
        'triangular', or 'zero'.
    mcmc_data_weights : dict
        Dictionary of weights for each data type, e.g. {'ZPF': 20, 'HM': 2}

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

    # initialize the priors
    if isinstance(prior, dict):
        logging.info('Initializing a {} prior for the parameters.'.format(prior['name']))
    elif isinstance(prior, PriorSpec):
        logging.info('Initializing a {} prior for the parameters.'.format(prior.name))
    elif prior is None:
        prior = {'name': 'zero'}
    prior_specs = build_prior_specs(prior, initial_parameters)
    rv_priors = []
    for spec, param, fit_symbol in zip(prior_specs, initial_parameters, symbols_to_fit):
        if isinstance(spec, PriorSpec):
            logging.debug('Initializing a {} prior for {} with parameters: {}.'.format(spec.name, fit_symbol, spec.parameters))
            rv_priors.append(spec.get_prior(param))
        elif hasattr(spec, "logpdf"):
            logging.debug('Using a user-specified prior for {}.'.format(fit_symbol))
            rv_priors.append(spec)

    # construct the models for each phase, substituting in the SymPy symbol to fit.
    logging.log(TRACE, 'Building phase models (this may take some time)')
    phases = sorted(dbf.phases.keys())
    orig_parameters = {sym: p for sym, p in zip(symbols_to_fit, initial_parameters)}
    models = instantiate_models(dbf, comps, phases, parameters=orig_parameters)
    eq_callables = build_callables(dbf, comps, phases, models, parameter_symbols=symbols_to_fit,
                        output='GM', build_gradients=True, build_hessians=False,
                        additional_statevars={v.N, v.P, v.T})
    thermochemical_data = get_thermochemical_data(dbf, comps, phases, datasets, weight_dict=mcmc_data_weights)
    logging.log(TRACE, 'Finished building phase models')

    # context for the log probability function
    # for all cases, parameters argument addressed in MCMC loop
    error_context = {
        'prior_rvs': rv_priors,
        'symbols_to_fit': symbols_to_fit,
        'zpf_kwargs': {
            'dbf': dbf, 'phases': phases, 'zpf_data': get_zpf_data(comps, phases, datasets),
            'phase_models': models, 'callables': eq_callables,
            'data_weight': mcmc_data_weights.get('ZPF', 1.0),
        },
        'thermochemical_kwargs': {
            'dbf': dbf, 'comps': comps, 'thermochemical_data': thermochemical_data,
        },
        'activity_kwargs': {
            'dbf': dbf, 'comps': comps, 'phases': phases, 'datasets': datasets,
            'phase_models': models, 'callables': eq_callables,
            'data_weight': mcmc_data_weights.get('ACR', 1.0),
        },
    }

    def save_sampler_state(sampler):
        if tracefile:
            logging.log(TRACE, 'Writing trace to {}'.format(tracefile))
            np.save(tracefile, sampler.chain)
        if probfile:
            logging.log(TRACE, 'Writing lnprob to {}'.format(probfile))
            np.save(probfile, sampler.lnprobability)


    # initialize the walkers either fresh or from the restart
    if restart_trace is not None:
        walkers = restart_trace[np.nonzero(restart_trace)].reshape(
            (restart_trace.shape[0], -1, restart_trace.shape[2]))[:, -1, :]
        nwalkers = walkers.shape[0]
        ndim = walkers.shape[1]
        initial_parameters = walkers.mean(axis=0)
        logging.info('Restarting from previous calculation with {} chains ({} per parameter).'.format(nwalkers, nwalkers / ndim))
        logging.log(TRACE, 'Means of restarting parameters are {}'.format(initial_parameters))
        logging.log(TRACE, 'Standard deviations of restarting parameters are {}'.format(walkers.std(axis=0)))
    else:
        logging.log(TRACE, 'Initial parameters: {}'.format(initial_parameters))
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
                logging.log(TRACE, 'Acceptance ratios for parameters: {}'.format(sampler.acceptance_fraction))
            n = int((progbar_width + 1) * float(i) / iterations)
            logging.info("\r[{0}{1}] ({2} of {3})\n".format('#' * n, ' ' * (progbar_width - n), i + 1, iterations))
        n = int((progbar_width + 1) * float(i + 1) / iterations)
        sys.stdout.write("\r[{0}{1}] ({2} of {3})\n".format('#' * n, ' ' * (progbar_width - n), i + 1, iterations))
    except KeyboardInterrupt:
        pass
    # final processing
    save_sampler_state(sampler)
    optimal_params = optimal_parameters(sampler.chain, sampler.lnprobability)
    logging.log(TRACE, 'Intial parameters: {}'.format(initial_parameters))
    logging.log(TRACE, 'Optimal parameters: {}'.format(optimal_params))
    logging.log(TRACE, 'Change in parameters: {}'.format(np.abs(initial_parameters - optimal_params) / initial_parameters))
    for param_name, value in zip(symbols_to_fit, optimal_params):
        dbf.symbols[param_name] = value
    logging.info('MCMC complete.')
    return dbf, sampler
