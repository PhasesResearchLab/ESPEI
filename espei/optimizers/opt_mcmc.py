import logging
import time
import numpy as np
import emcee
from espei.error_functions import calculate_zpf_error, calculate_activity_error, \
    calculate_non_equilibrium_thermochemical_probability, \
    calculate_equilibrium_thermochemical_probability
from espei.priors import PriorSpec, build_prior_specs
from espei.utils import unpack_piecewise, optimal_parameters
from espei.error_functions.context import setup_context
from .opt_base import OptimizerBase
from .graph import OptNode


TRACE = 15


class EmceeOptimizer(OptimizerBase):
    """
    An optimizer using an EnsembleSampler based on Goodman and Weare [1]
    implemented in emcee [2]

    Attributes
    ----------
    scheduler : mappable
        An object implementing a `map` function
    save_interval : int
        Interval of iterations to save the tracefile and probfile.
    tracefile : str
        Filename to store the trace with NumPy.save. Array has shape
        (chains, iterations, parameters). Defaults to None.
    probfile : str
        filename to store the log probability with NumPy.save. Has shape (chains, iterations)

    References
    ----------
    [1] Goodman and Weare, Ensemble Samplers with Affine Invariance. Commun. Appl. Math. Comput. Sci. 5, 65-80 (2010).
    [2] Foreman-Mackey, Hogg, Lang, Goodman, emcee: The MCMC Hammer. Publ. Astron. Soc. Pac. 125, 306-312 (2013).
    """
    def __init__(self, dbf, scheduler=None):
        super(EmceeOptimizer, self).__init__(dbf)
        self.scheduler = scheduler
        self.save_interval = 1
        # These are set by the _fit method
        self.sampler = None
        self.tracefile = None
        self.probfile = None

    @staticmethod
    def initialize_new_chains(params, chains_per_parameter, std_deviation, deterministic=True):
        """
        Return an array of num_samples from a Gaussian distribution about each parameter.

        Parameters
        ----------
        params : ndarray
            1D array of initial parameters that will be the mean of the distribution.
        num_samples : int
            Number of chains to initialize.
        chains_per_parameter : int
            number of chains for each parameter. Must be an even integer greater or
            equal to 2. Defaults to 2.
        std_deviation : float
            Fractional standard deviation of the parameters to use for initialization.
        deterministic : bool
            True if the parameters should be generated deterministically.

        Returns
        -------
        ndarray

        Notes
        -----
        Parameters are sampled from ``normal(loc=param, scale=param*std_deviation)``.
        A parameter of zero will produce a standard deviation of zero and
        therefore only zeros will be sampled. This will break emcee's
        StretchMove for this parameter and only zeros will be selected.

        """
        logging.log(TRACE, 'Initial parameters: {}'.format(params))
        params = np.array(params)
        num_zero_params = np.nonzero(params == 0)[0].size
        if num_zero_params > 0:
            logging.warning(f"{num_zero_params} initial parameter{' is' if num_zero_params == 1 else 's are'} "
                            "initialized to zero. The ensemble of chains for zero parameters will be all initialized "
                            "to zero and all proposed values for these parameter will be zero. If possible, it's "
                            "better to make a good guess at a reasonable parameter value to start with. "
                            "Alternatively, you can start with a small value near zero and let the ensemble search "
                            "parameter space.")
        nchains = params.size * chains_per_parameter
        logging.info('Initializing {} chains with {} chains per parameter.'.format(nchains, chains_per_parameter))
        if deterministic:
            rng = np.random.RandomState(1769)
        else:
            rng = np.random.RandomState()
        # apply a Gaussian random to each parameter with std dev of std_deviation*parameter
        tiled_parameters = np.tile(params, (nchains, 1))
        chains = rng.normal(tiled_parameters, np.abs(tiled_parameters * std_deviation))
        return chains

    @staticmethod
    def initialize_chains_from_trace(restart_trace):
        tr = restart_trace
        walkers = tr[np.nonzero(tr)].reshape((tr.shape[0], -1, tr.shape[2]))[:, -1, :]
        nchains = walkers.shape[0]
        ndim = walkers.shape[1]
        initial_parameters = walkers.mean(axis=0)
        logging.info('Restarting from previous calculation with {} chains ({} per parameter).'.format(nchains, nchains / ndim))
        logging.log(TRACE, 'Means of restarting parameters are {}'.format(initial_parameters))
        logging.log(TRACE, 'Standard deviations of restarting parameters are {}'.format(walkers.std(axis=0)))
        return walkers

    @staticmethod
    def get_priors(prior, symbols, params):
        """
        Build priors for a particular set of fitting symbols and initial parameters.
        Returns a dict that should be used to update the context.

        Parameters
        ----------
        prior : dict or PriorSpec or None
            Prior to initialize. See the docs on
        symbols : list of str
            List of symbols that will be fit
        params : list of float
            List of parameter values corresponding to the symbols. These should
            be the initial parameters that the priors will be based off of.

        Returns
        -------

        """
        if isinstance(prior, dict):
            logging.info('Initializing a {} prior for the parameters.'.format(prior['name']))
        elif isinstance(prior, PriorSpec):
            logging.info('Initializing a {} prior for the parameters.'.format(prior.name))
        elif prior is None:
            prior = {'name': 'zero'}
        prior_specs = build_prior_specs(prior, params)
        rv_priors = []
        for spec, param, fit_symbol in zip(prior_specs, params, symbols):
            if isinstance(spec, PriorSpec):
                logging.debug('Initializing a {} prior for {} with parameters: {}.'.format(spec.name, fit_symbol, spec.parameters))
                rv_priors.append(spec.get_prior(param))
            elif hasattr(spec, "logpdf"):
                logging.debug('Using a user-specified prior for {}.'.format(fit_symbol))
                rv_priors.append(spec)
        return {'prior_rvs': rv_priors}

    def save_sampler_state(self):
        """
        Convenience function that saves the trace and lnprob if
        they haven't been set to None by the user.

        Requires that the sampler attribute be set.
        """
        tr = self.tracefile
        if tr is not None:
            logging.log(TRACE, 'Writing trace to {}'.format(tr))
            np.save(tr, self.sampler.chain)
        prob = self.probfile
        if prob is not None:
            logging.log(TRACE, 'Writing lnprob to {}'.format(prob))
            np.save(prob, self.sampler.lnprobability)

    def do_sampling(self, chains, iterations):
        progbar_width = 30
        logging.info('Running MCMC for {} iterations.'.format(iterations))
        try:
            for i, result in enumerate(self.sampler.sample(chains, iterations=iterations)):
                # progress bar
                if (i + 1) % self.save_interval == 0:
                    self.save_sampler_state()
                    logging.log(TRACE, 'Acceptance ratios for parameters: {}'.format(self.sampler.acceptance_fraction))
                n = int((progbar_width + 1) * float(i) / iterations)
                logging.info("\r[{0}{1}] ({2} of {3})\n".format('#' * n, ' ' * (progbar_width - n), i + 1, iterations))
            n = int((progbar_width + 1) * float(i + 1) / iterations)
            logging.info("\r[{0}{1}] ({2} of {3})\n".format('#' * n, ' ' * (progbar_width - n), i + 1, iterations))
        except KeyboardInterrupt:
            pass
        logging.info('MCMC complete.')
        self.save_sampler_state()

    def _fit(self, symbols, ds, prior=None, iterations=1000,
             chains_per_parameter=2, chain_std_deviation=0.1, deterministic=True,
             restart_trace=None, tracefile=None, probfile=None,
             mcmc_data_weights=None, approximate_equilibrium=False,
             ):
        """

        Parameters
        ----------
        symbols : list of str
        ds : PickleableTinyDB
        prior : str
            Prior to use to generate priors. Defaults to 'zero', which keeps
            backwards compatibility. Can currently choose 'normal', 'uniform',
            'triangular', or 'zero'.
        iterations : int
            Number of iterations to calculate in MCMC. Default is 1000.
        chains_per_parameter : int
            number of chains for each parameter. Must be an even integer greater
            or equal to 2. Defaults to 2.
        chain_std_deviation : float
            Standard deviation of normal for parameter initialization as a
            fraction of each parameter. Must be greater than 0. Defaults to 0.1.
        deterministic : bool
            If True, the emcee sampler will be seeded to give deterministic sampling
            draws. This will ensure that the runs with the exact same database,
            chains_per_parameter, and chain_std_deviation (or restart_trace) will
            produce exactly the same results.
        restart_trace : np.ndarray
            ndarray of the previous trace. Should have shape (chains, iterations, parameters)
        tracefile : str
            filename to store the trace with NumPy.save. Array has shape
            (chains, iterations, parameters)
        probfile : str
            filename to store the log probability with NumPy.save. Has shape (chains, iterations)
        mcmc_data_weights : dict
            Dictionary of weights for each data type, e.g. {'ZPF': 20, 'HM': 2}

        Returns
        -------
        OptNode

        """
        cbs = self.scheduler is None
        ctx = setup_context(self.dbf, ds, symbols, data_weights=mcmc_data_weights, make_callables=cbs)
        symbols_to_fit = ctx['symbols_to_fit']
        initial_guess = np.array([unpack_piecewise(self.dbf.symbols[s]) for s in symbols_to_fit])

        prior_dict = self.get_priors(prior, symbols_to_fit, initial_guess)
        ctx.update(prior_dict)
        ctx['zpf_kwargs']['approximate_equilibrium'] = approximate_equilibrium
        ctx['equilibrium_thermochemical_kwargs']['approximate_equilibrium'] = approximate_equilibrium
        # Run the initial parameters for guessing purposes:
        logging.log(TRACE, "Probability for initial parameters")
        self.predict(initial_guess, **ctx)
        if restart_trace is not None:
            chains = self.initialize_chains_from_trace(restart_trace)
            # TODO: check that the shape is valid with the existing parameters
        else:
            chains = self.initialize_new_chains(initial_guess, chains_per_parameter, chain_std_deviation, deterministic)
        sampler = emcee.EnsembleSampler(chains.shape[0], initial_guess.size, self.predict, kwargs=ctx, pool=self.scheduler)
        if deterministic:
            from espei.rstate import numpy_rstate
            sampler.random_state = numpy_rstate
            logging.info('Using a deterministic ensemble sampler.')
        self.sampler = sampler
        self.tracefile = tracefile
        self.probfile = probfile
        # Run the MCMC simulation
        self.do_sampling(chains, iterations)

        # Post process
        optimal_params = optimal_parameters(sampler.chain, sampler.lnprobability)
        logging.log(TRACE, 'Initial parameters: {}'.format(initial_guess))
        logging.log(TRACE, 'Optimal parameters: {}'.format(optimal_params))
        logging.log(TRACE, 'Change in parameters: {}'.format(np.abs(initial_guess - optimal_params) / initial_guess))
        parameters = dict(zip(symbols_to_fit, optimal_params))
        return OptNode(parameters, ds)

    @staticmethod
    def predict(params, **ctx):
        """
        Calculate lnprob = lnlike + lnprior
        """
        logging.debug('Parameters - {}'.format(params))

        # lnprior
        prior_rvs = ctx['prior_rvs']
        lnprior_multivariate = [rv.logpdf(theta) for rv, theta in zip(prior_rvs, params)]
        logging.debug('Priors: {}'.format(lnprior_multivariate))
        lnprior = np.sum(lnprior_multivariate)
        if np.isneginf(lnprior):
            # It doesn't matter what the likelihood is. We can skip calculating it to save time.
            logging.log(TRACE, 'Proposal - lnprior: {:0.4f}, lnlike: {}, lnprob: {:0.4f}'.format(lnprior, np.nan, lnprior))
            return lnprior

        # lnlike
        parameters = {param_name: param for param_name, param in zip(ctx['symbols_to_fit'], params)}
        zpf_kwargs = ctx.get('zpf_kwargs')
        activity_kwargs = ctx.get('activity_kwargs')
        non_equilibrium_thermochemical_kwargs = ctx.get('thermochemical_kwargs')
        equilibrium_thermochemical_kwargs = ctx.get('equilibrium_thermochemical_kwargs')
        starttime = time.time()
        if zpf_kwargs is not None:
            try:
                multi_phase_error = calculate_zpf_error(parameters=np.array(params), **zpf_kwargs)
            except (ValueError, np.linalg.LinAlgError) as e:
                raise e
                print(e)
                multi_phase_error = -np.inf
        else:
            multi_phase_error = 0
        if equilibrium_thermochemical_kwargs is not None:
            eq_thermochemical_prob = calculate_equilibrium_thermochemical_probability(parameters=np.array(params), **equilibrium_thermochemical_kwargs)
        else:
            eq_thermochemical_prob = 0
        if activity_kwargs is not None:
            actvity_error = calculate_activity_error(parameters=parameters, **activity_kwargs)
        else:
            actvity_error = 0
        if non_equilibrium_thermochemical_kwargs is not None:
            non_eq_thermochemical_prob = calculate_non_equilibrium_thermochemical_probability(parameters=np.array(params), **non_equilibrium_thermochemical_kwargs)
        else:
            non_eq_thermochemical_prob = 0
        total_error = multi_phase_error + eq_thermochemical_prob + non_eq_thermochemical_prob + actvity_error
        logging.log(TRACE, f'Likelihood - {time.time() - starttime:0.2f}s - Non-equilibrium thermochemical: {non_eq_thermochemical_prob:0.3f}. Equilibrium thermochemical: {eq_thermochemical_prob:0.3f}. ZPF: {multi_phase_error:0.3f}. Activity: {actvity_error:0.3f}. Total: {total_error:0.3f}.')
        lnlike = np.array(total_error, dtype=np.float64)

        lnprob = lnprior + lnlike
        logging.log(TRACE, 'Proposal - lnprior: {:0.4f}, lnlike: {:0.4f}, lnprob: {:0.4f}'.format(lnprior, lnlike, lnprob))
        return lnprob
