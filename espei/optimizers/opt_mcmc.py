import logging
import sys
import time
import warnings

import numpy as np
import emcee
from espei.priors import PriorSpec, build_prior_specs, rv_zero
from espei.utils import unpack_piecewise, optimal_parameters, ImmediateClient
from espei.error_functions.context import setup_context
from .opt_base import OptimizerBase

_log = logging.getLogger(__name__)


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
    def __init__(self, dbf, phase_models=None, scheduler=None):
        super(EmceeOptimizer, self).__init__(dbf)
        self.phase_models = phase_models
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
        _log.trace('Initial parameters: %s', params)
        params = np.array(params)
        num_zero_params = np.nonzero(params == 0)[0].size
        if num_zero_params > 0:
            _log.warning(
                "%s initial parameters are initialized to zero. The ensemble of chains "
                "for zero parameters will be all initialized to zero and all proposed "
                "values for these parameter will be zero. If possible, it's better to "
                "make a good guess at a reasonable parameter value to start with. "
                "Alternatively, you can start with a small value near zero and let the "
                "ensemble search parameter space.", num_zero_params)
        nchains = params.size * chains_per_parameter
        _log.info('Initializing %s chains with %s chains per parameter.', nchains, chains_per_parameter)
        if deterministic:
            rng = np.random.RandomState(1769)
        else:
            rng = np.random.RandomState()
        # apply a Gaussian random to each parameter with std dev of std_deviation*parameter
        tiled_parameters = np.tile(params, (nchains, 1))
        chains = rng.normal(tiled_parameters, np.abs(tiled_parameters * std_deviation))
        chains[0] = params #Ensure the initial guess is always included in the set, as the std_deviation may be too large to generate feasible points around a "good" initial point.
        return chains

    @staticmethod
    def initialize_chains_from_trace(restart_trace):
        tr = restart_trace
        walkers = tr[np.nonzero(tr)].reshape((tr.shape[0], -1, tr.shape[2]))[:, -1, :]
        nchains = walkers.shape[0]
        ndim = walkers.shape[1]
        initial_parameters = walkers.mean(axis=0)
        _log.info('Restarting from previous calculation with %s chains (%s per parameter).', nchains, nchains / ndim)
        _log.trace('Means of restarting parameters are %s', initial_parameters)
        _log.trace('Standard deviations of restarting parameters are %s', walkers.std(axis=0))
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
            _log.info('Initializing a %s prior for the parameters.', prior['name'])
        elif isinstance(prior, PriorSpec):
            _log.info('Initializing a %s prior for the parameters.', prior.name)
        elif prior is None:
            prior = {'name': 'zero'}
        prior_specs = build_prior_specs(prior, params)
        rv_priors = []
        for spec, param, fit_symbol in zip(prior_specs, params, symbols):
            if isinstance(spec, PriorSpec):
                _log.debug('Initializing a %s prior for %s with parameters: %s.', spec.name, fit_symbol, spec.parameters)
                rv_priors.append(spec.get_prior(param))
            elif hasattr(spec, "logpdf"):
                _log.debug('Using a user-specified prior for %s.', fit_symbol)
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
            _log.trace('Writing trace to %s', tr)
            np.save(tr, self.sampler.chain)
        prob = self.probfile
        if prob is not None:
            _log.trace('Writing lnprob to %s', prob)
            np.save(prob, self.sampler.lnprobability)

    def do_sampling(self, chains, iterations):
        progbar_width = 30
        _log.info('Running MCMC for %s iterations.', iterations)
        iter_start_time = time.time()
        try:
            for i, result in enumerate(self.sampler.sample(chains, iterations=iterations)):
                # progress bar
                if (i + 1) % self.save_interval == 0:
                    self.save_sampler_state()
                    _log.trace('Acceptance ratios for parameters: %s', self.sampler.acceptance_fraction)
                n = int((progbar_width) * float(i + 1) / iterations)
                _log.info("\r[%s%s] (%d of %d)", '#' * n, ' ' * (progbar_width - n), i + 1, iterations)
                _log.info("Current time (s): %.1f", time.time() - iter_start_time)
                # Specific case to ImmediateClient to output memory usage
                if isinstance(self.scheduler, ImmediateClient):
                    scheduler_info = self.scheduler.scheduler_info()
                    memory = [float(worker['metrics'].get('memory', 0)) for worker in scheduler_info['workers'].values()]
                    _log.info("Total memory (GB): %.3f, Min/max worker memory (GB): [%.3f, %.3f]", np.sum(memory)/1e9, np.amin(memory)/1e9, np.amax(memory)/1e9)
        except KeyboardInterrupt:
            pass
        _log.info('MCMC complete.')
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
        Dict[str, float]

        """
        # Set NumPy print options so logged arrays print on one line. Reset at the end.
        np.set_printoptions(linewidth=sys.maxsize)
        cbs = self.scheduler is None
        ctx = setup_context(self.dbf, ds, symbols, data_weights=mcmc_data_weights, phase_models=self.phase_models, make_callables=cbs)
        symbols_to_fit = ctx['symbols_to_fit']
        initial_guess = np.array([unpack_piecewise(self.dbf.symbols[s]) for s in symbols_to_fit])

        if approximate_equilibrium:
            warnings.warn(f"Approximate equilibrium is deprecated and will be removed in ESPEI 0.10. Got {approximate_equilibrium}.", DeprecationWarning)

        prior_dict = self.get_priors(prior, symbols_to_fit, initial_guess)
        ctx.update(prior_dict)
        # Run the initial parameters for guessing purposes:
        _log.trace("Probability for initial parameters")
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
            _log.info('Using a deterministic ensemble sampler.')
        self.sampler = sampler
        self.tracefile = tracefile
        self.probfile = probfile
        # Run the MCMC simulation
        t0 = time.time()
        self.do_sampling(chains, iterations)
        tf = time.time()
        _log.info('Fit time (s): %.1f', tf-t0)

        # Post process
        optimal_params = optimal_parameters(sampler.chain, sampler.lnprobability)
        _log.trace('Initial parameters: %s', initial_guess)
        _log.trace('Optimal parameters: %s', optimal_params)
        _log.trace('Change in parameters: %s', np.abs(initial_guess - optimal_params) / initial_guess)
        parameters = dict(zip(symbols_to_fit, optimal_params))
        np.set_printoptions(linewidth=75)
        return parameters

    @staticmethod
    def predict(params, **ctx):
        """
        Calculate lnprob = lnlike + lnprior
        """
        _log.debug('Parameters - %s', params)

        # Important to coerce to floats here because the values _must_ be floats if
        # they are used to update PhaseRecords directly
        params = np.asarray(params, dtype=np.float64)

        # lnprior
        prior_rvs = ctx.get('prior_rvs', [rv_zero() for _ in range(params.size)])
        lnprior_multivariate = [rv.logpdf(theta) for rv, theta in zip(prior_rvs, params)]
        _log.debug('Priors: %s', lnprior_multivariate)
        lnprior = np.sum(lnprior_multivariate)
        if np.isneginf(lnprior):
            # It doesn't matter what the likelihood is. We can skip calculating it to save time.
            _log.trace('Proposal - lnprior: %0.4f, lnlike: %0.4f, lnprob: %0.4f', lnprior, np.nan, lnprior)
            return lnprior

        # lnlike
        starttime = time.time()
        lnlike = 0.0
        likelihoods = {}
        for residual_obj in ctx.get("residual_objs", []):
            residual_starttime = time.time()
            likelihood = residual_obj.get_likelihood(params)
            residual_time = time.time() - residual_starttime
            likelihoods[type(residual_obj).__name__] = (likelihood, residual_time)
            lnlike += likelihood
        liketime = time.time() - starttime

        like_str = ". ".join([f"{ky}: {vl[0]:0.3f} ({vl[1]:0.2f} s)" for ky, vl in likelihoods.items()])
        lnlike = np.array(lnlike, dtype=np.float64)
        _log.trace('Likelihood - %0.2fs - %s. Total: %0.3f.', liketime, like_str, lnlike)

        lnprob = lnprior + lnlike
        _log.trace('Proposal - lnprior: %0.4f, lnlike: %0.4f, lnprob: %0.4f', lnprior, lnlike, lnprob)
        return lnprob
