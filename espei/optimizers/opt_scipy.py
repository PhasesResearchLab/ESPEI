import time
import logging
import numpy as np
from scipy.optimize import minimize
from espei.utils import unpack_piecewise
from espei.error_functions.context import setup_context
from espei.error_functions import calculate_activity_error
from .opt_base import OptimizerBase

_log = logging.getLogger(__name__)


class SciPyOptimizer(OptimizerBase):
    def _fit(self, symbols, ds, method='Powell', maxiter=50, verbose=False, **kwargs):
        # Use Scipy's minimize with the Powell method to optimize Lnprob
        ctx = setup_context(self.dbf, ds, symbols)

        symbols_to_fit = ctx['symbols_to_fit']
        initial_guess = np.array([unpack_piecewise(self.dbf.symbols[s]) for s in symbols_to_fit])
        if verbose:
            print('Fitting', symbols_to_fit)
            print('Initial guess', initial_guess)
        s = time.time()
        out = minimize(self.predict, initial_guess, args=(ctx,), method=method,
                       options={'maxiter': maxiter}, **kwargs)
        xs = np.atleast_1d(out.x).tolist()
        if verbose:
            print('Found', xs, 'in', int(time.time() - s), 's')
        parameters = dict(zip(symbols_to_fit, xs))
        return parameters

    @staticmethod
    def predict(params, ctx):
        starttime = time.time()
        lnlike = 0.0
        likelihoods = {}
        for residual_obj in ctx.get("residual_objs", []):
            likelihood = residual_obj.get_likelihood(params)
            likelihoods[type(residual_obj).__name__] = likelihood
            lnlike += likelihood
        like_str = ". ".join([f"{ky}: {vl:0.3f}" for ky, vl in likelihoods.items()])
        _log.trace('Likelihood - %0.2fs. %s. Total: %0.3f.', time.time() - starttime, like_str, lnlike)
        error = np.array(lnlike, dtype=np.float64)
        return error
