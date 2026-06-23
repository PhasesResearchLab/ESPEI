import time
import logging
import numpy as np
from scipy.optimize import minimize
from espei.utils import unpack_piecewise
from espei.error_functions.context import setup_context
from .opt_base import OptimizerBase

_log = logging.getLogger(__name__)


class SciPyOptimizer(OptimizerBase):
    """ESPEI optimizer using SciPy's optimize.minimize function"""
    def _fit(self, symbols, ds, method='CG', maxiter=50, verbose=False, data_weights=None, **kwargs):
        # Use Scipy's minimize with the Powell method to optimize Lnprob
        symbols_to_fit = symbols
        initial_guess = np.array([unpack_piecewise(self.dbf.symbols[s]) for s in symbols])
        ctx = setup_context(self.dbf, ds, symbols, data_weights=data_weights) # data_weights=mcmc_data_weights, phase_models=self.phase_models, make_callables=cbs)
        if verbose:
            print('Fitting', symbols_to_fit)
            print('Initial guess', initial_guess)
        s = time.time()
        out = minimize(self.predict, initial_guess, args=(ctx,), method=method, jac=True,
                       options={'maxiter': maxiter}, **kwargs)
        self.minimize_result = out
        xs = np.atleast_1d(out.x).tolist()
        if verbose:
            print('Found', xs, 'in', int(time.time() - s), 's')
        parameters = dict(zip(symbols_to_fit, xs))
        return parameters

    @staticmethod
    def predict(params, ctx):
        starttime = time.time()
        lnlike = 0.0
        grads = []
        likelihoods = {}
        for residual_obj in ctx.get("residual_objs", []):
            likelihood, likelihood_grad = residual_obj.get_likelihood(params)
            likelihoods[type(residual_obj).__name__] = likelihood
            lnlike += likelihood
            grads.append(likelihood_grad)
        grad = np.sum(grads, axis=0)

        like_str = ". ".join([f"{ky}: {vl:0.3f}" for ky, vl in likelihoods.items()])
        grad_L_str = "[" + ",".join(f"{x:0.3f}" for x in grad) + "]"
        _log.trace('Likelihood - %0.2fs. %s. Total: %0.3f.', time.time() - starttime, like_str, lnlike)
        _log.trace('∇L = {%s}', grad_L_str)
        error = np.array(lnlike, dtype=np.float64)
        return -error, -grad
