import time
import logging
import numpy as np
from scipy.optimize import minimize
from espei.utils import unpack_piecewise
from espei.error_functions.context import setup_context
from espei.error_functions import calculate_activity_error, calculate_zpf_error, \
    calculate_non_equilibrium_thermochemical_probability
from .opt_base import OptimizerBase
from .graph import OptNode

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
        return OptNode(parameters, ds)

    @staticmethod
    def predict(params, ctx):
        parameters = {param_name: param for param_name, param in zip(ctx['symbols_to_fit'], params)}
        zpf_kwargs = ctx.get('zpf_kwargs')
        activity_kwargs = ctx.get('activity_kwargs')
        thermochemical_kwargs = ctx.get('thermochemical_kwargs')
        starttime = time.time()
        if zpf_kwargs is not None:
            try:
                multi_phase_error = calculate_zpf_error(parameters=parameters, **zpf_kwargs)
            except (ValueError, np.linalg.LinAlgError) as e:
                raise e
                print(e)
                multi_phase_error = -np.inf
        else:
            multi_phase_error = 0
        if activity_kwargs is not None:
            actvity_error = calculate_activity_error(parameters=parameters, **activity_kwargs)
        else:
            actvity_error = 0
        if thermochemical_kwargs is not None:
            single_phase_error = calculate_non_equilibrium_thermochemical_probability(parameters=parameters, **thermochemical_kwargs)
        else:
            single_phase_error = 0
        total_error = multi_phase_error + single_phase_error + actvity_error
        _log.trace('Likelihood - %0.2fs - Thermochemical: %0.3f. ZPF: %0.3f. Activity: %0.3f. Total: %0.3f.', time.time() - starttime, single_phase_error, multi_phase_error, actvity_error, total_error)
        error = np.array(total_error, dtype=np.float64)
        return error

