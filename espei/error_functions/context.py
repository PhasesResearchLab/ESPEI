"""Convenience function to create a context for the built in error functions"""

import logging
import copy
import symengine
from pycalphad import variables as v
from pycalphad.codegen.callables import build_callables
from pycalphad.core.utils import instantiate_models, filter_phases, unpack_components
from espei.error_functions import get_thermochemical_data, get_equilibrium_thermochemical_data
from espei.utils import database_symbols_to_fit, get_model_dict
from espei.error_functions.residual_base import residual_function_registry
from espei.error_functions.zpf_error import ZPFResidual

_log = logging.getLogger(__name__)


def setup_context(dbf, datasets, symbols_to_fit=None, data_weights=None, phase_models=None, make_callables=True):
    """
    Set up a context dictionary for calculating error.

    Parameters
    ----------
    dbf : Database
        A pycalphad Database that will be fit
    datasets : PickleableTinyDB
        A database of single- and multi-phase data to fit
    symbols_to_fit : list of str
        List of symbols in the Database that will be fit. If None (default) are
        passed, then all parameters prefixed with `VV` followed by a number,
        e.g. VV0001 will be fit.

    Returns
    -------

    Notes
    -----
    A copy of the Database is made and used in the context. To commit changes
    back to the original database, the dbf.symbols.update method should be used.
    """
    dbf = copy.deepcopy(dbf)
    if phase_models is not None:
        comps = sorted(phase_models['components'])
    else:
        comps = sorted([sp for sp in dbf.elements])
    if symbols_to_fit is None:
        symbols_to_fit = database_symbols_to_fit(dbf)
    else:
        symbols_to_fit = sorted(symbols_to_fit)
    data_weights = data_weights if data_weights is not None else {}

    if len(symbols_to_fit) == 0:
        raise ValueError('No degrees of freedom. Database must contain symbols starting with \'V\' or \'VV\', followed by a number.')
    else:
        _log.info('Fitting %s degrees of freedom.', len(symbols_to_fit))

    for x in symbols_to_fit:
        if isinstance(dbf.symbols[x], symengine.Piecewise):
            _log.debug('Replacing %s in database', x)
            dbf.symbols[x] = dbf.symbols[x].args[0]

    # construct the models for each phase, substituting in the SymEngine symbol to fit.
    if phase_models is not None:
        model_dict = get_model_dict(phase_models)
    else:
        model_dict = {}
    _log.trace('Building phase models (this may take some time)')
    import time
    t1 = time.time()
    phases = sorted(filter_phases(dbf, unpack_components(dbf, comps), dbf.phases.keys()))
    parameters = dict(zip(symbols_to_fit, [0]*len(symbols_to_fit)))
    models = instantiate_models(dbf, comps, phases, model=model_dict, parameters=parameters)
    if make_callables:
        eq_callables = build_callables(dbf, comps, phases, models, parameter_symbols=symbols_to_fit,
                            output='GM', build_gradients=True, build_hessians=True,
                            additional_statevars={v.N, v.P, v.T})
    else:
        eq_callables = None
    t2 = time.time()
    _log.trace('Finished building phase models (%0.2fs)', t2-t1)
    residual_objs = []
    for residual_func_class in residual_function_registry.get_registered_residual_functions():
        _log.trace('Getting residual object for %s', residual_func_class.__qualname__)
        t1 = time.time()
        residual_obj = residual_func_class(dbf, datasets, phase_models, symbols_to_fit, data_weights)
        residual_objs.append(residual_obj)
        t2 = time.time()
        _log.trace('Finished getting residual object for %s in %0.2f s', residual_func_class.__qualname__, t2-t1)


    # context for the log probability function
    # for all cases, parameters argument addressed in MCMC loop
    error_context = {
        'symbols_to_fit': symbols_to_fit,
        "residual_objs": residual_objs,
        'activity_kwargs': {
            'dbf': dbf, 'comps': comps, 'phases': phases, 'datasets': datasets,
            'phase_models': models, 'callables': eq_callables,
            'data_weight': data_weights.get('ACR', 1.0),
        },
    }
    return error_context
