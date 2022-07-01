"""Convenience function to create a context for the built in error functions"""

import logging
import copy
import time

import symengine

from espei.phase_models import PhaseModelSpecification
from espei.utils import database_symbols_to_fit
from espei.error_functions.residual_base import residual_function_registry

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
    phase_models : Optional[Dict[str, Any]]
        Phase model dictionary that will be converted to PhaseModelSpecification
        if it is provided.

    Returns
    -------

    Notes
    -----
    A copy of the Database is made and used in the context. To commit changes
    back to the original database, the dbf.symbols.update method should be used.
    """
    data_weights = data_weights if data_weights is not None else {}
    if phase_models is not None:
        phase_models = PhaseModelSpecification(**phase_models)

    # Copy the database because we replace Piecewise symbols and want to preserve the original database
    dbf = copy.deepcopy(dbf)
    if symbols_to_fit is None:
        symbols_to_fit = database_symbols_to_fit(dbf)
    else:
        symbols_to_fit = sorted(symbols_to_fit)
    if len(symbols_to_fit) == 0:
        raise ValueError("No degrees of freedom. Database must contain symbols starting with 'V' or 'VV', followed by a number.")
    else:
        _log.info("Fitting %s degrees of freedom.", len(symbols_to_fit))
    for x in symbols_to_fit:
        if isinstance(dbf.symbols[x], symengine.Piecewise):
            _log.debug("Replacing %s in database", x)
            dbf.symbols[x] = dbf.symbols[x].args[0]

    residual_objs = []
    for residual_func_class in residual_function_registry.get_registered_residual_functions():
        _log.trace("Getting residual object for %s", residual_func_class.__qualname__)
        t1 = time.time()
        residual_obj = residual_func_class(dbf, datasets, phase_models, symbols_to_fit, data_weights)
        residual_objs.append(residual_obj)
        t2 = time.time()
        _log.trace("Finished getting residual object for %s in %0.2f s", residual_func_class.__qualname__, t2-t1)

    # context for the log probability function
    # for all cases, parameters argument addressed in MCMC loop
    error_context = {
        "symbols_to_fit": symbols_to_fit,
        "residual_objs": residual_objs,
    }
    return error_context
