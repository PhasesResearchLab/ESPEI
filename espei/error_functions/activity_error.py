"""
Calculate error due to measured activities.

The residual function implemented in this module needs to exist because it is
currently not possible to compute activity as a property via equilibrium
calculations because as PyCalphad does not yet have a suitable notion of a
reference state that could be used for equilibrium chemical potentials.

"""

import logging
from typing import Dict, List, Optional, Union

import numpy as np
import numpy.typing as npt
import tinydb
from pycalphad import Database, equilibrium, variables as v
from pycalphad.plot.eqplot import _map_coord_to_variable
from pycalphad.core.utils import filter_phases, unpack_components
from scipy.stats import norm

from espei.core_utils import ravel_conditions
from espei.error_functions.residual_base import ResidualFunction, residual_function_registry
from espei.phase_models import PhaseModels
from espei.typing import SymbolName
from espei.utils import database_symbols_to_fit, PickleableTinyDB

_log = logging.getLogger(__name__)


def target_chempots_from_activity(component, target_activity, temperatures, reference_result):
    """
    Return an array of experimental chemical potentials for the component

    Parameters
    ----------
    component : str
        Name of the component
    target_activity : numpy.ndarray
        Array of experimental activities
    temperatures : numpy.ndarray
        Ravelled array of temperatures (of same size as ``exp_activity``).
    reference_result : xarray.Dataset
        Dataset of the equilibrium reference state. Should contain a singe point calculation.

    Returns
    -------
    numpy.ndarray
        Array of experimental chemical potentials
    """
    # acr_i = exp((mu_i - mu_i^{ref})/(RT))
    # so mu_i = R*T*ln(acr_i) + mu_i^{ref}
    ref_chempot = reference_result["MU"].sel(component=component).values.flatten()
    return v.R * temperatures * np.log(target_activity) + ref_chempot


def chempot_error(sample_chempots, target_chempots, std_dev=10.0):
    """
    Return the sum of square error from chemical potentials

    sample_chempots : numpy.ndarray
        Calculated chemical potentials
    target_activity : numpy.ndarray
        Chemical potentials to target
    std_dev : float
        Standard deviation of activity measurements in J/mol. Corresponds to the
        standard deviation of differences in chemical potential in typical
        measurements of activity.

    Returns
    -------
    float
        Error due to chemical potentials
    """
    # coerce the chemical potentials to float64s, fixes an issue where SymPy NaNs don't work
    return norm(loc=0, scale=std_dev).logpdf(np.array(target_chempots - sample_chempots, dtype=np.float64))


def calculate_activity_error(dbf, comps, phases, datasets, parameters=None, phase_models=None, callables=None, data_weight=1.0):
    """
    Return the sum of square error from activity data

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
        Dictionary of symbols that will be overridden in pycalphad.equilibrium
    phase_models : dict
        Phase models to pass to pycalphad calculations
    callables : dict
        Callables to pass to pycalphad
    data_weight : float
        Weight for standard deviation of activity measurements, dimensionless.
        Corresponds to the standard deviation of differences in chemical
        potential in typical measurements of activity, in J/mol.

    Returns
    -------
    float
        A single float of the sum of square errors

    Notes
    -----
    General procedure:
    1. Get the datasets
    2. For each dataset

        a. Calculate reference state equilibrium
        b. Calculate current chemical potentials
        c. Find the target chemical potentials
        d. Calculate error due to chemical potentials

    """
    std_dev = 500  # J/mol

    if parameters is None:
        parameters = {}

    activity_datasets = datasets.search(
        (tinydb.where('output').test(lambda x: 'ACR' in x)) &
        (tinydb.where('components').test(lambda x: set(x).issubset(comps))))

    error = 0
    if len(activity_datasets) == 0:
        return error

    for ds in activity_datasets:
        acr_component = ds['output'].split('_')[1]  # the component of interest
        # calculate the reference state equilibrium
        ref = ds['reference_state']
        # data_comps and data_phases ensures that we only do calculations on
        # the subsystem of the system defining the data.
        data_comps = ds['components']
        data_phases = filter_phases(dbf, unpack_components(dbf, data_comps), candidate_phases=phases)
        ref_conditions = {_map_coord_to_variable(coord): val for coord, val in ref['conditions'].items()}
        ref_result = equilibrium(dbf, data_comps, ref['phases'], ref_conditions,
                                 model=phase_models, parameters=parameters,
                                 callables=callables)

        # calculate current chemical potentials
        # get the conditions
        conditions = {}
        # first make sure the conditions are paired
        # only get the compositions, P and T are special cased
        conds_list = [(cond, value) for cond, value in ds['conditions'].items() if cond not in ('P', 'T')]
        # ravel the conditions
        # we will ravel each composition individually, since they all must have the same shape
        for comp_name, comp_x in conds_list:
            P, T, X = ravel_conditions(ds['values'], ds['conditions']['P'], ds['conditions']['T'], comp_x)
            conditions[v.P] = P
            conditions[v.T] = T
            conditions[_map_coord_to_variable(comp_name)] = X
        # do the calculations
        # we cannot currently turn broadcasting off, so we have to do equilibrium one by one
        # invert the conditions dicts to make a list of condition dicts rather than a condition dict of lists
        # assume now that the ravelled conditions all have the same size
        conditions_list = [{c: conditions[c][i] for c in conditions.keys()} for i in range(len(conditions[v.T]))]
        current_chempots = []
        for conds in conditions_list:
            sample_eq_res = equilibrium(dbf, data_comps, data_phases, conds,
                                        model=phase_models, parameters=parameters,
                                        callables=callables)
            current_chempots.append(sample_eq_res.MU.sel(component=acr_component).values.flatten()[0])
        current_chempots = np.array(current_chempots)

        # calculate target chempots
        samples = np.array(ds['values']).flatten()
        target_chempots = target_chempots_from_activity(acr_component, samples, conditions[v.T], ref_result)
        # calculate the error
        weight = ds.get('weight', 1.0)
        pe = chempot_error(current_chempots, target_chempots, std_dev=std_dev/data_weight/weight)
        error += np.sum(pe)
        _log.debug('Data: %s, chemical potential difference: %s, probability: %s, reference: %s', samples, current_chempots-target_chempots, pe, ds["reference"])

    # TODO: write a test for this
    if np.any(np.isnan(np.array([error], dtype=np.float64))):  # must coerce sympy.core.numbers.Float to float64
        return -np.inf
    return error


# TODO: the __init__ method should pre-compute Model and PhaseRecord objects
#       similar to the other residual functions, which will be much more performant.
# TODO: it seems possible (likely?) that "global" callables that were used
#       previously could be incorrect if there are activity datasets with
#       different sets of active components. Usually models, callables, and
#       phase records are tied 1:1 with a set of components. For now, callables
#       will never be built, but this will almost certainly cause a performance
#       regression. Model will also not be pre-built so we can properly use
#       custom user models
class ActivityResidual(ResidualFunction):
    def __init__(
        self,
        database: Database,
        datasets: PickleableTinyDB,
        phase_models: Union[PhaseModels, None],
        symbols_to_fit: Optional[List[SymbolName]] = None,
        weight: Optional[Dict[str, float]] = None,
        ):
        super().__init__(database, datasets, phase_models, symbols_to_fit)

        if weight is not None:
            self.weight = weight.get("ACR", 1.0)
        else:
            self.weight = 1.0

        if phase_models is not None:
            comps = sorted(phase_models.components)
            model_dict = phase_models.get_model_dict()
        else:
            comps = sorted(database.elements)
            model_dict = dict()
        phases = sorted(filter_phases(database, unpack_components(database, comps), database.phases.keys()))
        if symbols_to_fit is None:
            symbols_to_fit = database_symbols_to_fit(database)
        self._symbols_to_fit = symbols_to_fit

        self._activity_likelihood_kwargs = {
            "dbf": database, "comps": comps, "phases": phases, "datasets": datasets,
            "phase_models": model_dict,
            "callables": None,
            "data_weight": self.weight,
        }

    def get_residual(self, parameters: npt.ArrayLike) -> float:
        # TODO: implement via refactoring calculate_activity_error
        raise NotImplementedError("Getting residual for activity data not implemented.")

    def get_likelihood(self, parameters: npt.NDArray) -> float:
        parameters = {param_name: param for param_name, param in zip(self._symbols_to_fit, parameters.tolist())}
        likelihood = calculate_activity_error(parameters=parameters, **self._activity_likelihood_kwargs)
        return likelihood


residual_function_registry.register(ActivityResidual)
