"""
Calculate error due to measured activities.
"""

import logging

import numpy as np
import tinydb
from pycalphad import equilibrium, variables as v
from pycalphad.plot.eqplot import _map_coord_to_variable
from pycalphad.core.utils import filter_phases, unpack_components
from scipy.stats import norm

from espei.core_utils import ravel_conditions

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
