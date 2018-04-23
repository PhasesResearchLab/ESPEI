"""
Calculate error due to measured activities.
"""

import numpy as np
import tinydb

from pycalphad import equilibrium, variables as v
from pycalphad.plot.eqplot import _map_coord_to_variable

from espei.core_utils import ravel_conditions


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
    ref_chempot = reference_result.MU.sel(component=component).values.flatten()
    return v.R * temperatures * np.log(target_activity) + ref_chempot


def chempot_error(sample_chempots, target_chempots):
    """
    Return the sum of square error from chemical potentials

    sample_chempots : numpy.ndarray
        Calculated chemical potentials
    target_activity : numpy.ndarray
        Chemical potentials to target

    Returns
    -------
    float
        Error due to chemical potentials
    """
    return -np.sum(np.square(target_chempots - sample_chempots))


def calculate_activity_error(dbf, comps, phases, datasets, parameters=None, phase_models=None, callables=None, grad_callables=None, hess_callables=None, massfuncs=None, massgradfuncs=None):
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
    grad_callables : dict
        Gradient callables to pass to pycalphad
    hess_callables : dict
        Hessian callables to pass to pycalphad
    massfuncs : dict
        Callables of mass derivatives to pass to pycalphad
    massgradfuncs : dict
        Gradient callables of mass derivatives to pass to pycalphad

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
        ref_conditions = {_map_coord_to_variable(coord): val for coord, val in ref['conditions'].items()}
        ref_result = equilibrium(dbf, ds['components'], ref['phases'], ref_conditions,
                                 model=phase_models, parameters=parameters,
                                 massfuncs=massfuncs,
                                 massgradfuncs=massgradfuncs,
                                 callables=callables,
                                 grad_callables=grad_callables,
                                 hess_callables=hess_callables,
                                 )

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
            sample_eq_res = equilibrium(dbf, ds['components'], phases, conds,
                                        model=phase_models, parameters=parameters,
                                        massfuncs=massfuncs,
                                        massgradfuncs=massgradfuncs,
                                        callables=callables,
                                        grad_callables=grad_callables,
                                        hess_callables=hess_callables,
                                        )
            current_chempots.append(sample_eq_res.MU.sel(component=acr_component).values.flatten()[0])
        current_chempots = np.array(current_chempots)

        # calculate target chempots
        target_chempots = target_chempots_from_activity(acr_component, np.array(ds['values']).flatten(), conditions[v.T], ref_result)
        # calculate the error
        error += chempot_error(current_chempots, target_chempots)
    # TODO: write a test for this
    if np.any(np.isnan(np.array([error], dtype=np.float64))):  # must coerce sympy.core.numbers.Float to float64
        return -np.inf
    return error
