"""
Calculate error due to thermochemical quantities: heat capacity, entropy, enthalpy.
"""

import itertools
import logging

from scipy.stats import norm
import numpy as np

from pycalphad import calculate, Model
from pycalphad.core.utils import unpack_components

from espei.core_utils import ravel_conditions, get_prop_data


def calculate_points_array(phase_constituents, configuration, occupancies=None):
    """
    Calculate the points array to use in pycalphad calculate calls.

    Converts the configuration data (and occupancies for mixing data) into the
    points array by looking up the indices in the active phase constituents.

    Parameters
    ----------
    phase_constituents : list
        List of active constituents in a phase
    configuration : list
        List of the sublattice configuration
    occupancies : list
        List of sublattice occupancies. Required for mixing sublattices, otherwise takes no effect.

    Returns
    -------
    numpy.ndarray

    Notes
    -----
    Errors will be raised if components in the configuration are not in the
    corresponding phase constituents sublattice.
    """
    # pad the occupancies for zipping if none were passed (the case for non-mixing)
    if occupancies is None:
        occupancies = [0] * len(configuration)

    # construct the points array from zeros
    points = np.zeros(sum(len(subl) for subl in phase_constituents))
    current_subl_idx = 0  # index that marks the beginning of the sublattice
    for phase_subl, config_subl, subl_occupancies in zip(phase_constituents, configuration, occupancies):
        phase_subl = list(phase_subl)
        if isinstance(config_subl, (tuple, list)):
            # we have mixing on the sublattice
            for comp, comp_occupancy in zip(config_subl, subl_occupancies):
                points[current_subl_idx + phase_subl.index(comp)] = comp_occupancy
        else:
            points[current_subl_idx + phase_subl.index(config_subl)] = 1
        current_subl_idx += len(phase_subl)
    return points


def get_prop_samples(dbf, comps, phase_name, desired_data):
    """
    Return data values and the conditions to calculate them by pycalphad
    calculate from the datasets

    Parameters
    ----------
    dbf : pycalphad.Database
        Database to consider
    comps : list
        List of active component names
    phase_name : str
        Name of the phase to consider from the Database
    desired_data : list
        List of dictionary datasets that contain the values to sample

    Returns
    -------
    dict
        Dictionary of condition kwargs for pycalphad's calculate and the expected values

    """
    # TODO: assumes T, P as conditions
    # sublattice constituents are Species objects, so we need to be doing intersections with those
    species_comps = unpack_components(dbf, comps)
    phase_constituents = dbf.phases[phase_name].constituents
    # phase constituents must be filtered to only active:
    phase_constituents = [[c.name for c in sorted(subl_constituents.intersection(set(species_comps)))] for subl_constituents in phase_constituents]

    # calculate needs points, state variable lists, and values to compare to
    calculate_dict = {
        'P': np.array([]),
        'T': np.array([]),
        'points': np.atleast_2d([[]]).reshape(-1, sum([len(subl) for subl in phase_constituents])),
        'values': np.array([]),
        'weights': [],
        'references': [],
    }

    for datum in desired_data:
        # extract the data we care about
        datum_T = datum['conditions']['T']
        datum_P = datum['conditions']['P']
        configurations = datum['solver']['sublattice_configurations']
        occupancies = datum['solver'].get('sublattice_occupancies')
        values = np.array(datum['values'])

        # broadcast and flatten the conditions arrays
        P, T = ravel_conditions(values, datum_P, datum_T)
        if occupancies is None:
            occupancies = [None] * len(configurations)

        # calculate the points arrays, should be 2d array of points arrays
        points = np.array([calculate_points_array(phase_constituents, config, occup) for config, occup in zip(configurations, occupancies)])

        # add everything to the calculate_dict
        calculate_dict['P'] = np.concatenate([calculate_dict['P'], P])
        calculate_dict['T'] = np.concatenate([calculate_dict['T'], T])
        calculate_dict['points'] = np.concatenate([calculate_dict['points'], np.repeat(points, len(T)/points.shape[0], axis=0)], axis=0)
        calculate_dict['values'] = np.concatenate([calculate_dict['values'], values.flatten()])
        calculate_dict['weights'].extend([datum.get('weight', 1.0) for _ in range(values.flatten().size)])
        calculate_dict['references'].extend([datum.get('reference', "") for _ in range(values.flatten().size)])

    return calculate_dict


def calculate_thermochemical_error(dbf, comps, phases, datasets, parameters=None, phase_models=None, callables=None):
    """
    Calculate the weighted single phase error in the Database

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
        Dictionary of symbols that will be overridden in pycalphad.calculate
    phase_models : dict
        Phase models to pass to pycalphad calculations. Ideal mixing
        contributions must be removed.
    callables : dict
        Dictionary of {output_property: callables_dict} where callables_dict is
        a dictionary of {phase_name: callables}
        to pass to pycalphad. These must have ideal mixing portions removed.

    Returns
    -------
    float
        A single float of the residual sum of square errors

    Notes
    -----
    There are different single phase values, HM_MIX, SM_FORM, CP_FORM, etc.
    Each of these have different units and the error cannot be compared directly.
    To normalize all of the errors, a normalization factor must be used.
    Equation 2.59 and 2.60 in Lukas, Fries, and Sundman "Computational Thermodynamics" shows how this can be considered.
    Each type of error will be weighted by the reciprocal of the estimated uncertainty in the measured value and conditions.
    The weighting factor is calculated by
    $ p_i = (\Delta L_i)^{-1} $
    where $ \Delta L_i $ is the uncertainty in the measurement.
    We will neglect the uncertainty for quantities such as temperature, assuming they are small.

    """
    if parameters is None:
        parameters = {}

    if phase_models is None:
        # create phase models with ideal mixing removed
        phase_models = {}
        for phase_name in phases:
            phase_models[phase_name] = Model(dbf, comps, phase_name)
            phase_models[phase_name].models['idmix'] = 0


    # estimated from NIST TRC uncertainties
    property_std_deviation = {
        'HM': 500.0,  # J/mol
        'SM':   0.2,  # J/K-mol
        'CPM':  0.2,  # J/K-mol
    }
    property_suffixes = ('_FORM', '_MIX')
    # the kinds of properties, e.g. 'HM'+suffix =>, 'HM_FORM', 'HM_MIX'
    # we could also include the bare property ('' => 'HM'), but these are rarely used in ESPEI
    properties = [''.join(prop) for prop in itertools.product(property_std_deviation.keys(), property_suffixes)]

    whitelist_properties = ['HM', 'SM', 'CPM', 'HM_MIX', 'SM_MIX', 'CPM_MIX']
    # if callables is None, construct empty callables dicts, which will be JIT compiled by pycalphad later
    callables = callables if callables is not None else {prop: None for prop in whitelist_properties}

    prob_error = 0.0
    for phase_name in phases:
        for prop in properties:
            desired_data = get_prop_data(comps, phase_name, prop, datasets)
            if len(desired_data) == 0:
                continue
            calculate_dict = get_prop_samples(dbf, comps, phase_name, desired_data)
            if prop.endswith('_FORM'):
                calculate_dict['output'] = ''.join(prop.split('_')[:-1])
                params = parameters.copy()
                params.update({'GHSER' + (c.upper() * 2)[:2]: 0 for c in comps})
            else:
                calculate_dict['output'] = prop
                params = parameters
            sample_values = calculate_dict.pop('values')
            weights = calculate_dict.pop('weights')
            dataset_refs = calculate_dict.pop('references')
            results = calculate(dbf, comps, phase_name, broadcast=False, parameters=params, model=phase_models,
                                callables=callables[calculate_dict['output']],
                                **calculate_dict)[calculate_dict['output']].values
            std_dev = property_std_deviation[prop.split('_')[0]]
            probabilities = []
            for result, sample_value, weight, ref in zip(results, sample_values, weights, dataset_refs):
                probabilities.append(norm(loc=0, scale=std_dev/weight).logpdf(result-sample_value))
            logging.debug('Thermochemical error - data: {}, probabilities: {}, references: {}'.format(sample_values, probabilities, dataset_refs))
            prob_error = np.sum(probabilities)
    return prob_error
