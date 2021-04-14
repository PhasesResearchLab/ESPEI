"""
Internal utilities for developer use. May not be useful to users.
"""
import copy
import itertools
import numpy as np
import tinydb
from espei.sublattice_tools import canonicalize, recursive_tuplify
from espei.parameter_selection.redlich_kister import calc_interaction_product


def get_data(comps, phase_name, configuration, symmetry, datasets, prop):
    """
    Return list of cleaned single phase datasets matching the passed arguments.

    Parameters
    ----------
    comps : list
        List of string component names
    phase_name : str
        Name of phase
    configuration : tuple
        Sublattice configuration as a tuple, e.g. ("CU", ("CU", "MG"))
    symmetry : list of lists
        List of sublattice indices with symmetry
    datasets : espei.utils.PickleableTinyDB
        Database of datasets to search for data
    prop : list
        String name of the property of interest.

    Returns
    -------
    list
        List of datasets matching the arguments.

    """
    desired_data = datasets.search((tinydb.where('output').test(lambda x: x in prop)) &
                                   (tinydb.where('components').test(lambda x: set(x).issubset(comps))) &
                                   (tinydb.where('solver').test(symmetry_filter, configuration, recursive_tuplify(symmetry) if symmetry else symmetry)) &
                                   (tinydb.where('phases') == [phase_name]))
    # This seems to be necessary because the 'values' member does not modify 'datasets'
    # But everything else does!
    desired_data = copy.deepcopy(desired_data)

    def recursive_zip(a, b):
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            return list(recursive_zip(x, y) for x, y in zip(a, b))
        else:
            return list(zip(a, b))

    for idx, data in enumerate(desired_data):
        # Filter output values to only contain data for matching sublattice configurations
        matching_configs = np.array([(canonicalize(sblconf, symmetry) == canonicalize(configuration, symmetry))
                                     for sblconf in data['solver']['sublattice_configurations']])
        matching_configs = np.arange(len(data['solver']['sublattice_configurations']))[matching_configs]
        # Rewrite output values with filtered data
        desired_data[idx]['values'] = np.array(data['values'], dtype=np.float_)[..., matching_configs]
        desired_data[idx]['solver']['sublattice_configurations'] = recursive_tuplify(np.array(data['solver']['sublattice_configurations'],
                                                                                              dtype=np.object_)[matching_configs].tolist())
        try:
            desired_data[idx]['solver']['sublattice_occupancies'] = np.array(data['solver']['sublattice_occupancies'],
                                                                             dtype=np.object_)[matching_configs].tolist()
        except KeyError:
            pass
        # Filter out temperatures below 298.15 K (for now, until better refstates exist)
        temp_filter = np.atleast_1d(data['conditions']['T']) >= 298.15
        desired_data[idx]['conditions']['T'] = np.atleast_1d(data['conditions']['T'])[temp_filter]
        # Don't use data['values'] because we rewrote it above; not sure what 'data' references now
        desired_data[idx]['values'] = desired_data[idx]['values'][..., temp_filter, :]
    return desired_data


def _zpf_conditions_shape(zpf_values):
    """
    Calculate the shape of the conditions for ZPF values

    Parameters
    ----------
    zpf_values : list
        ZPF values. A multidimensional list where the innermost dimension looks like
        ['PHASE_NAME', ['C1', 'C2'], [x1, x2]]

    Returns
    -------
    Tuple
        Shape of the conditions
    """
    # create a list of the length of each dimension, if that dimension is a list
    # should terminate (return []) when we reach the 'PHASE_NAME'
    def _recursive_shape(x):
        if isinstance(x, list):
            return [len(x)] + _recursive_shape(x[0])
        else:
            return []

    shape = tuple(_recursive_shape(zpf_values))
    # the dimensions are something like (..., x, y)
    # x is the number of equilibria in the first equilibria set
    # y is the length of the single equlibrium and should always be 3:
    # (phase name, component names, component amounts)
    # the shape of conditions is the shape where these last two dimensions are removed
    return shape[-2]


def ravel_conditions(values, *conditions, **kwargs):
    """
    Broadcast and flatten conditions to the shape dictated by the values.

    Special handling for ZPF data that does not have nice array values.

    Parameters
    ----------
    values : list
        Multidimensional lists of values
    conditions : list
        List of conditions to broadcast. Must be the same length as the number
        of dimensions of the values array. In code, the following must be True:
        `all([s == len(cond) for s, cond in zip(values.shape, conditions)])`

    zpf : bool, optional
        Whether to consider values as a special case of ZPF data (not an even grid of conditions)
        Default is False

    Returns
    -------
    tuple
        Tuple of ravelled conditions

    Notes
    -----
    The current implementation of ZPF data only has the shape for one condition
    and this assumption is hardcoded in various places.

    Here we try to be as general as possible by explicitly calculating the shape
    of the ZPF values.

    A complication of this is that the user of this function must pass the
    correct conditions because usually T and P are specified in ZPF (but, again,
    only one can actually be a condition given the current shape).
    """
    # we have to parse the `zpf` kwarg manually because py27 does not allow named args after *args
    zpf = kwargs.get('zpf')
    if zpf:
        values_shape = _zpf_conditions_shape(values)
        # we have to make our integers tuples
        if not isinstance(values_shape, tuple):
            values_shape = tuple([values_shape])
    else:
        values_shape = np.array(values).shape
    ravelled_conditions = []
    for cond_idx, cond in enumerate(conditions):
        cond_shape = [1 for _ in values_shape]
        cond_shape[cond_idx] = -1
        x = np.array(cond).reshape(cond_shape)
        for shape_idx, dim_len in enumerate(values_shape):
            if shape_idx != cond_idx:
                x = x.repeat(dim_len, axis=shape_idx)
        ravelled_conditions.append(x.flatten())
    return tuple(ravelled_conditions)


def ravel_zpf_values(desired_data, independent_comps, conditions=None):
    """
    Unpack the phases and compositions from ZPF data. Dependent components are converted to independent components.

    Parameters
    ----------
    desired_data : espei.utils.PickleableTinyDB
        The selected data
    independent_comps : list
        List of indepdendent components. Used for mass balance component conversion
    conditions : dict
        Conditions to filter for. Right now only considers fixed temperatures

    Returns
    -------
    dict
        A dictonary of list of lists of tuples. Each dictionary key is the
        number of phases in equilibrium, e.g. a key "2" might have values
        [[(PHASE_NAME_1, {'C1': X1, 'C2': X2}, refkey), (PHASE_NAME_2, {'C1': X1, 'C2': X2}, refkey)], ...]
        Three would have three inner tuples and so on.

    """

    # integers are the number of equilibria in the phase that are lists of the individual points
    independent_comps = set(independent_comps)
    equilibria_dict = {}

    for data in desired_data:
        values = data['values']
        T = ravel_conditions(data['values'], data['conditions']['T'], zpf=True)[0]

        for equilbrium, temperature in zip(values, T):
            if conditions is not None:
                # make sure that the conditions match
                if temperature != conditions['T']:
                    continue
            # go through each equilibrium phase and ravel it correctly
            this_equilibrium = []
            for eq_phase in equilbrium:
                phase_name = eq_phase[0]
                components = eq_phase[1]
                compositions = eq_phase[2]

                # fix up any independent component issues and fill out the component dict
                comp_dict = {}
                # assume that there are len(independent_comps)+1 components
                # therefore mass balance applies
                if independent_comps != set(components):
                    mass_balance_dependent_comp = list(set(components).difference(independent_comps))[0]
                else:
                    mass_balance_dependent_comp = None
                for c, x in zip(components, compositions):
                    if c == mass_balance_dependent_comp:
                        c = list(independent_comps.difference(set(components)))[0]
                        x = None if any([xx is None for xx in compositions]) else 1 - sum(compositions)
                    comp_dict[c] = x
                if conditions is None:
                    # assume the other condition is temperature:
                    comp_dict['T'] = temperature
                this_equilibrium.append((phase_name, comp_dict, data['reference']))

            # add this set of equilibrium phases to the correct key in the equilibria dict
            n_phases_in_equilibrium = len(equilbrium)
            list_of_n_phase_equilibria = equilibria_dict.get(n_phases_in_equilibrium, [])
            list_of_n_phase_equilibria.append(this_equilibrium)
            equilibria_dict[n_phases_in_equilibrium] = list_of_n_phase_equilibria
    return equilibria_dict


def recursive_map(f, x):
    """
    map, but over nested lists

    Parameters
    ----------
    f : callable
        Function to apply to x
    x : list or value
        Value passed to v

    Returns
    -------
    list or value
    """
    if isinstance(x, list):
        if [isinstance(xx, list) for xx in x]:
            # we got a nested list
            return [recursive_map(f, xx) for xx in x]
        else:
            # it's a list with some values inside
            return list(map(f, x))
    else:
        # not a list, probably just a singular value
        return f(x)


def symmetry_filter(x, config, symmetry):
    """
    Return True if the candidate sublattice configuration has any symmetry
    which matches the phase model symmetry.

    Parameters
    ----------
    x : dict
        the candidate dataset 'solver' dict. Must contain the "sublattice_configurations" key
    config : list
        the configuration of interest: e.g. ['AL', ['AL', 'NI'], 'VA']
    symmetry : list
        tuple of tuples where each inner tuple is a group of equivalent
        sublattices. A value of ((0, 1), (2, 3, 4)) means that sublattices
        at indices 0 and 1 are symmetrically equivalent to each other and
        sublattices at indices 2, 3, and 4 are symetrically equivalent to
        each other.

    Returns
    -------
    bool

    """
    if x['mode'] == 'manual':
        if len(config) != len(x['sublattice_configurations'][0]):
            return False
        # If even one matches, it's a match
        # We do more filtering downstream
        for data_config in x['sublattice_configurations']:
            if canonicalize(config, symmetry) == canonicalize(data_config, symmetry):
                return True
    return False


def get_prop_data(comps, phase_name, prop, datasets, additional_query=None):
    """
    Return datasets that match the components, phase and property

    Parameters
    ----------
    comps : list
        List of components to get data for
    phase_name : str
        Name of the phase to get data for
    prop : str
        Property to get data for
    datasets : espei.utils.PickleableTinyDB
        Datasets to search for data
    additional_query : tinydb.Query
        A TinyDB Query object to search for. If None, a Query() will be created that does nothing.

    Returns
    -------
    list
        List of dictionary datasets that match the criteria

    """
    if additional_query is None:
        additional_query = tinydb.Query()
    # TODO: we should only search and get phases that have the same sublattice_site_ratios as the phase in the database
    desired_data = datasets.search(
        (tinydb.where('output').test(lambda x: x in prop)) &
        (tinydb.where('components').test(lambda x: set(x).issubset(comps))) &
        (tinydb.where('phases') == [phase_name]) &
        additional_query
    )
    return desired_data
