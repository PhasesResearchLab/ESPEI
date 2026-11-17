"""
Utilities for querying, modifiying, and extracting data from Datasets.
"""
import copy
from typing import List
import numpy as np
import tinydb
from espei.datasets import Dataset
from espei.sublattice_tools import canonical_order, canonicalize


def _select_values(data: Dataset, index) -> None:
    """Keep the values at an index, e.g. np.s_[..., selected_configurations], and the weights of those values."""
    values = np.asarray(data['values'], dtype=np.float64)
    data['values'] = values[index]
    data['weight'] = np.broadcast_to(data.get('weight', 1.0), values.shape)[index]  # weights broadcast to the values, e.g. one for each configuration


def select_configurations(data: Dataset, selected) -> None:
    """Keep the values, weights, sublattice configurations and occupancies of the selected configurations of a fixed configuration dataset."""
    selected = np.asarray(selected, dtype=np.bool_)
    _select_values(data, np.s_[..., selected])
    for key in ('sublattice_configurations', 'sublattice_occupancies'):
        data['solver'][key] = [x for x, keep in zip(data['solver'][key], selected) if keep]


def filter_configurations(desired_data: List[Dataset], configuration, symmetry) -> List[Dataset]:
    """
    Return non-equilibrium thermochemical datasets with only the configurations that are the configuration by symmetry.

    The sublattices of the configurations and their occupancies are in
    canonical order (see ``canonicalize``), which is the order of the
    sublattices of the canonical configuration.

    Parameters
    ----------
    desired_data : List[Dataset]
        List of non-equilibrium thermochemical datasets
    configuration : tuple
        Sublattice configuration as a tuple, e.g. (("CU",), ("CU", "MG"))
    symmetry : list of lists
        List of sublattice indices with symmetry

    Returns
    -------
    List[Dataset]

    """
    canonical_config = canonicalize(configuration, symmetry)
    for data in desired_data:
        solver = data['solver']
        orders = [canonical_order(config, symmetry) for config in solver['sublattice_configurations']]
        for key in ('sublattice_configurations', 'sublattice_occupancies'):
            solver[key] = [tuple(sublattices[i] for i in order) for sublattices, order in zip(solver[key], orders)]
        select_configurations(data, [config == canonical_config for config in solver['sublattice_configurations']])
    return desired_data


def filter_temperatures(desired_data: List[Dataset]) -> List[Dataset]:
    """
    Return non-equilibrium thermochemical datasets with temperatures below 298.15 K removed.

    The currently provided unary reference data from ESPEI use the SGTE unary data that
    are defined as piecewise in temperature with a lower limit of 298.15 K for most
    elements. Since pycalphad does not extrapolate outside of piecewise temperature
    limits, this filter prevents fitting data to regions of temperature space where
    the energy is zero.

    Parameters
    ----------
    desired_data : List[Dataset]
        List of non-equilibrium thermochemical datasets

    Returns
    -------
    List[Dataset]

    """
    for data in desired_data:
        temp_filter = np.atleast_1d(data['conditions']['T']) >= 298.15
        data['conditions']['T'] = np.atleast_1d(data['conditions']['T'])[temp_filter]
        _select_values(data, np.s_[..., temp_filter, :])
    return desired_data


def ravel_conditions(values, *conditions):
    """
    Broadcast and flatten conditions to the shape dictated by the values.

    Parameters
    ----------
    values : list
        Multidimensional lists of values
    conditions : list
        List of conditions to broadcast. Must be the same length as the number
        of dimensions of the values array. In code, the following must be True:
        `all([s == len(cond) for s, cond in zip(values.shape, conditions)])`

    Returns
    -------
    tuple
        Tuple of ravelled conditions
    """
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
    desired_data : List[Dataset]
        The selected data
    independent_comps : List[str]
        List of indepdendent components. Used for mass balance component conversion
    conditions : Dict[str, Union[float, ArrayLike]]
        Conditions to filter for. Right now only considers fixed temperatures

    Returns
    -------
    Dict[int, List[List[Tuple[str, Dict[str, float], str]]] ]
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
        data_temperatures = np.atleast_1d(data["conditions"]["T"])
        if data_temperatures.squeeze().ndim == 0:
            data_temperatures = np.asarray(len(data["values"]) * [data_temperatures[0]])
        T = data_temperatures.ravel()
        data_pressures = np.atleast_1d(data["conditions"]["P"])
        if data_pressures.squeeze().ndim == 0:
            data_pressures = np.asarray(len(data["values"]) * [data_pressures[0]])
        P = data_pressures.ravel()
        for equilbrium, temperature, pressure in zip(values, T, P):
            if conditions is not None:
                # make sure that the conditions match
                temperature_condition = conditions.get('T')
                if temperature_condition is not None and temperature != temperature_condition:
                    continue
                pressure_condition = conditions.get('P')
                if pressure_condition is not None and pressure != pressure_condition:
                    continue
            # go through each equilibrium phase and ravel it correctly
            this_equilibrium = []
            for eq_phase in equilbrium:
                phase_name = eq_phase[0]
                components = eq_phase[1]
                compositions = eq_phase[2]

                # fix up any independent component issues and fill out the component dict
                comp_dict = {}
                # always add T and P
                comp_dict['T'] = temperature
                comp_dict['P'] = pressure
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

                this_equilibrium.append((phase_name, comp_dict, data.get('reference', '')))

            # add this set of equilibrium phases to the correct key in the equilibria dict
            # don't count __HYPERPLANE__ meta phases towards dimensionality of tie-region
            n_phases_in_equilibrium = len(list(filter(lambda eq: eq[0] != "__HYPERPLANE__", equilbrium)))
            list_of_n_phase_equilibria = equilibria_dict.get(n_phases_in_equilibrium, [])
            list_of_n_phase_equilibria.append(this_equilibrium)
            equilibria_dict[n_phases_in_equilibrium] = list_of_n_phase_equilibria
    return equilibria_dict


def symmetry_filter(x, config, symmetry):
    """
    Return True if the candidate sublattice configuration has any symmetry
    which matches the phase model symmetry.

    Parameters
    ----------
    x : dict
        the candidate dataset 'solver' dict. Must contain the "sublattice_configurations" key
    config : list
        the configuration of interest: e.g. [['AL'], ['AL', 'NI'], ['VA']]
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
    if len(config) != len(x['sublattice_configurations'][0]):
        return False
    # If even one matches, it's a match
    # We do more filtering downstream
    canonical_config = canonicalize(config, symmetry)
    return any(canonicalize(data_config, symmetry) == canonical_config for data_config in x['sublattice_configurations'])


def get_prop_data(comps, phase_name, prop, datasets, additional_query=None) -> List[Dataset]:
    """
    Return a copy of datasets that match the components, phase and property.

    The queried datasets are copied to ensure that any modifications are safe.

    Parameters
    ----------
    comps : list
        List of components to get data for
    phase_name : str
        Name of the phase to get data for
    prop : Union[str, List[str]]
        Property or properties to get data for including reference state in the string.
    datasets : espei.utils.PickleableTinyDB
        Datasets to search for data
    additional_query : tinydb.Query
        A TinyDB Query object to search for. If None, no additional query is applied.

    Returns
    -------
    List[Dataset]

    """
    props = [prop] if isinstance(prop, str) else list(prop)
    query = (
        (tinydb.where('output').test(lambda x: x in props)) &
        (tinydb.where('components').test(lambda x: set(x).issubset(comps))) &
        (tinydb.where('phases') == [phase_name])
    )
    if additional_query is not None:
        query = query & additional_query
    # TODO: we should only search and get phases that have the same sublattice_site_ratios as the phase in the database
    desired_data = datasets.search(query)
    return copy.deepcopy(desired_data)
