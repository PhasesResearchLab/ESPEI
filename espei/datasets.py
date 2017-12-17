import fnmatch, warnings, json, os

import numpy as np
from six import string_types
from tinydb.storages import MemoryStorage

from espei.utils import PickleableTinyDB


class DatasetError(Exception):
    """Exception raised when datasets are invalid."""
    pass


def check_dataset(dataset):
    """Ensure that the dataset is valid and consistent.

    Currently supports the following validation checks:
    * data shape is valid
    * phases and components used match phases and components entered

    Planned validation checks:
    * all required keys are present
    * individual shapes of keys, such as ZPF, sublattice configs and site ratios

    Note that this follows some of the implicit assumptions in ESPEI at the time
    of writing, such that conditions are only P, T, configs for single phase and
    essentially only T for ZPF data.

    Parameters
    ----------
    dataset : dict
        Dictionary of the standard ESPEI dataset.

    Returns
    -------
    None

    Raises
    ------
    DatasetError
        If an error is found in the dataset
    """
    is_single_phase = dataset['output'] != 'ZPF'
    components = dataset['components']
    conditions = dataset['conditions']
    values = dataset['values']
    phases = dataset['phases']
    if is_single_phase:
        solver = dataset['solver']
        sublattice_configurations = solver['sublattice_configurations']
        sublattice_site_ratios = solver['sublattice_site_ratios']
        sublattice_occupancies = solver.get('sublattice_occupancies', None)
        # check for mixing
        is_mixing = any([any([isinstance(subl, list) for subl in config]) for config in sublattice_configurations])
        # pad the values of sublattice occupancies if there is no mixing
        if sublattice_occupancies is None and not is_mixing:
            sublattice_occupancies = [None]*len(sublattice_configurations)
        elif sublattice_occupancies is None:
            raise DatasetError('At least one sublattice in the following sublattice configurations is mixing, but the "sublattice_occupancies" key is empty: {}'.format(sublattice_configurations))

    # check that the shape of conditions match the values
    num_pressure = np.atleast_1d(conditions['P']).size
    num_temperature = np.atleast_1d(conditions['T']).size
    if is_single_phase:
        values_shape = np.array(values).shape
        num_configs = len(dataset['solver']['sublattice_configurations'])
        conditions_shape = (num_pressure, num_temperature, num_configs)
        if conditions_shape != values_shape:
            raise DatasetError('Shape of conditions (P, T, configs): {} does not match the shape of the values {}.'.format(conditions_shape, values_shape))
    else:
        values_shape = (len(values))
        conditions_shape = (num_temperature)
        if conditions_shape != values_shape:
            raise DatasetError('Shape of conditions (T): {} does not match the shape of the values {}.'.format(conditions_shape, values_shape))

    # check that all of the correct phases are present
    if not is_single_phase:
        phases_entered = set(phases)
        phases_used = set()
        for zpf in values:
            for tieline in zpf:
                phases_used.add(tieline[0])
        if len(phases_entered - phases_used) > 0:
            raise DatasetError('Phases entered {} do not match phases used {}.'.format(phases_entered, phases_used))

    # check that all of the components used match the components entered
    components_entered = set(components)
    components_used = set()
    if is_single_phase:
        for config in sublattice_configurations:
            for sl in config:
                if isinstance(sl, list):
                    components_used.update(set(sl))
                else:
                    components_used.add(sl)
        comp_dof = 0
    else:
        for zpf in values:
            for tieline in zpf:
                tieline_comps = set(tieline[1])
                components_used.update(tieline_comps)
                if len(components_entered - tieline_comps - {'VA'}) != 1:
                    raise DatasetError('Degree of freedom error for entered components {} in tieline {} of ZPF {}'.format(components_entered, tieline, zpf))
        # handle special case of mass balance in ZPFs
        comp_dof = 1
    if len(components_entered - components_used - {'VA'}) > comp_dof or len(components_used - components_entered) > 0:
        raise DatasetError('Components entered {} do not match components used {}.'.format(components_entered, components_used))

    # check that the ZPF values are formatted properly
    if not is_single_phase:
        for zpf in values:
            for tieline in zpf:
                phase = tieline[0]
                component_list = tieline[1]
                mole_fraction_list = tieline[2]
                # check that the phase is a string, components a list of strings,
                #  and the fractions are a list of float
                if not isinstance(phase, string_types):
                    raise DatasetError('The first element in the tieline {} for the ZPF point {} should be a string. Instead it is a {} of value {}'.format(tieline, zpf, type(phase), phase))
                if not all([isinstance(comp, string_types) for comp in component_list]):
                    raise DatasetError('The second element in the tieline {} for the ZPF point {} should be a list of strings. Instead it is a {} of value {}'.format(tieline, zpf, type(component_list), component_list))
                if not all([(isinstance(mole_frac, (int, float)) or mole_frac is None)  for mole_frac in mole_fraction_list]):
                    raise DatasetError('The last element in the tieline {} for the ZPF point {} should be a list of numbers. Instead it is a {} of value {}'.format(tieline, zpf, type(mole_fraction_list), mole_fraction_list))
                # check that the shape of components list and mole fractions list is the same
                if len(component_list) != len(mole_fraction_list):
                    raise DatasetError('The length of the components list and mole fractions list in tieline {} for the ZPF point {} should be the same.'.format(tieline, zpf))

    # check that the site ratios are valid as well as site occupancies, if applicable
    if is_single_phase:
        for configuration, occupancy in zip(sublattice_configurations, sublattice_occupancies):
            if len(configuration) != len(sublattice_site_ratios):
                raise DatasetError('Sublattice configuration {} and sublattice site ratio {} describe different numbers of sublattices ({} and {}).'.format(configuration, sublattice_site_ratios, len(configuration), len(sublattice_site_ratios)))
            if is_mixing:
                configuration_shape = tuple(len(sl) if isinstance(sl, list) else 1 for sl in configuration)
                occupancy_shape = tuple(len(sl) if isinstance(sl, list) else 1 for sl in occupancy)
                if configuration_shape != occupancy_shape:
                    raise DatasetError('The shape of sublattice configuration {} ({}) does not match the shape of occupancies {} ({})'.format(configuration, configuration_shape, occupancy, occupancy_shape))


def load_datasets(dataset_filenames):
    """Create a PickelableTinyDB with the data from a list of filenames.

    Parameters
    ----------
    dataset_filenames : [str]
        List of filenames to load as datasets

    Returns
    -------
    PickleableTinyDB
    """
    ds_database = PickleableTinyDB(storage=MemoryStorage)
    for fname in dataset_filenames:
        with open(fname) as file_:
            try:
                d = json.load(file_)
                check_dataset(d)
                ds_database.insert(d)
            except ValueError as e:
                raise ValueError('JSON Error in {}: {}'.format(fname, e))
            except DatasetError as e:
                raise DatasetError('Dataset Error in {}: {}'.format(fname, e))
    return ds_database


def recursive_glob(start, pattern):
    """Recursively glob for the given pattern from the start directory.

    Parameters
    ----------
    start : str
        Path of the directory to walk while for file globbing
    pattern : str
        Filename pattern to match in the glob

    Returns
    -------
    [str]
        List of matched filenames

    """
    matches = []
    for root, dirnames, filenames in os.walk(start):
        for filename in fnmatch.filter(filenames, pattern):
            matches.append(os.path.join(root, filename))
    return sorted(matches)
