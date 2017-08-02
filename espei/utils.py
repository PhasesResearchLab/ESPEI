"""
Utilities for ESPEI

Classes and functions defined here should have some reuse potential.
"""

import json

import fnmatch
import os
import numpy as np
from distributed import Client
from tinydb import TinyDB
from tinydb.storages import MemoryStorage

class DatasetError(Exception):
    """Exception raised when datasets are invalid."""
    pass

class PickleableTinyDB(TinyDB):
    """A pickleable version of TinyDB that uses MemoryStorage as a default."""
    def __getstate__(self):
        # first remove the query cache. The cache speed is not important to us.
        for table_name in self.tables():
            self.table(table_name)._query_cache = {}
        pickle_dict = {}
        for key, value in self.__dict__.items():
            if key == '_table':
                pickle_dict[key] = value.all()
            else:
                pickle_dict[key] = value
        return pickle_dict

    def __setstate__(self, state):
        self.__init__(storage=MemoryStorage)
        self.insert_multiple(state['_table'])


class ImmediateClient(Client):
    """
    A subclass of distributed.Client that automatically unwraps the Futures
    returned by map.
    """
    def map (self, *args, **kwargs):
        result = super(ImmediateClient, self).map(*args, **kwargs)
        result = [x.result() for x in result]
        return result


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
                components_used.update(set(tieline[1]))
        # handle special case of mass balance in ZPFs
        comp_dof = 1
    if len(components_entered - components_used) != comp_dof or len(components_used - components_entered) > 0:
        raise DatasetError('Components entered {} do not match components used.'.format(components_entered, components_used))


def load_datasets(dataset_filenames):
    """Create a PickelableTinyDB with the data from a list of filenames.

    Parameters
    ----------
    dataset_filenames : [str]
        List of filenames to load as datasets

    Returns:
    --------
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


def sigfigs(x, n):
    """Round x to n significant digits"""
    if x != 0:
        return np.around(x, -(np.floor(np.log10(np.abs(x)))).astype(np.int) + (n - 1))
    else:
        return x
