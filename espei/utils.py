"""
Utilities for ESPEI

Classes and functions defined here should have some reuse potential.
"""

import json

import fnmatch
import numpy as np
import os
from distributed import Client
from tinydb import TinyDB
from tinydb.storages import MemoryStorage


class PickleableTinyDB(TinyDB):
    """
    A pickleable version of TinyDB that uses MemoryStorage as a default.
    """
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


def load_datasets(dataset_filenames):
    """
    Create a PickelableTinyDB with the data from a list of filenames.

    Args:
        dataset_filenames ([str]):

    Returns:
        PickleableTinyDB
    """
    ds_database = PickleableTinyDB(storage=MemoryStorage)
    for fname in dataset_filenames:
        with open(fname) as file_:
            try:
                ds_database.insert(json.load(file_))
            except ValueError as e:
                print('JSON Error in {}: {}'.format(fname, e))
    return ds_database

def recursive_glob(start, pattern):
    """
    Recursively glob for the given pattern from the start directory.

    Args:
        start (str): Path of the directory to walk while for file globbing
        pattern (str): Filename pattern to match in the glob

    Returns:
        [str]: List of matched filenames
    """
    matches = []
    for root, dirnames, filenames in os.walk(start):
        for filename in fnmatch.filter(filenames, pattern):
            matches.append(os.path.join(root, filename))
    return sorted(matches)


def sigfigs(x, n):
    """
    Round x to n significant digits
    """
    if x != 0:
        return np.around(x, -(np.floor(np.log10(np.abs(x)))).astype(np.int) + (n - 1))
    else:
        return x
