"""
Utilities for ESPEI

Classes and functions defined here should have some reuse potential.
"""

import numpy as np
from distributed import Client
from tinydb import TinyDB
from tinydb.storages import MemoryStorage


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


def sigfigs(x, n):
    """Round x to n significant digits"""
    if x != 0:
        return np.around(x, -(np.floor(np.log10(np.abs(x)))).astype(np.int) + (n - 1))
    else:
        return x
