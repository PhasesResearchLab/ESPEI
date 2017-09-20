"""
Test espei.utils classes and functions.
"""
import pickle

from tinydb import where
from espei.utils import ImmediateClient, PickleableTinyDB, MemoryStorage

def test_immediate_client_returns_map_results_directly():
    """Calls ImmediateClient.map should return the results, instead of Futures."""
    from distributed import LocalCluster
    cli = ImmediateClient(LocalCluster(n_workers=1))
    num_list = range(0, 11)
    square = lambda x: x**2
    map_result = cli.map(square, num_list)
    assert map_result == [square(x) for x in num_list]


def test_pickelable_tinydb_can_be_pickled_and_unpickled():
    test_dict = {'test_key': ['test', 'values']}
    db = PickleableTinyDB(storage=MemoryStorage)
    db.insert(test_dict)
    db = pickle.loads(pickle.dumps(db))
    assert db.search(where('test_key').exists())[0] == test_dict
