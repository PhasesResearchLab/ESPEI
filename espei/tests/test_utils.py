"""
Test espei.utils classes and functions.
"""
import pickle, time, os

from tinydb import where
from espei.utils import ImmediateClient, PickleableTinyDB, MemoryStorage, flexible_open_string

import pytest

MULTILINE_HIPSTER_IPSUM = """Lorem ipsum dolor amet wayfarers kale chips chillwave
adaptogen schlitz lo-fi jianbing ennui occupy pabst health goth chicharrones.
Glossier enamel pin pitchfork PBR&B ennui. Actually small batch marfa edison
bulb poutine, chicharrones neutra swag farm-to-table lyft meggings mixtape
pork belly. DIY iceland schlitz YOLO, four loko pok pok single-origin coffee
normcore. Shabby chic helvetica mustache taxidermy tattooed kombucha cliche
gastropub gentrify ramps hexagon waistcoat authentic snackwave."""


@pytest.fixture
def tmp_file():
    """Create a temporary file and return the file name"""
    fname = 'tmp_file-' + str(time.time()).split('.')[0]
    with open(fname, 'w') as fp:
        fp.write(MULTILINE_HIPSTER_IPSUM)
    yield fname
    os.remove(fname)


def test_immediate_client_returns_map_results_directly():
    """Calls ImmediateClient.map should return the results, instead of Futures."""
    from distributed import LocalCluster
    cli = ImmediateClient(LocalCluster(n_workers=1))
    num_list = range(0, 11)
    square = lambda x: x**2
    map_result = cli.map(square, num_list)
    assert map_result == [square(x) for x in num_list]


def test_pickelable_tinydb_can_be_pickled_and_unpickled():
    """PickleableTinyDB should be able to be pickled and unpickled."""
    test_dict = {'test_key': ['test', 'values']}
    db = PickleableTinyDB(storage=MemoryStorage)
    db.insert(test_dict)
    db = pickle.loads(pickle.dumps(db))
    assert db.search(where('test_key').exists())[0] == test_dict


def test_flexible_open_string_raw_string():
    """Raw multiline strings should be directly returned by flexible_open_string."""
    returned_string = flexible_open_string(MULTILINE_HIPSTER_IPSUM)
    assert returned_string == MULTILINE_HIPSTER_IPSUM


def test_flexible_open_string_file_like(tmp_file):
    """File-like objects support read methods should have their content returned by flexible_open_string."""
    fname = tmp_file
    with open(fname) as fp:
        returned_string = flexible_open_string(fp)
    assert returned_string == MULTILINE_HIPSTER_IPSUM


def test_flexible_open_string_path_like(tmp_file):
    """Path-like strings should be opened, read and returned"""
    fname = tmp_file
    returned_string = flexible_open_string(fname)
    assert returned_string == MULTILINE_HIPSTER_IPSUM
