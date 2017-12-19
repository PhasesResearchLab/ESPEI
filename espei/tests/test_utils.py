"""
Test espei.utils classes and functions.
"""
import pickle

from tinydb import where
from espei.utils import ImmediateClient, PickleableTinyDB, MemoryStorage, \
    flexible_open_string, add_bibtex_to_bib_database, bib_marker_map

import pytest
from espei.tests.fixtures import datasets_db, tmp_file

MULTILINE_HIPSTER_IPSUM = """Lorem ipsum dolor amet wayfarers kale chips chillwave
adaptogen schlitz lo-fi jianbing ennui occupy pabst health goth chicharrones.
Glossier enamel pin pitchfork PBR&B ennui. Actually small batch marfa edison
bulb poutine, chicharrones neutra swag farm-to-table lyft meggings mixtape
pork belly. DIY iceland schlitz YOLO, four loko pok pok single-origin coffee
normcore. Shabby chic helvetica mustache taxidermy tattooed kombucha cliche
gastropub gentrify ramps hexagon waistcoat authentic snackwave."""


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
    fname = tmp_file(MULTILINE_HIPSTER_IPSUM)
    with open(fname) as fp:
        returned_string = flexible_open_string(fp)
    assert returned_string == MULTILINE_HIPSTER_IPSUM


def test_flexible_open_string_path_like(tmp_file):
    """Path-like strings should be opened, read and returned"""
    fname = tmp_file(MULTILINE_HIPSTER_IPSUM)
    returned_string = flexible_open_string(fname)
    assert returned_string == MULTILINE_HIPSTER_IPSUM


def test_adding_bibtex_entries_to_bibliography_db(datasets_db):
    """Adding a BibTeX entries to a database works and the database can be searched."""
    TEST_BIBTEX = """@article{Roe1952gamma,
author = {Roe, W. P. and Fishel, W. P.},
journal = {Trans. Am. Soc. Met.},
keywords = {Fe-Cr,Fe-Ti,Fe-Ti-Cr},
pages = {1030--1041},
title = {{Gamma Loop Studies in the Fe-Ti, Fe-Cr, and Fe-Ti-Cr Systems}},
volume = {44},
year = {1952}
}

@phdthesis{shin2007thesis,
author = {Shin, D},
keywords = {Al-Cu,Al-Cu-Mg,Al-Cu-Si,Al-Mg,Al-Mg-Si,Al-Si,Cu-Mg,Mg-Si,SQS},
number = {May},
school = {The Pennsylvania State University},
title = {{Thermodynamic properties of solid solutions from special quasirandom structures and CALPHAD modeling: Application to aluminum-copper-magnesium-silicon and hafnium-silicon-oxygen}},
year = {2007}
}"""
    db = add_bibtex_to_bib_database(TEST_BIBTEX, datasets_db)
    search_res = db.search(where('ID') == 'Roe1952gamma')
    assert len(search_res) == 1
    assert len(db.all()) == 2

def test_bib_marker_map():
    """bib_marker_map should return a proper dict"""
    marker_dict = bib_marker_map(['otis2016', 'bocklund2018'])
    EXEMPLAR_DICT = {
        'bocklund2018': {
            'formatted': 'bocklund2018',
            'markers': {'fillstyle': 'none', 'marker': 'o'}
        },
        'otis2016': {
            'formatted': 'otis2016',
            'markers': {'fillstyle': 'none', 'marker': 'v'}
        }
    }
    assert EXEMPLAR_DICT == marker_dict
