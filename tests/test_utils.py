"""
Test espei.utils classes and functions.
"""
import pickle

import pytest
import symengine
from pycalphad import Database, variables as v
from tinydb import where
import espei
from espei.utils import PickleableTinyDB, MemoryStorage, \
    bib_marker_map, docs_url, extract_aliases, formatted_parameter, parameter_term

from .fixtures import datasets_db, tmp_file
from .testing_data import CU_MG_TDB


def test_immediate_client_returns_map_results_directly():
    """Calls ImmediateClient.map should return the results, instead of Futures."""
    pytest.importorskip("distributed")
    from distributed import LocalCluster
    from espei.dask_utils import ImmediateClient
    cli = ImmediateClient(LocalCluster(n_workers=1))
    num_list = range(0, 11)
#    square = lambda x: x**2
    def square(x):
      return x**2
    map_result = cli.map(square, num_list)
    assert map_result == [square(x) for x in num_list]


def test_pickelable_tinydb_can_be_pickled_and_unpickled():
    """PickleableTinyDB should be able to be pickled and unpickled."""
    test_dict = {'test_key': ['test', 'values']}
    db = PickleableTinyDB(storage=MemoryStorage)
    db.insert(test_dict)
    db = pickle.loads(pickle.dumps(db))
    assert db.search(where('test_key').exists())[0] == test_dict


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


@pytest.mark.parametrize('reason, phase_models, expected_aliases', [
    (
        "No phases should give no aliases",
        {"phases": {}},
        {}
    ),
    (
        "A phase has an alias for itself and works without given aliases",
        {"phases": {"ALPHA": {}}},
        {"ALPHA": "ALPHA"}
    ),
    (
        "Empty aliases list works",
        {"phases": {"ALPHA": {"aliases": []}}},
        {"ALPHA": "ALPHA"}
    ),
    (
        "Basic test for adding aliases correctly",
        {"phases": {
            "ALPHA": {"aliases": ["FCC_A1"]}
        }},
        {"ALPHA": "ALPHA", "FCC_A1": "ALPHA"}
    ),
    (
        "A phase can have mulitple aliases",
        {"phases": {
            "ALPHA": {"aliases": ["FCC_A1", "A1", "FCC"]}
        }},
        {"ALPHA": "ALPHA", "FCC_A1": "ALPHA", "FCC": "ALPHA", "A1": "ALPHA"}
    ),
    (
        "Cannot have two phases with the same alias",
        {"phases": {
            "ALPHA": {"aliases": ["FCC_A1"]},
            "GAMMA": {"aliases": ["FCC_A1"]},
        }},
        None
    ),
    (
        "Cannot have a prescribed phase as an alias",
        {"phases": {
            "ALPHA": {"aliases": ["BETA"]},
            "BETA": {"aliases": []},
        }},
        None
    ),
]
)
def test_extract_aliases(reason, phase_models, expected_aliases):
    if expected_aliases is None:
        with pytest.raises(ValueError):
            aliases = extract_aliases(phase_models)
            print(aliases)
    else:
        assert extract_aliases(phase_models) == expected_aliases, reason


@pytest.mark.parametrize("version, url", [("0.10.0", "https://espei.org/0.10.0/index.html"), ("0.10.1.dev3+gc184894", "https://espei.org/dev/index.html")])
def test_docs_url_of_releases_and_development_versions(monkeypatch, version, url):
    """Documentation links go to the docs of a release, or to dev for other versions."""
    monkeypatch.setattr(espei, "__version__", version)
    assert docs_url("index.html") == url


def test_formatted_parameter():
    """Symbols in Cu-Mg parameters format to their phase, interaction and type."""
    dbf = Database(CU_MG_TDB)
    G = formatted_parameter(dbf, 'VV0009')
    assert (G.phase_name, G.interaction, G.parameter_type, G.term) == ('LAVES_C15', 'MG:CU', 'G', 1)
    L = formatted_parameter(dbf, 'VV0001')
    assert (L.phase_name, L.interaction, L.parameter_type, L.term) == ('FCC_A1', 'CU,MG:VA', 'L2', 1)


def test_parameter_term():
    """Temperature terms multiplying a symbol are found; nonlinear or missing symbols raise."""
    VV0000, VV0001 = symengine.Symbol('VV0000'), symengine.Symbol('VV0001')
    expr = symengine.Piecewise((VV0000 + VV0001*v.T*symengine.log(v.T), symengine.And(1.0 <= v.T, v.T < 10000.0)), (0, True))
    assert parameter_term(expr, VV0001) == v.T*symengine.log(v.T)
    with pytest.raises(ValueError):
        parameter_term(VV0000**2, VV0000)
    with pytest.raises(ValueError):
        parameter_term(VV0000, VV0001)
