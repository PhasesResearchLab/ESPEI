"""Test ESPEI's database utilities

Tests in here are heavily parameterized and represent a large fraction of the
number of tests, but a small amount of coverage.
"""

from pycalphad import Database, variables as v
import pytest
import symengine
from symengine import And, Piecewise, Symbol

import espei.refdata
from espei.database_utils import initialize_database, _get_ser_data

from .testing_data import SGTE91_PURE_ELEMENTS


@pytest.mark.parametrize("element_name", SGTE91_PURE_ELEMENTS)
def test_get_ser_data_SGTE91(element_name):
    """Test that all SGTE91 elements can be read from the primary SGTE91 dataset without error"""
    # Make a fake fallback dataset so we can confirm that it's pulling from the primary
    FAKE_FALLBACK = "FAKE_FALLBACK_"
    setattr(espei.refdata, FAKE_FALLBACK + "SER", {})
    data = _get_ser_data(element_name, "SGTE91", fallback_ref_state=FAKE_FALLBACK)
    assert len(data) > 0
    assert isinstance(data['phase'], str)
    assert isinstance(data['H298'], float)
    assert isinstance(data['S298'], float)
    assert isinstance(data['mass'], float)
    delattr(espei.refdata, FAKE_FALLBACK + "SER")


@pytest.mark.parametrize("element_name", SGTE91_PURE_ELEMENTS)
def test_get_ser_data_falls_back_on_SGTE91(element_name):
    """Test that a reference dataset with no SER data falls back on SGTE91"""
    data = _get_ser_data(element_name, "FAKE_REF_STATE")
    assert len(data) > 0
    assert isinstance(data['phase'], str)
    assert isinstance(data['H298'], float)
    assert isinstance(data['S298'], float)
    assert isinstance(data['mass'], float)


def test_get_ser_data_is_successful_without_refdata():
    """Test that an element not in reference data or fallback data returns an empty dict"""
    assert _get_ser_data("FAKE ELEMENT", "SGTE91") == {}
    assert _get_ser_data("FAKE ELEMENT", "FAKE REF DATA") == {}


def test_database_initialization_custom_refstate():
    """Test that a custom reference state with ficticious pure elements can be used to construct a Database"""
    refdata_stable = {
        "Q": Piecewise((symengine.oo, True)),
        "ZX": Piecewise((symengine.oo, True)),
    }
    refdata = {
        ("Q", "ALPHA"): Symbol("GHSERQQ"),
        ("Q", "BETA"): Symbol("GHSERQQ") + 10000.0,
        ("ZX", "BETA"): Symbol("GHSERZX"),
    }
    refdata_ser = {
        'Q': {'phase': 'ALPHA', 'mass': 8.0, 'H298': 80.0, 'S298': 0.80},
        'ZX': {'phase': 'BETA', 'mass': 52.0, 'H298': 520.0, 'S298': 5.20},
    }

    # Setup refdata
    CUSTOM_REFDATA_NAME = "CUSTOM"
    setattr(espei.refdata, CUSTOM_REFDATA_NAME + "Stable", refdata_stable)
    setattr(espei.refdata, CUSTOM_REFDATA_NAME, refdata)
    setattr(espei.refdata, CUSTOM_REFDATA_NAME + "SER", refdata_ser)

    # Test
    phase_models = {
        "components": ["Q", "ZX"],
        "phases": {
            "ALPHA": {
                "sublattice_model": [["Q"]],
                "sublattice_site_ratios": [1],
            },
            "BCC": {
                "aliases": ["BETA"],
                "sublattice_model": [["Q", "ZX"]],
                "sublattice_site_ratios": [1.0],
            },
        }
    }
    dbf = initialize_database(phase_models, CUSTOM_REFDATA_NAME)
    assert set(dbf.phases.keys()) == {"ALPHA", "BCC"}
    assert dbf.elements == {"Q", "ZX"}
    assert dbf.species == {v.Species("Q"), v.Species("ZX")}
    assert 'GHSERQQ' in dbf.symbols
    assert 'GHSERZX' in dbf.symbols
    assert dbf.refstates["Q"]["phase"] == "ALPHA"
    assert dbf.refstates["ZX"]["phase"] == "BCC"

    # Teardown refdata
    delattr(espei.refdata, CUSTOM_REFDATA_NAME + "Stable")
    delattr(espei.refdata, CUSTOM_REFDATA_NAME)
    delattr(espei.refdata, CUSTOM_REFDATA_NAME + "SER")


def test_database_initialization_adds_GHSER_data():
    phase_models = {
        "components": ["CR", "NI"],
        "phases": {
            "FCC_A1": {
                "sublattice_model": [["CR", "NI"]],
                "sublattice_site_ratios": [1],
            },
            "BCC": {
                "aliases": ["BCC_A2"],
                "sublattice_model": [["CR", "NI"]],
                "sublattice_site_ratios": [1.0],
            },
        }
    }
    dbf = initialize_database(phase_models, "SGTE91")
    assert dbf.symbols["GHSERCR"] != symengine.S.Zero
    assert dbf.symbols["GHSERNI"] != symengine.S.Zero


CU_O_SPECIES_PHASE_MODELS = {
    "components": ["CU", "O"],
    "species": {
        "CU2O": {"constituents": {"CU": 2, "O": 1}},
        "CU+1": {"constituents": {"CU": 1}, "charge": 1},
    },
    "phases": {
        "LIQUID": {"sublattice_model": [["CU", "CU+1", "CU2O", "O", "S"]], "sublattice_site_ratios": [1]},
    }
}


def test_database_initialization_adds_species():
    """Species in the phase models are added to the Database and the phase constituents"""
    dbf = initialize_database(CU_O_SPECIES_PHASE_MODELS, "SGTE91")
    species = {sp.name: sp for sp in dbf.species}
    assert (species["CU2O"].constituents, species["CU2O"].charge) == ({"CU": 2, "O": 1}, 0)
    assert (species["CU+1"].constituents, species["CU+1"].charge) == ({"CU": 1}, 1)
    assert [sorted(sp.name for sp in subl) for subl in dbf.phases["LIQUID"].constituents] == [["CU", "CU+1", "CU2O", "O"]]


CU_O_TDB = "ELEMENT CU FCC_A1 63.546 5004.1 33.15 !\nELEMENT O 1/2_MOLE_O2(G) 15.999 4341.0 102.52 !\n"


@pytest.mark.parametrize("components, species, tdb, message", [
    (["CU", "O"], {"CU2S": {"constituents": {"CU": 2, "S": 1}}}, None, "must be elements"),
    (["CU", "O"], {"CU": {"constituents": {"CU": 1}}}, None, "name of an element"),
    (["CU", "CU2O", "O"], {"CU2O": {"constituents": {"CU": 2, "O": 1}}}, None, "Component CU2O is a species"),
    (["CU", "CU2O", "O"], {}, CU_O_TDB + "SPECIES CU2O CU2O1 !", "Component CU2O is a species in the Database"),
    (["CU", "O"], {"CU2O": {"constituents": {"CU": 1, "O": 1}}}, CU_O_TDB + "SPECIES CU2O CU2O1 !", "in the Database"),
    (["CU", "H"], {}, None, "H has no reference data"),
])
def test_database_initialization_raises_for_invalid_species(components, species, tdb, message):
    """Invalid species and components raise"""
    phase_models = {"components": components, "species": species, "phases": {}}
    dbf = Database.from_string(tdb, fmt="tdb") if tdb is not None else None
    with pytest.raises(ValueError, match=message):
        initialize_database(phase_models, "SGTE91", dbf=dbf)


def test_database_initialization_uses_elements_from_input_database():
    """An element without reference data can be defined by the input Database, and VA and /- get no GHSER functions"""
    dbf = Database.from_string("ELEMENT /- ELECTRON_GAS 0 0 0 !\nELEMENT VA VACUUM 0 0 0 !\nELEMENT H 1/2_MOLE_H2(G) 1.0079 4234.0 65.285 !\nFUNCTION GHSERHH 298.15 -1000; 6000 N !", fmt="tdb")
    dbf = initialize_database({"components": ["H", "VA"], "phases": {}}, "SGTE91", dbf=dbf)
    assert dbf.symbols["GHSERHH"] == Piecewise((-1000.0, And(298.15 <= v.T, v.T < 6000.0)), (0, True))
    assert "GHSERVA" not in dbf.symbols
    assert "GHSER/-" not in dbf.symbols
