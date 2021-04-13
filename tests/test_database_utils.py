"""Test ESPEI's database utilities

Tests in here are heavily parameterized and represent a large fraction of the
number of tests, but a small amount of coverage.
"""

from pycalphad import variables as v
import pytest
import sympy
from sympy import Piecewise, Symbol

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
        "Q": Piecewise((sympy.oo, True)),
        "ZX": Piecewise((sympy.oo, True)),
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
    assert dbf.symbols["GHSERCR"] != sympy.S.Zero
    assert dbf.symbols["GHSERNI"] != sympy.S.Zero
