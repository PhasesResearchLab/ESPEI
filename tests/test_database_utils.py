"""Test ESPEI's database utilities

Tests in here are heavily parameterized and represent a large fraction of the
number of tests, but a small amount of coverage.
"""

import pytest

import espei.refdata
from espei.database_utils import initialize_database, _get_ser_data

from .testing_data import SGTE91_PURE_ELEMENTS


@pytest.mark.parametrize("element_name", SGTE91_PURE_ELEMENTS)
def test_get_ser_data_SGTE91(element_name):
    """Test that all SGTE91 elements can be read from the primary SGTE91 dataset without error"""
    # Make a fake fallback dataset so we can confirm that it's pulling from the primary
    FAKE_FALLBACK = "FAKE_FALLBACK_"
    setattr(espei.refdata, FAKE_FALLBACK + "SER", {})
    _get_ser_data(element_name, "SGTE91", fallback_ref_state=FAKE_FALLBACK)
    delattr(espei.refdata, FAKE_FALLBACK + "SER")


@pytest.mark.parametrize("element_name", SGTE91_PURE_ELEMENTS)
def test_get_ser_data_falls_back_on_SGTE91(element_name):
    """Test that a reference dataset with no SER data falls back on SGTE91"""
    _get_ser_data(element_name, "FAKE_REF_STATE")


def test_get_ser_data_is_successful_without_refdata():
    """Test that an element not in reference data or fallback data returns an empty dict"""
    assert _get_ser_data("FAKE ELEMENT", "SGTE91") == {}
    assert _get_ser_data("FAKE ELEMENT", "FAKE REF DATA") == {}
