"""
Test different error functions as isolated units.
"""

import numpy as np
from tinydb import where

from pycalphad import Database

from espei.paramselect import generate_parameters
from espei.error_functions import calculate_activity_error, calculate_thermochemical_error, calculate_zpf_error, get_thermochemical_data, get_zpf_data

from .fixtures import datasets_db
from .testing_data import *


def test_activity_error(datasets_db):
    """Test that activity error returns a correct result"""

    datasets_db.insert(CU_MG_EXP_ACTIVITY)

    dbf = Database(CU_MG_TDB)
    error = calculate_activity_error(dbf, ['CU','MG','VA'], list(dbf.phases.keys()), datasets_db, {}, {}, {})
    assert np.isclose(error, -257.41020886970756, rtol=1e-6)


def test_thermochemical_error_with_multiple_X_points(datasets_db):
    """Multiple composition datapoints in a dataset for a mixing phase should be successful."""
    datasets_db.insert(CU_MG_CPM_MIX_X_HCP_A3)

    dbf = Database(CU_MG_TDB)
    comps = ['CU','MG','VA']
    thermochemical_data = get_thermochemical_data(dbf, comps, )
    error = calculate_thermochemical_error(dbf, comps, thermochemical_data)

    assert np.isclose(error, -4061.119001241541, rtol=1e-6)


def test_thermochemical_error_with_multiple_T_points(datasets_db):
    """Multiple temperature datapoints in a dataset for a stoichiometric comnpound should be successful."""
    datasets_db.insert(CU_MG_HM_MIX_T_CUMG2)

    dbf = Database(CU_MG_TDB)
    error = calculate_thermochemical_error(dbf, ['CU','MG','VA'], list(dbf.phases.keys()), datasets_db)
    assert np.isclose(error,-14.287293263253728, rtol=1e-6)


def test_thermochemical_error_with_multiple_T_X_points(datasets_db):
    """Multiple temperature and composition datapoints in a dataset for a mixing phase should be successful."""
    datasets_db.insert(CU_MG_SM_MIX_T_X_FCC_A1)

    dbf = Database(CU_MG_TDB)
    error = calculate_thermochemical_error(dbf, ['CU', 'MG', 'VA'], list(dbf.phases.keys()), datasets_db)
    assert np.isclose(float(error), -3282497.2380024833, rtol=1e-6)

def test_thermochemical_error_for_mixing_entropy_error_is_excess_only(datasets_db):
    """Tests that error in mixing entropy data is excess only (the ideal part is removed)."""
    # If this fails, make sure the ideal mixing contribution is removed.
    phase_models = {
        "components": ["AL", "B"],
        "phases": {
            "LIQUID" : {
                "sublattice_model": [["AL", "B"]],
                "sublattice_site_ratios": [1]
            },
            "FCC_A1" : {
                "sublattice_model": [["AL", "B"]],
                "sublattice_site_ratios": [1]
            }
        }
    }

    dataset_excess_mixing = {
        "components": ["AL", "B"],
        "phases": ["FCC_A1"],
        "solver": {
            "sublattice_site_ratios": [1],
            "sublattice_occupancies": [[[0.5, 0.5]]],
            "sublattice_configurations": [[["AL", "B"]]],
            "mode": "manual"
        },
        "conditions": {
            "P": 101325,
            "T": 298.15
        },
        "output": "SM_MIX",
        "values": [[[10]]]
    }
    datasets_db.insert(dataset_excess_mixing)

    dbf = generate_parameters(phase_models, datasets_db, 'SGTE91', 'linear')
    assert dbf.elements == {'AL', 'B'}
    assert set(dbf.phases.keys()) == {'LIQUID', 'FCC_A1'}
    assert len(dbf._parameters.search(where('parameter_type') == 'L')) == 1

    # the error should be exactly 0 because we are only fitting to one point
    # the dataset is excess only
    from espei.error_functions import calculate_thermochemical_error
    assert np.isclose(calculate_thermochemical_error(dbf, sorted(dbf.elements), sorted(dbf.phases.keys()), datasets_db), 0.6904993792294276, atol=1e-6)


def test_thermochemical_error_for_of_enthalpy_mixing(datasets_db):
    """Tests that error in mixing entropy data is excess only (the ideal part is removed)."""
    # If this fails, make sure the ideal mixing contribution is removed.
    phase_models = {
        "components": ["AL", "B"],
        "phases": {
            "LIQUID" : {
                "sublattice_model": [["AL", "B"]],
                "sublattice_site_ratios": [1]
            },
            "FCC_A1" : {
                "sublattice_model": [["AL", "B"]],
                "sublattice_site_ratios": [1]
            }
        }
    }

    dataset_excess_mixing = {
        "components": ["AL", "B"],
        "phases": ["FCC_A1"],
        "solver": {
            "sublattice_site_ratios": [1],
            "sublattice_occupancies": [[[0.5, 0.5]]],
            "sublattice_configurations": [[["AL", "B"]]],
            "mode": "manual"
        },
        "conditions": {
            "P": 101325,
            "T": 298.15
        },
        "output": "HM_MIX",
        "values": [[[10000]]]
    }
    datasets_db.insert(dataset_excess_mixing)

    dbf = generate_parameters(phase_models, datasets_db, 'SGTE91', 'linear')
    assert dbf.elements == {'AL', 'B'}
    assert set(dbf.phases.keys()) == {'LIQUID', 'FCC_A1'}
    assert len(dbf._parameters.search(where('parameter_type') == 'L')) == 1

    # the error should be exactly 0 because we are only fitting to one point
    # the dataset is excess only
    from espei.error_functions import calculate_thermochemical_error
    assert np.isclose(calculate_thermochemical_error(dbf, sorted(dbf.elements), sorted(dbf.phases.keys()), datasets_db), -7.133546631626864, rtol=1e-6)


def test_zpf_error_zero(datasets_db):
    """Test that sum of square ZPF errors returns 0 for an exactly correct result"""

    datasets_db.insert(CU_MG_DATASET_ZPF_ZERO_ERROR)

    dbf = Database(CU_MG_TDB)

    error = calculate_zpf_error(dbf, ['CU','MG','VA'], list(dbf.phases.keys()), datasets_db, {}, {}, {})
    assert np.isclose(error, -15.653387624373618, rtol=1e-6)
