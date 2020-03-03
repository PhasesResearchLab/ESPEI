"""
Test different error functions as isolated units.
"""

import numpy as np
from tinydb import where

from pycalphad import Database

from espei.paramselect import generate_parameters
from espei.error_functions import calculate_activity_error, calculate_non_equilibrium_thermochemical_probability, calculate_zpf_error, get_thermochemical_data, get_zpf_data
import scipy.stats

from .fixtures import datasets_db
from .testing_data import *


def test_activity_error(datasets_db):
    """Test that activity error returns a correct result"""

    datasets_db.insert(CU_MG_EXP_ACTIVITY)

    dbf = Database(CU_MG_TDB)
    error = calculate_activity_error(dbf, ['CU','MG','VA'], list(dbf.phases.keys()), datasets_db, {}, {}, {})
    assert np.isclose(error, -257.41020886970756, rtol=1e-6)


def test_subsystem_activity_probability(datasets_db):
    """Test binary Cr-Ni data produces the same probability regardless of whether the main system is a binary or ternary."""

    datasets_db.insert(CR_NI_ACTIVITY)

    dbf_bin = Database(CR_NI_TDB)
    dbf_tern = Database(CR_FE_NI_TDB)
    phases = list(dbf_bin.phases.keys())

    # Truth
    bin_prob = calculate_activity_error(dbf_bin, ['CR','NI','VA'], phases, datasets_db, {}, {}, {})

    # Getting binary subsystem data explictly (from binary input)
    prob = calculate_activity_error(dbf_tern, ['CR','NI','VA'], phases, datasets_db, {}, {}, {})
    assert np.isclose(prob, bin_prob)

    # Getting binary subsystem from ternary input
    prob = calculate_activity_error(dbf_tern, ['CR', 'FE', 'NI','VA'], phases, datasets_db, {}, {}, {})
    assert np.isclose(prob, bin_prob)


def test_non_equilibrium_thermochemical_error_with_multiple_X_points(datasets_db):
    """Multiple composition datapoints in a dataset for a mixing phase should be successful."""
    datasets_db.insert(CU_MG_CPM_MIX_X_HCP_A3)

    dbf = Database(CU_MG_TDB)
    phases = list(dbf.phases.keys())
    comps = ['CU', 'MG', 'VA']
    thermochemical_data = get_thermochemical_data(dbf, comps, phases, datasets_db)
    error = calculate_non_equilibrium_thermochemical_probability(dbf, thermochemical_data)

    assert np.isclose(error, -4061.119001241541, rtol=1e-6)


def test_non_equilibrium_thermochemical_error_with_multiple_T_points(datasets_db):
    """Multiple temperature datapoints in a dataset for a stoichiometric comnpound should be successful."""
    datasets_db.insert(CU_MG_HM_MIX_T_CUMG2)

    dbf = Database(CU_MG_TDB)
    phases = list(dbf.phases.keys())
    comps = ['CU', 'MG', 'VA']
    thermochemical_data = get_thermochemical_data(dbf, comps, phases, datasets_db)
    error = calculate_non_equilibrium_thermochemical_probability(dbf, thermochemical_data)
    assert np.isclose(error,-14.287293263253728, rtol=1e-6)


def test_non_equilibrium_thermochemical_error_with_multiple_T_X_points(datasets_db):
    """Multiple temperature and composition datapoints in a dataset for a mixing phase should be successful."""
    datasets_db.insert(CU_MG_SM_MIX_T_X_FCC_A1)

    dbf = Database(CU_MG_TDB)
    phases = list(dbf.phases.keys())
    comps = ['CU', 'MG', 'VA']
    thermochemical_data = get_thermochemical_data(dbf, comps, phases, datasets_db)
    error = calculate_non_equilibrium_thermochemical_probability(dbf, thermochemical_data)
    assert np.isclose(float(error), -3282497.2380024833, rtol=1e-6)

def test_non_equilibrium_thermochemical_error_for_mixing_entropy_error_is_excess_only(datasets_db):
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
        "values": [[[10]]],
        "excluded_model_contributions": ["idmix"]
    }
    datasets_db.insert(dataset_excess_mixing)

    dbf = generate_parameters(phase_models, datasets_db, 'SGTE91', 'linear')
    assert dbf.elements == {'AL', 'B'}
    assert set(dbf.phases.keys()) == {'LIQUID', 'FCC_A1'}
    assert len(dbf._parameters.search(where('parameter_type') == 'L')) == 1
    phases = list(dbf.phases.keys())
    comps = list(dbf.elements)

    # the error should be exactly 0 because we are only fitting to one point
    # the dataset is excess only
    zero_error_prob = scipy.stats.norm(loc=0, scale=0.2).logpdf(0.0)  # SM weight = 0.2
    # Explicitly pass parameters={} to not try fitting anything
    thermochemical_data = get_thermochemical_data(dbf, comps, phases, datasets_db, symbols_to_fit=[])
    error = calculate_non_equilibrium_thermochemical_probability(dbf, thermochemical_data)
    assert np.isclose(error, zero_error_prob, atol=1e-6)


def test_non_equilibrium_thermochemical_error_for_of_enthalpy_mixing(datasets_db):
    """Tests that error in mixing enthalpy data is calculated correctly"""
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
        "values": [[[10000]]],
        "excluded_model_contributions": ["idmix"]
    }
    datasets_db.insert(dataset_excess_mixing)

    dbf = generate_parameters(phase_models, datasets_db, 'SGTE91', 'linear')
    assert dbf.elements == {'AL', 'B'}
    assert set(dbf.phases.keys()) == {'LIQUID', 'FCC_A1'}
    assert len(dbf._parameters.search(where('parameter_type') == 'L')) == 1
    phases = list(dbf.phases.keys())
    comps = list(dbf.elements)

    # the error should be exactly 0 because we are only fitting to one point
    # the dataset is excess only
    zero_error_prob = scipy.stats.norm(loc=0, scale=500.0).logpdf(0.0)  # HM weight = 500
    # Explicitly pass parameters={} to not try fitting anything
    thermochemical_data = get_thermochemical_data(dbf, comps, phases, datasets_db, symbols_to_fit=[])
    error = calculate_non_equilibrium_thermochemical_probability(dbf, thermochemical_data)
    assert np.isclose(error, zero_error_prob, atol=1e-6)


def test_subsystem_non_equilibrium_thermochemcial_probability(datasets_db):
    """Test binary Cr-Ni data produces the same probability regardless of whether the main system is a binary or ternary."""

    datasets_db.insert(CR_NI_LIQUID_DATA)

    dbf_bin = Database(CR_NI_TDB)
    dbf_tern = Database(CR_FE_NI_TDB)
    phases = list(dbf_bin.phases.keys())

    # Truth
    thermochemical_data = get_thermochemical_data(dbf_bin, ['CR', 'NI', 'VA'], phases, datasets_db)
    bin_prob = calculate_non_equilibrium_thermochemical_probability(dbf_bin, thermochemical_data)

    # Getting binary subsystem data explictly (from binary input)
    thermochemical_data = get_thermochemical_data(dbf_tern, ['CR', 'NI', 'VA'], phases, datasets_db)
    prob = calculate_non_equilibrium_thermochemical_probability(dbf_tern, thermochemical_data)
    assert np.isclose(prob, bin_prob)

    # Getting binary subsystem from ternary input
    thermochemical_data = get_thermochemical_data(dbf_tern, ['CR', 'FE', 'NI', 'VA'], phases, datasets_db)
    prob = calculate_non_equilibrium_thermochemical_probability(dbf_tern, thermochemical_data)
    assert np.isclose(prob, bin_prob)


def test_zpf_error_zero(datasets_db):
    """Test that sum of square ZPF errors returns 0 for an exactly correct result"""
    datasets_db.insert(CU_MG_DATASET_ZPF_ZERO_ERROR)

    dbf = Database(CU_MG_TDB)
    comps = ['CU','MG','VA']
    phases = list(dbf.phases.keys())

    # ZPF weight = 1 kJ and there are two points in the tieline
    zero_error_prob = 2 * scipy.stats.norm(loc=0, scale=1000.0).logpdf(0.0)

    zpf_data = get_zpf_data(dbf, comps, phases, datasets_db, {})
    error = calculate_zpf_error(zpf_data, np.array([]))
    assert np.isclose(error, zero_error_prob, rtol=1e-6)


def test_subsystem_zpf_probability(datasets_db):
    """Test binary Cr-Ni data produces the same probability regardless of whether the main system is a binary or ternary."""

    datasets_db.insert(CR_NI_ZPF_DATA)

    dbf_bin = Database(CR_NI_TDB)
    dbf_tern = Database(CR_FE_NI_TDB)
    phases = list(dbf_bin.phases.keys())

    # Truth
    zpf_data = get_zpf_data(dbf_bin, ['CR', 'NI', 'VA'], phases, datasets_db, {})
    bin_prob = calculate_zpf_error(zpf_data, np.array([]))

    # Getting binary subsystem data explictly (from binary input)
    zpf_data = get_zpf_data(dbf_tern, ['CR', 'NI', 'VA'], phases, datasets_db, {})
    prob = calculate_zpf_error(zpf_data, np.array([]))
    assert np.isclose(prob, bin_prob)

    # Getting binary subsystem from ternary input
    zpf_data = get_zpf_data(dbf_tern, ['CR', 'FE', 'NI', 'VA'], phases, datasets_db, {})
    prob = calculate_zpf_error(zpf_data, np.array([]))
    assert np.isclose(prob, bin_prob)
