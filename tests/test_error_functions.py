# pylint: disable=redefined-outer-name
"""
Test different error functions as isolated units.
"""

from unittest import mock
from espei.optimizers.opt_mcmc import EmceeOptimizer
from espei.utils import database_symbols_to_fit
import numpy as np
import pytest
import pickle
import scipy.stats
from tinydb import where

from pycalphad import Database, Model, variables as v

from espei.paramselect import generate_parameters
from espei.error_functions import *
from espei.error_functions.activity_error import ActivityResidual
from espei.error_functions.equilibrium_thermochemical_error import calc_prop_differences, EquilibriumPropertyResidual
from espei.error_functions.non_equilibrium_thermochemical_error import FixedConfigurationPropertyResidual
from espei.error_functions.zpf_error import calculate_zpf_driving_forces, ZPFResidual
from espei.error_functions.context import setup_context
from espei.utils import unpack_piecewise, ModelTestException

from .fixtures import datasets_db
from .testing_data import *


def test_activity_error(datasets_db):
    """Test that activity error returns a correct result"""

    datasets_db.insert(CU_MG_EXP_ACTIVITY)

    dbf = Database(CU_MG_TDB)
    error = calculate_activity_error(dbf, ['CU','MG','VA'], list(dbf.phases.keys()), datasets_db, {}, {}, {})
    assert np.isclose(error, -257.41020886970756, rtol=1e-6)


def test_activity_residual_function(datasets_db):
    dbf = Database(CU_MG_TDB)
    datasets_db.insert(CU_MG_EXP_ACTIVITY)

    residual_func = ActivityResidual(dbf, datasets_db, phase_models=None, symbols_to_fit=[])

    # Regression test "truth" values - got values by running
    residuals, weights = residual_func.get_residuals(np.asarray([]))
    assert len(residuals) == len(weights)
    assert np.allclose(residuals, [6522.652187085958, -1890.1414208991046, -4793.211215856485, -3018.311675280318, -1062.6724585088668, -2224.814500229084, -2256.9820026771777, -1735.8692674535414, -805.219891012428, 0.0])
    likelihood = residual_func.get_likelihood(np.asarray([]))
    assert np.isclose(likelihood, -257.41020886970756, rtol=1e-6)


def test_subsystem_activity_probability(datasets_db):
    """Test binary Cr-Ni data produces the same probability regardless of whether the main system is a binary or ternary."""

    datasets_db.insert(CR_NI_ACTIVITY)

    dbf_bin = Database(CR_NI_TDB)
    dbf_tern = Database(CR_FE_NI_TDB)
    phases = list(dbf_tern.phases.keys())

    # Truth
    bin_prob = calculate_activity_error(dbf_bin, ['CR','NI','VA'], phases, datasets_db, {}, {}, {})

    # Getting binary subsystem data explictly (from binary input)
    prob = calculate_activity_error(dbf_tern, ['CR','NI','VA'], phases, datasets_db, {}, {}, {})
    assert np.isclose(prob, bin_prob)

    # Getting binary subsystem from ternary input
    prob = calculate_activity_error(dbf_tern, ['CR', 'FE', 'NI', 'VA'], phases, datasets_db, {}, {}, {})
    assert np.isclose(prob, bin_prob)


def test_get_thermochemical_data_filters_invalid_sublattice_configurations(datasets_db):
    datasets_db.insert(CU_MG_HM_MIX_CUMG2_ANTISITE)

    dbf = Database(CU_MG_TDB)
    comps = ["CU", "MG", "VA"]
    phases = ["CUMG2"]
    thermochemical_data = get_thermochemical_data(dbf, comps, phases, datasets_db)
    print('thermochemical data:', thermochemical_data)
    assert thermochemical_data[0]["calculate_dict"]["values"].shape == (2,)

    error = calculate_non_equilibrium_thermochemical_probability(thermochemical_data)
    assert np.isclose(error, -14.28729)


def test_fixed_configuration_residual_function(datasets_db):
    dbf = Database(CU_MG_TDB)
    datasets_db.insert(CU_MG_HM_MIX_CUMG2_ANTISITE)

    residual_func = FixedConfigurationPropertyResidual(dbf, datasets_db, phase_models=None, symbols_to_fit=[])

    # Regression test "truth" values - got values by running
    residuals, weights = residual_func.get_residuals(np.asarray([]))
    assert len(residuals) == len(weights)
    assert np.allclose(residuals, [-10.0, -100.0])
    likelihood = residual_func.get_likelihood(np.asarray([]))
    assert np.isclose(likelihood, -14.28729, rtol=1e-6)


def test_get_thermochemical_data_filters_configurations_when_all_configurations_are_invalid(datasets_db):
    datasets_db.insert(CU_MG_HM_MIX_CUMG2_ALL_INVALID)  # No valid configurations

    dbf = Database(CU_MG_TDB)
    comps = ["CU", "MG", "VA"]
    phases = ["CUMG2"]
    thermochemical_data = get_thermochemical_data(dbf, comps, phases, datasets_db)
    print('thermochemical data:', thermochemical_data)
    assert thermochemical_data[0]["calculate_dict"]["values"].shape == (0,)

    error = calculate_non_equilibrium_thermochemical_probability(thermochemical_data)
    assert np.isclose(error, 0)


def test_non_equilibrium_thermochemical_error_with_multiple_X_points(datasets_db):
    """Multiple composition datapoints in a dataset for a mixing phase should be successful."""
    datasets_db.insert(CU_MG_CPM_MIX_X_HCP_A3)

    dbf = Database(CU_MG_TDB)
    phases = list(dbf.phases.keys())
    comps = ['CU', 'MG', 'VA']
    thermochemical_data = get_thermochemical_data(dbf, comps, phases, datasets_db)
    error = calculate_non_equilibrium_thermochemical_probability(thermochemical_data)

    assert np.isclose(error, -4061.119001241541, rtol=1e-6)


def test_non_equilibrium_thermochemical_error_with_multiple_T_points(datasets_db):
    """Multiple temperature datapoints in a dataset for a stoichiometric comnpound should be successful."""
    datasets_db.insert(CU_MG_HM_MIX_T_CUMG2)

    dbf = Database(CU_MG_TDB)
    phases = list(dbf.phases.keys())
    comps = ['CU', 'MG', 'VA']
    thermochemical_data = get_thermochemical_data(dbf, comps, phases, datasets_db)
    error = calculate_non_equilibrium_thermochemical_probability(thermochemical_data)
    assert np.isclose(error,-14.287293263253728, rtol=1e-6)


def test_non_equilibrium_thermochemical_error_with_multiple_T_X_points(datasets_db):
    """Multiple temperature and composition datapoints in a dataset for a mixing phase should be successful."""
    datasets_db.insert(CU_MG_SM_MIX_T_X_FCC_A1)

    dbf = Database(CU_MG_TDB)
    phases = list(dbf.phases.keys())
    comps = ['CU', 'MG', 'VA']
    thermochemical_data = get_thermochemical_data(dbf, comps, phases, datasets_db)
    error = calculate_non_equilibrium_thermochemical_probability(thermochemical_data)
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
    error = calculate_non_equilibrium_thermochemical_probability(thermochemical_data)
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
    error = calculate_non_equilibrium_thermochemical_probability(thermochemical_data)
    assert np.isclose(error, zero_error_prob, atol=1e-6)


def test_subsystem_non_equilibrium_thermochemcial_probability(datasets_db):
    """Test binary Cr-Ni data produces the same probability regardless of whether the main system is a binary or ternary."""

    datasets_db.insert(CR_NI_LIQUID_DATA)

    dbf_bin = Database(CR_NI_TDB)
    dbf_tern = Database(CR_FE_NI_TDB)
    phases = list(dbf_tern.phases.keys())

    # Truth
    thermochemical_data = get_thermochemical_data(dbf_bin, ['CR', 'NI', 'VA'], phases, datasets_db)
    bin_prob = calculate_non_equilibrium_thermochemical_probability(thermochemical_data)

    # Getting binary subsystem data explictly (from binary input)
    thermochemical_data = get_thermochemical_data(dbf_tern, ['CR', 'NI', 'VA'], phases, datasets_db)
    prob = calculate_non_equilibrium_thermochemical_probability(thermochemical_data)
    assert np.isclose(prob, bin_prob)

    # Getting binary subsystem from ternary input
    thermochemical_data = get_thermochemical_data(dbf_tern, ['CR', 'FE', 'NI', 'VA'], phases, datasets_db)
    prob = calculate_non_equilibrium_thermochemical_probability(thermochemical_data)
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


def test_zpf_residual_function(datasets_db):
    dbf = Database(CU_MG_TDB)
    datasets_db.insert(CU_MG_DATASET_ZPF_ZERO_ERROR)

    residual_func = ZPFResidual(dbf, datasets_db, phase_models=None, symbols_to_fit=[])

    # Regression test "truth" values - got values by running
    residuals, weights = residual_func.get_residuals(np.asarray([]))
    assert len(residuals) == len(weights)
    assert np.allclose(residuals, [0.0, 0.0], atol=1e-3)  # looser tolerance due to numerical instabilities
    likelihood = residual_func.get_likelihood(np.asarray([]))
    # ZPF weight = 1 kJ and there are two points in the tieline
    zero_error_prob = np.sum(scipy.stats.norm(loc=0, scale=1000.0).logpdf([0.0, 0.0]))
    assert np.isclose(likelihood, zero_error_prob, rtol=1e-6)


def test_subsystem_zpf_probability(datasets_db):
    """Test binary Cr-Ni data produces the same probability regardless of whether the main system is a binary or ternary."""

    datasets_db.insert(CR_NI_ZPF_DATA)

    dbf_bin = Database(CR_NI_TDB)
    dbf_tern = Database(CR_FE_NI_TDB)
    phases = list(dbf_tern.phases.keys())

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


def test_zpf_error_species(datasets_db):
    """Tests that ZPF error works if a species is used."""

    # Note that the liquid is stabilized by the species for the equilibrium
    # used in the data. If the SPECIES is removed from the database (and LIQUID
    # constituents), then the resulting likelihood will NOT match this (and be
    # closer to 93, according to a test.)

    datasets_db.insert(LI_SN_ZPF_DATA)

    dbf = Database(LI_SN_TDB)
    comps = ['LI', 'SN']
    phases = list(dbf.phases.keys())

    # ZPF weight = 1 kJ and there are two points in the tieline
    zero_error_probability = 2 * scipy.stats.norm(loc=0, scale=1000.0).logpdf(0.0)

    zpf_data = get_zpf_data(dbf, comps, phases, datasets_db, {})
    exact_likelihood = calculate_zpf_error(zpf_data, approximate_equilibrium=False)
    assert np.isclose(exact_likelihood, zero_error_probability)
    approx_likelihood = calculate_zpf_error(zpf_data, approximate_equilibrium=True)
    # accept higher tolerance for approximate
    assert np.isclose(approx_likelihood, zero_error_probability, rtol=1e-4)


def test_zpf_error_equilibrium_failure(datasets_db):
    """Test that a target hyperplane producing NaN chemical potentials gives a driving force of zero."""
    datasets_db.insert(CU_MG_DATASET_ZPF_NAN_EQUILIBRIUM)

    dbf = Database(CU_MG_TDB)
    comps = ['CU','MG','VA']
    phases = list(dbf.phases.keys())

    # ZPF weight = 1 kJ and there are two points in the tieline
    zero_error_probability = 2 * scipy.stats.norm(loc=0, scale=1000.0).logpdf(0.0)
    zpf_data = get_zpf_data(dbf, comps, phases, datasets_db, {})

    with mock.patch('espei.error_functions.zpf_error.estimate_hyperplane', return_value=np.array([np.nan, np.nan])):
        exact_likelihood = calculate_zpf_error(zpf_data)
        assert np.isclose(exact_likelihood, zero_error_probability, rtol=1e-6)
        approx_likelihood = calculate_zpf_error(zpf_data)
        assert np.isclose(approx_likelihood, zero_error_probability, rtol=1e-6)


def test_zpf_error_works_for_stoichiometric_cmpd_tielines(datasets_db):
    """A stochimetric compound with approximate composition can be in the datasets and work"""
    datasets_db.insert(CU_MG_DATASET_ZPF_STOICH_COMPOUND)

    dbf = Database(CU_MG_TDB)
    comps = ['CU','MG']
    phases = list(dbf.phases.keys())

    # ZPF weight = 1 kJ and there are two points in the tieline
    zero_error_probability = 2 * scipy.stats.norm(loc=0, scale=1000.0).logpdf(0.0)

    zpf_data = get_zpf_data(dbf, comps, phases, datasets_db, {})
    exact_likelihood = calculate_zpf_error(zpf_data)
    assert np.isclose(exact_likelihood, zero_error_probability, rtol=1e-6)
    approx_likelihood = calculate_zpf_error(zpf_data)
    assert np.isclose(approx_likelihood, zero_error_probability, rtol=1e-6)


def test_non_equilibrium_thermochemcial_species(datasets_db):
    """Test species work for non-equilibrium thermochemical data."""

    datasets_db.insert(LI_SN_LIQUID_DATA)

    dbf = Database(LI_SN_TDB)
    phases = ['LIQUID']

    thermochemical_data = get_thermochemical_data(dbf, ['LI', 'SN'], phases, datasets_db)
    prob = calculate_non_equilibrium_thermochemical_probability(thermochemical_data)
    # Near zero error and non-zero error
    assert np.isclose(prob, (-7.13354663 + -22.43585011))


def test_equilibrium_thermochemcial_error_species(datasets_db):
    """Test species work for equilibrium thermochemical data."""

    datasets_db.insert(LI_SN_LIQUID_EQ_DATA)

    dbf = Database(LI_SN_TDB)
    phases = list(dbf.phases.keys())

    eqdata = get_equilibrium_thermochemical_data(dbf, ['LI', 'SN'], phases, datasets_db)
    # Thermo-Calc
    truth_values = np.array([0.0, -28133.588, -40049.995, 0.0])
    # Approximate
    errors_approximate, weights = calc_prop_differences(eqdata[0], np.array([]), True)
    # Looser tolerances because the equilibrium is approximate, note that this is pdens dependent
    assert np.all(np.isclose(errors_approximate, truth_values, atol=1e-5, rtol=1e-3))
    # Exact
    errors_exact, weights = calc_prop_differences(eqdata[0], np.array([]), False)
    assert np.all(np.isclose(errors_exact, truth_values, atol=1e-5))


def test_equilibrium_thermochemical_error_unsupported_property(datasets_db):
    """Test that an equilibrium property that is not explictly supported will work."""
    # This test specifically tests Curie temperature
    datasets_db.insert(CR_NI_LIQUID_EQ_TC_DATA)
    EXPECTED_VALUES = np.array([374.6625, 0.0, 0.0])  # the TC should be 374.6625 in both cases, but "values" are [0 and 382.0214], so the differences should be flipped.

    dbf = Database(CR_NI_TDB)
    phases = list(dbf.phases.keys())

    eqdata = get_equilibrium_thermochemical_data(dbf, ['CR', 'NI'], phases, datasets_db)
    errors_exact, weights = calc_prop_differences(eqdata[0], np.array([]))
    assert np.all(np.isclose(errors_exact, EXPECTED_VALUES, atol=1e-3))


def test_equilibrium_property_residual_function(datasets_db):
    dbf = Database(CR_NI_TDB)
    datasets_db.insert(CR_NI_LIQUID_EQ_TC_DATA)

    residual_func = EquilibriumPropertyResidual(dbf, datasets_db, phase_models=None, symbols_to_fit=[])

    residuals, weights = residual_func.get_residuals(np.asarray([]))
    assert len(residuals) == len(weights)
    assert np.allclose(residuals, [374.6625, 0.0, 0.0])
    # Regression test "truth" values - got values by running
    likelihood = residual_func.get_likelihood(np.asarray([]))
    assert np.isclose(likelihood, -70188.75126872442, rtol=1e-6)


def test_equilibrium_thermochemical_error_computes_correct_probability(datasets_db):
    """Integration test for equilibrium thermochemical error."""
    datasets_db.insert(CU_MG_EQ_HMR_LIQUID)
    dbf = Database(CU_MG_TDB)
    phases = list(dbf.phases.keys())

    # Test that errors update in response to changing parameters
    # no parameters
    eqdata = get_equilibrium_thermochemical_data(dbf, ['CU', 'MG'], phases, datasets_db)
    errors, weights = calc_prop_differences(eqdata[0], np.array([]))
    expected_vals = [-31626.6*0.5*0.5]
    assert np.all(np.isclose(errors, expected_vals))

    # VV0017 (LIQUID, L0)
    eqdata = get_equilibrium_thermochemical_data(dbf, ['CU', 'MG'], phases, datasets_db, parameters={'VV0017': -31626.6})
    # unchanged, should be the same as before
    errors, weights = calc_prop_differences(eqdata[0], np.array([-31626.6]))
    assert np.all(np.isclose(errors, [-31626.6*0.5*0.5]))
    # change to -40000
    errors, weights = calc_prop_differences(eqdata[0], np.array([-40000], np.float_))
    assert np.all(np.isclose(errors, [-40000*0.5*0.5]))


def test_driving_force_miscibility_gap(datasets_db):
    datasets_db.insert(A_B_DATASET_ALPHA)
    dbf = Database(A_B_REGULAR_SOLUTION_TDB)
    parameters = {"L_ALPHA": None}
    zpf_data = get_zpf_data(dbf, ["A", "B"], ["ALPHA"], datasets_db, parameters)

    # probability for zero error error with ZPF weight = 1000.0
    zero_error_prob = scipy.stats.norm(loc=0, scale=1000.0).logpdf(0.0)

    # Ideal solution case
    params = np.array([0.0])
    prob = calculate_zpf_error(zpf_data, parameters=params, approximate_equilibrium=False)
    assert np.isclose(prob, zero_error_prob)
    prob = calculate_zpf_error(zpf_data, parameters=params, approximate_equilibrium=True)
    assert np.isclose(prob, zero_error_prob)

    # Negative interaction case
    params = np.array([-10000.0])
    prob = calculate_zpf_error(zpf_data, parameters=params, approximate_equilibrium=False)
    assert np.isclose(prob, zero_error_prob)
    prob = calculate_zpf_error(zpf_data, parameters=params, approximate_equilibrium=True)
    assert np.isclose(prob, zero_error_prob)

    # Miscibility gap case
    params = np.array([10000.0])
    prob = calculate_zpf_error(zpf_data, parameters=params, approximate_equilibrium=False)
    # Remember these are log probabilities, so more negative means smaller probability and larger error
    assert prob < zero_error_prob
    prob = calculate_zpf_error(zpf_data, parameters=params, approximate_equilibrium=True)
    assert prob < zero_error_prob


def test_setting_up_context_with_custom_models(datasets_db):
    phase_models = {
      "components": ["CU", "MG", "VA"],
      "phases": {
             "LIQUID" : {
                "sublattice_model": [["CU", "MG"]],
                "sublattice_site_ratios": [1],
                "model": "espei.utils.ErrorModel"
             },
             "FCC_A1": {
                "sublattice_model": [["CU", "MG"], ["VA"]],
                "sublattice_site_ratios": [1, 1]
             }
        }
    }

    dbf = Database(CU_MG_TDB)

    # Should work without error
    ctx = setup_context(dbf, datasets_db, phase_models=phase_models)

    # Once we have data, the ErrorModel should be built and raise
    datasets_db.insert(CU_MG_DATASET_ZPF_ZERO_ERROR)
    with pytest.raises(ModelTestException):
        ctx = setup_context(dbf, datasets_db, phase_models=phase_models)


def test_zpf_context_is_pickleable(datasets_db):
    """Test that the context for ZPF data is pickleable"""
    datasets_db.insert(CU_MG_DATASET_ZPF_ZERO_ERROR)
    dbf = Database(CU_MG_TDB)

    symbols_to_fit = database_symbols_to_fit(dbf)
    initial_guess = np.array([unpack_piecewise(dbf.symbols[s]) for s in symbols_to_fit])
    prior_dict = EmceeOptimizer.get_priors(None, symbols_to_fit, initial_guess)
    ctx = setup_context(dbf, datasets_db)
    ctx.update(prior_dict)

    ctx_pickle = pickle.dumps(ctx)
    ctx_unpickled = pickle.loads(ctx_pickle)

    regular_predict  = EmceeOptimizer.predict(initial_guess, **ctx)
    unpickle_predict = EmceeOptimizer.predict(initial_guess, **ctx_unpickled)
    assert np.isclose(regular_predict, unpickle_predict)


def test_activity_context_is_pickleable(datasets_db):
    """Test that the context for activity data is pickleable"""
    datasets_db.insert(CU_MG_EXP_ACTIVITY)
    dbf = Database(CU_MG_TDB)

    symbols_to_fit = database_symbols_to_fit(dbf)
    initial_guess = np.array([unpack_piecewise(dbf.symbols[s]) for s in symbols_to_fit])
    prior_dict = EmceeOptimizer.get_priors(None, symbols_to_fit, initial_guess)
    ctx = setup_context(dbf, datasets_db)
    ctx.update(prior_dict)

    ctx_pickle = pickle.dumps(ctx)
    ctx_unpickled = pickle.loads(ctx_pickle)

    regular_predict  = EmceeOptimizer.predict(initial_guess, **ctx)
    unpickle_predict = EmceeOptimizer.predict(initial_guess, **ctx_unpickled)
    assert np.isclose(regular_predict, unpickle_predict)


def test_non_equilibrium_thermochemical_context_is_pickleable(datasets_db):
    """Test that the context for non-equilibrium thermochemical data is pickleable"""
    datasets_db.insert(CU_MG_CPM_MIX_X_HCP_A3)
    datasets_db.insert(CU_MG_SM_MIX_T_X_FCC_A1)
    dbf = Database(CU_MG_TDB)

    symbols_to_fit = database_symbols_to_fit(dbf)
    initial_guess = np.array([unpack_piecewise(dbf.symbols[s]) for s in symbols_to_fit])
    prior_dict = EmceeOptimizer.get_priors(None, symbols_to_fit, initial_guess)
    ctx = setup_context(dbf, datasets_db)
    ctx.update(prior_dict)

    ctx_pickle = pickle.dumps(ctx)
    ctx_unpickled = pickle.loads(ctx_pickle)

    regular_predict  = EmceeOptimizer.predict(initial_guess, **ctx)
    unpickle_predict = EmceeOptimizer.predict(initial_guess, **ctx_unpickled)
    assert np.isclose(regular_predict, unpickle_predict)


def test_equilibrium_thermochemical_context_is_pickleable(datasets_db):
    """Test that the context for equilibrium thermochemical data is pickleable"""
    datasets_db.insert(CU_MG_EQ_HMR_LIQUID)
    dbf = Database(CU_MG_TDB)

    symbols_to_fit = database_symbols_to_fit(dbf)
    initial_guess = np.array([unpack_piecewise(dbf.symbols[s]) for s in symbols_to_fit])
    prior_dict = EmceeOptimizer.get_priors(None, symbols_to_fit, initial_guess)
    ctx = setup_context(dbf, datasets_db)
    ctx.update(prior_dict)

    ctx_pickle = pickle.dumps(ctx)
    ctx_unpickled = pickle.loads(ctx_pickle)

    regular_predict  = EmceeOptimizer.predict(initial_guess, **ctx)
    unpickle_predict = EmceeOptimizer.predict(initial_guess, **ctx_unpickled)
    assert np.isclose(regular_predict, unpickle_predict)


def test_zpf_error_for_prescribed_hyperplane_composition(datasets_db):
    """Test a dataset with __HYPERPLANE__ defined works"""
    datasets_db.insert(A_B_DATASET_ALPHA)
    dbf = Database(A_B_REGULAR_SOLUTION_TDB)  # Ideal solution case by default
    zpf_data = get_zpf_data(dbf, ["A", "B"], ["ALPHA"], datasets_db, {})
    driving_forces, weights = calculate_zpf_driving_forces(zpf_data)
    flat_driving_forces = np.asarray(driving_forces).flatten()
    assert len(flat_driving_forces) == 1
    assert np.isclose(flat_driving_forces[0], 0.0)


def test_zpf_error_hyperplane_with_null_phases(datasets_db):
    """Test typical use case of __HYPERPLANE__, where no phase compositions are defined."""
    datasets_db.insert(CU_MG_DATASET_ZPF_HYPERPLANE_TWOPHASE)
    dbf = Database(CU_MG_TDB)  # Ideal solution case by default
    zpf_data = get_zpf_data(dbf, ["CU", "MG"], list(dbf.phases.keys()), datasets_db, {})
    driving_forces, weights = calculate_zpf_driving_forces(zpf_data)
    flat_driving_forces = np.asarray(driving_forces).flatten()
    assert len(flat_driving_forces) == 2  # One for each vertex, HCP_A3 and CUMG2
    assert np.allclose(flat_driving_forces, [-18.05883506, -780.50836135])
