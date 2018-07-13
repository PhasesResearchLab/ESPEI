"""The test_parameter_generation module tests that parameter selection is correct"""

from tinydb import where
import sympy
from pycalphad import variables as v
from collections import OrderedDict

from espei.tests.fixtures import datasets_db
from espei.parameter_selection.ternary_parameters import build_candidate_models
from espei.paramselect import generate_parameters, generate_symmetric_group, sorted_interactions
from espei.tests.testing_data import *

def test_mixing_energies_are_fit(datasets_db):
    """Tests that given mixing energy data, the excess parameter is fit."""
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
        "values": [[[-10000]]]
    }
    datasets_db.insert(dataset_excess_mixing)

    dbf = generate_parameters(phase_models, datasets_db, 'SGTE91', 'linear')

    assert dbf.elements == {'AL', 'B'}
    assert set(dbf.phases.keys()) == {'LIQUID', 'FCC_A1'}
    assert len(dbf._parameters.search(where('parameter_type') == 'L')) == 1
    assert dbf.symbols['VV0000'] == -40000

    # check that read/write is ok
    read_dbf = dbf.from_string(dbf.to_string(fmt='tdb'), fmt='tdb')
    assert read_dbf.elements == {'AL', 'B'}
    assert set(read_dbf.phases.keys()) == {'LIQUID', 'FCC_A1'}
    assert len(read_dbf._parameters.search(where('parameter_type') == 'L')) == 1

    # the error should be exactly 0 because we are only fitting to one point
    from espei.error_functions import calculate_thermochemical_error
    assert calculate_thermochemical_error(read_dbf, sorted(dbf.elements), sorted(dbf.phases.keys()), datasets_db) == 0


def test_mixing_data_is_excess_only(datasets_db):
    """Tests that given an entropy of mixing datapoint of 0, no excess parameters are fit (meaning datasets do not include ideal mixing)."""
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
        "values": [[[0]]]
    }
    datasets_db.insert(dataset_excess_mixing)

    dbf = generate_parameters(phase_models, datasets_db, 'SGTE91', 'linear')
    assert dbf.elements == {'AL', 'B'}
    assert set(dbf.phases.keys()) == {'LIQUID', 'FCC_A1'}
    assert len(dbf._parameters.search(where('parameter_type') == 'L')) == 0


def test_multi_sublattice_mixing_energies_are_fit(datasets_db):
    """Tests the correct excess parameter is fit for phases with multiple sublattices with vacancies."""
    phase_models = {
        "components": ["AL", "B", "VA"],
        "phases": {
            "FCC_A1" : {
                "sublattice_model": [["AL", "B"], ["AL", "VA"]],
                "sublattice_site_ratios": [1, 3]
            }
        }
    }

    dataset_excess_mixing = {
        "components": ["AL", "B", "VA"],
        "phases": ["FCC_A1"],
        "solver": {
            "sublattice_site_ratios": [1, 3],
            "sublattice_occupancies": [[[0.5, 0.5], 1], [[0.5, 0.5], 1]],
            "sublattice_configurations": [[["AL", "B"], "VA"], [["AL", "B"], "AL"]],
            "mode": "manual"
        },
        "conditions": {
            "P": 101325,
            "T": 298.15
        },
        "output": "HM_MIX",
        "values": [[[-10000, -10000]]]
    }
    datasets_db.insert(dataset_excess_mixing)

    dbf = generate_parameters(phase_models, datasets_db, 'SGTE91', 'linear')

    assert set(dbf.phases.keys()) == {'FCC_A1'}
    assert len(dbf._parameters.search(where('parameter_type') == 'L')) == 2
    assert dbf.symbols['VV0000'] == -160000.0
    assert dbf.symbols['VV0001'] == -40000.0


def test_sgte_reference_state_naming_is_correct_for_character_element(datasets_db):
    """Elements with single character names should get the correct GHSER reference state name (V => GHSERVV)"""
    phase_models = {
        "components": ["AL", "V"],
        "phases": {
            "LIQUID" : {
                "sublattice_model": [["AL", "V"]],
                "sublattice_site_ratios": [1]
            },
            "BCC_A2" : {
                "sublattice_model": [["AL", "V"]],
                "sublattice_site_ratios": [1]
            }
        }
    }

    dbf = generate_parameters(phase_models, datasets_db, 'SGTE91', 'linear')
    assert dbf.symbols['GBCCV'].args[0][0].__str__() == 'GHSERVV'
    assert 'GHSERVV' in dbf.symbols.keys()
    assert 'GHSERAL' in dbf.symbols.keys()


def test_symmetric_group_can_be_generated_for_2_sl_mixing_with_symmetry():
    """A phase with two sublattices that are mixing should generate a cross interaction"""
    symm_groups = generate_symmetric_group((('AL', 'CO'), ('AL', 'CO')), [[0, 1]])
    assert symm_groups == [(('AL', 'CO'), ('AL', 'CO'))]


def test_symmetric_group_can_be_generated_for_2_sl_endmembers_with_symmetry():
    """A phase with symmetric sublattices should find a symmetric endmember """
    symm_groups = generate_symmetric_group(('AL', 'CO'), [[0, 1]])
    assert symm_groups == [('AL', 'CO'), ('CO', 'AL')]


def test_interaction_sorting_is_correct():
    """High order (order >= 3) interactions should sorted correctly"""
    # Correct sorting of n-order interactions should sort first by number of
    # interactions of order n, then n-1, then n-2... to 1
    unsorted_interactions = [
        ('AL', ('AL', 'CO', 'CR')),
        (('AL', 'CO'), ('AL', 'CO', 'CR')),
        (('AL', 'CO', 'CR'), ('AL', 'CO', 'CR')),
        (('AL', 'CO', 'CR'), 'AL'),
        (('AL', 'CO', 'CR'), ('AL', 'CO')),
        (('AL', 'CO', 'CR'), ('AL', 'CR')),
        (('AL', 'CO', 'CR'), 'CO'),
        (('AL', 'CO', 'CR'), ('CO', 'CR')),
        (('AL', 'CO', 'CR'), 'CR'),
        (('AL', 'CR'), ('AL', 'CO', 'CR')),
        ('CO', ('AL', 'CO', 'CR')),
        (('CO', 'CR'), ('AL', 'CO', 'CR')),
        ('CR', ('AL', 'CO', 'CR')),
    ]
    interactions = sorted_interactions(unsorted_interactions, max_interaction_order=3, symmetry=None)

    # the numbers are the different sort scores. Two of the same sort scores mean
    # the order doesn't matter
    assert interactions == [
        ('AL', ('AL', 'CO', 'CR')),                # (1, 0, 1)
        (('AL', 'CO', 'CR'), 'AL'),                # (1, 0, 1)
        (('AL', 'CO', 'CR'), 'CO'),                # (1, 0, 1)
        (('AL', 'CO', 'CR'), 'CR'),                # (1, 0, 1)
        ('CO', ('AL', 'CO', 'CR')),                # (1, 0, 1)
        ('CR', ('AL', 'CO', 'CR')),                # (1, 0, 1)
        (('AL', 'CO'), ('AL', 'CO', 'CR')),        # (1, 1, 0)
        (('AL', 'CO', 'CR'), ('AL', 'CO')),        # (1, 1, 0)
        (('AL', 'CO', 'CR'), ('AL', 'CR')),        # (1, 1, 0)
        (('AL', 'CO', 'CR'), ('CO', 'CR')),        # (1, 1, 0)
        (('AL', 'CR'), ('AL', 'CO', 'CR')),        # (1, 1, 0)
        (('CO', 'CR'), ('AL', 'CO', 'CR')),        # (1, 1, 0)
        (('AL', 'CO', 'CR'), ('AL', 'CO', 'CR')),  # (2, 0, 0)
    ]


def test_symmetric_ternary_parameter_can_be_generated(datasets_db):
    """A symmetric ternary parameter should be generated correctly."""
    datasets_db.insert(AL_CO_CR_A2_TERNARY_SYMMETRIC_DATASET)

    dbf = generate_parameters(AL_CO_CR_A2_PHASE_MODELS, datasets_db, 'SGTE91', 'linear')

    assert dbf.elements == {'AL', 'CO', 'CR'}
    assert set(dbf.phases.keys()) == {'BCC_A2'}
    # rounded to 6 digits by `numdigits`, this is confirmed to be a correct value.
    assert len(dbf._parameters.search(where('parameter_type') == 'L')) == 1
    assert dbf.symbols['VV0000'] == -212221.0


def test_symmetric_ternary_parameter_can_be_generated_in_presence_of_binary_data(datasets_db):
    """A symmetric ternary paramer should be generated correctly when low order binary data is also fit."""
    datasets_db.insert(AL_CO_A2_BINARY_SYMMETRIC_DATASET)
    datasets_db.insert(AL_CO_CR_A2_TERNARY_SYMMETRIC_DATASET)

    dbf = generate_parameters(AL_CO_CR_A2_PHASE_MODELS, datasets_db, 'SGTE91', 'linear')

    assert dbf.elements == {'AL', 'CO', 'CR'}
    assert set(dbf.phases.keys()) == {'BCC_A2'}
    # rounded to 6 digits by `numdigits`, this is confirmed to be a correct value.
    assert len(dbf._parameters.search(where('parameter_type') == 'L')) == 2
    assert dbf.symbols['VV0000'] == -4000.0
    assert dbf.symbols['VV0001'] == -200245.0



def test_asymmetric_ternary_parameters_can_be_generated(datasets_db):
    """3 asymmetric ternary parameters should be generated correctly."""
    datasets_db.insert(AL_CO_CR_BCC_A2_TERNARY_NON_SYMMETRIC_DATASET)

    dbf = generate_parameters(AL_CO_CR_A2_PHASE_MODELS, datasets_db, 'SGTE91', 'linear')

    assert dbf.elements == {'AL', 'CO', 'CR'}
    assert set(dbf.phases.keys()) == {'BCC_A2'}
    # rounded to 6 digits by `numdigits`, this is confirmed to be a correct value.
    assert len(dbf._parameters.search(where('parameter_type') == 'L')) == 3
    assert dbf.symbols['VV0000'] == -4000.0
    assert dbf.symbols['VV0001'] == -200245.0
    assert dbf.symbols['VV0002'] == -200245.0


def test_ternary_candidate_models_are_constructed_correctly():
    """Candidate models should generate all combinations of possible models"""
    features = OrderedDict([("CPM_FORM",
                 (v.T*sympy.log(v.T), v.T**2)),
                ("SM_FORM", (v.T,)),
                ("HM_FORM", (sympy.S.One,))
                ])
    YS = sympy.Symbol('YS')
    V_i, V_j, V_k = sympy.Symbol('V_i'), sympy.Symbol('V_j'), sympy.Symbol('V_k')
    candidate_models = build_candidate_models((('A', 'B', 'C'), 'A'), features)
    assert candidate_models == OrderedDict([
        ('CPM_FORM', [
            [v.T*YS*sympy.log(v.T)],
            [v.T*YS*sympy.log(v.T), v.T**2*YS],
            [v.T*V_i*YS*sympy.log(v.T), v.T*V_j*YS*sympy.log(v.T), v.T*V_k*YS*sympy.log(v.T)],
            [v.T*V_i*YS*sympy.log(v.T), v.T**2*V_i*YS, v.T*V_j*YS*sympy.log(v.T), v.T**2*V_j*YS, v.T*V_k*YS*sympy.log(v.T), v.T**2*V_k*YS],
        ]),
        ('SM_FORM', [
            [v.T*YS],
            [v.T*V_i*YS, v.T*V_j*YS, v.T*V_k*YS]
        ]),
        ('HM_FORM', [
            [YS],
            [V_i*YS, V_j*YS, V_k*YS]
        ])
    ])
