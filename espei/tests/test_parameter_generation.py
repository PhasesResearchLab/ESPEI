"""The test_parameter_generation module tests that parameter selection is correct"""

from tinydb import where

from espei.tests.fixtures import datasets_db
from espei.paramselect import generate_parameters

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

    # check that read/write is ok
    read_dbf = dbf.from_string(dbf.to_string(fmt='tdb'), fmt='tdb')
    assert read_dbf.elements == {'AL', 'B'}
    assert set(read_dbf.phases.keys()) == {'LIQUID', 'FCC_A1'}
    assert len(read_dbf._parameters.search(where('parameter_type') == 'L')) == 1

    # the error should be exactly 0 because we are only fitting to one point
    from espei.error_functions import calculate_thermochemical_error
    assert calculate_thermochemical_error(read_dbf, sorted(dbf.elements), sorted(dbf.phases.keys()), datasets_db) == 0

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
