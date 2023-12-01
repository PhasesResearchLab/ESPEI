"""The test_parameter_generation module tests that parameter selection is correct"""

import copy
from tinydb import where
from tinydb.storages import MemoryStorage
import numpy as np
from pycalphad import Database, variables as v
import scipy
from symengine import Symbol

from espei.logger import config_logger
from espei.paramselect import generate_parameters
from espei.utils import PickleableTinyDB
from espei.parameter_selection.fitting_descriptions import molar_volume_gibbs_energy_fitting_description
from .testing_data import *
from .fixtures import datasets_db

import pytest


def test_formation_energies_are_fit(datasets_db):
    """Tests that given formation energy data, the parameter is fit."""
    phase_models = {
        "components": ["CU", "MG"],
        "phases": {
            "CUMG2" : {
                "sublattice_model": [["CU"], ["MG"]],
                "sublattice_site_ratios": [1, 2]
            }
        }
    }

    dataset_cumg2_hm_form = {
        "components": ["CU", "MG"],
        "phases": ["CUMG2"],
        "solver": {
            "sublattice_site_ratios": [1, 2],
            "sublattice_configurations": [["CU", "MG"]],
            "mode": "manual"
        },
        "conditions": {
            "P": 101325,
            "T": 298.15
        },
        "output": "HM_FORM",
        "values": [[[-5000]]]
    }

    dataset_cumg2_sm_form = {
        "components": ["CU", "MG"],
        "phases": ["CUMG2"],
        "solver": {
            "sublattice_site_ratios": [1, 2],
            "sublattice_configurations": [["CU", "MG"]],
            "mode": "manual"
        },
        "conditions": {
            "P": 101325,
            "T": 298.15
        },
        "output": "SM_FORM",
        "values": [[[-2]]]
    }

    dataset_cumg2_cpm_form = {
        "components": ["CU", "MG"],
        "phases": ["CUMG2"],
        "solver": {
            "sublattice_site_ratios": [1, 2],
            "sublattice_configurations": [["CU", "MG"]],
            "mode": "manual"
        },
        "conditions": {
            "P": 101325,
            "T": 298.15
        },
        "output": "CPM_FORM",
        "values": [[[0.3]]]
    }
    datasets_db.insert(dataset_cumg2_hm_form)
    datasets_db.insert(dataset_cumg2_sm_form)
    datasets_db.insert(dataset_cumg2_cpm_form)

    dbf = generate_parameters(phase_models, datasets_db, 'SGTE91', 'linear')

    assert dbf.elements == {'CU', 'MG'}
    assert set(dbf.phases.keys()) == {'CUMG2'}
    assert len(dbf._parameters.search((where('parameter_type') == 'G') & (where('phase_name') == 'CUMG2'))) == 1
    assert dbf.symbols['VV0000'] == -0.9  # heat capacity
    assert dbf.symbols['VV0001'] == 12.0278  # entropy
    assert dbf.symbols['VV0002'] == -15268.3  # enthalpy


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

    from espei.error_functions import calculate_non_equilibrium_thermochemical_probability, get_thermochemical_data
    # the error should be exactly 0 because we are only fitting to one point
    zero_error_prob = scipy.stats.norm(loc=0, scale=500.0).logpdf(0.0)  # HM weight = 500
    # Explicitly pass parameters={} to not try fitting anything
    thermochemical_data = get_thermochemical_data(dbf, sorted(read_dbf.elements), list(read_dbf.phases.keys()), datasets_db, symbols_to_fit=[])
    error = calculate_non_equilibrium_thermochemical_probability(thermochemical_data)
    assert np.isclose(error, zero_error_prob, atol=1e-6)

def test_duplicate_parameters_are_not_added_with_input_database(datasets_db):
    phase_models = {
        "components": ["AL", "B"],
        "phases": {
            "LIQUID" : {
                "sublattice_model": [["AL", "B"]],
                "sublattice_site_ratios": [1]
            }
        }
    }

    dbf = generate_parameters(phase_models, datasets_db, 'SGTE91', 'linear', 1e-2)
    assert len(dbf._parameters.search(where('parameter_type') == 'G')) == 2 # each endmember
    dbf = generate_parameters(phase_models, datasets_db, 'SGTE91', 'linear', 1e-2, dbf=dbf)
    assert len(dbf._parameters.search(where('parameter_type') == 'G')) == 2 # each endmember

def test_mixing_energies_are_reduced_with_ridge_alpha(datasets_db):
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

    dbf = generate_parameters(phase_models, datasets_db, 'SGTE91', 'linear', 1e-2)

    assert dbf.elements == {'AL', 'B'}
    assert set(dbf.phases.keys()) == {'LIQUID', 'FCC_A1'}
    assert len(dbf._parameters.search(where('parameter_type') == 'L')) == 1
    assert dbf.symbols['VV0000'] == -34482.8


def test_mixing_energies_are_fit_with_higher_order_data(datasets_db):
    """Tests that given mixing energy data with high order terms, the excess parameter is fit."""
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

    dataset_excess_mixing_sm = {
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
        "values": [[[0.1]]],
        "excluded_model_contributions": ["idmix", "mag"]
    }
    datasets_db.insert(dataset_excess_mixing)
    datasets_db.insert(dataset_excess_mixing_sm)

    dbf = generate_parameters(phase_models, datasets_db, 'SGTE91', 'linear')

    assert dbf.elements == {'AL', 'B'}
    assert set(dbf.phases.keys()) == {'LIQUID', 'FCC_A1'}
    assert len(dbf._parameters.search(where('parameter_type') == 'L')) == 1
    assert dbf.symbols['VV0000'] == -0.4  # entropy
    assert dbf.symbols['VV0001'] == -40000  # heat capacity


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
        "values": [[[0]]],
        "excluded_model_contributions": ["idmix", "mag"]
    }
    datasets_db.insert(dataset_excess_mixing)

    dbf = generate_parameters(phase_models, datasets_db, 'SGTE91', 'linear')
    assert dbf.elements == {'AL', 'B'}
    assert set(dbf.phases.keys()) == {'LIQUID', 'FCC_A1'}
    try:
        assert len(dbf._parameters.search(where('parameter_type') == 'L')) == 0
    except AssertionError:
        # Also accept a parameter that's nearly zero (precision issues)
        assert len(dbf._parameters.search(where('parameter_type') == 'L')) == 1
        assert np.isclose(dbf.symbols['VV0000'], 0.0, atol=1e-14)


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
    assert dbf.symbols['GBCCV'].args[0].__str__() == 'GHSERVV'
    assert 'GHSERVV' in dbf.symbols.keys()
    assert 'GHSERAL' in dbf.symbols.keys()


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
    assert dbf.symbols['VV0000'] == -3000.0
    assert dbf.symbols['VV0001'] == -2000.0
    assert dbf.symbols['VV0002'] == -1000.0



def test_asymmetric_ternary_parameters_can_be_generated_for_2_sublattice(datasets_db):
    """3 asymmetric ternary parameters should be generated correctly in a 2 sublattice model."""
    datasets_db.insert(AL_CO_CR_BCC_B2_TERNARY_NON_SYMMETRIC_DATASET)

    dbf = generate_parameters(AL_CO_CR_B2_PHASE_MODELS, datasets_db, 'SGTE91', 'linear')

    assert dbf.elements == {'AL', 'CO', 'CR'}
    assert set(dbf.phases.keys()) == {'BCC_B2'}
    # rounded to 6 digits by `numdigits`, this is confirmed to be a correct value.
    interaction_parameters = dbf._parameters.search(where('parameter_type') == 'L')
    assert len(interaction_parameters) == 3
    assert dbf.symbols['VV0000'] == -6000.0
    assert dbf.symbols['VV0001'] == -4000.0
    assert dbf.symbols['VV0002'] == -2000.0


def test_cpm_sm_data_can_be_fit_successively(datasets_db):
    """CPM_MIX data should be able to be fit, followed by SM_MIX data, producing parameters that reproduce the original data"""
    # An issue was brought up where having CPM_MIX data and SM_MIX data in the same fit raised an error:
    # TypeError("can't convert expression to float") in line 188 of fit_formation_energy
    # caused by having leftover YS symbols that were fit in the fixed portions.
    datasets_db.insert(CU_ZN_CPM_MIX_EXPR_TO_FLOAT)
    datasets_db.insert(CU_ZN_SM_MIX_EXPR_TO_FLOAT)
    dbf = generate_parameters(CU_ZN_LIQUID_PHASE_MODEL, datasets_db, 'SGTE91', 'linear')
    # beware that the calculate() results will not match up exactly with the original data due to rounding of parameters

    assert dbf.symbols['VV0000'] == 105.255  # T*ln(T) L2 term
    assert dbf.symbols['VV0001'] == -40.953  # T*ln(T) L1 term
    assert dbf.symbols['VV0002'] == -44.57 # T*ln(T) L0 term
    assert dbf.symbols['VV0003'] == 36.6556 # L0 T term, found after CPM_MIX addition


def test_high_order_interaction_terms_no_spurious_symbols(datasets_db):
    """Test that no spurious symbols (e.g. `Z`) slip into the symbolic part of generated excess parameters"""
    config_logger(verbosity=3)
    datasets_db.insert(CU_ZN_SM_MIX_L1)
    dbf = generate_parameters(CU_ZN_LIQUID_PHASE_MODEL, datasets_db, 'SGTE91', 'linear')

    assert 'VV0000' in dbf.symbols
    assert 'VV0001' in dbf.symbols

    params = dbf.search((where("parameter_order") == 1) & (where("parameter_type") == "L"))
    assert len(params) == 1
    param = params[0]['parameter']
    print(param)
    assert param == Symbol("VV0000") * v.T


def test_initial_database_can_be_supplied(datasets_db):
    """Initial Databases can be passed to parameter generation"""
    initial_dbf = Database(CR_FE_INITIAL_TDB_CONTRIBUTIONS)
    assert len(initial_dbf._parameters.all()) == 11
    dbf = generate_parameters(CR_FE_PHASE_MODELS, datasets_db, 'SGTE91', 'linear', dbf=initial_dbf)
    assert len(dbf._parameters.all()) == 13  # 11 initial parameters + 2 generated endmember parameters


def test_model_contributions_can_be_excluded(datasets_db):
    """Model contributions excluded in the datasets should not be fit"""
    datasets_db.insert(CR_FE_HM_MIX_EXCLUDED_MAG)
    dbf = generate_parameters(CR_FE_PHASE_MODELS, datasets_db, 'SGTE91', 'linear', dbf=Database(CR_FE_INITIAL_TDB_CONTRIBUTIONS))
    assert dbf.symbols['VV0000'] == 40000  # 4 mol-atom/mol-form * 10000 J/mol-atom, verified with no initial Database


def test_multiple_excluded_contributions(datasets_db):
    """Model contributions excluded more than once in the datasets still produce correct results"""
    double_exclude_dataset = copy.deepcopy(CR_FE_HM_MIX_EXCLUDED_MAG)
    double_exclude_dataset['excluded_model_contributions'] = ['mag', 'mag']
    datasets_db.insert(double_exclude_dataset)
    dbf = generate_parameters(CR_FE_PHASE_MODELS, datasets_db, 'SGTE91', 'linear', dbf=Database(CR_FE_INITIAL_TDB_CONTRIBUTIONS))
    assert dbf.symbols['VV0000'] == 40000  # 4 mol-atom/mol-form * 10000 J/mol-atom, verified with no initial Database


def test_model_contributions_can_be_excluded_mixed_datasets(datasets_db):
    """Model contributions excluded in the datasets should not be fit and should still work when different types of datasets are mixed"""
    datasets_db.insert(CR_FE_HM_MIX_EXCLUDED_MAG)
    datasets_db.insert(CR_FE_HM_MIX_WITH_MAG)
    dbf = generate_parameters(CR_FE_PHASE_MODELS, datasets_db, 'SGTE91', 'linear', dbf=Database(CR_FE_INITIAL_TDB_CONTRIBUTIONS))
    assert dbf.symbols['VV0000'] == 40000  # 4 mol-atom/mol-form * 10000 J/mol-atom, verified with no initial Database

def test_parameters_can_be_generated_with_component_subsets(datasets_db):
    CR_FE_PHASE_MODELS = {
        "components": ["CR", "FE"],
        "phases": {
               "BCC_A2": {
                  "sublattice_model": [["CR", "FE", "NI", "V"]],
                  "sublattice_site_ratios": [1]
               },
               "INACTIVE": {
                    "sublattice_model": [["NI", "V"]],
                    "sublattice_site_ratios": [1]
               }
          }
      }

    generate_parameters(CR_FE_PHASE_MODELS, datasets_db, 'SGTE91', 'linear')


def test_weighting_invariance():
    """Test that weights do not affect model selection using perfect L0 and L1 cases."""
    phase_models = {
        "components": ["AL", "B"],
        "phases": {
            "ALPHA" : {
                "sublattice_model": [["AL", "B"]],
                "sublattice_site_ratios": [1]
            }
        }
    }

    L0_data = {
        "components": ["AL", "B"],
        "phases": ["ALPHA"],
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
        "values": [[[-1000]]]
    }

    L1_data = {
        "components": ["AL", "B"],
        "phases": ["ALPHA"],
        "solver": {
            "sublattice_site_ratios": [1],
            "sublattice_occupancies": [[[0.25, 0.75]], [[0.5, 0.5]], [[0.75, 0.25]]],
            "sublattice_configurations": [[["AL", "B"]], [["AL", "B"]], [["AL", "B"]]],
            "mode": "manual"
        },
        "conditions": {
            "P": 101325,
            "T": 298.15
        },
        "output": "HM_MIX",
        "values": [[[-1000.0, 0, 1000.0]]]
    }

    # Perfect L0, no weight
    datasets_db = PickleableTinyDB(storage=MemoryStorage)
    datasets_db.insert(L0_data)
    dbf = generate_parameters(phase_models, datasets_db, 'SGTE91', 'linear')
    datasets_db.close()
    params = dbf._parameters.search(where('parameter_type') == 'L')
    print([f"L{p['parameter_order']}: {p['parameter']}" for p in params])
    print({str(p['parameter']): dbf.symbols[str(p['parameter'])] for p in params})
    assert len(params) == 1
    assert dbf.symbols['VV0000'] == -4000

    # Perfect L0, with weight
    datasets_db = PickleableTinyDB(storage=MemoryStorage)
    L0_data['weight'] = 0.1  # lower weight
    datasets_db.insert(L0_data)
    dbf = generate_parameters(phase_models, datasets_db, 'SGTE91', 'linear')
    datasets_db.close()
    params = dbf._parameters.search(where('parameter_type') == 'L')
    print([f"L{p['parameter_order']}: {p['parameter']}" for p in params])
    print({str(p['parameter']): dbf.symbols[str(p['parameter'])] for p in params})
    assert len(params) == 1
    assert dbf.symbols['VV0000'] == -4000


    # Perfect L1, no weight
    datasets_db = PickleableTinyDB(storage=MemoryStorage)
    datasets_db.insert(L1_data)
    dbf = generate_parameters(phase_models, datasets_db, 'SGTE91', 'linear')
    datasets_db.close()
    params = dbf._parameters.search(where('parameter_type') == 'L')
    print([f"L{p['parameter_order']}: {p['parameter']}" for p in params])
    print({str(p['parameter']): dbf.symbols[str(p['parameter'])] for p in params})
    assert len(params) == 2
    assert np.isclose(dbf.symbols['VV0000'], 1000*32/3)  # L1
    assert np.isclose(dbf.symbols['VV0001'], 0)  # L0

    # Perfect L1, with weight
    datasets_db = PickleableTinyDB(storage=MemoryStorage)
    L1_data['weight'] = 0.1  # lower weight
    datasets_db.insert(L1_data)
    dbf = generate_parameters(phase_models, datasets_db, 'SGTE91', 'linear')
    datasets_db.close()
    params = dbf._parameters.search(where('parameter_type') == 'L')
    print([f"L{p['parameter_order']}: {p['parameter']}" for p in params])
    print({str(p['parameter']): dbf.symbols[str(p['parameter'])] for p in params})
    # TODO: sometimes the presence of L0 terms can be flaky
    # assert len(params) == 2
    assert np.isclose(dbf.symbols['VV0000'], 1000*32/3)  # L1
    # assert np.isclose(dbf.symbols['VV0001'], 0)  # L0


def test_G_lattice_stabilities_do_not_prevent_fitting_other_parameters(datasets_db):
    # ESPEI skips adding G/L parameters if a lattice stability or an input
    # database is provided that has those parameters already. The presence of G
    # parameters (or any parameter X) should not prevent other parameters to be
    # fit as long as the fitting step parameter name != G (or != X).
    # This also should act as a test for fitting G and non-G parameters at the same time

    datasets_db.insert({
        "components": ["HF"], "phases": ["HCP_A3"],
        "conditions": {"P": 101315, "T": 298.15},
        "solver": {"mode": "manual", "sublattice_site_ratios": [1], "sublattice_configurations": [["HF"]], "sublattice_occupancies": [[1.0]]},
        "output": "V0", "values": [[[10.1092e-6]]],
        "reference": "Lu (2005)", "bibtex": "lu2005", "comment": "From Table 1",
    })

    phase_models = {
        "components": ["HF"],
        "phases": {"HCP_A3" : {"sublattice_model": [["HF"]], "sublattice_site_ratios": [1]}}
    }

    dbf = generate_parameters(phase_models, datasets_db, 'SGTE91', 'linear', fitting_description=molar_volume_gibbs_energy_fitting_description)
    print(dbf._parameters.all())
    assert len(dbf._parameters.search(where('parameter_type') == 'G')) == 1 # pure element lattice stability added
    assert len(dbf._parameters.search(where('parameter_type') == 'V0')) == 1 # volume parameter added


def test_volume_parameters_are_not_fit_if_present_in_database(datasets_db):
    # Use Lu 2005 digitized data as an input_db and provide some unary volume
    # data and make sure parameters aren't fit.
    datasets_db.insert({
        "components": ["HF"], "phases": ["HCP_A3"],
        "conditions": {"P": 101315, "T": 298.15},
        "solver": {"mode": "manual", "sublattice_site_ratios": [1], "sublattice_configurations": [["HF"]], "sublattice_occupancies": [[1.0]]},
        "output": "V0", "values": [[[10.1092e-6]]],
        "reference": "Lu (2005)", "bibtex": "lu2005", "comment": "From Table 1",
    })

    # fake data
    datasets_db.insert({
        "components": ["HF", "ZR"], "phases": ["HCP_A3"],
        "conditions": {"T": 298.15, "P": 101325},
        "solver": {"mode": "manual", "sublattice_site_ratios": [1], "sublattice_configurations": [[["HF", "ZR"]]], "sublattice_occupancies": [[[0.5, 0.5]]]},
        "output": "V0_MIX", "values": [[[1.0e-05]]],  # "fixed" value (absolute value from mix data with calphd reference state) extracted manually from above
    })

    dbf = Database(dbf_vol)

    phase_models = {
        "components": ["HF"],
        "phases": {"HCP_A3" : {"sublattice_model": [["HF"]], "sublattice_site_ratios": [1]}}
    }

    dbf = generate_parameters(phase_models, datasets_db, 'SGTE91', 'linear', dbf=dbf, fitting_description=molar_volume_gibbs_energy_fitting_description)
    assert len(dbf._parameters.search(where('parameter_type') == 'G')) == 2 # pure element lattice stability added
    assert len(dbf._parameters.search(where('parameter_type') == 'V0')) == 3 # 2 volume parameters already exist in database, 1 added


def test_elastic_fitting_description_works(datasets_db):
    # We can implement and pass a custom model and fitting description to generate parameters
    # Essentially matches the tutorial
    import tinydb
    from pycalphad import Model

    class ElasticModel(Model):
        def build_phase(self, dbe):
            phase = dbe.phases[self.phase_name]
            param_search = dbe.search
            for prop in ['C11', 'C12', 'C44']:
                prop_param_query = (
                    (tinydb.where('phase_name') == phase.name) & \
                    (tinydb.where('parameter_type') == prop) & \
                    (tinydb.where('constituent_array').test(self._array_validity))
                    )
                prop_val = self.redlich_kister_sum(phase, param_search, prop_param_query).subs(dbe.symbols)
                setattr(self, prop, prop_val)

    from espei.parameter_selection.fitting_descriptions import ModelFittingDescription
    from espei.parameter_selection.fitting_steps import AbstractRKMPropertyStep

    class StepElasticC11(AbstractRKMPropertyStep):
        parameter_name = "C11"
        data_types_read = "C11"

    class StepElasticC12(AbstractRKMPropertyStep):
        parameter_name = "C12"
        data_types_read = "C12"

    class StepElasticC44(AbstractRKMPropertyStep):
        parameter_name = "C44"
        data_types_read = "C44"

    elastic_fitting_description = ModelFittingDescription([StepElasticC11, StepElasticC12, StepElasticC44], model=ElasticModel)

    datasets_db.insert({
    "components": ["TI", "VA"], "phases": ["BCC_A2"],
    "output": "C12", "values": [[[115]]],
    "conditions": {"T": 298.15, "P": 101325},
    "solver": {"mode": "manual", "sublattice_site_ratios": [1, 3], "sublattice_configurations": [["TI", "VA"]], "sublattice_occupancies": [[1.0, 1.0]]},
    "reference": "Marker (2018)", "bibtex": "marker2018binary_elastic", "comment": "Values pulled from Table 4 (DFT calculations).",
    })
    datasets_db.insert({
    "components": ["MO", "VA"], "phases": ["BCC_A2"],
    "output": "C12", "values": [[[164]]],
    "conditions": {"T": 298.15, "P": 101325},
    "solver": {"mode": "manual", "sublattice_site_ratios": [1, 3], "sublattice_configurations": [["MO", "VA"]], "sublattice_occupancies": [[1.0, 1.0]]},
    "reference": "Marker (2018)", "bibtex": "marker2018binary_elastic", "comment": "Values pulled from Table 4 (DFT calculations).",
    })
    datasets_db.insert({
    "components": ["MO", "TI", "VA"], "phases": ["BCC_A2"],
    "output": "C12", "values": [[[111, 113, 123, 136, 146, 158, 163]]],
    "conditions": {"T": 298.15, "P": 101325},
    "solver": {"mode": "manual", "sublattice_site_ratios": [1, 3], "sublattice_configurations": [[["MO", "TI"], "VA"], [["MO", "TI"], "VA"], [["MO", "TI"], "VA"], [["MO", "TI"], "VA"], [["MO", "TI"], "VA"], [["MO", "TI"], "VA"], [["MO", "TI"], "VA"]], "sublattice_occupancies": [[[0.06, 0.94], 1.0], [[0.13, 0.87], 1.0], [[0.25, 0.75], 1.0], [[0.50, 0.50], 1.0], [[0.75, 0.25], 1.0], [[0.94, 0.06], 1.0], [[0.98, 0.02], 1.0]]},
    "reference": "Marker (2018)", "bibtex": "marker2018binary_elastic", "comment": "Values pulled from Table 4 (DFT calculations).",
    })

    phase_models = {
        "components": ["MO", "TI", "VA"],
        "phases": {"BCC_A2" : {"sublattice_model": [["MO", "TI"], ["VA"]], "sublattice_site_ratios": [1, 3]}}
    }

    config_logger(3)
    dbf = generate_parameters(phase_models, datasets_db, 'SGTE91', 'linear', fitting_description=elastic_fitting_description)
    assert len(dbf._parameters.search(where('parameter_type') == 'C12')) == 3 # 3 added
    assert dbf.symbols['VV0000'] == 164.0
    assert dbf.symbols['VV0001'] == 115.0
    assert dbf.symbols['VV0002'] == -27.9687
    assert len(dbf._parameters.all()) == 3 # nothing else added


def test_property_models_for_phases_with_more_than_one_mole_formula_fit_correctly(datasets_db):
    # Test that phases that have more than one mole of formula units fit
    # correctly normalized parameters.
    # Might be missing some normalizations for some properties.
    raise NotImplementedError()

def test_molar_volume_model_fits(datasets_db):
    # test V0, VM unary, binary, and ternary parameters from the notebook work
    # TODO: performance of mixing models with lots of features is really bad
    #   (CPM, AbstractRKMProperty, VA parameters). Think if we can find a
    #   solution to that. Maybe some code in the feature generation to take only
    #   the first N features for mixing or something? Maybe limiting the number
    #   of non-mixing features to 2 to limit combinatorics? Can it be tuned by users?
    raise NotImplementedError()


def test_integration_of_qualified_fitting_descrption_in_schema():
    # test that using a fully qualified import path works to run ESPEI via the high level API in the schema
    raise NotImplementedError()