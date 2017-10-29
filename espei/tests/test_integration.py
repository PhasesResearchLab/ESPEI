"""The test_integration module contains integration tests that ensure ESPEI behaves correctly"""

import os
import json
from tinydb import where

from espei.tests.fixtures import datasets_db
from espei.paramselect import fit

# TODO: clean up this test when the need for phase_models to be a file is decoupled from fitting
def test_mixing_energies_are_fit(datasets_db):
    """Tests that given mixing energy data, the excess parameter is fit."""
    phases_dict = {
        "components": ["AL", "B"],
        "refdata": "SGTE91",
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

    # create a dummy file
    dummy_phase_models = 'temp_test_mixing_energies.json'
    with open(dummy_phase_models, 'w') as fp:
        fp.write(json.dumps(phases_dict))

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

    dbf, sampler, parameters = fit(dummy_phase_models, datasets_db, run_mcmc=False)

    # clean up the temporary file we created
    os.remove(dummy_phase_models)

    assert dbf.elements == {'AL', 'B'}
    assert set(dbf.phases.keys()) == {'LIQUID', 'FCC_A1'}
    assert len(dbf._parameters.search(where('parameter_type') == 'L')) == 1

    # check that read/write is ok
    read_dbf = dbf.from_string(dbf.to_string(fmt='tdb'), fmt='tdb')
    assert read_dbf.elements == {'AL', 'B'}
    assert set(read_dbf.phases.keys()) == {'LIQUID', 'FCC_A1'}
    assert len(read_dbf._parameters.search(where('parameter_type') == 'L')) == 1

