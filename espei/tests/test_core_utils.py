import numpy as np

from espei.core_utils import get_data
from espei.utils import PickleableTinyDB, MemoryStorage


def test_get_data_for_a_minimal_example():
    """Given a dataset and the congfiguration pertaining to that dataset, we should find the values."""
    SAMPLE_DATASET = {
        "components": ["CU", "MG", "VA"],
        "phases": ["LAVES_C15"],
        "solver": {
            "mode": "manual",
            "sublattice_site_ratios": [2, 1],
            "sublattice_configurations": [["CU", "MG"],
                                          ["MG", "CU"],
                                          ["MG", "MG"],
                                          ["CU", "CU"]]
        },
        "conditions": {
            "P": 101325,
            "T": 298.15
        },
        "output": "HM_FORM",
            "values":   [[[-15720, 34720, 7000, 15500]]]
    }
    datasets = PickleableTinyDB(storage=MemoryStorage)
    datasets.insert(SAMPLE_DATASET)
    comps = ['CU', 'MG', 'VA']
    phase_name = 'LAVES_C15'
    configuration = ('MG', 'CU')
    symmetry = None
    desired_props = ['HM_FORM']

    desired_data = get_data(comps, phase_name, configuration, symmetry, datasets, desired_props)
    assert len(desired_data) == 1
    desired_data = desired_data[0]
    assert desired_data['components'] == comps
    assert desired_data['phases'][0] == phase_name
    assert desired_data['solver']['sublattice_site_ratios'] == [2, 1]
    assert desired_data['solver']['sublattice_configurations'] == (('MG', 'CU'),)
    assert desired_data['conditions']['P'] == 101325
    assert desired_data['conditions']['T'] == 298.15
    assert desired_data['output'] == 'HM_FORM'
    assert desired_data['values'] == np.array([[[34720.0]]])
