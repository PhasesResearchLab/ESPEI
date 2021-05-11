import numpy as np
import tinydb

from espei.core_utils import get_prop_data, filter_configurations, filter_temperatures, symmetry_filter
from espei.datasets import recursive_map
from espei.sublattice_tools import recursive_tuplify
from espei.utils import PickleableTinyDB, MemoryStorage
from espei.error_functions.non_equilibrium_thermochemical_error import get_prop_samples


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

    # The following lines replace "get_data" in a more functional form
    solver_qry = (tinydb.where('solver').test(symmetry_filter, configuration, recursive_tuplify(symmetry) if symmetry else symmetry))
    desired_data = get_prop_data(comps, phase_name, desired_props, datasets, additional_query=solver_qry)
    desired_data = filter_configurations(desired_data, configuration, symmetry)
    desired_data = filter_temperatures(desired_data)

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


def test_recursive_map():
    """Test that recursive map function works"""

    strings = [[["1.0"], ["5.5", "8.8"], ["10.7"]]]
    floats = [[[1.0], [5.5, 8.8], [10.7]]]

    assert recursive_map(float, strings) == floats
    assert recursive_map(str, floats) == strings
    assert recursive_map(float, "1.234") == 1.234
    assert recursive_map(int, ["1", "2", "5"]) == [1, 2, 5]
    assert recursive_map(float, ["1.0", ["0.5", "0.5"]]) == [1.0, [0.5, 0.5]]


def test_get_prop_samples_ravels_correctly():
    """get_prop_samples should ravel non-equilibrium thermochemical data correctly"""
    desired_data = [{
        "solver": {
            "sublattice_site_ratios": [1],
            "sublattice_occupancies": [[[0, 0]], [[1, 1]]],
            "sublattice_configurations": [[["CU", "MG"]], [["CU", "MG"]]],
            "mode": "manual"
        },
        "conditions": {
            "P": [0, 1], "T": [0, 1, 2, 3]},
        "values": [[[0, 1], [2, 3], [4, 5], [6, 7]], [[8, 9], [10, 11], [12, 13], [14, 15]]],
        "weights": None  # SET ME!
    }]

    calculate_dict = get_prop_samples(desired_data, [['CU', 'MG']])
    print(calculate_dict)
    # Unravel by (P, T, configs), where the left-most dimensions unravel the slowest
    assert np.all(calculate_dict['P'] == np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]))
    assert np.all(calculate_dict['T'] == np.array([0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 1, 1, 2, 2, 3, 3]))
    assert np.all(calculate_dict['points'] == np.array([[0, 0], [1, 1], [0, 0], [1, 1], [0, 0], [1, 1], [0, 0], [1, 1], [0, 0], [1, 1], [0, 0], [1, 1], [0, 0], [1, 1], [0, 0], [1, 1]]))
    assert np.all(calculate_dict['values'] == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]))


def test_get_prop_samples_broadcasts_weights_correctly():
    """get_prop_samples should broadcast weights to match the values"""
    desired_data = [{
        "solver": {
            "sublattice_site_ratios": [1],
            "sublattice_occupancies": [[[0, 0]], [[1, 1]]],
            "sublattice_configurations": [[["CU", "MG"]], [["CU", "MG"]]],
            "mode": "manual"
        },
        "conditions": {
            "P": [0, 1], "T": [0, 1, 2, 3]},
        "values": [[[0, 1], [2, 3], [4, 5], [6, 7]], [[8, 9], [10, 11], [12, 13], [14, 15]]],
    }]

    # No weight
    calculate_dict = get_prop_samples(desired_data, [['CU', 'MG']])
    assert calculate_dict["values"].shape == (16,)
    assert calculate_dict["values"].size == len(calculate_dict["weights"])
    assert np.all(np.isclose(np.asarray(calculate_dict["weights"]), 1.0))

    # Scalar weight
    desired_data[0]["weight"] = 5.0
    calculate_dict = get_prop_samples(desired_data, [['CU', 'MG']])
    assert calculate_dict["values"].shape == (16,)
    assert calculate_dict["values"].size == len(calculate_dict["weights"])
    assert np.all(np.isclose(np.asarray(calculate_dict["weights"]), 5.0))

    # 1D weights aligned in...
    # ... P
    desired_data[0]["weight"] = [[[1]], [[2]]]
    calculate_dict = get_prop_samples(desired_data, [['CU', 'MG']])
    print("P",calculate_dict)
    assert calculate_dict["values"].shape == (16,)
    assert calculate_dict["values"].size == len(calculate_dict["weights"])

    # ... T
    desired_data[0]["weight"] = [[[1], [2], [3], [4]]]
    calculate_dict = get_prop_samples(desired_data, [['CU', 'MG']])
    print("T",calculate_dict)
    assert calculate_dict["values"].shape == (16,)
    assert calculate_dict["values"].size == len(calculate_dict["weights"])

    # ... configs
    desired_data[0]["weight"] = [[[3, 4]]]
    calculate_dict = get_prop_samples(desired_data, [['CU', 'MG']])
    print("CONFIGS", calculate_dict)
    assert calculate_dict["values"].shape == (16,)
    assert calculate_dict["values"].size == len(calculate_dict["weights"])

    # 3D weights aligned
    num_P = 2
    num_T = 4
    prescribed = [[(np.array([1, 2])*i*j).tolist() for j in range(1, num_T+1)] for i in range(1, num_P+1)]
    desired_data[0]["weight"] = prescribed
    calculate_dict = get_prop_samples(desired_data, [['CU', 'MG']])
    print(calculate_dict)
    assert calculate_dict["values"].shape == (16,)
    assert calculate_dict["values"].size == len(calculate_dict["weights"])
