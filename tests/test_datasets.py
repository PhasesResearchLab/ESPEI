import pytest
import numpy as np
from espei.datasets import DatasetError, check_dataset, clean_dataset, apply_tags

from .testing_data import CU_MG_EXP_ACTIVITY, CU_MG_DATASET_THERMOCHEMICAL_STRING_VALUES, CU_MG_DATASET_ZPF_STRING_VALUES, LI_SN_LIQUID_DATA
from .fixtures import datasets_db

dataset_single_valid = {
    "components": ["AL", "NI", "VA"],
    "phases": ["BCC_B2"],
    "solver": {
        "sublattice_site_ratios": [0.5, 0.5, 1],
        "sublattice_occupancies": [[1, [0.25, 0.75], 1]],
        "sublattice_configurations": [["AL", ["AL", "NI"], "VA"]],
        "comment": "NiAl sublattice configuration (2SL)"
    },
    "conditions": {
        "P": 101325,
        "T": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
    },
    "output": "CPM_FORM",
    "values": [[[0],
                [-0.0173],
                [-0.01205],
                [0.12915],
                [0.24355],
                [0.13305],
                [-0.1617],
                [-0.51625],
                [-0.841],
                [-1.0975],
                [-1.28045],
                [-1.3997]]]
}

dataset_single_misaligned = {
    "components": ["AL", "NI", "VA"],
    "phases": ["BCC_B2"],
    "solver": {
        "sublattice_occupancies": [[1, [0.25, 0.75], 1]],
        "sublattice_site_ratios": [0.5, 0.5, 1],
        "sublattice_configurations": [["AL", ["AL", "NI"], "VA"]],
        "comment": "NiAl sublattice configuration (2SL)"
    },
    "conditions": {
        "P": 101325,
        "T": [0]
    },
    "output": "CPM_FORM",
    "values": [[[0],
                [-0.0173]]]
}

dataset_multi_valid = {
    "components": ["AL", "NI", "VA"],
    "phases": ["AL3NI2", "BCC_B2"],
    "conditions": {
        "P": 101325,
        "T": [1348, 1176, 977]
    },
    "output": "ZPF",
    "values": [
        [["AL3NI2", ["NI"], [0.4083]], ["BCC_B2", ["NI"], [None]]],
        [["AL3NI2", ["NI"], [0.4114]], ["BCC_B2", ["NI"], [0.4456]]],
        [["AL3NI2", ["NI"], [0.4114]], ["BCC_B2", ["NI"], [0.4532]]]
    ],
}

dataset_multi_valid_ternary = {
    "components": ["AL", "CR", "NI", "VA"],
    "phases": ["AL3NI2", "BCC_B2"],
    "conditions": {
        "P": 101325,
        "T": [1348, 1176, 977]
    },
    "output": "ZPF",
    "values": [
        [["AL3NI2", ["CR", "NI"], [0.2, 0.4083]], ["BCC_B2", ["CR", "NI"], [None, None]]],
        [["AL3NI2", ["CR", "NI"], [0.2, 0.4114]], ["BCC_B2", ["CR", "NI"], [0.2, 0.4456]]],
        [["AL3NI2", ["CR", "NI"], [0.2, 0.4114]], ["BCC_B2", ["CR", "NI"], [0.2, 0.4532]]]
    ],
}

dataset_multi_misaligned = {
    "components": ["AL", "NI"],
    "phases": ["AL3NI2", "BCC_B2"],
    "conditions": {
        "P": 101325,
        "T": [1348, 977]
    },
    "output": "ZPF",
    "values": [
        [["AL3NI2", ["NI"], [0.4114]], ["BCC_B2", ["NI"], [0.4532]]]
    ],
}

dataset_multi_incorrect_phases = {
    "components": ["AL", "NI"],
    "phases": ["AL3NI2", "BCC_A2"],
    "conditions": {
        "P": 101325,
        "T": [1348, 1176, 977]
    },
    "output": "ZPF",
    "values": [
        [["AL3NI2", ["NI"], [0.4083]], ["BCC_B2", ["NI"], [0.4340]]],
        [["AL3NI2", ["NI"], [0.4114]], ["BCC_B2", ["NI"], [0.4456]]],
        [["AL3NI2", ["NI"], [0.4114]], ["BCC_B2", ["NI"], [0.4532]]]
    ],
}

dataset_single_incorrect_components_underspecified = {
    "components": ["AL"],
    "phases": ["BCC_B2"],
    "solver": {
        "sublattice_site_ratios": [0.5, 0.5, 1],
        "sublattice_occupancies": [[1, [0.25, 0.75], 1]],
        "sublattice_configurations": [["AL", ["AL", "NI"], "VA"]],
        "comment": "NiAl sublattice configuration (2SL)"
    },
    "conditions": {
        "P": 101325,
        "T": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
    },
    "output": "CPM_FORM",
    "values": [[[0],
                [-0.0173],
                [-0.01205],
                [0.12915],
                [0.24355],
                [0.13305],
                [-0.1617],
                [-0.51625],
                [-0.841],
                [-1.0975],
                [-1.28045],
                [-1.3997]]]
}

dataset_single_incorrect_components_overspecified = {
    "components": ["AL", "NI", "VA", "FE"],
    "phases": ["BCC_B2"],
    "solver": {
        "sublattice_site_ratios": [0.5, 0.5, 1],
        "sublattice_occupancies": [[1, [0.25, 0.75], 1]],
        "sublattice_configurations": [["AL", ["AL", "NI"], "VA"]],
        "comment": "NiAl sublattice configuration (2SL)"
    },
    "conditions": {
        "P": 101325,
        "T": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
    },
    "output": "CPM_FORM",
    "values": [[[0],
                [-0.0173],
                [-0.01205],
                [0.12915],
                [0.24355],
                [0.13305],
                [-0.1617],
                [-0.51625],
                [-0.841],
                [-1.0975],
                [-1.28045],
                [-1.3997]]]
}

dataset_multi_incorrect_components_underspecified = {
    "components": ["NI"],
    "phases": ["AL3NI2", "BCC_B2"],
    "conditions": {
        "P": 101325,
        "T": [1348, 1176, 977]
    },
    "output": "ZPF",
    "values": [
        [["AL3NI2", ["NI"], [0.4083]], ["BCC_B2", ["NI"], [0.4340]]],
        [["AL3NI2", ["NI"], [0.4114]], ["BCC_B2", ["NI"], [0.4456]]],
        [["AL3NI2", ["NI"], [0.4114]], ["BCC_B2", ["NI"], [0.4532]]]
    ],
}

dataset_multi_incorrect_components_overspecified = {
    "components": ["AL", "NI", "FE"],
    "phases": ["AL3NI2", "BCC_B2"],
    "conditions": {
        "P": 101325,
        "T": [1348, 1176, 977]
    },
    "output": "ZPF",
    "values": [
        [["AL3NI2", ["NI"], [0.4083]], ["BCC_B2", ["NI"], [0.4340]]],
        [["AL3NI2", ["NI"], [0.4114]], ["BCC_B2", ["NI"], [0.4456]]],
        [["AL3NI2", ["NI"], [0.4114]], ["BCC_B2", ["NI"], [0.4532]]]
    ],
}

dataset_multi_malformed_zpfs_components_not_list = {
    "components": ["AL", "NI"],
    "phases": ["AL3NI2", "BCC_B2"],
    "conditions": {
        "P": 101325,
        "T": [1348, 977]
    },
    "output": "ZPF",
    "values": [
        [["AL3NI2", ["NI"], [0.4083]], ["BCC_B2", ["NI"], [0.4340]]],
        [["AL3NI2", ["NI"], [0.4114]], ["BCC_B2", "NI", [0.4532]]]
    ],
}

dataset_multi_malformed_zpfs_fractions_do_not_match_components = {
    "components": ["AL", "NI"],
    "phases": ["AL3NI2", "BCC_B2"],
    "conditions": {
        "P": 101325,
        "T": [1348, 977]
    },
    "output": "ZPF",
    "values": [
        [["AL3NI2", ["NI"], [0.4083]], ["BCC_B2", ["NI"], [0.4340]]],
        [["AL3NI2", ["NI"], [0.4114]], ["BCC_B2", ["NI"], [0.4532, 0.4532]]]
    ],
}

dataset_multi_malformed_zpfs_components_do_not_match_fractions = {
    "components": ["AL", "NI"],
    "phases": ["AL3NI2", "BCC_B2"],
    "conditions": {
        "P": 101325,
        "T": [1348, 977]
    },
    "output": "ZPF",
    "values": [
        [["AL3NI2", ["NI"], [0.4083]], ["BCC_B2", ["NI"], [0.4340]]],
        [["AL3NI2", ["NI"], [0.4114]], ["BCC_B2", ["NI", "AL"], [0.4532, ]]]
    ],
}

dataset_single_malformed_site_occupancies = {
    "components": ["AL", "NI", "VA"],
    "phases": ["BCC_B2"],
    "solver": {
        "sublattice_site_ratios": [0.5, 0.5, 1],
        "sublattice_occupancies": [[1, 0.75, 1]],
        "sublattice_configurations": [["AL", ["AL", "NI"], "VA"]],
        "comment": "NiAl sublattice configuration (2SL)"
    },
    "conditions": {
        "P": 101325,
        "T": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
    },
    "output": "CPM_FORM",
    "values": [[[0],
                [-0.0173],
                [-0.01205],
                [0.12915],
                [0.24355],
                [0.13305],
                [-0.1617],
                [-0.51625],
                [-0.841],
                [-1.0975],
                [-1.28045],
                [-1.3997]]]
}

dataset_single_malformed_site_ratios = {
    "components": ["AL", "NI", "VA"],
    "phases": ["BCC_B2"],
    "solver": {
        "sublattice_site_ratios": [0.5, 0.5, 1, 1],
        "sublattice_occupancies": [[1, [0.25, 0.75], 1]],
        "sublattice_configurations": [["AL", ["AL", "NI"], "VA"]],
        "comment": "NiAl sublattice configuration (2SL)"
    },
    "conditions": {
        "P": 101325,
        "T": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
    },
    "output": "CPM_FORM",
    "values": [[[0],
                [-0.0173],
                [-0.01205],
                [0.12915],
                [0.24355],
                [0.13305],
                [-0.1617],
                [-0.51625],
                [-0.841],
                [-1.0975],
                [-1.28045],
                [-1.3997]]]
}

dataset_multi_mole_fractions_as_percents = {
    "components": ["AL", "NI", "VA"],
    "phases": ["AL3NI2", "BCC_B2"],
    "conditions": {
        "P": 101325,
        "T": [1348, 1176, 977]
    },
    "output": "ZPF",
    "values": [
        [["AL3NI2", ["NI"], [40.83]], ["BCC_B2", ["NI"], [None]]], # mole fraction is a percent
        [["AL3NI2", ["NI"], [0.4114]], ["BCC_B2", ["NI"], [0.4456]]],
        [["AL3NI2", ["NI"], [0.4114]], ["BCC_B2", ["NI"], [0.4532]]]
    ],
}

dataset_single_unsorted_interaction = {
    "components": ["AL", "NI", "VA"],
    "phases": ["BCC_B2"],
    "solver": {
        "sublattice_site_ratios": [0.5, 0.5, 1],
        "sublattice_occupancies": [[1, [0.75, 0.25], 1]],
        "sublattice_configurations": [["AL", ["NI", "AL"], "VA"]],  # NI comes before AL, which causes a sorting bug
        "comment": "NiAl sublattice configuration (2SL)"
    },
    "conditions": {
        "P": 101325,
        "T": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
    },
    "output": "CPM_FORM",
    "values": [[[0],
                [-0.0173],
                [-0.01205],
                [0.12915],
                [0.24355],
                [0.13305],
                [-0.1617],
                [-0.51625],
                [-0.841],
                [-1.0975],
                [-1.28045],
                [-1.3997]]]
}


dataset_mismatched_configs_occupancies = {
# 2 configurations (values) 1 occupancy
    "components": ["AL", "NI", "VA"],
    "phases": ["BCC_B2"],
    "solver": {
        "sublattice_site_ratios": [0.5, 0.5, 1],
        "sublattice_occupancies": [
        [1, [0.25, 0.75], 1]
        ],
        "sublattice_configurations": [
        ["AL", ["AL", "NI"], "VA"],
        ["AL", ["AL", "NI"], "VA"]
        ],
        "comment": "NiAl sublattice configuration (2SL)"
    },
    "conditions": {
        "P": 101325,
        "T": [0, 10]
    },
    "output": "CPM_FORM",
    "values": [[[0, 1], [-0.0173, -1.0173]]]
}


def test_check_datasets_run_on_good_data():
    """Passed valid datasets that should raise DatasetError."""
    check_dataset(dataset_single_valid)
    check_dataset(dataset_multi_valid)
    check_dataset(dataset_multi_valid_ternary)


def test_check_datasets_raises_on_misaligned_data():
    """Passed datasets that have misaligned data and conditions should raise DatasetError."""
    with pytest.raises(DatasetError):
        check_dataset(dataset_single_misaligned)
    with pytest.raises(DatasetError):
        check_dataset(dataset_multi_misaligned)


def test_check_datasets_raises_with_incorrect_zpf_phases():
    """Passed datasets that have incorrect phases entered than used should raise."""
    with pytest.raises(DatasetError):
        check_dataset(dataset_multi_incorrect_phases)


def test_check_datasets_raises_with_incorrect_components():
    """Passed datasets that have incorrect components entered vs. used should raise."""
    with pytest.raises(DatasetError):
        check_dataset(dataset_single_incorrect_components_overspecified)
    with pytest.raises(DatasetError):
        check_dataset(dataset_single_incorrect_components_underspecified)
    with pytest.raises(DatasetError):
        check_dataset(dataset_multi_incorrect_components_overspecified)
    with pytest.raises(DatasetError):
        check_dataset(dataset_multi_incorrect_components_underspecified)


def test_check_datasets_raises_with_malformed_zpf():
    """Passed datasets that have malformed ZPF values should raise."""
    with pytest.raises(DatasetError):
        check_dataset(dataset_multi_malformed_zpfs_components_not_list)
    with pytest.raises(DatasetError):
        check_dataset(dataset_multi_malformed_zpfs_fractions_do_not_match_components)
    with pytest.raises(DatasetError):
        check_dataset(dataset_multi_malformed_zpfs_components_do_not_match_fractions)


def test_check_datasets_raises_with_malformed_sublattice_configurations():
    """Passed datasets that have malformed ZPF values should raise."""
    with pytest.raises(DatasetError):
        check_dataset(dataset_single_malformed_site_occupancies)
    with pytest.raises(DatasetError):
       check_dataset(dataset_single_malformed_site_ratios)


def test_check_datasets_works_on_activity_data():
    """Passed activity datasets should work correctly."""
    check_dataset(CU_MG_EXP_ACTIVITY)


def test_check_datasets_raises_with_zpf_fractions_greater_than_one():
    """Passed datasets that have mole fractions greater than one should raise."""
    with pytest.raises(DatasetError):
        check_dataset(dataset_multi_mole_fractions_as_percents)


def test_check_datasets_raises_with_unsorted_interactions():
    """Passed datasets that have sublattice interactions not in sorted order should raise."""
    with pytest.raises(DatasetError):
        check_dataset(dataset_single_unsorted_interaction)


def test_datasets_convert_thermochemical_string_values_producing_correct_value(datasets_db):
    """Strings where floats are expected should give correct answers for thermochemical datasets"""
    ds = clean_dataset(CU_MG_DATASET_THERMOCHEMICAL_STRING_VALUES)
    assert np.issubdtype(np.array(ds['values']).dtype, np.number)
    assert np.issubdtype(np.array(ds['conditions']['T']).dtype, np.number)
    assert np.issubdtype(np.array(ds['conditions']['P']).dtype, np.number)


def test_datasets_convert_zpf_string_values_producing_correct_value(datasets_db):
    """Strings where floats are expected should give correct answers for ZPF datasets"""
    ds = clean_dataset(CU_MG_DATASET_ZPF_STRING_VALUES)
    assert np.issubdtype(np.array([t[0][2] for t in ds['values']]).dtype, np.number)
    assert np.issubdtype(np.array(ds['conditions']['T']).dtype, np.number)
    assert np.issubdtype(np.array(ds['conditions']['P']).dtype, np.number)

def test_check_datasets_raises_if_configs_occupancies_not_aligned(datasets_db):
    """Checking datasets that don't have the same number/shape of configurations/occupancies should raise."""
    with pytest.raises(DatasetError):
        check_dataset(dataset_mismatched_configs_occupancies)


# Expected to fail, since the dataset checker cannot determine that species are used in the configurations and components should only contain pure elements.
@pytest.mark.xfail
def test_non_equilibrium_thermo_data_with_species_passes_checker():
    """Non-equilibrium thermochemical data that use species in the configurations should pass the dataset checker.
    """
    check_dataset(LI_SN_LIQUID_DATA)


def test_applying_tags(datasets_db):
    """Test that applying tags updates the appropriate values"""
    dataset = clean_dataset(CU_MG_DATASET_THERMOCHEMICAL_STRING_VALUES)
    # overwrite tags for this test
    dataset["tags"] = ["testtag"]
    datasets_db.insert(dataset)
    assert len(datasets_db.all()) == 1
    assert "newkey_from_tag" not in datasets_db.all()[0]
    apply_tags(datasets_db, {"testtag": {"newkey_from_tag": ["tag", "values"]}})
    assert len(datasets_db.all()) == 1
    assert "newkey_from_tag" in datasets_db.all()[0]
    assert datasets_db.all()[0]["newkey_from_tag"] == ["tag", "values"]
