import pytest
from espei.datasets import DatasetError, check_dataset

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


def test_check_datasets_run_on_good_data():
    """Passed valid datasets that should raise DatasetError."""
    check_dataset(dataset_single_valid)
    check_dataset(dataset_multi_valid)


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
