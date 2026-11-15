from copy import deepcopy
import json, re
import pytest
import numpy as np
from espei.datasets import DatasetError, DatasetWarning, validate_dataset, apply_tags, load_datasets, check_dataset, clean_dataset
from espei.datasets import FixedConfigurationDataset, EquilibriumPropertyDataset, ActivityDataset, ZPFDataset
from espei.espei_script import main as espei_main
from espei.phase_models import PhaseModelSpecification

from .testing_data import CU_MG_EXP_ACTIVITY, CU_MG_DATASET_THERMOCHEMICAL_STRING_VALUES, CU_MG_DATASET_ZPF_STRING_VALUES, LI_SN_LIQUID_DATA, dataset_multi_valid_ternary
from .fixtures import datasets_db


def load(tmp_path, *datasets, **kwargs):
    """Load datasets from JSON files like ESPEI does."""
    paths = [tmp_path / f"{i}.json" for i in range(len(datasets))]
    for path, dataset in zip(paths, datasets):
        path.write_text(json.dumps(dataset))
    return load_datasets(list(map(str, paths)), **kwargs).all()

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

dataset_zpf_negative_mole_fraction = {
    "components": ["AL", "NI", "VA"],
    "phases": ["AL3NI2", "BCC_B2"],
    "conditions": {
        "P": 101325,
        "T": [1348]
    },
    "output": "ZPF",
    "values": [
        [["AL3NI2", ["NI"], [-0.5]], ["BCC_B2", ["NI"], [None]]], # mole fraction is negative
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
    validate_dataset(dataset_single_valid)
    validate_dataset(dataset_multi_valid)
    validate_dataset(dataset_multi_valid_ternary)


def test_check_datasets_raises_on_misaligned_data():
    """Passed datasets that have misaligned data and conditions should raise DatasetError."""
    with pytest.raises(DatasetError):
        validate_dataset(dataset_single_misaligned)
    with pytest.raises(DatasetError):
        validate_dataset(dataset_multi_misaligned)


def test_check_datasets_raises_with_incorrect_zpf_phases():
    """Passed datasets that have incorrect phases entered than used should raise."""
    with pytest.raises(DatasetError):
        validate_dataset(dataset_multi_incorrect_phases)


def test_check_datasets_raises_with_incorrect_components():
    """Passed datasets that have incorrect components entered vs. used should raise."""
    with pytest.raises(DatasetError):
        validate_dataset(dataset_single_incorrect_components_overspecified)
    with pytest.raises(DatasetError):
        # without the phase models, NI can't be told apart from a species
        validate_dataset(dataset_single_incorrect_components_underspecified, species=PhaseModelSpecification(components=["AL", "NI", "VA"], phases={}).get_species())
    with pytest.raises(DatasetError):
        validate_dataset(dataset_multi_incorrect_components_overspecified)
    with pytest.raises(DatasetError):
        validate_dataset(dataset_multi_incorrect_components_underspecified)

    # equilibrium datasets underspecified
    ds_eq_underspecified = {
    "components": ["NI"],
    "phases": ["LIQUID"],
    "conditions": {
        "P": 101325,
        "T": [1348, 1176, 977],
        "X_NI": 0.5
    },
    "output": "HM",
    "values": [[[-1000], [-900], [-800]]]
    }
    with pytest.raises(DatasetError):
        validate_dataset(ds_eq_underspecified)

    # equilibrium datasets overspecified
    ds_eq_overspecified = {
    "components": ["CU", "MG", "NI"],
    "phases": ["LIQUID"],
    "conditions": {
        "P": 101325,
        "T": [1348, 1176, 977],
        "X_NI": 0.5
    },
    "output": "HM",
    "values": [[[-1000], [-900], [-800]]]
    }
    with pytest.raises(DatasetError):
        validate_dataset(ds_eq_overspecified)


def test_check_datasets_raises_with_malformed_zpf():
    """Passed datasets that have malformed ZPF values should raise."""
    with pytest.raises(DatasetError):
        validate_dataset(dataset_multi_malformed_zpfs_components_not_list)
    with pytest.raises(DatasetError):
        validate_dataset(dataset_multi_malformed_zpfs_fractions_do_not_match_components)
    with pytest.raises(DatasetError):
        validate_dataset(dataset_multi_malformed_zpfs_components_do_not_match_fractions)


def test_check_datasets_raises_with_malformed_sublattice_configurations():
    """Passed datasets that have malformed ZPF values should raise."""
    with pytest.raises(DatasetError):
        validate_dataset(dataset_single_malformed_site_occupancies)
    with pytest.raises(DatasetError):
       validate_dataset(dataset_single_malformed_site_ratios)


def test_check_datasets_raises_with_equilibrium_conditions_and_values_shapes_mismatch():
    """Passed equilibrium datasets that have mismatched condition and values shapes should raise."""
    COND_VALS_SHAPE_GOOD = {
        "components": ["CU", "MG"],
        "phases": ["LIQUID"],
        "conditions": {"P": [101325, 1e5], "T": [1400, 1500, 1600], "X_MG": [0.5, 0.6, 0.7, 0.8]},
        "reference_states": {
            "CU": {"phase": "LIQUID"},
            "MG": {"phase": "LIQUID"}
        },
        "output": "HMR",
        "values": np.zeros((2, 3, 4)).tolist(),
        "reference": "equilibrium thermochemical tests",
    }
    # Good shape should not raise
    validate_dataset(COND_VALS_SHAPE_GOOD)

    COND_VALS_SHAPE_DISAGREEMENT_1_1_2 = {
        "components": ["CU", "MG", "NI"],
        "phases": ["LIQUID"],
        "conditions": {"P": 101325, "T": [1400], "X_MG": [0.5, 0.6], "X_NI": [0.5, 0.6]},
        "reference_states": {
            "CU": {"phase": "LIQUID"},
            "MG": {"phase": "LIQUID"},
            "NI": {"phase": "LIQUID"}
        },
        "output": "HMR",
        "values": [[[0]]],
        "reference": "equilibrium thermochemical tests",
    }
    with pytest.raises(DatasetError):
        validate_dataset(COND_VALS_SHAPE_DISAGREEMENT_1_1_2)

    COND_VALS_SHAPE_DISAGREEMENT_1_2_2 = {
        "components": ["CU", "MG"],
        "phases": ["LIQUID"],
        "conditions": {"P": 101325, "T": [1400, 1500], "X_MG": [0.5, 0.6]},
        "reference_states": {
            "CU": {"phase": "LIQUID"},
            "MG": {"phase": "LIQUID"}
        },
        "output": "HMR",
        "values": [[[0, 0]]],
        "reference": "equilibrium thermochemical tests",
    }
    with pytest.raises(DatasetError):
        validate_dataset(COND_VALS_SHAPE_DISAGREEMENT_1_2_2)

    # we don't broadcast over compositions, so composition conditions shapes need to match
    MISMATCHED_COMPOSITION_CONDS = {
        "components": ["CU", "MG", "NI"],
        "phases": ["LIQUID"],
        "conditions": {"P": 101325, "T": [1400], "X_MG": [0.5, 0.6], "X_NI": [0.5]},
        "reference_states": {
            "CU": {"phase": "LIQUID"},
            "MG": {"phase": "LIQUID"},
            "NI": {"phase": "LIQUID"}
        },
        "output": "HMR",
        "values": [[[0, 0]]],
        "reference": "equilibrium thermochemical tests",
    }
    with pytest.raises(DatasetError):
        validate_dataset(MISMATCHED_COMPOSITION_CONDS)


def test_check_datasets_works_on_activity_data():
    """Passed activity datasets should work correctly."""
    validate_dataset(CU_MG_EXP_ACTIVITY)


def test_check_datasets_raises_with_zpf_fractions_greater_than_one():
    """Passed datasets that have mole fractions greater than one should raise."""
    with pytest.raises(DatasetError):
        validate_dataset(dataset_multi_mole_fractions_as_percents)


def test_check_datasets_raises_with_negative_zpf_fractions():
    """Passed datasets that have negative mole fractions should raise."""
    with pytest.raises(DatasetError):
        validate_dataset(dataset_zpf_negative_mole_fraction)


def test_check_datasets_raises_with_unsorted_interactions():
    """Passed datasets that have sublattice interactions not in sorted order should raise."""
    with pytest.raises(DatasetError):
        validate_dataset(dataset_single_unsorted_interaction)


def test_datasets_convert_thermochemical_string_values_producing_correct_value(tmp_path):
    """Strings where floats are expected should give correct answers for thermochemical datasets"""
    ds, = load(tmp_path, CU_MG_DATASET_THERMOCHEMICAL_STRING_VALUES)
    assert np.issubdtype(np.array(ds['values']).dtype, np.number)
    assert np.issubdtype(np.array(ds['conditions']['T']).dtype, np.number)
    assert np.issubdtype(np.array(ds['conditions']['P']).dtype, np.number)


def test_datasets_convert_zpf_string_values_producing_correct_value(tmp_path):
    """Strings where floats are expected should give correct answers for ZPF datasets"""
    ds, = load(tmp_path, CU_MG_DATASET_ZPF_STRING_VALUES)
    assert np.issubdtype(np.array([t[0][2] for t in ds['values']]).dtype, np.number)
    assert np.issubdtype(np.array(ds['conditions']['T']).dtype, np.number)
    assert np.issubdtype(np.array(ds['conditions']['P']).dtype, np.number)

def test_check_datasets_raises_if_configs_occupancies_not_aligned(datasets_db):
    """Checking datasets that don't have the same number/shape of configurations/occupancies should raise."""
    with pytest.raises(DatasetError):
        validate_dataset(dataset_mismatched_configs_occupancies)


LI_SN_SPECIES = PhaseModelSpecification(components=["LI", "SN"], species={"LI4SN": {"constituents": {"LI": 4, "SN": 1}}}, phases={}).get_species()


@pytest.mark.parametrize("dataset", [
    LI_SN_LIQUID_DATA,
    {**LI_SN_LIQUID_DATA, "components": ["CU", "MG"]},  # outside the system, so species aren't resolved
])
def test_non_equilibrium_thermo_data_with_species_passes_checker(dataset):
    """Non-equilibrium thermochemical data that use species in the configurations should pass the dataset checker."""
    validate_dataset(dataset, species=LI_SN_SPECIES)


@pytest.mark.parametrize("dataset, species, message", [
    ({**LI_SN_LIQUID_DATA, "components": ["LI", "LI4SN", "SN"]}, LI_SN_SPECIES, "are species"),
    ({**LI_SN_LIQUID_DATA, "solver": {**LI_SN_LIQUID_DATA["solver"], "sublattice_configurations": [[["LI", "LI5SN"]], [["LI5SN", "SN"]]]}}, LI_SN_SPECIES, "LI5SN is not defined"),
])
def test_non_equilibrium_thermo_data_with_invalid_species_raises(dataset, species, message):
    """Species in configurations must be defined, and species can't be components."""
    with pytest.raises(DatasetError, match=message):
        validate_dataset(dataset, species=species)


def test_species_without_phase_models_are_reported_not_checked(tmp_path):
    """Without phase models, species in configurations give a DatasetWarning, and the other checks still run."""
    endmember = {**LI_SN_LIQUID_DATA, "solver": {"mode": "manual", "sublattice_site_ratios": [1], "sublattice_configurations": [["LI4SN"]]}, "values": [[[0]]]}
    with pytest.warns(DatasetWarning, match="LI4SN.*--phase-models"):
        validate_dataset(endmember)
    with pytest.raises(DatasetError, match="sorted"):
        validate_dataset({**LI_SN_LIQUID_DATA, "solver": {**LI_SN_LIQUID_DATA["solver"], "sublattice_configurations": [[["LI4SN", "LI"]], [["LI4SN", "SN"]]]}})
    path = tmp_path / "li-sn.json"
    path.write_text(json.dumps(LI_SN_LIQUID_DATA))
    with pytest.warns(DatasetWarning, match="li-sn.json: Species LI4SN"):
        load_datasets([str(path)])


def test_applying_tags(datasets_db):
    """Test that applying tags updates the appropriate values"""
    dataset = deepcopy(CU_MG_DATASET_THERMOCHEMICAL_STRING_VALUES)
    # overwrite tags for this test
    dataset["tags"] = ["testtag"]
    datasets_db.insert(dataset)
    assert len(datasets_db.all()) == 1
    assert "newkey_from_tag" not in datasets_db.all()[0]
    apply_tags(datasets_db, {"testtag": {"newkey_from_tag": ["tag", "values"]}})
    assert len(datasets_db.all()) == 1
    assert "newkey_from_tag" in datasets_db.all()[0]
    assert datasets_db.all()[0]["newkey_from_tag"] == ["tag", "values"]


# Datasets from ESPEI-datasets (Cu-Mg), Kawin (Cu-Ni mobility) and H in LaNi5
HM_FORM_CUMG2 = {"components": ["CU", "MG"], "phases": ["CUMG2"], "conditions": {"P": 101325.0, "T": 299.0}, "output": "HM_FORM", "values": [[[-9539.5]]], "reference": "King1964",
                 "solver": {"mode": "manual", "sublattice_site_ratios": [1.0, 2.0], "sublattice_configurations": [["CU", "MG"]]}}
HM_MIX_FCC = {"components": ["CU", "MG", "VA"], "phases": ["FCC_A1"], "conditions": {"P": 101325.0, "T": 298.15}, "output": "HM_MIX", "values": [[[-566.0, -1944.0, -3611.0]]], "reference": "Gao2014first",
              "solver": {"mode": "manual", "sublattice_site_ratios": [1.0, 1.0], "sublattice_configurations": [[["CU", "MG"], "VA"]] * 3,
                         "sublattice_occupancies": [[[0.96875, 0.03125], 1.0], [[0.75, 0.25], 1.0], [[0.5, 0.5], 1.0]]}}
HM_MIX_LIQUID = {"components": ["CU", "MG"], "phases": ["LIQUID"], "conditions": {"P": 101325.0, "T": 1100.0, "X_MG": [0.2, 0.4]}, "output": "HM_MIX", "values": [[[-3070.0, -4310.0]]], "reference": "sommer1983calorimetric_investigations"}
ACR_MG = {"components": ["CU", "MG", "VA"], "phases": ["LIQUID"], "reference_state": {"phases": ["LIQUID"], "conditions": {"P": 101325.0, "T": 1200.0, "X_CU": 0.0}},
          "conditions": {"P": 101325.0, "T": 1200.0, "X_CU": [0.9, 0.8]}, "output": "ACR_MG", "values": [[[0.0057, 0.0264]]], "reference": "garg1973thermodynamic"}
ZPF_C15_LIQUID = {"components": ["CU", "MG", "VA"], "phases": ["LIQUID", "LAVES_C15"], "conditions": {"P": 101325.0, "T": [1056.11423, 1062.70164]}, "output": "ZPF", "reference": "Bagnoud1978",
                  "values": [[["LIQUID", ["MG"], [0.29042]], ["LAVES_C15", ["MG"], [None]]], [["LIQUID", ["MG"], [0.30246]], ["LAVES_C15", ["MG"], [None], "disordered"]]]}
TRACER_D0_CU = {"components": ["CU", "NI", "VA"], "phases": ["FCC_A1"], "conditions": {"P": 101325, "T": 1273}, "output": "TRACER_D0_CU", "values": [[[1.87e-4, 2.16e-4]]], "reference": "Monma1964",
                "solver": {"mode": "manual", "sublattice_site_ratios": [1, 3], "sublattice_configurations": [[["CU", "NI"], "VA"]] * 2, "sublattice_occupancies": [[[0.15, 0.85], 1], [[0.50, 0.50], 1]]}}
INTER_DIFF = {"components": ["CU", "NI", "VA"], "phases": ["FCC_A1"], "ref_el": ["CU"], "dependent_el": ["NI", "NI"], "conditions": {"P": 101325, "T": 1273, "X_CU": [0.047, 0.101]},
              "output": "INTER_DIFF", "values": [[[1.546e-15, 1.67e-15]]], "reference": "Heumann1972"}
H_IN_LANI5 = {"components": ["LN", "H", "VA"], "phases": ["LAN5H7", "GAS"], "conditions": {"T": [293, 293], "P": [1103873.4, 954120.72], "N": [1.0, 1.0]}, "output": "ZPF", "reference": "Shang (2025)",
              "values": [[["LAN5H7", ["H"], [0.5]], ["GAS", ["H"], [None]]], [["LAN5H7", ["H"], [0.4991]], ["GAS", ["H"], [None]]]]}


def test_load_datasets_returns_the_datasets_as_written(tmp_path):
    """Loaded datasets keep the keys and values of their files and get no defaults (gh-269)."""
    datasets = [
        {**HM_MIX_FCC, "weight": [0.5, 1.0, 2.0], "comment": "fully relaxed", "solver": {**HM_MIX_FCC["solver"], "comment": "SQS"}},
        {**HM_FORM_CUMG2, "excluded_model_contributions": ["idmix"], "tags": ["dft"]},
        {**HM_MIX_LIQUID, "weight": 2.0, "bibtex": "sommer1983", "dataset_author": "bocklund", "primary_source": ["a", "b"]},
        ACR_MG,
        ZPF_C15_LIQUID,
    ]
    assert load(tmp_path, *datasets) == datasets


@pytest.mark.parametrize("dataset", [
    pytest.param({**HM_FORM_CUMG2, "solver": {k: x for k, x in HM_FORM_CUMG2["solver"].items() if k != "mode"}}, id="fixed configuration without solver mode"),
    pytest.param({k: x for k, x in ZPF_C15_LIQUID.items() if k != "reference"}, id="ZPF without reference"),
    pytest.param({k: x for k, x in ACR_MG.items() if k != "reference"}, id="activity without reference"),
    pytest.param({**ZPF_C15_LIQUID, "conditions": {"P": 101325.0, "T": 1060.0}}, id="ZPF with one condition for every phase region"),
    pytest.param(H_IN_LANI5, id="ZPF with a pressure for each phase region"),
    pytest.param({**HM_MIX_LIQUID, "output": "HMR", "reference_states": {"CU": {"phase": "LIQUID"}, "MG": {"phase": "LIQUID"}, "NI": {"phase": "FCC_A1"}}}, id="reference states of more components"),
    pytest.param({"components": ["CU"], "phases": ["LIQUID"], "conditions": {"P": 101325, "T": [1400, 1500]}, "output": "HM", "values": [[[48262.0], [51399.0]]]}, id="equilibrium property of one component"),
    pytest.param(TRACER_D0_CU, id="Kawin tracer diffusivity"),
    pytest.param(INTER_DIFF, id="Kawin interdiffusivity"),
])
def test_datasets_load_in_every_supported_form(tmp_path, dataset):
    """Datasets load in every form the code using them supports (gh-269)."""
    load(tmp_path, dataset)


@pytest.mark.parametrize("dataset, location, rule", [
    pytest.param({**HM_MIX_LIQUID, "conditions": {"T": 1100.0, "X_MG": [0.2, 0.4]}}, "", "conditions must include T and P", id="missing pressure"),
    pytest.param({**HM_FORM_CUMG2, "phases": ["CUMG2", "LAVES_C15"]}, "phases", "", id="fixed configuration data of two phases"),
    pytest.param({**HM_FORM_CUMG2, "tags": "dft"}, "tags", "", id="tags not a list"),
    pytest.param({**HM_MIX_FCC, "weight": [1.0, 2.0]}, "", "weight must be positive and broadcast to the shape of the values (1, 1, 3)", id="weights not one per configuration"),
    pytest.param({**HM_MIX_FCC, "weight": -1.0}, "", "weight must be positive", id="negative weight"),
    pytest.param({**HM_MIX_FCC, "solver": {**HM_MIX_FCC["solver"], "sublattice_occupancies": [[[0.75, 0.75], 1.0]] * 3}}, "", "sum to one", id="mixing occupancies not summing to one"),
    pytest.param({**HM_FORM_CUMG2, "solver": {**HM_FORM_CUMG2["solver"], "sublattice_occupancies": [[0.5, 1.0]]}}, "", "sum to one", id="occupancy of one species not one"),
    pytest.param({**HM_MIX_FCC, "solver": {**HM_MIX_FCC["solver"], "sublattice_occupancies": [[[0.5, 0.5], [1.0]]] * 3}}, "", "must have the same shape as the sublattice VA", id="occupancies not shaped like the configuration"),
    pytest.param({**HM_MIX_FCC, "solver": {**HM_MIX_FCC["solver"], "sublattice_configurations": [[["MG", "CU"], "VA"]] * 3}}, "", "must be sorted and unique", id="unsorted mixing sublattice"),
    pytest.param({**HM_MIX_LIQUID, "conditions": {**HM_MIX_LIQUID["conditions"], "W_MG": [0.1, 0.2]}}, "", "conditions must be T, P, N or X_<component>", id="mass fraction condition"),
    pytest.param({**HM_MIX_LIQUID, "conditions": {"P": 101325.0, "T": 1100.0}, "values": [[[-3070.0]]]}, "", "leaving one dependent component", id="binary without X_ condition"),
    pytest.param({**HM_MIX_LIQUID, "components": ["CU", "MG", "NI"], "conditions": {**HM_MIX_LIQUID["conditions"], "X_NI": [0.1]}}, "", "X_ conditions must have the same number of values", id="X_ conditions of different lengths"),
    pytest.param({**HM_MIX_LIQUID, "reference_states": {"CU": {"phase": "LIQUID"}, "MG": {"phase": "LIQUID"}}}, "", 'ending in "R"', id="reference states of an unreferenced output"),
    pytest.param({**HM_MIX_LIQUID, "output": "HMR", "reference_states": {"CU": {"phase": "LIQUID"}}}, "", "reference_states must be given for every component, missing ['MG']", id="reference states missing a component"),
    pytest.param({**ACR_MG, "output": "ACR_NI"}, "", 'output must be "ACR_<component>"', id="activity of a component not in the components"),
    pytest.param({**ZPF_C15_LIQUID, "phases": ["LIQUID"]}, "", "must be the phases used in the values ['LAVES_C15', 'LIQUID']", id="ZPF phase not listed"),
    pytest.param({**ZPF_C15_LIQUID, "values": [ZPF_C15_LIQUID["values"][0], [["LIQUID", ["MG"], [0.3]], ["LAVES_C15", ["MG"], [0.3], "ordered"]]]}, "values[1][1].flag", "", id="ZPF flag not disordered"),
    pytest.param({**ZPF_C15_LIQUID, "values": [ZPF_C15_LIQUID["values"][0], [["LIQUID", ["MG"], [None]], ["LAVES_C15", ["MG"], [None]]]]}, "", "values[1] needs a phase composition with known mole fractions", id="ZPF phase region without known composition"),
    pytest.param({**ZPF_C15_LIQUID, "values": [ZPF_C15_LIQUID["values"][0], [["__HYPERPLANE__", ["MG"], [None]], ["LIQUID", ["MG"], [0.3]], ["LAVES_C15", ["MG"], [None]]]]}, "values[1][0]", "__HYPERPLANE__", id="ZPF hyperplane with unknown composition"),
    pytest.param({**ZPF_C15_LIQUID, "conditions": {"P": 101325.0, "T": [1056.0, 1060.0, 1062.0]}}, "", "one value per phase region (2)", id="ZPF temperatures not one per phase region"),
])
def test_invalid_datasets_raise_with_the_file_location_and_rule(tmp_path, dataset, location, rule):
    """Invalid datasets raise an error with the file, the location in the dataset and the rule (gh-269)."""
    with pytest.raises(DatasetError) as error:
        load(tmp_path, dataset)
    header, *errors = str(error.value).split("\n")
    assert header.startswith(f"Dataset Error in {tmp_path / '0.json'}: ")
    assert any(e.startswith(f"  {location}: " if location else "  ") and rule in e for e in errors)


def test_dataset_errors_list_every_error_with_its_location_and_value(tmp_path):
    """All errors of a dataset are reported at once, each with its location and the given value (gh-269)."""
    zpf = {**ZPF_C15_LIQUID, "phases": "LIQUID", "values": [[["LIQUID", ["MG"], ["0.3x"]], ["LAVES_C15", "MG", [None]]]]}
    with pytest.raises(DatasetError) as error:
        load(tmp_path, zpf)
    header, *errors = str(error.value).split("\n")
    assert header.endswith("3 error(s) in ZPFDataset:")
    assert [e.split(": ")[0].strip() for e in errors] == ["phases", "values[0][0].fractions[0]", "values[0][1].components"]
    assert [e.endswith(f"(got {value!r})") for e, value in zip(errors, ["LIQUID", "0.3x", "MG"])] == [True] * 3


def test_disabled_datasets_are_not_checked(tmp_path):
    """Disabled datasets are skipped without being checked, unless they are included."""
    disabled = {**HM_FORM_CUMG2, "disabled": True, "values": [[[-9539.5, 0.0]]]}
    assert load(tmp_path, disabled, HM_MIX_LIQUID) == [HM_MIX_LIQUID]
    with pytest.raises(DatasetError):
        load(tmp_path, disabled, HM_MIX_LIQUID, include_disabled=True)


def test_check_datasets_reports_every_invalid_dataset(tmp_path, capsys, monkeypatch):
    """espei --check-datasets reports the errors of every invalid dataset and exits with 1 (gh-269)."""
    paths = [tmp_path / name for name in ("a-valid.json", "b-no-configurations.json", "c-misaligned.json")]
    no_configurations = {**HM_FORM_CUMG2, "solver": {"mode": "manual", "sublattice_site_ratios": [1, 2]}}
    for path, dataset in zip(paths, [HM_FORM_CUMG2, no_configurations, {**ZPF_C15_LIQUID, "conditions": {"P": 101325.0, "T": [1056.0, 1060.0, 1062.0]}}]):
        path.write_text(json.dumps(dataset))
    monkeypatch.setattr("sys.argv", ["espei", "--check-datasets", str(tmp_path)])
    assert espei_main() == 1
    output = capsys.readouterr().out
    assert "b-no-configurations.json" in output and "solver.sublattice_configurations" in output
    assert "c-misaligned.json" in output and "a-valid.json" not in output


def test_deprecated_check_dataset_and_clean_dataset_work_like_espei_0_9():
    """check_dataset and clean_dataset are deprecated, and check and clean datasets in place like ESPEI 0.9 (gh-269)."""
    dataset = deepcopy(CU_MG_DATASET_ZPF_STRING_VALUES)
    with pytest.warns(DeprecationWarning, match="validate_dataset"):
        assert check_dataset(dataset) is None
    with pytest.warns(DeprecationWarning, match="validate_dataset"):
        assert clean_dataset(dataset) is dataset
    assert dataset["conditions"]["T"] == [1337.97, 1262.238]
    with pytest.warns(DeprecationWarning), pytest.raises(DatasetError):
        check_dataset(dataset_multi_misaligned)


@pytest.mark.parametrize("update, expected", [
    pytest.param({"excluded_model_contributions": ["mag"]}, ["idmix", "mag"], id="list"),
    pytest.param({"weight": 5.0}, 5.0, id="number"),
    pytest.param({"reference_states": {"MG": {"phase": "HCP_A3"}}}, {"CU": {"phase": "FCC_A1"}, "MG": {"phase": "HCP_A3"}}, id="dict"),
])
def test_apply_tags_update_rules(datasets_db, update, expected):
    """Tags extend lists, update dicts and replace other values of tagged datasets."""
    datasets_db.insert({"tags": ["dft"], "excluded_model_contributions": ["idmix"], "weight": 1.0, "reference_states": {"CU": {"phase": "FCC_A1"}, "MG": {"phase": "LIQUID"}}})
    apply_tags(datasets_db, {"dft": update})
    key, = update
    assert datasets_db.all()[0][key] == expected


@pytest.mark.parametrize("dataset, model", [
    pytest.param(HM_FORM_CUMG2, FixedConfigurationDataset, id="HM_FORM"), pytest.param(TRACER_D0_CU, FixedConfigurationDataset, id="TRACER_D0_CU"),
    pytest.param(HM_MIX_LIQUID, EquilibriumPropertyDataset, id="HM_MIX"), pytest.param(INTER_DIFF, EquilibriumPropertyDataset, id="INTER_DIFF"),
    pytest.param(ACR_MG, ActivityDataset, id="ACR_MG"), pytest.param(ZPF_C15_LIQUID, ZPFDataset, id="ZPF"),
])
def test_validate_dataset_returns_the_model_of_the_type_of_dataset(dataset, model):
    """Datasets are validated by the model of their type, from their output and solver (gh-269)."""
    assert type(validate_dataset(dataset)) is model


def test_zpf_dataset_models_give_phase_compositions():
    """ZPF dataset models give the phase, components, mole fractions and flag of each phase composition (gh-269)."""
    vertex = validate_dataset({**ZPF_C15_LIQUID, "values": [[["LAVES_C15", ["MG"], [0.31], "disordered"], ["LIQUID", ["MG"], [0.43]]]] * 2}).values[0][0]
    assert (vertex.phase, vertex.components, vertex.fractions, vertex.flag) == ("LAVES_C15", ["MG"], [0.31], "disordered")


def test_one_species_sublattices_given_as_lists_warn(tmp_path):
    """One species sublattices given as lists warn, because parameter selection ignores them."""
    as_list = {**HM_MIX_FCC, "solver": {**HM_MIX_FCC["solver"], "sublattice_configurations": [[["CU", "MG"], ["VA"]]] * 3, "sublattice_occupancies": [[[0.5, 0.5], [1.0]]] * 3}}
    with pytest.warns(DatasetWarning, match=re.escape("0.json: Sublattice ['VA'] of configuration [['CU', 'MG'], ['VA']] has one species")):
        load(tmp_path, as_list)


@pytest.mark.xfail(strict=True, reason="Bug: like ESPEI 0.9, load_datasets drops the disordered flag of known compositions")
def test_load_datasets_keeps_the_disordered_flag_of_known_compositions(tmp_path):
    """Loaded ZPF datasets keep the disordered flag of known compositions."""
    zpf = {**ZPF_C15_LIQUID, "values": [[["LAVES_C15", ["MG"], [0.31], "disordered"], ["LIQUID", ["MG"], [0.43]]]] * 2}
    assert load(tmp_path, zpf) == [zpf]
