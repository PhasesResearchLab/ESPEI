from typing import Any, Literal, Union, TypeAlias, Self
import warnings
from pydantic import BaseModel, Field, model_validator, field_validator
import numpy as np
import fnmatch
import json
import os
from tinydb.storages import MemoryStorage
from tinydb import where

from espei.utils import PickleableTinyDB

__all__ = [
    # Models
    "Dataset",
    "BroadcastSinglePhaseFixedConfigurationDataset",
    "ActivityPropertyDataset",
    "EquilibriumPropertyDataset",
    "ZPFDataset",

    # Errors (when validating models)
    "DatasetError",

    # User-facing API
    "load_datasets",
    "recursive_glob",
    "apply_tags",
    "to_Dataset",

    # Deprecated
    "check_dataset",
    "clean_dataset",
]


# Type aliases - used to clarify intent
# e.g. when we want a ComponentName rather than a str (even though that's what it is)
ComponentName: TypeAlias = str
PhaseName: TypeAlias = str
PhaseCompositionType: TypeAlias = Union[
    tuple[PhaseName, list[ComponentName], list[float | None]],       # The usual definition ["LIQUID", ["B"], [0.5]]
    tuple[PhaseName, list[ComponentName], list[float | None], bool]  # Handle the disordered flag
]
PhaseRegionType: TypeAlias = list[PhaseCompositionType]


class DatasetError(Exception):
    """Exception raised when datasets are invalid."""
    pass


class Dataset(BaseModel):
    pass


class Solver(BaseModel):
    mode: Literal["manual"] = Field(default="manual")
    sublattice_site_ratios: list[float]
    # TODO: migrate to list[list[list[float]]]
    sublattice_configurations: list[list[ComponentName | list[ComponentName]]]
    sublattice_occupancies: list[list[float | list[float]]] | None = Field(default=None)


class BroadcastSinglePhaseFixedConfigurationDataset(Dataset):
    components: list[ComponentName] = Field(min_length=1)
    phases: list[PhaseName] = Field(min_length=1, max_length=1)
    solver: Solver
    conditions: dict[str, float | list[float]]
    output: str
    values: list[list[list[float]]]
    excluded_model_contributions: list[str] = Field(default_factory=list)
    reference: str = Field(default="")
    bibtex: str = Field(default="")
    dataset_author: str = Field(default="")
    comment: str = Field(default="")
    disabled: bool = Field(default=False)
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_components_entered_match_components_used(self) -> Self:
        components_entered = set(self.components)
        components_used = set()
        for config in self.solver.sublattice_configurations:
            for subl in config:
                if isinstance(subl, list):
                    components_used.update(set(subl))
                else:
                    components_used.add(subl)
        # Don't count vacancies as a component here
        components_difference = components_entered.symmetric_difference(components_used) - {"VA"}
        if len(components_difference) != 0:
            raise DatasetError(f'Components entered {components_entered} do not match components used {components_used} ({components_difference} different).')
        return self

    @model_validator(mode="after")
    def validate_condition_value_shape_agreement(self) -> Self:
        values_shape = np.array(self.values).shape
        num_configs = len(self.solver.sublattice_configurations)
        num_temperature = np.atleast_1d(self.conditions["T"]).size
        num_pressure = np.atleast_1d(self.conditions["P"]).size
        conditions_shape = (num_pressure, num_temperature, num_configs)
        if conditions_shape != values_shape:
            raise DatasetError(f'Shape of conditions (P, T, configs): {conditions_shape} does not match the shape of the values {values_shape}.')
        return self

    @model_validator(mode="after")
    def validate_configuration_occupancy_shape_agreement(self) -> Self:
        sublattice_configurations = self.solver.sublattice_configurations
        sublattice_site_ratios = self.solver.sublattice_site_ratios
        sublattice_occupancies = self.solver.sublattice_occupancies
        # check for mixing
        is_mixing = any([any([isinstance(subl, list) for subl in config]) for config in sublattice_configurations])
        # pad the values of sublattice occupancies if there is no mixing
        # just for the purposes of checking validity
        if sublattice_occupancies is None and not is_mixing:
            sublattice_occupancies = [None]*len(sublattice_configurations)
        elif sublattice_occupancies is None:
            raise DatasetError(f'At least one sublattice in the following sublattice configurations is mixing, but the "sublattice_occupancies" key is empty: {sublattice_configurations}')

        # check that the site ratios are valid as well as site occupancies, if applicable
        nconfigs = len(sublattice_configurations)
        noccupancies = len(sublattice_occupancies)
        if nconfigs != noccupancies:
            raise DatasetError(f'Number of sublattice configurations ({nconfigs}) does not match the number of sublattice occupancies ({noccupancies})')
        for configuration, occupancy in zip(sublattice_configurations, sublattice_occupancies):
            if len(configuration) != len(sublattice_site_ratios):
                raise DatasetError(f'Sublattice configuration {configuration} and sublattice site ratio {sublattice_site_ratios} describe different numbers of sublattices ({len(configuration)} and {len(sublattice_site_ratios)}).')
            if is_mixing:
                configuration_shape = tuple(len(sl) if isinstance(sl, list) else 1 for sl in configuration)
                occupancy_shape = tuple(len(sl) if isinstance(sl, list) else 1 for sl in occupancy)
                if configuration_shape != occupancy_shape:
                    raise DatasetError(f'The shape of sublattice configuration {configuration} ({configuration_shape}) does not match the shape of occupancies {occupancy} ({occupancy_shape})')
                # check that sublattice interactions are in sorted. Related to sorting in espei.core_utils.get_samples
                for subl in configuration:
                    if isinstance(subl, (list, tuple)) and sorted(subl) != subl:
                        raise DatasetError(f'Sublattice {subl} in configuration {configuration} is must be sorted in alphabetic order ({sorted(subl)})')
        return self


class ActivityDataReferenceState(Dataset):
    phases: list[PhaseName] = Field(min_length=1)
    conditions: dict[str, float]

# TODO: refactor ActivityPropertyDataset to merge with EquilibriumPropertyDataset
# The validator functions are exactly duplicated in EquilibriumPropertyDataset
# The duplication simplifies the implementation since the activity special case is
# ultimately meant to be removed once activity is a PyCalphad Workspace property
class ActivityPropertyDataset(Dataset):
    components: list[ComponentName] = Field(min_length=1)
    phases: list[PhaseName] = Field(min_length=1)
    conditions: dict[str, float | list[float]]
    reference_state: ActivityDataReferenceState
    output: str
    values: list[list[list[float]]]
    reference: str = Field(default="")
    bibtex: str = Field(default="")
    dataset_author: str = Field(default="")
    comment: str = Field(default="")
    disabled: bool = Field(default=False)
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_condition_value_shape_agreement(self) -> Self:
        conditions = self.conditions
        comp_conditions = {k: v for k, v in conditions.items() if k.startswith('X_')}
        num_temperature = np.atleast_1d(self.conditions["T"]).size
        num_pressure = np.atleast_1d(self.conditions["P"]).size
        # check each composition condition is the same shape
        num_x_conds = [np.atleast_1d(vals).size for _, vals in comp_conditions.items()]
        if num_x_conds.count(num_x_conds[0]) != len(num_x_conds):
            raise DatasetError(f'All compositions in conditions are not the same shape. Note that conditions cannot be broadcast. Composition conditions are {comp_conditions}')
        conditions_shape = (num_pressure, num_temperature, num_x_conds[0])
        values_shape = np.array(self.values).shape
        if conditions_shape != values_shape:
            raise DatasetError(f'Shape of conditions (P, T, compositions): {conditions_shape} does not match the shape of the values {values_shape}.')
        return self

    @model_validator(mode="after")
    def validate_components_entered_match_components_used(self) -> Self:
        conditions = self.conditions
        comp_conditions = {ky: vl for ky, vl in conditions.items() if ky.startswith('X_')}
        components_entered = set(self.components)
        components_used = set()
        components_used.update({c.split('_')[1] for c in comp_conditions.keys()})
        if not components_entered.issuperset(components_used):
            raise DatasetError(f"Components were used as conditions that are not present in the specified components: {components_used - components_entered}.")
        independent_components = components_entered - components_used - {'VA'}
        if len(independent_components) != 1:
            raise DatasetError(f"Degree of freedom error: expected 1 independent component, got {len(independent_components)} for entered components {components_entered} and {components_used} used in the conditions.")
        return self


class ReferenceStates(BaseModel):
    phase: PhaseName
    fixed_state_variables: dict[str, float] | None = Field(default=None, description="Fixed potentials for the reference state", examples=[{"T": 298.15, "P": 101325}])


class EquilibriumPropertyDataset(Dataset):
    components: list[ComponentName] = Field(min_length=1)
    phases: list[PhaseName] = Field(min_length=1)
    conditions: dict[str, float | list[float]]
    output: str
    values: list[list[list[float]]]
    reference_states: dict[ComponentName, ReferenceStates] | None = Field(default=None)
    reference: str = Field(default="")
    bibtex: str = Field(default="")
    dataset_author: str = Field(default="")
    comment: str = Field(default="")
    disabled: bool = Field(default=False)
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_condition_value_shape_agreement(self) -> Self:
        conditions = self.conditions
        comp_conditions = {k: v for k, v in conditions.items() if k.startswith('X_')}
        num_temperature = np.atleast_1d(self.conditions["T"]).size
        num_pressure = np.atleast_1d(self.conditions["P"]).size
        # check each composition condition is the same shape
        num_x_conds = [np.atleast_1d(vals).size for _, vals in comp_conditions.items()]
        if num_x_conds.count(num_x_conds[0]) != len(num_x_conds):
            raise DatasetError(f'All compositions in conditions are not the same shape. Note that conditions cannot be broadcast. Composition conditions are {comp_conditions}')
        conditions_shape = (num_pressure, num_temperature, num_x_conds[0])
        values_shape = np.array(self.values).shape
        if conditions_shape != values_shape:
            raise DatasetError(f'Shape of conditions (P, T, compositions): {conditions_shape} does not match the shape of the values {values_shape}.')
        return self

    @model_validator(mode="after")
    def validate_components_entered_match_components_used(self) -> Self:
        conditions = self.conditions
        comp_conditions = {ky: vl for ky, vl in conditions.items() if ky.startswith('X_')}
        components_entered = set(self.components)
        components_used = set()
        components_used.update({c.split('_')[1] for c in comp_conditions.keys()})
        if not components_entered.issuperset(components_used):
            raise DatasetError(f"Components were used as conditions that are not present in the specified components: {components_used - components_entered}.")
        independent_components = components_entered - components_used - {'VA'}
        if len(independent_components) != 1:
            raise DatasetError(f"Degree of freedom error: expected 1 independent component, got {len(independent_components)} for entered components {components_entered} and {components_used} used in the conditions.")
        return self

    @model_validator(mode="after")
    def validate_reference_state_fully_specified_if_used(self) -> Self:
        """If there is a reference state specified, the components in the reference state must match the dataset components"""
        components_entered = set(self.components) - {"VA"}
        if self.reference_states is not None:
            reference_state_components = set(self.reference_states.keys()) - {"VA"}
            if components_entered != reference_state_components:
                raise DatasetError(f"If used, reference states in equilibrium property must define a reference state for all components in the calculation. Got {components_entered} entered components and {reference_state_components} in the reference states ({components_entered.symmetric_difference(reference_state_components)} non-matching).")
        return self


class ZPFDataset(Dataset):
    components: list[ComponentName] = Field(min_length=1)
    phases: list[str] = Field(min_length=1)
    conditions: dict[str, float | list[float]]
    broadcast_conditions: Literal[False] = Field(default=False)  # TODO: migrate and remove, since True was never supported
    output: Literal["ZPF"]
    values: list[PhaseRegionType]  # TODO: validate to be of same shape as conditions
    excluded_model_contributions: list[str] = Field(default_factory=list)
    reference: str = Field(default="")
    bibtex: str = Field(default="")
    dataset_author: str = Field(default="")
    comment: str = Field(default="")
    disabled: bool = Field(default=False)
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_condition_value_shape_agreement(self) -> Self:
        values_shape = (len(self.values),)
        num_temperature = np.atleast_1d(self.conditions["T"]).size
        num_pressure = np.atleast_1d(self.conditions["P"]).size
        if num_pressure != 1:
            raise DatasetError("Non-scalar pressures are not currently supported")
        conditions_shape = (num_temperature,)
        if conditions_shape != values_shape:
            raise DatasetError("Shape of conditions (T): {} does not match the shape of the values {}.".format(conditions_shape, values_shape))
        return self

    @model_validator(mode="after")
    def validate_phases_entered_match_phases_used(self) -> Self:
        phases_entered = set(self.phases)
        phases_used = set()
        for phase_region in self.values:
            for phase_composition in phase_region:
                phases_used.add(phase_composition[0])
        if len(phases_entered - phases_used) > 0:
            raise DatasetError("Phases entered {} do not match phases used {}.".format(phases_entered, phases_used))
        return self

    @model_validator(mode="after")
    def validate_components_entered_match_components_used(self) -> Self:
        components_entered = set(self.components)
        for i, phase_region in enumerate(self.values):
            for j, phase_compositions in enumerate(phase_region):
                phase_composition_components = set(phase_compositions[1])
                if not components_entered.issuperset(phase_composition_components):
                    raise DatasetError("Components were used in phase region {} ({}) for phase composition {} ({}) that are not specified as components in the dataset ()", i,phase_region, j, phase_compositions, components_entered)
                independent_components = components_entered - phase_composition_components - {'VA'}
                if len(independent_components) != 1:
                    raise DatasetError('Degree of freedom error: expected 1 independent component, got {} for entered components {} and phase composition components {} in phase region {} ({}) for phase composition {} ({})'.format(len(independent_components), components_entered, phase_composition_components, i, phase_region, j, phase_compositions))
        return self

    @field_validator("values", mode="after")
    @classmethod
    def validate_phase_compositions(cls, values: list[PhaseRegionType]) -> list[PhaseRegionType]:
        for i, phase_region in enumerate(values):
            for j, phase_composition in enumerate(phase_region):
                phase = phase_composition[0]
                component_list = phase_composition[1]
                mole_fraction_list = phase_composition[2]
                # check that the phase is a string, components a list of strings,
                #  and the fractions are a list of float
                if not isinstance(phase, str):
                    raise DatasetError('The first element in phase composition {} ({}) for phase region {} ({}) should be a string. Instead it is a {} of value {}'.format(j, phase_composition, i, phase_region, type(phase), phase))
                if not all([isinstance(comp, str) for comp in component_list]):
                    raise DatasetError('The second element in phase composition {} ({}) for phase region {} ({}) should be a list of strings. Instead it is a {} of value {}'.format(j, phase_composition, i, phase_region, type(component_list), component_list))
                if not all([(isinstance(mole_frac, (int, float)) or mole_frac is None)  for mole_frac in mole_fraction_list]):
                    raise DatasetError('The last element in phase composition {} ({}) for phase region {} ({}) should be a list of numbers. Instead it is a {} of value {}'.format(j, phase_composition, i, phase_region, type(mole_fraction_list), mole_fraction_list))
                # check that the shape of components list and mole fractions list is the same
                if len(component_list) != len(mole_fraction_list):
                    raise DatasetError('The length of the components list and mole fractions list in phase composition {} ({}) for phase region {} ({}) should be the same.'.format(j, phase_composition, i, phase_region))
                # check that all mole fractions are less than one
                mf_sum = np.nansum(np.array(mole_fraction_list, dtype=np.float64))
                if any([mf is not None for mf in mole_fraction_list]) and mf_sum > 1.0:
                    raise DatasetError('Mole fractions for phase composition {} ({}) for phase region {} ({}) sum to greater than one.'.format(j, phase_composition, i, phase_region))
                if any([(mf is not None) and (mf < 0.0) for mf in mole_fraction_list]):
                    raise DatasetError('Got unallowed negative mole fraction for phase composition {} ({}) for phase region {} ({}).'.format(j, phase_composition, i, phase_region))
        return values


def to_Dataset(candidate: dict[str, Any]) -> Dataset:
    """Return a validated Dataset object for a dataset dict. Raises if a validated dataset cannot be created.

    Parameters
    ----------
    candidate : dict[str, Any]
        Dictionary describing an ESPEI dataset.

    Returns
    -------
    Dataset

    Raises
    ------
    DatasetError
        If an error is found in the dataset
    """
    if candidate["output"] == "ZPF":
        return ZPFDataset.model_validate(candidate)
    elif candidate['output'].startswith('ACR'):
        return ActivityPropertyDataset.model_validate(candidate)
    elif 'solver' in candidate.keys():
        return BroadcastSinglePhaseFixedConfigurationDataset.model_validate(candidate)
    else:
        return EquilibriumPropertyDataset.model_validate(candidate)


def apply_tags(datasets: PickleableTinyDB, tags):
    """
    Modify datasets using the tags system

    Parameters
    ----------
    datasets : PickleableTinyDB
        Datasets to modify
    tags : dict
        Dictionary of {tag: update_dict}

    Returns
    -------
    None

    Notes
    -----
    In general, everything replaces or is additive. We use the following update rules:
    1. If the update value is a list, extend the existing list (empty list if key does not exist)
    2. If the update value is scalar, override the previous (deleting any old value, if present)
    3. If the update value is a dict, update the exist dict (empty dict if dict does not exist)
    4. Otherwise, the value is updated, overriding the previous

    Examples
    --------
    >>> from espei.utils import PickleableTinyDB
    >>> from tinydb.storages import MemoryStorage
    >>> ds = PickleableTinyDB(storage=MemoryStorage)
    >>> doc_id = ds.insert({'tags': ['dft'], 'excluded_model_contributions': ['contrib']})
    >>> my_tags = {'dft': {'excluded_model_contributions': ['idmix', 'mag'], 'weight': 5.0}}
    >>> from espei.datasets import apply_tags
    >>> apply_tags(ds, my_tags)
    >>> all_data = ds.all()
    >>> all(d['excluded_model_contributions'] == ['contrib', 'idmix', 'mag'] for d in all_data)
    True
    >>> all(d['weight'] == 5.0 for d in all_data)
    True

    """
    for tag, update_dict in tags.items():
        matching_datasets = datasets.search(where("tags").test(lambda x: tag in x))
        for newkey, newval in update_dict.items():
            for match in matching_datasets:
                if isinstance(newval, list):
                    match[newkey] = match.get(newkey, []) + newval
                elif np.isscalar(newval):
                    match[newkey] = newval
                elif isinstance(newval, dict):
                    d = match.get(newkey, dict())
                    d.update(newval)
                    match[newkey] = d
                else:
                    match[newkey] = newval
                datasets.update(match, doc_ids=[match.doc_id])


def load_datasets(dataset_filenames, include_disabled=False) -> PickleableTinyDB:
    """
    Create a PickelableTinyDB with the data from a list of filenames.

    Parameters
    ----------
    dataset_filenames : [str]
        List of filenames to load as datasets

    Returns
    -------
    PickleableTinyDB
    """
    ds_database = PickleableTinyDB(storage=MemoryStorage)
    for fname in dataset_filenames:
        with open(fname) as file_:
            try:
                d = json.load(file_)
                if not include_disabled and d.get('disabled', False):
                    # The dataset is disabled and not included
                    continue
                ds_database.insert(to_Dataset(d).model_dump())
            except ValueError as e:
                raise ValueError('JSON Error in {}: {}'.format(fname, e))
            except DatasetError as e:
                raise DatasetError('Dataset Error in {}: {}'.format(fname, e))
    return ds_database


def recursive_glob(start, pattern='*.json'):
    """
    Recursively glob for the given pattern from the start directory.

    Parameters
    ----------
    start : str
        Path of the directory to walk while for file globbing
    pattern : str
        Filename pattern to match in the glob.

    Returns
    -------
    [str]
        List of matched filenames

    """
    matches = []
    for root, dirnames, filenames in os.walk(start, followlinks=True):
        for filename in fnmatch.filter(filenames, pattern):
            matches.append(os.path.join(root, filename))
    return sorted(matches)


def check_dataset(dataset: dict[str, Any]) -> dict[str, Any]:
    """Ensure that the dataset is valid and consistent by round-tripping through pydantic."""
    warnings.warn("check_dataset is deprecated will be removed in ESPEI 0.11. Behavior has been migrated to the pydantic dataset implementations in espei.datasets.dataset_models. To get a Dataset object, use espei.datasets.to_Dataset.", DeprecationWarning)
    return to_Dataset(dataset).model_dump()


def clean_dataset(dataset: dict[str, Any]) -> dict[str, Any]:
    """Ensure that the dataset is valid and consistent by round-tripping through pydantic."""
    warnings.warn("clean_dataset is deprecated will be removed in ESPEI 0.11. Behavior has been migrated to the pydantic dataset implementations in espei.datasets.dataset_models. To get a Dataset object, use espei.datasets.to_Dataset.", DeprecationWarning)
    return to_Dataset(dataset).model_dump()
