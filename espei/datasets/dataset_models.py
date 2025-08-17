from typing import Literal, Optional, Union, TypeAlias, Self
from pydantic import BaseModel, Field, model_validator, field_validator
import numpy as np

__all__ = [
    "Dataset",
    "BroadcastSinglePhaseFixedConfigurationDataset",
    "ActivityPropertyDataset",
    "EquilibriumPropertyDataset",
    "ZPFDataset",
]

class DatasetError(Exception):
    """Exception raised when datasets are invalid."""
    pass

ComponentName: TypeAlias = str
PhaseName: TypeAlias = str
PhaseCompositionType: TypeAlias = Union[
    tuple[PhaseName, list[ComponentName], list[float | None]],       # The usual definition ["LIQUID", ["B"], [0.5]]
    tuple[PhaseName, list[ComponentName], list[float | None], bool]  # Handle the disordered flag
]
PhaseRegionType: TypeAlias = list[PhaseCompositionType]

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


# TODO: would be great to remove
class ActivityDataReferenceState(BaseModel):
    phases: list[PhaseName] = Field(min_length=1)
    conditions: dict[str, float]


# TODO: refactor to merge this with EquilibriumPropertyDataset
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


class ReferenceStates(BaseModel):
    phase: PhaseName
    fixed_state_variables: dict[str, float] | None = Field(default=None, description="Fixed potentials for the reference state", examples=[{"T": 298.15, "P": 101325}])


class EquilibriumPropertyDataset(Dataset):
    components: list[ComponentName] = Field(min_length=1)
    phases: list[PhaseName] = Field(min_length=1)
    conditions: dict[str, float | list[float]]
    reference_states: dict[ComponentName, ReferenceStates]
    output: str
    values: list[list[list[float]]]
    reference: str = Field(default="")
    bibtex: str = Field(default="")
    dataset_author: str = Field(default="")
    comment: str = Field(default="")
    disabled: bool = Field(default=False)


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