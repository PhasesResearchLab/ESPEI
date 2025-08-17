from typing import Literal, Optional, Union, TypeAlias
from pydantic import BaseModel, Field

__all__ = [
    "Dataset",
    "BroadcastSinglePhaseFixedConfigurationDataset",
    "ActivityPropertyDataset",
    "EquilibriumPropertyDataset",
    "ZPFDataset",
]

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
    sublattice_occupancies: list[list[float | list[float]]] # TODO: optional and validate against configurations

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
    phases: list[PhaseName] = Field(min_length=1, max_length=1)
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
    phases: list[PhaseName] = Field(min_length=1, max_length=1)
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
