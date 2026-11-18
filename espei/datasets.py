"""
ESPEI datasets and the pydantic models of each type of dataset.
"""
import fnmatch, json, os, sys, warnings
from typing import Any, Iterable, Literal, Optional, Self

import numpy as np
from pycalphad import variables as v
from pydantic import BaseModel, ConfigDict, Field, ValidationError, ValidationInfo, model_serializer, model_validator
from tinydb.storages import MemoryStorage
from tinydb import where

from espei.utils import PickleableTinyDB, docs_url


class DatasetError(Exception):
    """Exception raised when datasets are invalid."""


class DatasetWarning(UserWarning):
    """Warning for parts of a dataset that could not be checked or are likely mistakes."""


class DatasetDeprecationWarning(DatasetWarning, DeprecationWarning):
    """Warning for parts of a dataset that are deprecated."""


# DeprecationWarnings are not shown by default, but users must see these to update their datasets
if not sys.warnoptions:
    warnings.filterwarnings("default", category=DatasetDeprecationWarning)


class _Model(BaseModel):
    # Keep keys that are not part of the schema, e.g. "comment". Build validators when first used, not on import.
    model_config = ConfigDict(extra="allow", defer_build=True)


Values = list[list[list[float]]]  # shape of (len(P), len(T), number of configurations or compositions)


class Dataset(_Model):
    """Keys common to all types of datasets."""
    components: list[str] = Field(min_length=1)
    phases: list[str] = Field(min_length=1)
    conditions: dict[str, float | list[float]]
    output: str
    reference: str = ""
    bibtex: str = ""
    dataset_author: str = ""
    weight: float = Field(1.0, gt=0)
    tags: list[str] = Field(default_factory=list)
    disabled: bool = False

    @model_validator(mode="after")
    def _check_conditions_and_components(self, info: ValidationInfo) -> Self:
        if missing := {"T", "P"} - self.conditions.keys():
            raise ValueError(f"conditions must include T and P, missing {sorted(missing)}")
        species = (info.context or {}).get("species")
        if species and (species_components := sorted(c for c in self.components if c in species and species[c].constituents != {c: 1})):
            raise ValueError(f"components must be elements, but {species_components} are species")
        return self


class Solver(_Model):
    mode: Literal["manual"] = "manual"
    sublattice_site_ratios: list[float] = Field(min_length=1)
    sublattice_configurations: list[tuple[tuple[str, ...], ...]] = Field(min_length=1)  # species of each sublattice
    sublattice_occupancies: Optional[list[tuple[tuple[float, ...], ...]]] = None  # site fractions of the species of each sublattice

    @model_validator(mode="before")
    @classmethod
    def _sublattices_of_one_species(cls, solver: Any) -> Any:
        # Deprecated: a sublattice of one species given as the species, e.g. "VA" for ["VA"], with the occupancy 1.0 for [1.0]
        keys = [key for key in ("sublattice_configurations", "sublattice_occupancies") if isinstance(solver, dict) and isinstance(solver.get(key), list)]
        deprecated = [(key, i) for key in keys for i, row in enumerate(solver[key]) if isinstance(row, (list, tuple)) and not all(isinstance(subl, (list, tuple)) for subl in row)]
        if not deprecated:
            return solver
        updated = {key: [[subl if isinstance(subl, (list, tuple)) else [subl] for subl in row] if isinstance(row, (list, tuple)) else row for row in solver[key]] for key in keys}
        i = deprecated[0][1]
        warnings.warn("Every sublattice must be a list of species, even if it has one species. Giving a single species without a list is deprecated. Update this dataset, for example:" +
                      "".join(f"\n  {key}[{i}] {json.dumps(solver[key][i])} should be {json.dumps(updated[key][i])}" for key in keys if (key, i) in deprecated) +
                      f"\nRead more at {docs_url('reference/dataset_schema.html#non_equilibrium_thermochemical_data')}", DatasetDeprecationWarning)
        return {**solver, **updated}

    @model_validator(mode="after")
    def _occupancies_without_mixing(self) -> Self:
        if self.sublattice_occupancies is None and all(len(subl) == 1 for config in self.sublattice_configurations for subl in config):
            self.sublattice_occupancies = [tuple((1.0,) for _ in config) for config in self.sublattice_configurations]
        return self


class FixedConfigurationDataset(Dataset):
    """Properties of one phase at the fixed site fractions of each sublattice configuration."""
    phases: list[str] = Field(min_length=1, max_length=1)
    solver: Solver
    values: Values
    weight: float | list[float] | list[list[float]] | Values = 1.0  # broadcast to the shape of the values
    excluded_model_contributions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_configurations(self) -> Self:
        configurations, occupancies = self.solver.sublattice_configurations, self.solver.sublattice_occupancies
        shape = (np.size(self.conditions["P"]), np.size(self.conditions["T"]), len(configurations))
        if np.shape(self.values) != shape:
            raise ValueError(f"values must have the shape (len(P), len(T), len(sublattice_configurations)) = {shape}, got {np.shape(self.values)}")
        weight = np.asarray(self.weight)  # broadcast to the values, e.g. one weight per configuration
        if np.any(weight <= 0) or weight.ndim > 3 or any(w not in (1, s) for w, s in zip(weight.shape[::-1], shape[::-1])):
            raise ValueError(f"weight must be positive and broadcast to the shape of the values {shape}, got {self.weight}")
        if occupancies is None:
            raise ValueError("sublattice_occupancies are required when there is mixing in a sublattice")
        if len(occupancies) != len(configurations):
            raise ValueError(f"sublattice_occupancies must have one entry per sublattice configuration ({len(configurations)}), got {len(occupancies)}")
        num_sublattices = len(self.solver.sublattice_site_ratios)
        for i, (config, occupancy) in enumerate(zip(configurations, occupancies)):
            if len(config) != num_sublattices:
                raise ValueError(f"sublattice_configurations[{i}] {json.dumps(config)} must have one entry for each of the {num_sublattices} sublattices")
            if [len(occ) for occ in occupancy] != [len(subl) for subl in config] or any(abs(sum(occ) - 1) > 0.01 for occ in occupancy):
                raise ValueError(f"sublattice_occupancies[{i}] {json.dumps(occupancy)} must have one site fraction per species of sublattice_configurations[{i}] {json.dumps(config)} and sum to one in each sublattice")
            if any(len(set(subl)) != len(subl) for subl in config):
                raise ValueError(f"sublattice_configurations[{i}] {json.dumps(config)} must list each species of a sublattice once")
            # Interactions are sorted, e.g. for espei.core_utils.get_samples
            if any(subl != tuple(sorted(subl)) for subl in config):
                pairs = [sorted(zip(subl, occ)) for subl, occ in zip(config, occupancy)]
                raise ValueError("the species of each sublattice must be sorted, with their occupancies in the same order:\n"
                                 f"    sublattice_configurations[{i}] {json.dumps(config)} should be {json.dumps([[sp for sp, _ in p] for p in pairs])}\n"
                                 f"    sublattice_occupancies[{i}] {json.dumps(occupancy)} should be {json.dumps([[x for _, x in p] for p in pairs])}")
        return self

    @model_validator(mode="after")
    def _check_components(self, info: ValidationInfo) -> Self:
        species = (info.context or {}).get("species")
        entered = set(self.components)
        if species is not None and not entered - {"VA"} <= species.keys():
            return self  # datasets outside the system being fit are never used, so their species aren't resolved
        used, unresolved = set(), set()
        for name in {sp for config in self.solver.sublattice_configurations for subl in config for sp in subl}:
            if name == "VA" or (species is None and name in entered):
                used.add(name)
            elif species is None:
                unresolved.add(name)  # can't tell a species from a misspelled component
            elif name in species:
                used.update(species[name].constituents)
            else:
                raise ValueError(f"species {name} is not defined in the phase models or the input database")
        if unresolved:
            warnings.warn(f"Species {', '.join(sorted(unresolved))} in the sublattice configurations were not checked. Provide the phase models (e.g. `espei --check-datasets PATH --phase-models FILE`) to check them.", DatasetWarning)
        elif used - entered or entered - used - {"VA"}:
            raise ValueError(f"components {sorted(entered)} must match the components in the sublattice configurations {sorted(used)}")
        return self


class ComponentReferenceState(_Model):
    phase: str
    fixed_state_variables: Optional[dict[str, float]] = None


class EquilibriumPropertyDataset(Dataset):
    """Properties at equilibrium for each pressure, temperature and composition (X_ conditions)."""
    values: Values
    reference_states: Optional[dict[str, ComponentReferenceState]] = None

    @model_validator(mode="after")
    def _check_conditions(self) -> Self:
        if other := [k for k in self.conditions if k not in ("T", "P", "N") and not k.startswith("X_")]:
            raise ValueError(f"conditions must be T, P, N or X_<component>, got {other}")
        num_compositions = {k[2:]: np.size(x) for k, x in self.conditions.items() if k.startswith("X_")}
        dependent = set(self.components) - num_compositions.keys() - {"VA"}
        if not num_compositions.keys() <= set(self.components) or len(dependent) != 1:
            raise ValueError(f"X_ conditions {sorted(num_compositions)} must be components {self.components}, leaving one dependent component, got {sorted(dependent)}")
        if len(set(num_compositions.values())) > 1:
            raise ValueError(f"X_ conditions must have the same number of values, because they are not broadcast, got {num_compositions}")
        shape = (np.size(self.conditions["P"]), np.size(self.conditions["T"]), *(set(num_compositions.values()) or {1}))  # one composition for one component
        if np.shape(self.values) != shape:
            raise ValueError(f"values must have the shape (len(P), len(T), len(X)) = {shape}, got {np.shape(self.values)}")
        if self.reference_states is not None:
            if missing := set(self.components) - {"VA"} - self.reference_states.keys():
                raise ValueError(f"reference_states must be given for every component, missing {sorted(missing)}")
            if not self.output.endswith("R"):
                raise ValueError(f'reference_states are only used for outputs referenced to them, ending in "R" (e.g. "HMR"), got "{self.output}"')
        return self


class ActivityReferenceState(_Model):
    phases: list[str] = Field(min_length=1)
    conditions: dict[str, float]


class ActivityDataset(EquilibriumPropertyDataset):
    """Activities of a component (output "ACR_<component>") relative to its reference state."""
    reference_state: ActivityReferenceState

    @model_validator(mode="after")
    def _check_output(self) -> Self:
        if self.output.removeprefix("ACR_") not in self.components:
            raise ValueError(f'output must be "ACR_<component>" for one of the components {self.components}, got "{self.output}"')
        return self


class PhaseComposition(_Model):
    """[phase, components, mole fractions] of a phase in a phase region, with an optional "disordered" flag."""
    phase: str
    components: list[str]
    fractions: list[Optional[float]]  # None for unknown
    flag: Optional[Literal["disordered"]] = None

    @model_validator(mode="before")
    @classmethod
    def _from_list(cls, value: Any) -> Any:
        if isinstance(value, (list, tuple)) and len(value) in (3, 4):
            return dict(zip(["phase", "components", "fractions", "flag"], value))
        raise ValueError(f'phase compositions must be [phase, components, mole fractions] or [phase, components, mole fractions, "disordered"], got {value}')

    @model_validator(mode="after")
    def _check_fractions(self) -> Self:
        known = [x for x in self.fractions if x is not None]
        if len(self.fractions) != len(self.components) or sum(known) > 1:
            raise ValueError(f"mole fractions {self.fractions} must be one per component {self.components} and sum to at most one")
        if self.phase.upper() == "__HYPERPLANE__" and len(known) < len(self.fractions):
            raise ValueError(f"__HYPERPLANE__ mole fractions must all be known, got {self.fractions}")
        return self

    @model_serializer
    def _to_list(self) -> list:
        return [self.phase, self.components, self.fractions] + ([self.flag] if self.flag else [])


class ZPFDataset(Dataset):
    """Phase equilibria. Each phase region of the values lists the phases in equilibrium and their compositions."""
    values: list[list[PhaseComposition]] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_phase_regions(self) -> Self:
        if invalid := {k: np.size(x) for k, x in self.conditions.items() if k not in ("T", "P", "N") or np.size(x) not in (1, len(self.values))}:
            raise ValueError(f"conditions must be T, P or N with one value or one value per phase region ({len(self.values)}), got {invalid} values")
        components, phases_used = set(self.components), set()
        for i, region in enumerate(self.values):
            if all(None in pc.fractions for pc in region):
                raise ValueError(f"phase region values[{i}] needs a phase composition with known mole fractions to define the target hyperplane")
            for pc in region:
                if not set(pc.components) <= components or len(components - set(pc.components) - {"VA"}) != 1:
                    raise ValueError(f"components {pc.components} of {pc.phase} in phase region values[{i}] must be all but one of the components {self.components}")
                phases_used.add(pc.phase)
        phases_used.discard("__HYPERPLANE__")
        if phases_used != set(self.phases):
            raise ValueError(f"phases {self.phases} must be the phases used in the values {sorted(phases_used)}")
        return self


def validate_dataset(dataset: dict[str, Any], species: Optional[Iterable[v.Species]] = None) -> Dataset:
    """Return the dataset as a validated model of its type of dataset.

    Parameters
    ----------
    dataset : dict[str, Any]
        Dataset, e.g. loaded from a JSON file.
    species : Optional[Iterable[v.Species]]
        Species, e.g. from ``PhaseModelSpecification.get_species``, used to
        resolve the names in sublattice configurations to their elements.
        Without them, names that are not components give a ``DatasetWarning``
        and the components of the dataset are not checked.

    Returns
    -------
    Dataset
        One of ``ZPFDataset``, ``ActivityDataset``, ``FixedConfigurationDataset``
        (datasets with a ``solver``) or ``EquilibriumPropertyDataset``.

    Raises
    ------
    DatasetError
        Listing every error found in the dataset, with its location.
    """
    output = str(dataset.get("output", ""))
    model = (ZPFDataset if output == "ZPF" else
             ActivityDataset if output.startswith("ACR") else
             FixedConfigurationDataset if "solver" in dataset else
             EquilibriumPropertyDataset)
    try:
        return model.model_validate(dataset, context={"species": None if species is None else {sp.name: sp for sp in species}})
    except ValidationError as e:
        errors = []
        for error in e.errors():
            location = "".join(f"[{x}]" if isinstance(x, int) else f".{x}" for x in error["loc"]).lstrip(".")
            got = "" if isinstance(error["input"], (dict, list)) else f" (got {error['input']!r})"
            errors.append(f"{location}{': ' if location else ''}{error['msg'].removeprefix('Value error, ')}{got}")
        raise DatasetError(f"{len(errors)} error(s) in {model.__name__}:\n  " + "\n  ".join(errors)) from None


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
                elif isinstance(newval, dict):
                    match[newkey] = {**match.get(newkey, {}), **newval}
                else:
                    match[newkey] = newval
                datasets.update(match, doc_ids=[match.doc_id])


def load_datasets(dataset_filenames, include_disabled=False, species=None) -> PickleableTinyDB:
    """
    Create a PickelableTinyDB with the data from a list of filenames.

    Parameters
    ----------
    dataset_filenames : [str]
        List of filenames to load as datasets
    species : Optional[Iterable[v.Species]]
        Species used to check the datasets, see ``validate_dataset``.

    Returns
    -------
    PickleableTinyDB
    """
    datasets = []
    for fname in dataset_filenames:
        with open(fname) as file_:
            try:
                d = json.load(file_)
            except ValueError as e:
                raise ValueError('JSON Error in {}: {}'.format(fname, e))
        if not include_disabled and d.get('disabled', False):
            # The dataset is disabled and not included
            continue
        # add the file name to dataset errors and warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DatasetWarning)
            try:
                # Only keys in the file are kept, so code using the datasets sees missing keys rather than defaults
                datasets.append(validate_dataset(d, species).model_dump(exclude_unset=True))
            except DatasetError as e:
                raise DatasetError('Dataset Error in {}: {}'.format(fname, e)) from None
        for w in caught:
            warnings.warn_explicit(f"{fname}: {w.message}" if issubclass(w.category, DatasetWarning) else w.message, w.category, w.filename, w.lineno)
    ds_database = PickleableTinyDB(storage=MemoryStorage)
    ds_database.insert_multiple(datasets)  # inserting one at a time is quadratic in the number of datasets
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


def check_dataset(dataset: dict[str, Any], species: Optional[Iterable[v.Species]] = None) -> None:
    """Deprecated, use ``validate_dataset``."""
    warnings.warn("check_dataset is deprecated and will be removed in ESPEI 0.11. Use espei.datasets.validate_dataset instead.", DeprecationWarning, stacklevel=2)
    validate_dataset(dataset, species)


def clean_dataset(dataset: dict[str, Any]) -> dict[str, Any]:
    """Deprecated, use ``validate_dataset``."""
    warnings.warn("clean_dataset is deprecated and will be removed in ESPEI 0.11. Use espei.datasets.validate_dataset instead.", DeprecationWarning, stacklevel=2)
    dataset.update(validate_dataset(dataset).model_dump(exclude_unset=True))
    return dataset
