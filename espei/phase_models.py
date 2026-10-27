from typing import Dict, List, Optional, Set, Type

from pydantic import BaseModel, PositiveFloat
from pydantic.types import ImportString
from pycalphad import Database, Model, variables as v

from espei.typing import ComponentName, PhaseName, SpeciesName


# TODO: validate that len(model) == len(coefficients)
class ModelMetadata(BaseModel):
    sublattice_model: List[List[SpeciesName]]
    sublattice_site_ratios: List[PositiveFloat]
    # Fully qualified import path for a Python class that follows the pycalphad.Model API
    model: Optional[ImportString] = None


class SpeciesSpecification(BaseModel):
    constituents: Dict[ComponentName, float]
    charge: int = 0


class PhaseModelSpecification(BaseModel):
    components: List[ComponentName]
    species: Dict[SpeciesName, SpeciesSpecification] = {}
    phases: Dict[PhaseName, ModelMetadata]

    def get_species(self, dbf: Optional[Database] = None) -> Set[v.Species]:
        """Return the species of the components, the Database and the phase models, unique by name."""
        # TODO: resolve v.Component names to elements here once pycalphad supports them
        components = {comp.upper() for comp in self.components}
        elements = set(components)
        species = {el: v.Species(el, {el: 1}) for el in elements}
        if dbf is not None:
            elements |= dbf.elements
            species.update({sp.name: sp for sp in dbf.species})
        for comp in sorted(components):
            if species[comp].constituents != {comp: 1}:
                raise ValueError(f"Component {comp} is a species in the Database. Components must be elements.")
        for name, spec in self.species.items():
            name = name.upper()
            constituents = {el.upper(): amount for el, amount in spec.constituents.items()}
            if name in components and constituents != {name: 1}:
                raise ValueError(f"Component {name} is a species. Components must be elements; remove {name} from `components`.")
            if name in elements:
                raise ValueError(f"Species {name} has the name of an element.")
            if not set(constituents).issubset(elements):
                raise ValueError(f"Constituents of species {name} must be elements, but {sorted(set(constituents) - elements)} are not components or Database elements.")
            new_species = v.Species(name, constituents, spec.charge)
            old_species = species.get(name)
            if old_species is not None and (old_species != new_species or old_species.charge != new_species.charge):
                raise ValueError(f"Species {name} is {constituents} with charge {spec.charge} in the phase models, but {old_species.constituents} with charge {old_species.charge} in the Database.")
            species[name] = new_species
        return set(species.values())

    # TODO: update type of Model to ModelProtocol if/when available
    def get_model_dict(self) -> Dict[str, Type[Model]]:
        """
        Return a pycalphad-style model dictionary mapping phase names to model classes.

        If a phase's "model" key is not specified, no entry is created. In practice, the
        behavior of the defaults would be handled by pycalphad.

        Returns
        -------
        Any

        """
        model_dict = {}
        for phase_name, phase_model_metadata in self.phases.items():
            if phase_model_metadata.model is not None:
                model_dict[phase_name] = phase_model_metadata.model
        return model_dict
