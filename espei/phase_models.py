from typing import Dict, List, Optional, Type

from pydantic import BaseModel, PositiveFloat, PyObject
from pycalphad import Model

from espei.typing import ComponentName, PhaseName


# TODO: validate that len(model) == len(coefficients)
class ModelMetadata(BaseModel):
    sublattice_model: List[List[ComponentName]]
    sublattice_site_ratios: List[PositiveFloat]
    # Fully qualified import path for a Python class that follows the pycalphad.Model API
    model: Optional[PyObject]


class PhaseModelSpecification(BaseModel):
    """

    Examples
    --------
    >>> from espei.phase_models import PhaseModelSpecification
    >>> data = {
    ...   "components": ["CU", "MG", "VA"],
    ...   "phases": {
    ...          "LIQUID" : {
    ...             "sublattice_model": [["CU", "MG"]],
    ...             "sublattice_site_ratios": [1],
    ...             "model": "pycalphad.model.Model"
    ...          },
    ...          "FCC_A1": {
    ...             "sublattice_model": [["CU", "MG"], ["VA"]],
    ...             "sublattice_site_ratios": [1, 1]
    ...          }
    ...     }
    ... }
    ...
    >>> phase_models = PhaseModelSpecification(**data)
    >>> assert len(phase_models.phases) == 2
    >>> assert phase_models.phases["LIQUID"].sublattice_model == [["CU", "MG"]]
    >>> assert phase_models.phases["LIQUID"].sublattice_site_ratios == [1]
    >>> assert type(phase_models.phases["LIQUID"].model) is type
    >>> assert phase_models.phases["FCC_A1"].model is None
    >>> assert "LIQUID" in phase_models.get_model_dict()
    >>> assert len(phase_models.get_model_dict()) == 1
    >>> assert phase_models.get_model_dict()["LIQUID"] == phase_models.phases["LIQUID"].model
    >>> assert phase_models.phases["FCC_A1"].model is None

    """
    components: List[ComponentName]
    phases: Dict[PhaseName, ModelMetadata]

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
