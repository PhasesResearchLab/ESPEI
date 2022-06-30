from typing import Dict, List, Optional, Type

from pydantic import BaseModel, PositiveFloat, PyObject
from pycalphad import Model

from espei.typing import ComponentName, PhaseName


# TODO: validate that len(model) == len(coefficients)
class PhaseModelMetadata(BaseModel):
    sublattice_model: List[List[ComponentName]]
    coefficients: List[PositiveFloat]
    # Fully qualified import path for a Python class that follows the pycalphad.Model API
    model_class: Optional[PyObject]


class PhaseModels(BaseModel):
    components: List[ComponentName]
    phases: Dict[PhaseName, PhaseModelMetadata]

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
            if phase_model_metadata.model_class is not None:
                model_dict[phase_name] = phase_model_metadata.model_class
        return model_dict
