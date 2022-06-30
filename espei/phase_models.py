from typing import Dict, List, Optional

from pydantic import BaseModel, PositiveFloat, PyObject

from espei.constants import ComponentName, PhaseName


# TODO: validate that len(model) == len(coefficients)
class PhaseModelMetadata(BaseModel):
    sublattice_model: List[List[ComponentName]]
    coefficients: List[PositiveFloat]
    # Fully qualified import path for a Python class that follows the pycalphad.Model API
    model_class: Optional[PyObject]


class PhaseModels(BaseModel):
    components: List[ComponentName]
    phases: Dict[PhaseName, PhaseModelMetadata]