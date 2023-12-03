from typing import Optional, Type
from pycalphad import Model
from espei.parameter_selection.fitting_steps import *

class ModelFittingDescription():
    """

    Attributes
    ----------
    fitting_steps: [FittingStep]
    model: Type[Model]
    """
    def __init__(self, fitting_steps: [FittingStep], model: Optional[Type[Model]] = Model) -> None:
        self.fitting_steps = fitting_steps
        self.model = model


molar_volume_fitting_description = ModelFittingDescription([StepV0, StepLogVA])
gibbs_energy_fitting_description = ModelFittingDescription([StepCPM, StepSM, StepHM])
molar_volume_gibbs_energy_fitting_description = ModelFittingDescription([StepV0, StepLogVA, StepCPM, StepSM, StepHM])