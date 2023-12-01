from typing import Optional, Type
from pycalphad import Model
import tinydb
from espei.parameter_selection.fitting_steps import *

# TODO: would it make sense to have ModelFittingDescription.fitting_steps to
# have type [Union[ModelFittingDescription, FittingStep]]? The idea being
# that we could compose multiple individual fitting descriptions into a larger
# one. We'd need to be able to resolve what happens if there are different model
# objects used.
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


molar_volume_fit_desc = ModelFittingDescription([StepV0, StepLogVA])
gibbs_energy_fitting_description = ModelFittingDescription([StepCPM, StepSM, StepHM])
molar_volume_gibbs_energy_fitting_description = ModelFittingDescription([StepV0, StepLogVA, StepCPM, StepSM, StepHM])