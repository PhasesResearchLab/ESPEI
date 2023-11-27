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
    def __init__(self, fitting_steps: [FittingStep], model: Optional[Type[Model]] = Model) -> None:
        self.fitting_steps = fitting_steps
        self.model = model


molar_volume_fit_desc = ModelFittingDescription([StepV0, StepLogVA])

class ElasticModel(Model):
    def build_phase(self, dbe):
        phase = dbe.phases[self.phase_name]
        param_search = dbe.search
        for prop in ['C11', 'C12', 'C44']:
            prop_param_query = (
                (tinydb.where('phase_name') == phase.name) & \
                (tinydb.where('parameter_type') == prop) & \
                (tinydb.where('constituent_array').test(self._array_validity))
                )
            prop_val = self.redlich_kister_sum(phase, param_search, prop_param_query).subs(dbe.symbols)
            setattr(self, prop, prop_val)

elastic_fitting_description = ModelFittingDescription([StepElasticC11, StepElasticC12, StepElasticC44], model=ElasticModel)
