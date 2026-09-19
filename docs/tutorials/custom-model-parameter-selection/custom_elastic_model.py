# This module defines a model and fitting description for ESPEI to use to fit
# elastic constants for a bcc phase, where only C11, C12, and C44 are unique
# model parameters.
# On its own, this module doesn't do any fitting. Its purpose is to describe a
# model and the steps to fit the model and that will be used by ESPEI.

# First, since are trying to fit model parameters that PyCalphad doesn't know
# about yet, we need to define a new model that uses these parameters.
# If PyCalphad's Model class already implements the parameters you are
# interested in fitting, you can skip this step and pass the `Model` object to
# the `model` arguement of `ModelFittingDescription` instead (that is optional
# as `model=Model` by default).
import tinydb
from pycalphad import Model

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

# Now define the fitting steps.
# AbstractLinearPropertyStep is a base class to use for properties that use
# standard Redlich-Kister-Muggianu mixing. It works best and is easiest to use
# when you have data that map 1:1 with PyCalphad model parameters.
# By default, the features use a power expansion in temperature as a sum of
# coefficients corresponding to the variables [1, T, T^2, T^3, T^(-1)].
from espei.parameter_selection.fitting_descriptions import ModelFittingDescription
from espei.parameter_selection.fitting_steps import AbstractLinearPropertyStep

# In this case, we are happy with the features and don't need to override them,
# so we just can set the parameter and data type class variables.
class StepElasticC11(AbstractLinearPropertyStep):
    parameter_name = "C11"
    data_types_read = "C11"

class StepElasticC12(AbstractLinearPropertyStep):
    parameter_name = "C12"
    data_types_read = "C12"

class StepElasticC44(AbstractLinearPropertyStep):
    parameter_name = "C44"
    data_types_read = "C44"

# Finally, define the fitting description as the list of steps and the custom
# model. This object is the only thing we need to give to ESPEI to know how to
# fit these model parameters.
# If you wanted to fit other parameters, you could add other steps here.
elastic_fitting_description = ModelFittingDescription([StepElasticC11, StepElasticC12, StepElasticC44], model=ElasticModel)
