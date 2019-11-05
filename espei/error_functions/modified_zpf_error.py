"""
Author: Jonathan Siegel

A modified zpf error which avoids equilibrium calculations.
It works as follows:

1. Find hyperplane passing through observed tie points at
energies predicted by the model.

2. Calculate hyperplane predicted by the model at the given chemical
potential.

3. Return energy gap between the two hyperplanes.
"""

import operator, logging
from collections import defaultdict, OrderedDict

import numpy as np
from scipy.stats import norm

from .zpf_error import get_zpf_data
from pycalphad.core.errors import ConditionError
from pycalphad import calculate, equilibrium, variables as v
from pycalphad.core.utils import instantiate_models
from pycalphad.codegen.callables import build_callables, build_phase_records

from .chemical_potential_optimizer import PhaseConditions, single_phase_opt_at_chem_pot, single_phase_energy_at_composition

"""
def single_phase_opt_at_chem_pot(value_funct, grad_funct, hess_funct, chempotential):
  print(value_funct(np.array([1.0, 101325.0, 1.0, 0.2, 0.3, 0.5])))
  return 0

def calculate_opts_at_chem_pot(dbf, comps, phases, chempotential, models=None):
  """"""
  Calculates the equilibrium state at a given chemical potential.
  """"""
  if (len(chempotential.shape) != 1):
    raise ConditionError('The chemical potential must be a vector.')
  if (len(comps) != chempotential.size + 1):
    raise ConditionError('The number of componentents and the chemical potential must match.')

  models = instantiate_models(dbf, comps, phases, model=models)
  print(comps)
  model_functions = build_callables(dbf, comps, phases, models, build_hessians=True, additional_statevars={v.P, v.T, v.N})
  phase_optima = {}
  for phase in phases:
    phase_optima[phase] = single_phase_opt_at_chem_pot(model_functions['GM']['callables'][phase],
                                                       model_functions['GM']['grad_callables'][phase],
                                                       model_functions['GM']['hess_callables'][phase],
                                                       chempotential)
  return phase_optima
"""

def calculate_zpf_error(dbf, phases, zpf_data, phase_models=None,
                        parameters=None, callables=None, data_weight=1.0, chem_samples=100
                        ):
  print('Running new zpf')
  comps = ['CU', 'MG', 'VA']
  models = instantiate_models(dbf, comps, phases, parameters=parameters)
  cbs = build_callables(dbf, comps, phases, models=models, parameter_symbols=parameters.keys(), 
                        additional_statevars={v.P, v.T, v.N}, build_gradients=True, build_hessians=True)
  if parameters is None:
    parameters = {}
  prob_error = 0
  phase_condition_data = {}
  phase_chem_pot_data = {}
  for phase in phases:
    phase_condition_data[phase] = PhaseConditions(models, phase, np.array([1.0, 101325.0, 300.0]), cbs['GM']['callables'][phase],
                                                                                                   cbs['GM']['grad_callables'][phase],
                                                                                                   cbs['GM']['hess_callables'][phase],
                                                                                                   parameters.values())
  prob_error = 0.0
  for data in zpf_data:
    # unpack the data
    phase_regions = data['phase_regions']
    data_comps = data['data_comps']
    if set(data_comps) != set(comps):
      print('Warning: Only implemented for CU-MG')
      return 0
    weight = data['weight']
    dataset_ref = data['dataset_reference']
    for region, region_eq in phase_regions.items():
      for current_statevars, comp_dicts in region_eq:
        state_array = np.array([1.0, current_statevars[v.P], current_statevars[v.T]])
        for phase in phases:
          phase_condition_data[phase].set_state_vars(state_array)
        sample_chemical_potentials = np.column_stack((np.linspace(-10000.0, 10000.0, chem_samples), np.zeros(chem_samples))).transpose()
        for phase in phases:
          phase_chem_pot_data[phase] = single_phase_opt_at_chem_pot(phase_condition_data[phase], sample_chemical_potentials)
        chem_pot_errors = np.zeros(chem_samples)
        chem_pot_minima = np.minimum.reduce(list(phase_chem_pot_data.values()))
        for current_phase, cond_dict in zip(region, comp_dicts):
          cond_dict, phase_flag = cond_dict
          for key, val in cond_dict.items():
            if key.__str__() != 'X_MG' and key.__str__() != 'X_CU':
              print(key)
              print('Not Implemented yet.')
              continue
            if val is None or val is np.nan:
              chem_pot_errors += np.square((phase_chem_pot_data[current_phase][0,:] - chem_pot_minima[0,:]).transpose())
            else:
              # X_MG is listed second.
              current_composition = np.array([1.0-val, val])
              if key.__str__() == 'X_CU':
                current_composition = np.array([val, 1.0-val])
              # If the driving force is negative, set it to zero. This is a hack since our sampling may not be dense enough. Should not be necessary when Newton's method has been implemented.
              chem_pot_errors += np.square(np.maximum(0, single_phase_energy_at_composition(phase_condition_data[current_phase], current_composition) \
                                 - np.matmul(chem_pot_minima.transpose(), current_composition)))
        driving_force = np.min(np.sqrt(chem_pot_errors))
        vertex_prob = norm(loc=0, scale=1000/data_weight/weight).logpdf(driving_force)
        prob_error += vertex_prob
  if np.isnan(prob_error):
    return -np.inf
  return prob_error
