"""
Author: Jonathan Siegel

Contains classes and functions for optimizing at chemical potentials.
"""

import numpy as np
from pycalphad.core.utils import point_sample
from pycalphad.core.constants import MIN_SITE_FRACTION

class PhaseConditions:
  """
  Contains data about a given phase at a given temperature and pressure.
  Evaluates objectives, gradients and Hessians of the Gibbs Free energy.
  Also contains information about the sublattices and site fractions.
  This is passed to the optimization function as an input.
  """
  def __init__(self, models, phase, state_vars, obj_f, grad_f, hess_f, parameters):
    self.state_vars = state_vars
    self.obj_f = obj_f
    self.grad_f = grad_f
    self.hess_f = hess_f
    self.phase = phase
    self.parameters = np.array(list(parameters))
    # Collect information about sublattices and which components they contain.
    self.site_fraction_counts = []
    self.site_ratios = {}
    ind = 0
    while True:
      count = len([x for x in models[phase].site_fractions if x.sublattice_index == ind])
      if count == 0:
        break
      self.site_fraction_counts.append(count)
      ind += 1
    # Generate matrix mapping site fractions to composition. Remove vacancies.
    self.composition_conversion_matrix = \
          np.zeros([len([x for x in models[phase].components if x.number_of_atoms > 0]), \
                    len(models[phase].site_fractions)])
    # Collect site ratios.
    for i, x in enumerate(models[phase].site_fractions):
      if x.species.number_of_atoms > 0:
        self.site_ratios[x.sublattice_index] = models[phase].site_ratios[x.sublattice_index]
    # Normalize the site ratios to sum to 1.
    self.site_ratios = np.array(list(self.site_ratios.values()))
    self.site_ratios /= np.sum(self.site_ratios)
    for i, x in enumerate(models[phase].site_fractions):
      if x.species.number_of_atoms > 0:
        comp_idx = models[phase].components.index(x.species)
        self.composition_conversion_matrix[comp_idx][i] = self.site_ratios[x.sublattice_index]

  def obj(self, x):
    x[x <= 0] = MIN_SITE_FRACTION
    # If the input is an array of compositions, we need to broadcast model evaluation.
    if len(x.shape) == 1:
      return self.obj_f(np.concatenate((self.state_vars, x, self.parameters)))
    else:
      temp = np.broadcast_to(self.state_vars, x.shape[:-1]+self.state_vars.shape)
      temp_parameters = np.broadcast_to(self.parameters, x.shape[:-1]+self.parameters.shape)
      return self.obj_f(np.concatenate((temp, x, temp_parameters), axis=-1))

  def grad(self, x):
    x[x <= 0] = MIN_SITE_FRACTION
    return np.array(self.grad_f(np.concatenate((self.state_vars, x, self.parameters))))[self.state_vars.size:]

  def hess(self, x):
    x[x <= 0] = MIN_SITE_FRACTION
    return np.array(self.hess_f(np.concatenate((self.state_vars, x, self.parameters))))[self.state_vars.size:,self.state_vars.size:]

  def set_state_vars(self, state_vars):
    self.state_vars = state_vars

def single_phase_opt_at_chem_pot(phase_conditions, chemical_potentials, pdof=1000):
  """
  Calculates the optimal gibbs energy plane for a single phase at a given chemical potential.
  Begins by sampling and then refines the estimate using Newton's method.
  """
  sample_points = point_sample(phase_conditions.site_fraction_counts, pdof=pdof)
  sample_energy = phase_conditions.obj(sample_points)
  sample_compositions = np.matmul(sample_points, phase_conditions.composition_conversion_matrix.transpose())
  sample_obj = sample_energy[:,np.newaxis] - np.matmul(sample_compositions, chemical_potentials)
  min_indices = np.argmin(sample_obj, axis=0)
  offset = (sample_compositions[min_indices].transpose() * chemical_potentials).sum(axis=0)
  return chemical_potentials + (sample_energy[min_indices] - offset)

def single_phase_energy_at_composition(phase_conditions, composition, pdof=1000):
  if phase_conditions.site_fraction_counts == [2]:
    return phase_conditions.obj(composition)    
  if phase_conditions.site_fraction_counts == [2,1]:
    return phase_conditions.obj(np.append(composition, [1.0]))
  if phase_conditions.site_fraction_counts == [1,1]:
    return phase_conditions.obj(np.array([1.0, 1.0]))
  if phase_conditions.site_fraction_counts == [2,2]:
    if composition.size > 2:
      raise NotImplementedError('This has not yet been implemented.')
    minimum = max(0, (composition[0] - phase_conditions.site_ratios[1]) / phase_conditions.site_ratios[0])
    maximum = min(1.0, (composition[0]) / phase_conditions.site_ratios[0])
    points = np.linspace(minimum, maximum, pdof)
    second_points = (composition[0] - points * phase_conditions.site_ratios[0]) / phase_conditions.site_ratios[1]
    energies = phase_conditions.obj(np.column_stack((points, 1.0 - points, second_points, 1.0 - second_points)))
    return np.min(energies)
  else:
    raise NotImplementedError('This has not yet been implemented.') 
