"""
Calculate driving_force due to ZPF tielines.

The general approach is similar to the PanOptimizer rough search method.

1. With all phases active, calculate the chemical potentials of the tieline
   endpoints via ``equilibrium`` calls. Done in ``estimate_hyperplane``.
2. Calculate the target chemical potentials, which are the average chemical
   potentials of all of the current chemical potentials at the tieline endpoints.
3. Calculate the current chemical potentials of the desired single phases
4. The error is the difference between these chemical potentials

There's some special handling for tieline endpoints where we do not know the
composition conditions to calculate chemical potentials at.
"""

import operator, logging
from collections import defaultdict, OrderedDict

import numpy as np
from scipy.stats import norm
import tinydb

from pycalphad.core.problem import Problem
from pycalphad.core.composition_set import CompositionSet
from pycalphad import calculate, equilibrium, variables as v
from pycalphad.core.solver import InteriorPointSolver
from pycalphad.core.eqsolver import pointsolve
from pycalphad.core.equilibrium import _adjust_conditions
from pycalphad.codegen.callables import build_phase_records
from pycalphad.core.utils import instantiate_models

def _safe_index(items, index):
    try:
        return items[index]
    except IndexError:
        return None

class PotentialProblem(Problem):
  """
  Finds tangent hyperplane at a given chemical potential.

  Parameters
  ----------
  cs_list : list
    List of CompositionSets
  species : list
    List of species contained in the problem
  str_conds : dict
    Dictionary mapping variables to conditions
  chem_pot : np.array
    Numpy array of chemical potentials

  Methods
  ----------
  objective and gradient are overloaded. Use with the pointsolve function.
  """
  def __init__(self, cs_list, species, str_conds, chem_pot):
    super(PotentialProblem, self).__init__(cs_list, species, str_conds)
    self.chem_pot = chem_pot

  def objective(self, x_in):
    output = super(PotentialProblem, self).objective(x_in)
    mass_grad = super(PotentialProblem, self).mass_gradient(x_in)
    # TODO Remove this hack. The final entry of x_in is the total mass, 
    # the derivative with respect to which gives the composition.
    composition = mass_grad[-1,:]
    return output + np.inner(self.chem_pot, composition)

  def gradient(self, x_in):
    original_grad = super(PotentialProblem, self).gradient(x_in)
    mass_grad = super(PotentialProblem, self).mass_gradient(x_in)
    return original_grad + np.matmul(mass_grad, self.chem_pot)

def get_zpf_data(comps, phases, datasets):
    """
    Return the ZPF data used in the calculation of ZPF error

    Parameters
    ----------
    comps : list
        List of active component names
    phases : list
        List of phases to consider
    datasets : espei.utils.PickleableTinyDB
        Datasets that contain single phase data

    Returns
    -------
    list
        List of data dictionaries with keys ``weight``, ``data_comps`` and
        ``phase_regions``. ``data_comps`` are the components for the data in
        question. ``phase_regions`` are the ZPF phases, state variables and compositions.
    """
    desired_data = datasets.search((tinydb.where('output') == 'ZPF') &
                                   (tinydb.where('components').test(lambda x: set(x).issubset(comps))) &
                                   (tinydb.where('phases').test(lambda x: len(set(phases).intersection(x)) > 0)))

    zpf_data = []
    for data in desired_data:
        payload = data['values']
        conditions = data['conditions']
        # create a dictionary of each set of phases containing a list of individual points on the tieline
        # individual tieline points are tuples of (conditions, {composition dictionaries})
        phase_regions = defaultdict(lambda: list())
        # TODO: Fix to only include equilibria listed in 'phases'
        for idx, p in enumerate(payload):
            phase_key = tuple(sorted(rp[0] for rp in p))
            if len(phase_key) < 2:
                # Skip single-phase regions for fitting purposes
                continue
            # Need to sort 'p' here so we have the sorted ordering used in 'phase_key'
            # rp[3] optionally contains additional flags, e.g., "disordered", to help the solver
            comp_dicts = [(dict(zip([v.X(x.upper()) for x in rp[1]], rp[2])), _safe_index(rp, 3))
                          for rp in sorted(p, key=operator.itemgetter(0))]
            cur_conds = {}
            for key, value in conditions.items():
                value = np.atleast_1d(np.asarray(value))
                if len(value) > 1:
                    value = value[idx]
                cur_conds[getattr(v, key)] = float(value)
            phase_regions[phase_key].append((cur_conds, comp_dicts))

        data_dict = {
            'weight': data.get('weight', 1.0),
            'data_comps': list(set(data['components']).union({'VA'})),
            'phase_regions': phase_regions,
            'dataset_reference': data['reference']
        }
        zpf_data.append(data_dict)
    return zpf_data

def calculate_driving_force_at_chem_potential(chem_pot, species, phase_dict, phase_records, str_conds):
  for key in phase_records:
    solver = InteriorPointSolver()
    cs_l = CompositionSet(phase_records[key])
    potential_problem = PotentialProblem([cs_l], species, str_conds, chem_pot)
    res = pointsolve([cs_l], species, str_conds, potential_problem, solver)
  return 0

def calculate_driving_force(dbf, data_comps, phases, current_statevars, ph_cond_dict, phase_models, parameters, callables):
  # TODO Refactor absurd unpacking which represents a significant overhead.
  species = list(map(v.Species, data_comps))
  conditions = current_statevars
  if conditions.get(v.N) is None:
    conditions[v.N] = 1
  if np.any(np.array(conditions[v.N]) != 1):
    raise ConditionError('N!=1 is not yet supported, got N={}'.format(conditions[v.N]))
  conds = conditions
  str_conds = OrderedDict([(str(key), conds[key]) for key in sorted(conds.keys(), key=str)])
  models = instantiate_models(dbf, data_comps, phases, model=phase_models, parameters=parameters)
  prxs = build_phase_records(dbf, species, phases, conds, models, build_gradients=True, build_hessians=True, callables=callables)
  phase_dict = {}
  for ph, cond in ph_cond_dict:
    ph_conds = cond[0]
    for key in ph_conds:
      if ph_conds[key] is None:
        phase_dict[ph] = None
    if ph not in phase_dict:
      ph_conds.update(conditions)
      phase_records = build_phase_records(dbf, species, [ph], ph_conds, models, build_gradients=True, build_hessians=True, callables=callables)
      phase_dict[ph] = {'phase_record': phase_records[ph], 'str_conds': OrderedDict([(str(key), ph_conds[key]) for key in sorted(ph_conds.keys(), key=str)])}
  calculate_driving_force_at_chem_potential(np.array([0, 0]), species, phase_dict, prxs, str_conds)
  return 0

def calculate_zpf_error(dbf, phases, zpf_data, phase_models=None,
                        parameters=None, callables=None, data_weight=1.0,
                        ):
    """
    Calculate error due to phase equilibria data

    Parameters
    ----------
    dbf : pycalphad.Database
        Database to consider
    phases : list
        List of phases to consider
    zpf_data : list
        Datasets that contain single phase data
    phase_models : dict
        Phase models to pass to pycalphad calculations
    parameters : dict
        Dictionary of symbols that will be overridden in pycalphad.equilibrium
    callables : dict
        Callables to pass to pycalphad
    data_weight : float
        Scaling factor for the standard deviation of the measurement of a
        tieline which has units J/mol. The standard deviation is 1000 J/mol
        and the scaling factor defaults to 1.0.

    Returns
    -------
    float
        Log probability of ZPF error

    Notes
    -----
    The physical picture of the standard deviation is that we've measured a ZPF
    line. That line corresponds to some equilibrium chemical potentials. The
    standard deviation is the standard deviation of those 'measured' chemical
    potentials.

    """
    if parameters is None:
        parameters = {}
    prob_error = 0.0
    for data in zpf_data:
        phase_regions = data['phase_regions']
        data_comps = data['data_comps']
        weight = data['weight']
        dataset_ref = data['dataset_reference']
        # for each set of phases in equilibrium and their individual tieline points
        for region, region_eq in phase_regions.items():
            # for each tieline region conditions and compositions
            for current_statevars, comp_dicts in region_eq:
                # a "region" is a set of phase equilibria
                eq_str = "conds: ({}), comps: ({})".format(current_statevars, ', '.join(['{}: {}'.format(ph,c[0]) for ph, c in zip(region, comp_dicts)]))
                driving_force = calculate_driving_force(dbf, data_comps, phases, current_statevars, list(zip(region, comp_dicts)), phase_models, parameters, callables=callables)
                data_prob = norm(loc=0, scale=1000/data_weight/weight).logpdf(driving_force)
                prob_error += data_prob
                logging.debug('ZPF error - Equilibria: ({}), driving force: {}, probability: {}, reference: {}'.format(eq_str, driving_force, data_prob, dataset_ref))
    if np.isnan(prob_error):
        return -np.inf
    return prob_error

