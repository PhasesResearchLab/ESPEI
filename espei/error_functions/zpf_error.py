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
from itertools import repeat

import math
import numpy as np
from scipy.stats import norm
import tinydb
import multiprocessing as mp

from pycalphad.core.problem import Problem
from pycalphad.core.composition_set import CompositionSet
from pycalphad import calculate, equilibrium, variables as v
from pycalphad.core.solver import InteriorPointSolver
from pycalphad.core.eqsolver import pointsolve
from pycalphad.core.equilibrium import _adjust_conditions
from pycalphad.codegen.callables import build_phase_records
from pycalphad.core.utils import instantiate_models, point_sample, generate_dof, unpack_components
from pycalphad.core.calculate import _sample_phase_constitution

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
    orig_obj = super(PotentialProblem, self).objective(x_in)
    mass_grad = super(PotentialProblem, self).mass_gradient(x_in)
    # TODO Remove this hack. The final entry of x_in is the total mass, 
    # the derivative with respect to which gives the composition.
    composition = mass_grad[-1,:]
    return orig_obj - np.inner(self.chem_pot, composition)

  def gradient(self, x_in):
    original_grad = super(PotentialProblem, self).gradient(x_in)
    mass_grad = super(PotentialProblem, self).mass_gradient(x_in)
    return original_grad - np.matmul(mass_grad, self.chem_pot)

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
            'phase_regions': list(phase_regions.items()),
            'dataset_reference': data['reference']
        }
        zpf_data.append(data_dict)
    return zpf_data

def minimize_single_phase(dbf, composition_set, phase_name, phase_min_problem, species, str_conds, phase_dict, approx):
    """
    Calculate the minimum at a single phase. Samples and, if approx is false, refines 
    the sampled minimum using ipopt.

    Parameters
    ----------
    dbf: Database
      Contains a database with relevant phase models.
    composition_set: CompositionSet
      Contains data concerning the phase to be minimized.
    phase_name: string
      Name of the desired phase to be optimized.
    phase_min_problem: Problem instance
      Contains the objective to be optimized
    species: list
      A list of the species involved in the calculation
    str_conds: string
      A string representation of the conditions which are common to all phases.
      Needed for the solver.
    models: dict
      A dictionary of model objects for each phase.
    approx: bool
      Indicates whether we tolerate an approximate solution obtained via sampling or
      whether we should use ipopt to refine the approximate solution.
    """
    energy_values = phase_dict[phase_name]['energy_values']
    values = energy_values - np.matmul(phase_dict[phase_name]['composition_values'], phase_min_problem.chem_pot)
    min_ind = np.argmin(values)
    state_variables = np.array([str_conds[key] for key in sorted(str_conds.keys(), key=str)])
    composition_set.py_update(phase_dict[phase_name]['sample_points'][min_ind,:], np.array([1.0]), state_variables, False)
    if not approx:
        solver = InteriorPointSolver()
        res = pointsolve([composition_set], species, str_conds, phase_min_problem, solver)
        if not res.converged:
            logging.debug('Pointsolver failed to Converge')
            return None

class _hyperplane_data:
    """
    This class is only used in the loss function calculation.
    """
    def __init__(self, intercept, active, point_left, point_right):
        self.intercept = intercept
        self.active = active
        self.point_left = point_left
        self.point_right = point_right

def calculate_driving_force_at_chem_potential(dbf, chem_pot, species, phase_dict, phase_records, str_conds, approx=False):
    """
    Calculate the driving force assuming a particular chemical potential.

    Parameters
    ----------
    dbf: Database
      The database containing the phase model information.
    chem_pot: np.array
      The chemical potential for the calculation
    species: list
      A list of the species involved in the calculation
    phase_dict: dict
      A dictionary mapping the phases which occurred in the experiment to a phase
      record and an str_cond which correspond to the tie point measured (if the tie point is known,
      None otherwise)
    phase_records: dict
      A dictionary of phase records for all phases with no composition specified
    str_conds: string
      A string representation of the conditions which are common to all phases.
      Needed for the solver.
    models: dict
      A dictionary of model objects for each phase.
    approx: bool
      Indicates whether we tolerate an approximate solution obtained via sampling or
      whether we should use ipopt to refine the approximate solution.

    Notes
    ---------
    By fixing the chemical potential, the whole calculation decouples over the different phases.
    """
    species = sorted(species, key=str)
    hyperplane_data_list = []
    active_phase_count = 0
    total_offset = 0.0
    # Find hyperplane equilibrium for each phase.
    for key in phase_records:
        cs_l = CompositionSet(phase_records[key])
        phase_min_problem = PotentialProblem([cs_l], species, str_conds, chem_pot)
        minimize_single_phase(dbf, cs_l, key, phase_min_problem, species, str_conds, phase_dict, approx)
        lower_plane = chem_pot + cs_l.energy - np.dot(np.asarray(cs_l.X), chem_pot)
        min_point = np.asarray(cs_l.X)
        if not phase_dict[key]['data']:
            hyperplane_data_list.append(_hyperplane_data(lower_plane[0], False, -1.0*min_point, np.zeros_like(min_point)))
        elif phase_dict[key]['phase_record'] is None:
            hyperplane_data_list.append(_hyperplane_data(lower_plane[0], True, -1.0*min_point, min_point))
            active_phase_count += 1
        else:
            if not 'min_energy' in phase_dict[key] or phase_dict[key]['min_energy'] is None:
                solver = InteriorPointSolver()
                cs = CompositionSet(phase_dict[key]['phase_record'])
                phase_composition_problem = PotentialProblem([cs], species, phase_dict[key]['str_conds'], np.zeros_like(chem_pot))
                res = pointsolve([cs], species, phase_dict[key]['str_conds'], phase_composition_problem, solver)
                if not res.converged:
                    logging.debug('Pointsolver failed to Converge')
                    return None
                phase_dict[key]['min_energy'] = cs.energy
                phase_dict[key]['composition'] = np.array(cs.X)
            point_plane = chem_pot + phase_dict[key]['min_energy'] - np.dot(phase_dict[key]['composition'], chem_pot)
            hyperplane_data_list.append(_hyperplane_data(0.5*lower_plane[0] + 0.5*point_plane[0], True, -1.0*min_point, phase_dict[key]['composition']))
            total_offset += 0.5*(point_plane[0] - lower_plane[0])
            active_phase_count += 1
    hyperplane_data_list.sort(key=lambda x: x.intercept)
    driving_force = total_offset
    grad_vect = np.zeros_like(chem_pot)
    optimal_intercept = hyperplane_data_list[0].intercept
    for data in hyperplane_data_list:
        if active_phase_count == 0:
            optimal_intercept = 0.5 * optimal_intercept + 0.5 * data.intercept
        if data.active:
            active_phase_count -= 2
        else:
            active_phase_count -= 1
        if active_phase_count == 0 or (active_phase_count == -1 and data.active):
            optimal_intercept = data.intercept
    for data in hyperplane_data_list:
        if data.intercept < optimal_intercept:
            driving_force += optimal_intercept - data.intercept
            grad_vect += data.point_left
        elif data.active and data.intercept > optimal_intercept:
            driving_force += data.intercept - optimal_intercept
            grad_vect += data.point_right
    output_dict = {'driving_force': driving_force, 'grad_vect': grad_vect}
    return output_dict

def generate_random_hyperplane(species):
    """
    Generates a random initial hyperplane.

    Parameters
    ----------
    species: list
      A list of species to be used in the calculation.
    """
    dim = len([sp for sp in species if sp.__str__() != 'VA'])
    circle_sample = np.random.randn(dim)
    hyperplane = circle_sample / np.abs(circle_sample[-1])
    hyperplane[-1] = 0
    return hyperplane

def calculate_driving_force(dbf, data_comps, phases, current_statevars, ph_cond_dict, phase_models, phase_dict, parameters, callables, tol=0.001, max_it=50):
    """
    Calculates driving force for a single data point.

    Parameters
    ----------
    dbf : pycalphad.Database
        Database to consider
    data_comps : list
        List of active component names
    phases : list
        List of phases to consider
    current_statevars : dict
        Dictionary of state variables, e.g. v.P and v.T, no compositions.
    ph_cond_dict : dict
        Dictionary mapping phases to the conditions at which they occurred in experiment.
    phase_models : dict
        Phase models to pass to pycalphad calculations
    parameters : dict
        Dictionary of symbols that will be overridden in pycalphad.equilibrium
    callables : dict
        Callables to pass to pycalphad
    tol: double
        The tolerance allowed for optimization over hyperplanes.
    max_it: int
        The maximum number of iterations allowed for optimization over hyperplanes.

    Notes
    ------
    Calculates the driving force by optimizing the driving force over the chemical potential.
    Allow calculation of the driving force even when both tie points are missing.
    """
    # TODO Refactor absurd unpacking which represents a significant overhead.
    species = list(map(v.Species, data_comps))
    conditions = current_statevars
    if conditions.get(v.N) is None:
        conditions[v.N] = 1.0
    if np.any(np.array(conditions[v.N]) != 1):
        raise ConditionError('N!=1 is not yet supported, got N={}'.format(conditions[v.N]))
    conds = conditions
    str_conds = OrderedDict([(str(key), conds[key]) for key in sorted(conds.keys(), key=str)])
    models = instantiate_models(dbf, data_comps, phases, model=phase_models, parameters=parameters)
    prxs = build_phase_records(dbf, species, phases, conds, models, build_gradients=True, build_hessians=True, callables=callables, parameters=parameters)
    # Collect data information in phase_dict.
    for phase in phases:
        phase_dict[phase]['data'] = False
    for ph, cond in ph_cond_dict:
        has_nones = False
        ph_conds = cond[0]
        phase_dict[ph]['data'] = True
        for key in ph_conds:
            if ph_conds[key] is None:
                has_nones = True
                phase_dict[ph]['phase_record'] = None
                phase_dict[ph]['str_conds'] = None
        if not has_nones:
            ph_conds.update(conditions)
            phase_records = build_phase_records(dbf, species, [ph], ph_conds, models, build_gradients=True, build_hessians=True, callables=callables, parameters=parameters)
            phase_dict[ph]['phase_record'] = phase_records[ph]
            phase_dict[ph]['str_conds'] = OrderedDict([(str(key), ph_conds[key]) for key in sorted(ph_conds.keys(), key=str)])
            phase_dict[ph]['min_energy'] = None
    # Collect sampling and equilibrium information in phase_dict.
    for phase in phases:
        # If sample points have not yet been calculated for this phase, calculate them.
        if not 'sample_points' in phase_dict[phase]:
            phase_obj = dbf.phases[phase]
            components = models[phase].components
            variables, sublattice_dof = generate_dof(phase_obj, components)
            sample_points = _sample_phase_constitution(phase, phase_obj.constituents, sublattice_dof, data_comps, tuple(variables), point_sample, True, 2000)
            phase_dict[phase]['sample_points'] = sample_points
        # If composition values have not yet been calculated for this phase, calculate them.
        if not 'composition_values' in phase_dict[phase]:
            composition_values = np.zeros((sample_points.shape[0], len([sp for sp in species if sp.__str__() != 'VA'])))
            temp_comp_set = CompositionSet(prxs[phase])
            current_state_variables = np.array([str_conds[key] for key in sorted(str_conds.keys(), key=str)])
            for i in range(sample_points.shape[0]):
                temp_comp_set.py_update(sample_points[i,:], np.array([1.0]), current_state_variables, False)
                composition_values[i,:] = temp_comp_set.X
            phase_dict[phase]['composition_values'] = composition_values
        energies = calculate(dbf, data_comps, [phase], points=phase_dict[phase]['sample_points'], to_xarray=False, **str_conds)
        phase_dict[phase]['energy_values'] = np.array(energies['GM'][0][0][0])
    hyperplane = generate_random_hyperplane(species)
    result = calculate_driving_force_at_chem_potential(dbf, hyperplane, species, phase_dict, prxs, str_conds, approx=True)
    # Ignore entire data point if pointsolver fails to converge.
    if result is None:
        return 0
    # Optimize over the hyperplane.
    it = 0
    step_size = 1.0
    current_driving_force = result['driving_force']
    grad_dir = result['grad_vect']
    while step_size > tol and it < max_it:
        it += 1
        new_hyperplane = hyperplane + step_size * grad_dir / np.linalg.norm(grad_dir)
        result = calculate_driving_force_at_chem_potential(dbf, new_hyperplane, species, phase_dict, prxs, str_conds, approx=True)
        # If step results in objective decrease, double the step size until decrease becomes maximal.
        if result['driving_force'] < current_driving_force:
            while result['driving_force'] < current_driving_force:
                current_driving_force = result['driving_force']
                current_hyperplane = new_hyperplane
                current_grad_dir = result['grad_vect']
                step_size *= 2
                new_hyperplane = hyperplane + step_size * grad_dir / np.linalg.norm(grad_dir)
                result = calculate_driving_force_at_chem_potential(dbf, new_hyperplane, species, phase_dict, prxs, str_conds, approx=True)
            hyperplane = current_hyperplane
            grad_dir = current_grad_dir
        # If step results in objective increase, halve the step size until decrease is obtained
        else:
            while result['driving_force'] > current_driving_force and step_size > tol:
                step_size /= 2
                new_hyperplane = hyperplane + step_size * grad_dir / np.linalg.norm(grad_dir)
                result = calculate_driving_force_at_chem_potential(dbf, new_hyperplane, species, phase_dict, prxs, str_conds, approx=True)
            current_driving_force = result['driving_force']
            hyperplane = new_hyperplane
            grad_dir = result['grad_vect']
    final_result = calculate_driving_force_at_chem_potential(dbf, hyperplane, species, phase_dict, prxs, str_conds, approx=True)
    final_driving_force = final_result['driving_force']
    return final_driving_force

def calculate_log_data_prob(arg_list):
    """
    Wrapper function used in parallelization.

    Parameters
    ----------
    arg_list: list
        List of parameters required to calculate the data probability
    """
    # Unpack parameters from list.
    data = arg_list[0]
    dbf = arg_list[1]
    phases = arg_list[2]
    phase_models = arg_list[3]
    parameters = arg_list[4]
    callables = arg_list[5]
    data_weight = arg_list[6]
    phase_dict = arg_list[7]
    # Perform calculation.
    prob_error = 0
    phase_regions = data['phase_regions']
    data_comps = data['data_comps']
    weight = data['weight']
    dataset_ref = data['dataset_reference']
    # for each set of phases in equilibrium and their individual tieline points
    for region, region_eq in phase_regions:
        # for each tieline region conditions and compositions
        for current_statevars, comp_dicts in region_eq:
            # a "region" is a set of phase equilibria
            eq_str = "conds: ({}), comps: ({})".format(current_statevars, ', '.join(['{}: {}'.format(ph,c[0]) for ph, c in zip(region, comp_dicts)]))
            driving_force = calculate_driving_force(dbf, data_comps, phases, current_statevars, list(zip(region, comp_dicts)), phase_models, phase_dict, parameters, callables=callables)
            data_prob = norm(loc=0, scale=1000/data_weight/weight).logpdf(driving_force)
            prob_error += data_prob
            logging.debug('ZPF error - Equilibria: ({}), driving force: {}, probability: {}, reference: {}'.format(eq_str, driving_force, data_prob, dataset_ref))
    if np.isnan(prob_error):
        return -np.inf
    return prob_error

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
    # logging.info("{}".format(parameters))
    phase_dict = {}
    for phase in phases:
        phase_dict[phase] = {}
    iter_data = zip(zpf_data, repeat(dbf),
                              repeat(phases),
                              repeat(phase_models),
                              repeat(parameters),
                              repeat(callables),
                              repeat(data_weight),
                              repeat(phase_dict))
    return sum(x for x in map(calculate_log_data_prob, iter_data))

