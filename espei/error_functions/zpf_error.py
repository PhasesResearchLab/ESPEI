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

import logging
from dataclasses import dataclass
from collections import OrderedDict
from typing import Sequence, Dict, Any, Union, List, Tuple, Type, Optional

import numpy as np
from numpy.typing import ArrayLike
from pycalphad import Database, Model, Workspace, variables as v
from pycalphad.property_framework import IsolatedPhase, JanssonDerivative
from pycalphad.core.utils import filter_phases, unpack_species
from pycalphad.codegen.phase_record_factory import PhaseRecordFactory
from scipy.stats import norm
import tinydb

from espei.phase_models import PhaseModelSpecification
from espei.shadow_functions import calculate_
from espei.typing import SymbolName
from espei.utils import PickleableTinyDB, database_symbols_to_fit
from .residual_base import ResidualFunction, residual_function_registry

_log = logging.getLogger(__name__)


@dataclass
class RegionVertex:
    phase_name: str
    composition: ArrayLike  # 1D of size (number nonvacant pure elements)
    comp_conds: Dict[v.X, float]
    phase_record_factory: PhaseRecordFactory
    is_disordered: bool
    has_missing_comp_cond: bool

@dataclass
class PhaseRegion:
    hyperplane_vertices: Sequence[RegionVertex]  # Vertices used to estimate the target hyperplane
    vertices: Sequence[RegionVertex]  # Vertices to compute driving forces
    potential_conds: Dict[v.StateVariable, float]
    species: Sequence[v.Species]
    phases: Sequence[str]
    models: Dict[str, Model]

    def eq_str(self):
        phase_compositions = ', '.join(f'{vtx.phase_name}: {vtx.comp_conds}' for vtx in self.vertices)
        return f"conds: ({self.potential_conds}), comps: ({phase_compositions})"


def _extract_pot_conds(all_conditions: Dict[v.StateVariable, np.ndarray], idx: int) -> Dict[v.StateVariable, float]:
    """Conditions are either scalar or 1d arrays for the conditions in the entire dataset.
    This function extracts the condition corresponding to the current region,
    based on the index in the 1d condition array.
    """
    pot_conds = {}  # e.g. v.P, v.T
    for cond_key, cond_val in all_conditions.items():
        cond_val = np.atleast_1d(np.asarray(cond_val))
        # If the conditions is an array, we want the corresponding value
        # Otherwise treat it as a scalar
        if len(cond_val) > 1:
            cond_val = cond_val[idx]
        pot_conds[getattr(v, cond_key)] = float(cond_val)
    return pot_conds


def _extract_phases_comps(vertex):
    """Extract the phase name, phase compositions and disordered flag from a vertex
    """
    if len(vertex) == 4:  # phase_flag within
        phase_name, components, compositions, flag = vertex
        if flag == "disordered":
            disordered_flag = True
        else:
            disordered_flag = False
    elif len(vertex) == 3:  # no phase_flag within
        phase_name, components, compositions = vertex
        disordered_flag = False
    else:
        raise ValueError("Wrong number of data in tie-line point")
    comp_conds = dict(zip(map(v.X, map(str.upper, components)), compositions))
    return phase_name, comp_conds, disordered_flag


def _phase_is_stoichiometric(mod):
    return all(len(subl) == 1 for subl in mod.constituents)


def _compute_vertex_composition(comps: Sequence[str], comp_conds: Dict[str, float]):
    """Compute the overall composition in a vertex assuming an N=1 normalization condition"""
    pure_elements = sorted(c for c in comps if c != 'VA')
    vertex_composition = np.empty(len(pure_elements), dtype=np.float64)
    unknown_indices = []
    for idx, el in enumerate(pure_elements):
        amt = comp_conds.get(v.X(el), None)
        if amt is None:
            unknown_indices.append(idx)
            vertex_composition[idx] = np.nan
        else:
            vertex_composition[idx] = amt
    if len(unknown_indices) == 1:
        # Determine the dependent component by mass balance
        vertex_composition[unknown_indices[0]] = 1 - np.nansum(vertex_composition)
    return vertex_composition


def get_zpf_data(dbf: Database, comps: Sequence[str], phases: Sequence[str], datasets: PickleableTinyDB, parameters: Dict[str, float], model: Optional[Dict[str, Type[Model]]] = None):
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
    parameters : dict
        Dictionary mapping symbols to optimize to their initial values
    model : Optional[Dict[str, Type[Model]]]
        Dictionary phase names to pycalphad Model classes.

    Returns
    -------
    list
        List of data dictionaries with keys ``weight``, ``phase_regions`` and ``dataset_references``.
    """
    desired_data = datasets.search((tinydb.where('output') == 'ZPF') &
                                   (tinydb.where('components').test(lambda x: set(x).issubset(comps))) &
                                   (tinydb.where('phases').test(lambda x: len(set(phases).intersection(x)) > 0)))
    wks = Workspace(dbf, comps, phases)
    if model is None:
        model = {}
        
    params_keys = []
    for key in parameters.keys():
        if not hasattr(v, key):
            setattr(v, key, v.IndependentPotential(key))
        params_keys.append(getattr(v, key))
        dbf.symbols.pop(key,None)

    zpf_data = []  # 1:1 correspondence with each dataset
    for data in desired_data:
        current_wks = wks.copy()
        data_comps = list(set(data['components']).union({'VA'}))
        all_regions = data['values']
        conditions = data['conditions']
        current_wks.components = data_comps
        current_wks.conditions = conditions
        current_wks.phases = phases
        # only init models that are defined for the phases; fallback to default pycalphad behavior if no custom model defined
        current_wks.models = {phase_name: model.get(phase_name, current_wks.models[phase_name])
                              for phase_name in current_wks.phases}
        phase_regions = []
        # Each phase_region is one set of phases in equilibrium (on a tie-line),
        # e.g. [["ALPHA", ["B"], [0.25]], ["BETA", ["B"], [0.5]]]
        for idx, phase_region in enumerate(all_regions):
            # Extract the conditions for entire phase region
            pot_conds = _extract_pot_conds(conditions, idx)
            pot_conds.setdefault(v.N, 1.0) # Add v.N condition, if missing
            # Extract all the phases and compositions from the tie-line points
            vertices = []
            hyperplane_vertices = []
            for vertex in phase_region:
                phase_name, comp_conds, disordered_flag = _extract_phases_comps(vertex)
                composition = _compute_vertex_composition(data_comps, comp_conds)
                # TODO: should maybe have different types of RegionVertex that are semantic for __HYPERPLANE__, known phase composition, unknown phase composition
                if phase_name.upper() == '__HYPERPLANE__':
                    if np.any(np.isnan(composition)):  # TODO: make this a part of the dataset checker
                        raise ValueError(f"__HYPERPLANE__ vertex ({vertex}) must have all independent compositions defined to make a well-defined hyperplane (from dataset: {data})")
                    vtx = RegionVertex(phase_name, composition, comp_conds, current_wks.phase_record_factory, disordered_flag, False)
                    hyperplane_vertices.append(vtx)
                    continue
                mod = current_wks.models[phase_name]
                if np.any(np.isnan(composition)):
                    has_missing_comp_cond = True
                elif _phase_is_stoichiometric(mod):
                    has_missing_comp_cond = False
                else:
                    has_missing_comp_cond = False
                vtx = RegionVertex(phase_name, composition, comp_conds, current_wks.phase_record_factory, disordered_flag, has_missing_comp_cond)
                vertices.append(vtx)
            if len(hyperplane_vertices) == 0:
                # Define the hyperplane at the vertices of the ZPF points
                hyperplane_vertices = vertices
            region = PhaseRegion(hyperplane_vertices, vertices, pot_conds, current_wks.components, current_wks.phases, current_wks.models.unwrap())
            phase_regions.append(region)

        data_dict = {
            'weight': data.get('weight', 1.0),
            'phase_regions': phase_regions,
            'dataset_reference': data['reference'],
            'dbf': dbf,  # TODO: not ideal to ship databases across the wire, but we can accept it for now.
            'parameter_dict': parameters
        }
        zpf_data.append(data_dict)
    return zpf_data


def estimate_hyperplane(phase_region: PhaseRegion, dbf: Database, parameter_dict, parameters: np.ndarray, approximate_equilibrium: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the chemical potentials for the target hyperplane, one vertex at a time

    Notes
    -----
    This takes just *one* set of phase equilibria, a phase region, e.g. a dataset point of
    [['FCC_A1', ['CU'], [0.1]], ['LAVES_C15', ['CU'], [0.3]]]
    and calculates the chemical potentials given all the phases possible at the
    given compositions. Then the average chemical potentials of each end point
    are taken as the target hyperplane for the given equilibria.

    """
    target_hyperplane_chempots = []
    target_hyperplane_chempots_grads = []
    species = phase_region.species
    phases = phase_region.phases
    models = phase_region.models
    for vertex in phase_region.hyperplane_vertices:
        cond_dict = {**vertex.comp_conds, **phase_region.potential_conds}
        params_keys = list(parameter_dict.keys())
        for index in range(len(parameters)):
            cond_dict.update({params_keys[index]: parameters[index]})
        if vertex.has_missing_comp_cond:
            # This composition is unknown -- it doesn't contribute to hyperplane estimation
            pass
        else:
            # Extract chemical potential hyperplane from multi-phase calculation
            # Note that we consider all phases in the system, not just ones in this tie region
            wks = Workspace(database=dbf, components=species, phases=phases, models=models, phase_record_factory=vertex.phase_record_factory, conditions=cond_dict)
            # TODO: active_pure_elements should be replaced with wks.components when wks.components no longer includes phase constituent Species
            active_pure_elements = [list(x.constituents.keys()) for x in species]
            active_pure_elements = sorted(set(el.upper() for constituents in active_pure_elements for el in constituents) - {"VA"})
            MU_values = [wks.get(v.MU(comp)) for comp in active_pure_elements] 
            gradient_params = [JanssonDerivative(v.MU(spc), key) for spc in active_pure_elements for key in parameter_dict]
            gradients = wks.get(*gradient_params)
            if type(gradients) is list:
                gradients_magnitude = [float(element) for element in gradients]
            else:
                gradients_magnitude = gradients
            gradients_magnitude = np.reshape(gradients_magnitude,(len(MU_values),len(parameter_dict)))
            num_phases = np.sum(wks.eq.Phase.squeeze() != '')
            Y_values = wks.eq.Y.squeeze()
            no_internal_dof = np.all((np.isclose(Y_values, 1.)) | np.isnan(Y_values))
            if (num_phases == 1) and no_internal_dof:
                target_hyperplane_chempots.append(np.full_like(MU_values, np.nan))
            else:
                target_hyperplane_chempots.append(MU_values)
                target_hyperplane_chempots_grads.append(gradients_magnitude)
    target_hyperplane_mean_chempots = np.nanmean(target_hyperplane_chempots, axis=0, dtype=np.float64)
    target_hyperplane_chempots_grads = np.nanmean(target_hyperplane_chempots_grads, axis=0, dtype=np.float64) 
    return target_hyperplane_mean_chempots, target_hyperplane_chempots_grads


def driving_force_to_hyperplane(target_hyperplane_chempots: np.ndarray, target_hyperplane_chempots_grads: np.ndarray,
                                phase_region: PhaseRegion, dbf: Database, parameter_dict, vertex: RegionVertex,
                                parameters: np.ndarray, approximate_equilibrium: bool = False) -> Tuple[float,List[float]]:
    """Calculate the integrated driving force between the current hyperplane and target hyperplane.
    """
    species = phase_region.species
    models = phase_region.models
    current_phase = vertex.phase_name
    cond_dict = {**phase_region.potential_conds, **vertex.comp_conds}
    params_keys = list(parameter_dict.keys())
    for index in range(len(parameters)):
        cond_dict.update({params_keys[index]: parameters[index]})
    str_statevar_dict = OrderedDict([(str(key),cond_dict[key]) for key in sorted(phase_region.potential_conds.keys(), key=str)])
    phase_record_factory = vertex.phase_record_factory
    if vertex.has_missing_comp_cond:
        for index in range(len(parameters)):
            str_statevar_dict.update({params_keys[index]: parameters[index]})
        # We don't have the phase composition here, so we estimate the driving force.
        # Can happen if one of the composition conditions is unknown or if the phase is
        # stoichiometric and the user did not specify a valid phase composition.
        single_eqdata = calculate_(species, [current_phase], str_statevar_dict, models, phase_record_factory, pdens=50) ## SOMETHING WEIRD HAPPENS WHEN PDENS IS TOO HIGH!
        df = np.multiply(target_hyperplane_chempots, single_eqdata.X).sum(axis=-1) - single_eqdata.GM
        
        if np.squeeze(single_eqdata.X).ndim == 1:
            vertex_comp_estimate = np.squeeze(single_eqdata.X)
        elif (np.isnan(df).all()):
            driving_force = np.nan
            driving_force_gradient = np.full(len(parameters), np.nan)
            return driving_force, driving_force_gradient
        else:
            vertex_comp_estimate = np.squeeze(single_eqdata.X)[np.nanargmax(df),:]
            
        counter = 0
        for comp in species:
            if v.Species(comp) != v.Species('VA'):
                if v.X(comp) in cond_dict.keys():
                    if vertex_comp_estimate[counter] < 5e-6:
                        vertex_comp_estimate[counter] = 5e-6
                    elif vertex_comp_estimate[counter] > 1 - 5e-6:
                        vertex_comp_estimate[counter] = 1 - 5e-6
                    cond_dict.update({v.X(comp): vertex_comp_estimate[counter]})
                counter = counter + 1
                
        # local_conds = dict(zip(single_eqdata.components, single_eqdata.X))
            
        wks = Workspace(database=dbf, components=species, phases=current_phase,  phase_record_factory=phase_record_factory, conditions=cond_dict)
        constrained_energy = wks.get(IsolatedPhase(current_phase,wks=wks)('GM'))
        driving_force = np.dot(np.squeeze(target_hyperplane_chempots), vertex_comp_estimate) - constrained_energy
        ip = IsolatedPhase(current_phase, wks=wks)
        constrained_energy_gradient = []
        for key in parameter_dict:
            constrained_energy_gradient.append(wks.get(ip('GM.'+key)))
        
        driving_force_gradient = np.squeeze(np.matmul(vertex_comp_estimate,target_hyperplane_chempots_grads) - constrained_energy_gradient)
        
    elif vertex.is_disordered:
        # Construct disordered sublattice configuration from composition dict
        # Compute energy
        # Compute residual driving force
        # TODO: Check that it actually makes sense to declare this phase 'disordered'
        num_dof = sum([len(subl) for subl in models[current_phase].constituents])
        desired_sitefracs = np.ones(num_dof, dtype=np.float64)
        dof_idx = 0
        for subl in models[current_phase].constituents:
            dof = sorted(subl, key=str)
            num_subl_dof = len(subl)
            if v.Species("VA") in dof:
                if num_subl_dof == 1:
                    _log.debug('Cannot predict the site fraction of vacancies in the disordered configuration %s of %s. Returning driving force of zero.', subl, current_phase)
                    return 0
                else:
                    sitefracs_to_add = [1.0]
            else:
                sitefracs_to_add = np.array([cond_dict.get(v.X(d)) for d in dof], dtype=np.float64)
                # Fix composition of dependent component
                sitefracs_to_add[np.isnan(sitefracs_to_add)] = 1 - np.nansum(sitefracs_to_add)
            desired_sitefracs[dof_idx:dof_idx + num_subl_dof] = sitefracs_to_add
            dof_idx += num_subl_dof
        single_eqdata = calculate_(species, [current_phase], str_statevar_dict, models, phase_record_factory, points=np.asarray([desired_sitefracs]))
        driving_force = np.multiply(target_hyperplane_chempots, single_eqdata.X).sum(axis=-1) - single_eqdata.GM
        driving_force = float(np.squeeze(driving_force))
    else:
        wks = Workspace(database=dbf, components=species, phases=current_phase, models=models, phase_record_factory=phase_record_factory, conditions=cond_dict)
        constrained_energy = wks.get(IsolatedPhase(current_phase,wks=wks)('GM'))
        driving_force = np.dot(np.squeeze(target_hyperplane_chempots), vertex.composition) - constrained_energy
        constrained_energy_gradient = []
        for key in parameter_dict:
            constrained_energy_gradient.append(wks.get(IsolatedPhase(current_phase,wks=wks)('GM.'+key)))
        driving_force_gradient = np.squeeze(np.matmul(np.reshape(vertex.composition,(1,-1)),target_hyperplane_chempots_grads) - constrained_energy_gradient)
    return driving_force, driving_force_gradient


def calculate_zpf_driving_forces(zpf_data: Sequence[Dict[str, Any]],
                                 parameters: ArrayLike = None,
                                 approximate_equilibrium: bool = False,
                                 short_circuit: bool = False
                                 ) -> Tuple[List[List[float]], List[List[float]], List[List[float]]]:
    """
    Calculate error due to phase equilibria data

    zpf_data : Sequence[Dict[str, Any]]
        Datasets that contain single phase data
    parameters : ArrayLike
        Array of parameters to calculate the error with.
    approximate_equilibrium : bool
        Whether or not to use an approximate version of equilibrium that does
        not refine the solution and uses ``starting_point`` instead.
    short_circuit: bool
        If True, immediately return a size 1 array with a driving force of
        ``np.nan`` (failed hyperplane) or ``np.inf`` (failed driving force).
        Can save computational time if the caller will aggregate driving forces.

    Returns
    -------
    Tuple[List[List[float]], List[List[float]]]
        Driving forces and weights as ragged 2D arrays with shape
        ``(len(zpf_data), len(vertices in each zpf_data))``

    Notes
    -----
    The physical picture of the standard deviation is that we've measured a ZPF
    line. That line corresponds to some equilibrium chemical potentials. The
    standard deviation is the standard deviation of those 'measured' chemical
    potentials.

    """
    if parameters is None:
        parameters = np.array([])
    driving_forces = []
    weights = []
    gradients = []
    for data in zpf_data:
        data_driving_forces = []
        data_weights = []
        data_gradients = []
        weight = data['weight']
        dataset_ref = data['dataset_reference']
        # for the set of phases and corresponding tie-line verticies in equilibrium
        for phase_region in data['phase_regions']:
            # 1. Calculate the average multiphase hyperplane
            eq_str = phase_region.eq_str()
            target_hyperplane, hyperplane_grads = estimate_hyperplane(phase_region, data['dbf'], data['parameter_dict'], parameters, approximate_equilibrium=approximate_equilibrium)
            if np.any(np.isnan(target_hyperplane)):
                _log.debug('NaN target hyperplane. Equilibria: (%s), driving force: 0.0, reference: %s.', eq_str, dataset_ref)
                data_driving_forces.extend([0]*len(phase_region.vertices))
                data_weights.extend([weight]*len(phase_region.vertices))
                if len(parameters) == 1:
                    data_gradients.extend(np.zeros(len(phase_region.vertices)))
                else:
                    data_gradients.extend([[0]*len(parameters)]*len(phase_region.vertices))
                continue
            # 2. Calculate the driving force to that hyperplane for each vertex
            for vertex in phase_region.vertices:
                driving_force, df_grad = driving_force_to_hyperplane(target_hyperplane, hyperplane_grads, phase_region, data['dbf'], data['parameter_dict'], vertex, parameters, approximate_equilibrium=approximate_equilibrium)
                if np.isinf(driving_force) and short_circuit:
                    _log.debug('Equilibria: (%s), current phase: %s, hyperplane: %s, driving force: %s, reference: %s. Short circuiting.', eq_str, vertex.phase_name, target_hyperplane, driving_force, dataset_ref)
                    return [[np.inf]], [[np.inf]]
                data_driving_forces.append(driving_force)
                data_weights.append(weight)
                data_gradients.append(df_grad)
                _log.debug('Equilibria: (%s), current phase: %s, hyperplane: %s, driving force: %s, reference: %s', eq_str, vertex.phase_name, target_hyperplane, driving_force, dataset_ref)
        driving_forces.append(data_driving_forces)
        weights.append(data_weights)
        gradients.append(data_gradients)
    return driving_forces, weights, gradients


def calculate_zpf_error(zpf_data: Sequence[Dict[str, Any]],
                        parameters: np.ndarray = None,
                        data_weight: int = 1.0,
                        approximate_equilibrium: bool = False) -> Tuple[float, List[float]]:
    """
    Calculate the likelihood due to phase equilibria data.

    For detailed documentation, see ``calculate_zpf_driving_forces``

    Returns
    -------
    float
        Log probability of ZPF driving forces

    """
    if len(zpf_data) == 0:
        return 0.0, []
    driving_forces, weights, gradients = calculate_zpf_driving_forces(zpf_data, parameters, approximate_equilibrium, short_circuit=True)
    # Driving forces and weights are 2D ragged arrays with the shape (len(zpf_data), len(zpf_data['values']))
    driving_forces = np.concatenate(driving_forces).T
    weights = np.concatenate(weights)
    gradients = np.concatenate(gradients)
    if np.any(np.logical_or(np.isinf(driving_forces), np.isnan(driving_forces))):
        if len(parameters) == 1:
            return -np.inf, -np.inf
        else:
            return -np.inf, np.ones(len(parameters))*(-np.inf)
        
    log_probabilites = norm.logpdf(driving_forces, loc=0, scale=1000/data_weight/weights)
    grad_log_probs = -driving_forces*gradients.T/(1000/data_weight/weights)**2

    if grad_log_probs.ndim == 1:
        grad_log_probs_sum = np.sum(grad_log_probs, axis =0)
    else:
        grad_log_probs_sum = np.sum(grad_log_probs, axis =1)
    _log.debug('Data weight: %s, driving forces: %s, weights: %s, probabilities: %s', data_weight, driving_forces, weights, log_probabilites)
    return np.sum(log_probabilites), grad_log_probs_sum


class ZPFResidual(ResidualFunction):
    def __init__(
        self,
        database: Database,
        datasets: PickleableTinyDB,
        phase_models: Union[PhaseModelSpecification, None],
        symbols_to_fit: Optional[List[SymbolName]] = None,
        weight: Optional[Dict[str, float]] = None,
        ):
        super().__init__(database, datasets, phase_models, symbols_to_fit, weight)
        if weight is not None:
            self.weight = weight.get("ZPF", 1.0)
        else:
            self.weight = 1.0
        if phase_models is not None:
            comps = sorted(phase_models.components)
            model_dict = phase_models.get_model_dict()
        else:
            comps = sorted(database.elements)
            model_dict = dict()
        phases = sorted(filter_phases(database, unpack_species(database, comps), database.phases.keys()))
        if symbols_to_fit is None:
            symbols_to_fit = database_symbols_to_fit(database)
        # okay if parameters are initialized to zero, we only need the symbol names
        parameters = dict(zip(symbols_to_fit, [0]*len(symbols_to_fit)))
        self.zpf_data = get_zpf_data(database, comps, phases, datasets, parameters, model_dict)

    def get_residuals(self, parameters: ArrayLike) -> Tuple[List[float], List[float]]:
        driving_forces, weights, grads = calculate_zpf_driving_forces(self.zpf_data, parameters, short_circuit=True)
        # Driving forces and weights are 2D ragged arrays with the shape (len(zpf_data), len(zpf_data['values']))
        residuals = np.concatenate(driving_forces).tolist()
        weights = np.concatenate(weights).tolist()
        return residuals, weights

    def get_likelihood(self, parameters) -> Tuple[float, List[float]]:
        likelihood, gradients = calculate_zpf_error(self.zpf_data, parameters, data_weight=self.weight)
        return likelihood, gradients


residual_function_registry.register(ZPFResidual)
