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
from pycalphad.property_framework import IsolatedPhase
from pycalphad.core.utils import filter_phases, instantiate_models, unpack_species
from pycalphad.codegen.phase_record_factory import PhaseRecordFactory
from scipy.stats import norm
import tinydb

from espei.phase_models import PhaseModelSpecification
from espei.shadow_functions import calculate_, update_phase_record_parameters
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


def _disordered_phase(dbf: Database, phase_name: str) -> Optional[str]:
    """Return the disordered phase of an order-disorder model that can disorder, which is the energy of its disordered configurations, otherwise None."""
    hints = dbf.phases[phase_name].model_hints
    return None if hints.get('never_disorder') else hints.get('disordered_phase')


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

    Raises
    ------
    ValueError
        If a phase composition is "disordered" for a phase without a disordered phase.
    """
    wks = Workspace(dbf, sorted(set(comps) | {'VA'}), phases, parameters=parameters)
    # Disordered phases, e.g. FCC_A1 of FCC_L12, are phases of the system even though pycalphad hides them with their ordered phase
    modeled_phases = set(wks.phases) | {_disordered_phase(dbf, name) for name in wks.phases} - {None}
    desired_data = datasets.search((tinydb.where('output') == 'ZPF') &
                                   (tinydb.where('components').test(lambda x: set(x).issubset(comps))) &
                                   (tinydb.where('phases').test(lambda x: set(x).issubset(modeled_phases))))

    zpf_data = []  # 1:1 correspondence with each dataset
    for data in desired_data:
        current_wks = wks.copy()
        data_comps = list(set(data['components']).union({'VA'}))
        pure_elements = sorted(set(data_comps) - {'VA'})
        all_regions = data['values']
        conditions = data['conditions']
        current_wks.components = data_comps
        current_wks.conditions = conditions
        current_wks.phases = phases
        # The system's phases, and the disordered phases of the dataset's phases for their disordered phase compositions
        model_phases = set(current_wks.phases) | {_disordered_phase(dbf, name) or name for name in data['phases']}
        models = instantiate_models(dbf, current_wks.components, sorted(model_phases), model=model, parameters=current_wks.parameters)
        phase_record_factory = PhaseRecordFactory(dbf, current_wks.components, current_wks.conditions, models, parameters=current_wks.parameters)
        phase_regions = []
        # Each phase_region is one set of phases in equilibrium (on a tie-line),
        # e.g. [["ALPHA", ["B"], [0.25]], ["BETA", ["B"], [0.5]]]
        for idx, phase_region in enumerate(all_regions):
            # Conditions are one value for the dataset or one value for each phase region
            pot_conds = {getattr(v, key): float(np.atleast_1d(value)[idx if np.size(value) > 1 else 0]) for key, value in conditions.items()}
            pot_conds.setdefault(v.N, 1.0) # Add v.N condition, if missing
            # Extract all the phases and compositions from the tie-line points
            vertices = []
            hyperplane_vertices = []
            for phase_name, components, fractions, *flag in phase_region:
                comp_conds = dict(zip(map(v.X, map(str.upper, components)), fractions))
                if flag == ["disordered"]:
                    if _disordered_phase(dbf, phase_name) is None:
                        raise ValueError(f'{phase_name} is "disordered" in values[{idx}] of the ZPF dataset "{data.get("reference", "")}", but it has no disordered phase that it can disorder to')
                    phase_name = _disordered_phase(dbf, phase_name)
                # Composition of the pure elements, with unknown mole fractions as NaN and the dependent element by mass balance (N=1)
                composition = np.array([comp_conds.get(v.X(el)) for el in pure_elements], dtype=np.float64)
                if np.isnan(composition).sum() == 1:
                    composition[np.isnan(composition)] = 1 - np.nansum(composition)
                vtx = RegionVertex(phase_name, composition, comp_conds, phase_record_factory, bool(np.any(np.isnan(composition))))
                # TODO: should maybe have different types of RegionVertex that are semantic for __HYPERPLANE__, known phase composition, unknown phase composition
                if phase_name.upper() == '__HYPERPLANE__':
                    hyperplane_vertices.append(vtx)
                else:
                    vertices.append(vtx)
            if len(hyperplane_vertices) == 0:
                # Define the hyperplane at the vertices of the ZPF points
                hyperplane_vertices = vertices
            region = PhaseRegion(hyperplane_vertices, vertices, pot_conds, current_wks.components, current_wks.phases, models)
            phase_regions.append(region)

        data_dict = {
            'weight': data.get('weight', 1.0),
            'phase_regions': phase_regions,
            'dataset_reference': data.get('reference', ''),
            'dbf': dbf,  # TODO: not ideal to ship databases across the wire, but we can accept it for now.
            'parameter_dict': parameters
        }
        zpf_data.append(data_dict)
    return zpf_data


def estimate_hyperplane(phase_region: PhaseRegion, dbf: Database, parameters: np.ndarray, approximate_equilibrium: bool = False) -> np.ndarray:
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
    species = phase_region.species
    phases = phase_region.phases
    models = phase_region.models
    param_keys = list(models.values())[0]._parameters_arg
    parameters_dict = dict(zip(sorted(map(str, param_keys)), parameters))
    for vertex in phase_region.hyperplane_vertices:
        update_phase_record_parameters(vertex.phase_record_factory, parameters)
        cond_dict = {**vertex.comp_conds, **phase_region.potential_conds}
        if vertex.has_missing_comp_cond:
            # This composition is unknown -- it doesn't contribute to hyperplane estimation
            pass
        else:
            # Extract chemical potential hyperplane from multi-phase calculation
            # Note that we consider all phases in the system, not just ones in this tie region
            wks = Workspace(database=dbf, components=species, phases=phases, models=models, phase_record_factory=vertex.phase_record_factory, conditions=cond_dict, parameters=parameters_dict)
            # TODO: active_pure_elements should be replaced with wks.components when wks.components no longer includes phase constituent Species
            active_pure_elements = [list(x.constituents.keys()) for x in species]
            active_pure_elements = sorted(set(el.upper() for constituents in active_pure_elements for el in constituents) - {"VA"})
            MU_values = [wks.get(v.MU(comp)) for comp in active_pure_elements]
            num_phases = np.sum(wks.eq.Phase.squeeze() != '')
            Y_values = wks.eq.Y.squeeze()
            # TODO: we need better detection for equilibria low-rank composition
            # subspaces of the composition space of interest.
            # This one covers stoichiometric phases, but would fail to capture
            # two stoichiometric phases in a ternary or a single phase
            # pseudo-binary in a ternary (and higher order generalizations).
            no_internal_dof = np.all((np.isclose(Y_values, 1.)) | np.isnan(Y_values))
            if (num_phases == 1) and no_internal_dof:
                _log.trace("The equilibrium computed at the vertex for %s with composition conditions %s in the phase region %s has underdetermined chemical potentials and will not be considered in the target hyperplane", vertex.phase_name, vertex.comp_conds, phase_region.eq_str())
                target_hyperplane_chempots.append(np.full_like(MU_values, np.nan))
            else:
                if np.any(np.isnan(MU_values)):
                    _log.info("The equilibrium computed at the vertex for %s with composition conditions %s in the phase region %s has NaN chemical potentials %s", vertex.phase_name, vertex.comp_conds, phase_region.eq_str(), np.asarray(MU_values))
                target_hyperplane_chempots.append(MU_values)
    target_hyperplane_mean_chempots = np.nanmean(target_hyperplane_chempots, axis=0, dtype=np.float64)
    return target_hyperplane_mean_chempots


def driving_force_to_hyperplane(target_hyperplane_chempots: np.ndarray,
                                phase_region: PhaseRegion, dbf: Database, parameter_dict, vertex: RegionVertex,
                                parameters: np.ndarray, approximate_equilibrium: bool = False) -> Tuple[float,List[float]]:
    """Calculate the integrated driving force between the current hyperplane and target hyperplane.
    """
    species = phase_region.species
    models = phase_region.models
    param_keys = list(models.values())[0]._parameters_arg
    parameters_dict = dict(zip(sorted(map(str, param_keys)), parameters))
    current_phase = vertex.phase_name
    cond_dict = {**phase_region.potential_conds, **vertex.comp_conds}
    str_statevar_dict = OrderedDict([(str(key),cond_dict[key]) for key in sorted(phase_region.potential_conds.keys(), key=str)])
    phase_record_factory = vertex.phase_record_factory
    update_phase_record_parameters(phase_record_factory, parameters)
    if vertex.has_missing_comp_cond:
        # We don't have the phase composition here, so we estimate the driving force.
        # Can happen if one of the composition conditions is unknown or if the phase is
        # stoichiometric and the user did not specify a valid phase composition.
        single_eqdata = calculate_(species, [current_phase], str_statevar_dict, models, phase_record_factory, pdens=50)
        df = np.multiply(target_hyperplane_chempots, single_eqdata.X).sum(axis=-1) - single_eqdata.GM
        driving_force = float(df.max())
    else:
        wks = Workspace(database=dbf, components=species, phases=current_phase, models=models, phase_record_factory=phase_record_factory, conditions=cond_dict, parameters=parameters_dict)
        constrained_energy = wks.get(IsolatedPhase(current_phase,wks=wks)('GM'))
        driving_force = np.dot(np.squeeze(target_hyperplane_chempots), vertex.composition) - constrained_energy
    return driving_force


def calculate_zpf_driving_forces(zpf_data: Sequence[Dict[str, Any]],
                                 parameters: ArrayLike = None,
                                 approximate_equilibrium: bool = False,
                                 short_circuit: bool = False
                                 ) -> Tuple[List[List[float]], List[List[float]]]:
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
    for data in zpf_data:
        data_driving_forces = []
        data_weights = []
        weight = data['weight']
        dataset_ref = data['dataset_reference']
        # for the set of phases and corresponding tie-line verticies in equilibrium
        for phase_region in data['phase_regions']:
            # 1. Calculate the average multiphase hyperplane
            eq_str = phase_region.eq_str()
            target_hyperplane = estimate_hyperplane(phase_region, data['dbf'], parameters, approximate_equilibrium=approximate_equilibrium)
            if np.any(np.isnan(target_hyperplane)):
                _log.debug('NaN target hyperplane. Equilibria: (%s), driving force: 0.0, reference: %s.', eq_str, dataset_ref)
                data_driving_forces.extend([0]*len(phase_region.vertices))
                data_weights.extend([weight]*len(phase_region.vertices))
                continue
            # 2. Calculate the driving force to that hyperplane for each vertex
            for vertex in phase_region.vertices:
                driving_force = driving_force_to_hyperplane(target_hyperplane, phase_region, data['dbf'], data['parameter_dict'], vertex, parameters,
                                                            approximate_equilibrium=approximate_equilibrium,
                                                            )
                if np.isinf(driving_force) and short_circuit:
                    _log.debug('Equilibria: (%s), current phase: %s, hyperplane: %s, driving force: %s, reference: %s. Short circuiting.', eq_str, vertex.phase_name, target_hyperplane, driving_force, dataset_ref)
                    return [[np.inf]], [[np.inf]]
                data_driving_forces.append(driving_force)
                data_weights.append(weight)
                _log.debug('Equilibria: (%s), current phase: %s, hyperplane: %s, driving force: %s, reference: %s', eq_str, vertex.phase_name, target_hyperplane, driving_force, dataset_ref)
        driving_forces.append(data_driving_forces)
        weights.append(data_weights)
    return driving_forces, weights


def calculate_zpf_error(zpf_data: Sequence[Dict[str, Any]],
                        parameters: np.ndarray = None,
                        data_weight: int = 1.0,
                        approximate_equilibrium: bool = False) -> float:
    """
    Calculate the likelihood due to phase equilibria data.

    For detailed documentation, see ``calculate_zpf_driving_forces``

    Returns
    -------
    float
        Log probability of ZPF driving forces

    """
    if len(zpf_data) == 0:
        return 0.0
    driving_forces, weights = calculate_zpf_driving_forces(zpf_data, parameters, approximate_equilibrium, short_circuit=True)
    # Driving forces and weights are 2D ragged arrays with the shape (len(zpf_data), len(zpf_data['values']))
    driving_forces = np.concatenate(driving_forces).T
    weights = np.concatenate(weights)
    if np.any(np.logical_or(np.isinf(driving_forces), np.isnan(driving_forces))):
        return -np.inf
    log_probabilites = norm.logpdf(driving_forces, loc=0, scale=1000/data_weight/weights)
    _log.debug('Data weight: %s, driving forces: %s, weights: %s, probabilities: %s', data_weight, driving_forces, weights, log_probabilites)
    return np.sum(log_probabilites)


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
        # parameters are given in the order of symbols_to_fit; the phase records sort them by name
        self._sorted_order = np.argsort(symbols_to_fit)

    def get_residuals(self, parameters: ArrayLike) -> Tuple[List[float], List[float]]:
        if len(self.zpf_data) == 0:
            return [], []
        parameters = np.asarray(parameters)[self._sorted_order]
        driving_forces, weights = calculate_zpf_driving_forces(self.zpf_data, parameters, short_circuit=True)
        # Driving forces and weights are 2D ragged arrays with the shape (len(zpf_data), len(zpf_data['values']))
        residuals = np.concatenate(driving_forces).tolist()
        weights = np.concatenate(weights).tolist()
        return residuals, weights

    def get_likelihood(self, parameters) -> float:
        parameters = np.asarray(parameters)[self._sorted_order]
        likelihood = calculate_zpf_error(self.zpf_data, parameters, data_weight=self.weight)
        return likelihood


residual_function_registry.register(ZPFResidual)