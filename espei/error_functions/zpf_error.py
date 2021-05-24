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
from typing import Sequence, Dict, Any, Union, List, Tuple

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm
import tinydb

from pycalphad import Database, Model, variables as v
from pycalphad.codegen.callables import build_phase_records
from pycalphad.core.utils import instantiate_models, filter_phases, unpack_components
from pycalphad.core.phase_rec import PhaseRecord
from espei.utils import PickleableTinyDB
from espei.shadow_functions import equilibrium_, calculate_, no_op_equilibrium_, update_phase_record_parameters, constrained_equilibrium
from pycalphad.core.calculate import _sample_phase_constitution
from pycalphad.core.utils import point_sample

_log = logging.getLogger(__name__)


@dataclass
class RegionVertex:
    phase_name: str
    comp_conds: Dict[v.X, float]
    points: ArrayLike
    phase_records: Dict[str, PhaseRecord]
    is_disordered: bool
    has_missing_comp_cond: bool

@dataclass
class PhaseRegion:
    vertices: Sequence[RegionVertex]
    potential_conds: Dict[v.StateVariable, float]
    dbf: Database
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


def _phase_is_stoichiometric(dbf, species, phase_name):
    phase_constituents = dbf.phases[phase_name].constituents
    # phase constituents must be filtered to only active:
    constituents = [[sp.name for sp in sorted(subl_constituents.intersection(species))] for subl_constituents in phase_constituents]
    return all(len(subl) == 1 for subl in constituents)


def get_zpf_data(dbf: Database, comps: Sequence[str], phases: Sequence[str], datasets: PickleableTinyDB, parameters: Dict[str, float]):
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

    zpf_data = []  # 1:1 correspondence with each dataset
    for data in desired_data:
        data_comps = list(set(data['components']).union({'VA'}))
        species = sorted(unpack_components(dbf, data_comps), key=str)
        data_phases = filter_phases(dbf, species, candidate_phases=phases)
        models = instantiate_models(dbf, species, data_phases, parameters=parameters)
        all_regions = data['values']
        conditions = data['conditions']
        phase_regions = []
        # Each phase_region is one set of phases in equilibrium (on a tie-line),
        # e.g. [["ALPHA", ["B"], [0.25]], ["BETA", ["B"], [0.5]]]
        for idx, phase_region in enumerate(all_regions):
            # Extract the conditions for entire phase region
            pot_conds = _extract_pot_conds(conditions, idx)
            pot_conds.setdefault(v.N, 1.0) # Add v.N condition, if missing
            # Extract all the phases and compositions from the tie-line points
            vertices = []
            for vertex in phase_region:
                phase_name, comp_conds, disordered_flag = _extract_phases_comps(vertex)
                phase_recs = build_phase_records(dbf, species, data_phases, {**pot_conds, **comp_conds}, models, parameters=parameters, build_gradients=True, build_hessians=True)
                # Construct single-phase points satisfying the conditions for each phase in the region
                if any(val is None for val in comp_conds.values()):
                    # We can't construct points because we don't have a known composition
                    has_missing_comp_cond = True
                    phase_points = None
                elif _phase_is_stoichiometric(dbf, species, phase_name):
                    has_missing_comp_cond = False
                    phase_points = None
                else:
                    has_missing_comp_cond = False
                    mod = models[phase_name]
                    phase_points = _sample_phase_constitution(mod, point_sample, True, 50)
                vtx = RegionVertex(phase_name, comp_conds, phase_points, phase_recs, disordered_flag, has_missing_comp_cond)
                vertices.append(vtx)
            region = PhaseRegion(vertices, pot_conds, dbf, species, data_phases, models)
            phase_regions.append(region)

        data_dict = {
            'weight': data.get('weight', 1.0),
            'data_comps': data_comps,
            'phase_regions': phase_regions,
            'dataset_reference': data['reference']
        }
        zpf_data.append(data_dict)
    return zpf_data


def estimate_hyperplane(phase_region: PhaseRegion, parameters: np.ndarray, approximate_equilibrium: bool = False) -> np.ndarray:
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
    if approximate_equilibrium:
        _equilibrium = no_op_equilibrium_
    else:
        _equilibrium = equilibrium_
    target_hyperplane_chempots = []
    target_hyperplane_phases = []
    dbf = phase_region.dbf
    species = phase_region.species
    phases = phase_region.phases
    models = phase_region.models
    for vertex in phase_region.vertices:
        phase_records = vertex.phase_records
        update_phase_record_parameters(phase_records, parameters)
        cond_dict = {**vertex.comp_conds, **phase_region.potential_conds}
        if vertex.has_missing_comp_cond:
            # This composition is unknown -- it doesn't contribute to hyperplane estimation
            pass
        else:
            # Extract chemical potential hyperplane from multi-phase calculation
            # Note that we consider all phases in the system, not just ones in this tie region
            str_statevar_dict = OrderedDict([(str(key), cond_dict[key]) for key in sorted(phase_region.potential_conds.keys(), key=str)])
            grid = calculate_(species, phases, str_statevar_dict, models, phase_records, pdens=50, fake_points=True)
            multi_eqdata = _equilibrium(species, phase_records, cond_dict, grid)
            target_hyperplane_phases.append(multi_eqdata.Phase.squeeze())
            # Does there exist only a single phase in the result with zero internal degrees of freedom?
            # We should exclude those chemical potentials from the average because they are meaningless.
            num_phases = np.sum(multi_eqdata.Phase.squeeze() != '')
            Y_values = multi_eqdata.Y.squeeze()
            no_internal_dof = np.all((np.isclose(Y_values, 1.)) | np.isnan(Y_values))
            MU_values = multi_eqdata.MU.squeeze()
            if (num_phases == 1) and no_internal_dof:
                target_hyperplane_chempots.append(np.full_like(MU_values, np.nan))
            else:
                target_hyperplane_chempots.append(MU_values)
    target_hyperplane_mean_chempots = np.nanmean(target_hyperplane_chempots, axis=0, dtype=np.float_)
    return target_hyperplane_mean_chempots


def driving_force_to_hyperplane(target_hyperplane_chempots: np.ndarray, comps: Sequence[str],
                                phase_region: PhaseRegion, vertex: RegionVertex,
                                parameters: np.ndarray, approximate_equilibrium: bool = False) -> float:
    """Calculate the integrated driving force between the current hyperplane and target hyperplane.
    """
    if approximate_equilibrium:
        _equilibrium = no_op_equilibrium_
    else:
        _equilibrium = equilibrium_
    dbf = phase_region.dbf
    species = phase_region.species
    models = phase_region.models
    current_phase = vertex.phase_name
    cond_dict = {**phase_region.potential_conds, **vertex.comp_conds}
    str_statevar_dict = OrderedDict([(str(key),cond_dict[key]) for key in sorted(phase_region.potential_conds.keys(), key=str)])
    phase_points = vertex.points
    phase_records = vertex.phase_records
    update_phase_record_parameters(phase_records, parameters)
    if phase_points is None:
        # We don't have the phase composition here, so we estimate the driving force.
        # Can happen if one of the composition conditions is unknown or if the phase is
        # stoichiometric and the user did not specify a valid phase composition.
        single_eqdata = calculate_(species, [current_phase], str_statevar_dict, models, phase_records, pdens=50)
        df = np.multiply(target_hyperplane_chempots, single_eqdata.X).sum(axis=-1) - single_eqdata.GM
        driving_force = float(df.max())
    elif vertex.is_disordered:
        # Construct disordered sublattice configuration from composition dict
        # Compute energy
        # Compute residual driving force
        # TODO: Check that it actually makes sense to declare this phase 'disordered'
        num_dof = sum([len(set(c).intersection(species)) for c in dbf.phases[current_phase].constituents])
        desired_sitefracs = np.ones(num_dof, dtype=np.float_)
        dof_idx = 0
        for c in dbf.phases[current_phase].constituents:
            dof = sorted(set(c).intersection(comps))
            if (len(dof) == 1) and (dof[0] == 'VA'):
                return 0
            # If it's disordered config of BCC_B2 with VA, disordered config is tiny vacancy count
            sitefracs_to_add = np.array([cond_dict.get(v.X(d)) for d in dof], dtype=np.float_)
            # Fix composition of dependent component
            sitefracs_to_add[np.isnan(sitefracs_to_add)] = 1 - np.nansum(sitefracs_to_add)
            desired_sitefracs[dof_idx:dof_idx + len(dof)] = sitefracs_to_add
            dof_idx += len(dof)
        # TODO: we probably should be passing desired_sitefracs to calculate_
        # here since the internal DOF is fixed. This should satisfy the same
        # effect as passing phase_points in the other calls here.
        single_eqdata = calculate_(species, [current_phase], str_statevar_dict, models, phase_records, pdens=50)
        driving_force = np.multiply(target_hyperplane_chempots, single_eqdata.X).sum(axis=-1) - single_eqdata.GM
        driving_force = float(np.squeeze(driving_force))
    else:
        # Extract energies from single-phase calculations
        grid = calculate_(species, [current_phase], str_statevar_dict, models, phase_records, points=phase_points, pdens=50, fake_points=True)
        converged, energy = constrained_equilibrium(species, phase_records, cond_dict, grid)

        if not converged:
            _log.debug('Calculation failure: constrained equilibrium not converged for %s, conditions: %s, parameters %s', current_phase, cond_dict, parameters)
            return np.inf
        select_energy = float(energy)
        # TODO: make region_comps part of the RegionVertex so we can take the dot product right away
        region_comps = []
        for comp in [c for c in sorted(comps) if c != 'VA']:
            region_comps.append(cond_dict.get(v.X(comp), np.nan))
        region_comps[region_comps.index(np.nan)] = 1 - np.nansum(region_comps)
        driving_force = np.multiply(target_hyperplane_chempots, region_comps).sum() - select_energy
        driving_force = float(driving_force)
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
        data_comps = data['data_comps']
        weight = data['weight']
        dataset_ref = data['dataset_reference']
        # for the set of phases and corresponding tie-line verticies in equilibrium
        for phase_region in data['phase_regions']:
            # 1. Calculate the average multiphase hyperplane
            eq_str = phase_region.eq_str()
            target_hyperplane = estimate_hyperplane(phase_region, parameters, approximate_equilibrium=approximate_equilibrium)
            if np.any(np.isnan(target_hyperplane)):
                _log.debug('NaN target hyperplane. Equilibria: (%s), driving force: 0.0, reference: %s.', eq_str, dataset_ref)
                data_driving_forces.extend([0]*len(phase_region.vertices))
                data_weights.extend([weight]*len(phase_region.vertices))
                continue
            # 2. Calculate the driving force to that hyperplane for each vertex
            for vertex in phase_region.vertices:
                driving_force = driving_force_to_hyperplane(target_hyperplane, data_comps,
                                                            phase_region, vertex, parameters,
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
    driving_forces = np.concatenate(driving_forces)
    weights = np.concatenate(weights)
    if np.any(np.logical_or(np.isinf(driving_forces), np.isnan(driving_forces))):
        return -np.inf
    log_probabilites = norm.logpdf(driving_forces, loc=0, scale=1000/data_weight/weights)
    _log.debug('Data weight: %s, driving forces: %s, weights: %s, probabilities: %s', data_weight, driving_forces, weights, log_probabilites)
    return np.sum(log_probabilites)
