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
import warnings
from collections import OrderedDict
from typing import Sequence, Dict, NamedTuple, Any, Union, List, Tuple

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm
import sympy
import tinydb

from pycalphad import Database, Model, variables as v
from pycalphad.codegen.callables import build_phase_records
from pycalphad.core.constants import MIN_SITE_FRACTION
from pycalphad.core.utils import instantiate_models, filter_phases, unpack_components
from pycalphad.core.phase_rec import PhaseRecord
from espei.utils import PickleableTinyDB
from espei.shadow_functions import equilibrium_, calculate_, no_op_equilibrium_, update_phase_record_parameters

_log = logging.getLogger(__name__)


def _safe_index(items, index):
    try:
        return items[index]
    except IndexError:
        return None


def _extract_symbols(soln):
    """Return dependent and independent symbols in the solution dictionary"""
    dependent_symbols = set(soln.keys())
    independent_symbols = set()
    for expr in soln.values():
        independent_symbols |= expr.free_symbols
    return dependent_symbols, independent_symbols


def _solve(equations, desired_syms, phase_name, comp_conds):
    """Return a unique solution dictionary to the desired equations.

    Phases and composition conditions are only used for error reporting.
    Will raise a ValueError if no solutions are possible.
    """
    soln_dicts = sympy.solve(equations, desired_syms, dict=True)
    if len(soln_dicts) == 0:
        raise ValueError(f'No possible solutions for phase "{phase_name}" to satisfy the composition conditions: {comp_conds}. Check whether these compositions are valid.')
    elif len(soln_dicts) > 1:
        warnings.warn(f'Multiple solutions possible for phase "{phase_name}" to satisfy the composition conditions: {comp_conds}. Taking the first solution')
    return soln_dicts[0] # take the first solution even if multiple


def _solve_sitefracs_composition(mod: Model, comp_conds: Dict[v.X, float]) -> Dict[v.Y, Union[sympy.Expr, v.Y]]:
    """Return a dictionary of site fraction expressions that solves the prescribed composition

    Given a Model and global composition conditions, solve for site fractions
    (dependent and independent) that can produce points (internal degrees of
    freedom) that satisfy both the global composition conditions AND internal
    phase constraints.

    Returns
    -------
    Dict[v.Y, Union[sympy.Expr, v.Y]]
        Mapping of dependent site fractions to independent site fractions (possibly in expressions)

    Notes
    -----
    The symbols in the solution dictionary may not seem like they are
    independent variables and that they would invalidate the site fraction sum
    condition if replaced with arrays. Instead, the dependent site fraction
    would become negative if the condition were invalidated and those points
    can be filtered out in a later step.
    """
    # populate equations list with the site fraction contraints
    eqns = list(mod.get_internal_constraints()) # assumes only sublattice constraints

    # Add the composition constraint equation for each constraint
    for comp_cond_key, comp_cond_val in comp_conds.items():
        # only one constraint because it's one condition at a time
        X_el = mod.get_multiphase_constraints({comp_cond_key: comp_cond_val})[0].subs({v.Symbol('NP'): 1.0})
        eqn = X_el - comp_cond_val  # = 0
        eqns.append(eqn)

    # Try to make vacancy-like species independent so we can sample their dilute constitutions effectively
    desired_dependent_sfs = [sf for sf in mod.site_fractions if sf.species.number_of_atoms != 0]
    soln = _solve(eqns, desired_dependent_sfs, mod.phase_name, comp_conds)

    # Verify that the solution has all the site fractions
    dep_sfs, indep_sfs = _extract_symbols(soln)
    missing_sfs = set(mod.site_fractions).difference(dep_sfs | indep_sfs)
    if len(missing_sfs) > 0:
        # The solution is missing some site fractions. They may be independent
        # variables. Add them and solve again.
        soln = _solve(eqns, list(set(desired_dependent_sfs) | missing_sfs), mod.phase_name, comp_conds)

    return soln


def _sample_solution_constitution(mod: Model, soln: Dict[v.Y, Union[sympy.Expr, v.Y]], pdens=101) -> ArrayLike:
    """Return an array of discrete points sampling the independent degrees of freedom of a solution

    The solution is responsible for guaranteeing that all points are valid
    under the desired constraints. The only validation/pruning performed here
    is to remove any points which have negative site fractions. It is the
    responsibility of the caller to enforce that the sum of positive site
    fractions is unity (which will prevent any site fractions > 1).

    Each degree of freedom will have

    * ``pdens`` points sampled linearly on the interval ``[0, 1]``
    * ``pdens`` points sampled in logspace on the intervals
       ``[MIN_SITE_FRACTION, 0.01]`` and ``[0.99, 1.0]``.

    Returns
    -------
    ArrayLike
        A 2D array of shape ``(points, mod.site_fractions)``
    """
    # identify independent site fractions (values of `soln`)
    indep_site_fracs = set()
    for expr in soln.values():
        indep_site_fracs |= expr.free_symbols
    indep_site_fracs = sorted(indep_site_fracs, key=str)

    # Sample dilute edges in logspace and linearly in the middle
    grid_1d = np.concatenate([
        np.logspace(np.log10(MIN_SITE_FRACTION), -2, pdens//2),  # [~0, 0.01]
        np.linspace(0, 1, pdens),
        np.logspace(np.log10(0.99), 0, pdens//2),  # [0.99, 1.0]
    ])
    # Create 1D arrays from 0 to 1 for each independent variable
    indep_site_frac_arrays = [grid_1d for _ in indep_site_fracs]

    # Broadcast the 1D arrays against each other and flatten them
    grids_Nd = np.meshgrid(*indep_site_frac_arrays)
    grids_1d = [grid.flatten() for grid in grids_Nd]
    # Need an OrderedDict here because the keys will become arguments to lambdify
    indep_site_frac_dict = OrderedDict(zip(indep_site_fracs, grids_1d))
    lambdify_syms = tuple(indep_site_frac_dict.keys())

    # Use the 2D array to fill the columns of the dependent site fractions (keys of `soln`) (points, indep. and dep. site fractions)
    indep_dep_dict = dict()
    indep_dep_dict.update(soln)
    indep_dep_dict.update(indep_site_frac_dict)
    if len(indep_site_frac_dict) > 0:
        npts = grids_1d[0].shape[0]
    else:
        # No independent degrees of freedom
        npts = 1
    points = np.empty((npts, len(mod.site_fractions)))
    # Enumerating over site fractions ensures that the array must be filled correctly
    for idof, sf in enumerate(mod.site_fractions):
        # A key error here would mean that a site fraction variable was not in
        # the indepedent or dependent site fractions
        sitefracs = indep_dep_dict[sf]
        if isinstance(sitefracs, sympy.Expr):
            # Site fractions are dependent and symbolic. Substitute the
            # independent site fraction arrays into the dependent ones.
            _f = sympy.lambdify(lambdify_syms, sitefracs)
            x = tuple(indep_site_frac_dict.values())
            points[:, idof] = _f(*x)
        else:
            # site fractions are independent already
            points[:, idof] = sitefracs

    # Remove any rows (points) where any site fractions are negative.
    # As long as the internal constraints were used to find the solution,
    # this should also ensure that any site fractions >1 are not possible.
    valid_site_frac_rows = np.nonzero(~np.any(points < 0, axis=1))[0]
    points = points[valid_site_frac_rows, :]

    return points


def extract_conditions(all_conditions: Dict[v.StateVariable, np.ndarray], idx: int) -> Dict[v.StateVariable, float]:
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


def extract_phases_comps(phase_region):
    """Extract the phase names, phase compositions and any phase flags from
    each tie-line point in the phase region
    """
    region_phases = []
    region_comp_conds = []
    phase_flags = []
    for tie_point in phase_region:
        if len(tie_point) == 4:  # phase_flag within
            phase_name, components, compositions, flag = tie_point
        elif len(tie_point) == 3:  # no phase_flag within
            phase_name, components, compositions = tie_point
            flag = None
        else:
            raise ValueError("Wrong number of data in tie-line point")
        region_phases.append(phase_name)
        region_comp_conds.append(dict(zip(map(v.X, map(lambda x: x.upper(), components)), compositions)))
        phase_flags.append(flag)
    return region_phases, region_comp_conds, phase_flags


PhaseRegion = NamedTuple('PhaseRegion', (('region_phases', Sequence[str]),
                                         ('potential_conds', Dict[v.StateVariable, float]),
                                         ('comp_conds', Sequence[Dict[v.X, float]]),
                                         ('phase_points', Sequence[ArrayLike]),
                                         ('phase_flags', Sequence[str]),
                                         ('dbf', Database),
                                         ('species', Sequence[v.Species]),
                                         ('phases', Sequence[str]),
                                         ('models', Dict[str, Model]),
                                         ('phase_records', Sequence[Dict[str, PhaseRecord]]),
                                         ))


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
            # We need to construct a PhaseRegion by matching up phases/compositions to the conditions
            # Extract the conditions for entire phase region
            region_potential_conds = extract_conditions(conditions, idx)
            region_potential_conds[v.N] = region_potential_conds.get(v.N) or 1.0  # Add v.N condition, if missing
            # Extract all the phases and compositions from the tie-line points
            region_phases, region_comp_conds, phase_flags = extract_phases_comps(phase_region)
            # Construct single-phase points satisfying the conditions for each phase in the region
            region_phase_points = []
            for phase_name, comp_conds in zip(region_phases, region_comp_conds):
                if any(val is None for val in comp_conds.values()):
                    # We can't construct points because we don't have a known composition
                    region_phase_points.append(None)
                    continue
                mod = models[phase_name]
                sitefrac_soln = _solve_sitefracs_composition(mod, comp_conds)
                phase_points = _sample_solution_constitution(mod, sitefrac_soln)
                region_phase_points.append(phase_points)
            region_phase_records = [build_phase_records(dbf, species, data_phases, {**region_potential_conds, **comp_conds}, models, parameters=parameters, build_gradients=True, build_hessians=True)
                                    for comp_conds in region_comp_conds]
            phase_regions.append(PhaseRegion(region_phases, region_potential_conds, region_comp_conds, region_phase_points, phase_flags, dbf, species, data_phases, models, region_phase_records))

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
    for comp_conds, phase_flag, phase_records in zip(phase_region.comp_conds, phase_region.phase_flags, phase_region.phase_records):
        # We are now considering a particular tie vertex
        update_phase_record_parameters(phase_records, parameters)
        cond_dict = {**comp_conds, **phase_region.potential_conds}
        for key, val in cond_dict.items():
            if val is None:
                cond_dict[key] = np.nan
        if np.any(np.isnan(list(cond_dict.values()))):
            # This composition is unknown -- it doesn't contribute to hyperplane estimation
            pass
        else:
            # Extract chemical potential hyperplane from multi-phase calculation
            # Note that we consider all phases in the system, not just ones in this tie region
            str_statevar_dict = OrderedDict([(str(key), cond_dict[key]) for key in sorted(phase_region.potential_conds.keys(), key=str)])
            grid = calculate_(dbf, species, phases, str_statevar_dict, models, phase_records, pdens=500, fake_points=True)
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
                                phase_region: PhaseRegion, vertex_idx: int,
                                parameters: np.ndarray, approximate_equilibrium: bool = False) -> float:
    """Calculate the integrated driving force between the current hyperplane and target hyperplane.
    """
    if approximate_equilibrium:
        _equilibrium = no_op_equilibrium_
    else:
        _equilibrium = equilibrium_
    dbf = phase_region.dbf
    species = phase_region.species
    phases = phase_region.phases
    models = phase_region.models
    current_phase = phase_region.region_phases[vertex_idx]
    cond_dict = {**phase_region.potential_conds, **phase_region.comp_conds[vertex_idx]}
    str_statevar_dict = OrderedDict([(str(key),cond_dict[key]) for key in sorted(phase_region.potential_conds.keys(), key=str)])
    phase_points = phase_region.phase_points[vertex_idx]
    phase_flag = phase_region.phase_flags[vertex_idx]
    phase_records = phase_region.phase_records[vertex_idx]
    update_phase_record_parameters(phase_records, parameters)
    for key, val in cond_dict.items():
        if val is None:
            cond_dict[key] = np.nan
    if np.any(np.isnan(list(cond_dict.values()))):
        # We don't actually know the phase composition here, so we estimate it
        single_eqdata = calculate_(dbf, species, [current_phase], str_statevar_dict, models, phase_records, pdens=500)
        df = np.multiply(target_hyperplane_chempots, single_eqdata.X).sum(axis=-1) - single_eqdata.GM
        driving_force = float(df.max())
    elif phase_flag == 'disordered':
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
        single_eqdata = calculate_(dbf, species, [current_phase], str_statevar_dict, models, phase_records, pdens=500)
        driving_force = np.multiply(target_hyperplane_chempots, single_eqdata.X).sum(axis=-1) - single_eqdata.GM
        driving_force = float(np.squeeze(driving_force))
    else:
        # Extract energies from single-phase calculations
        grid = calculate_(dbf, species, [current_phase], str_statevar_dict, models, phase_records, points=phase_points, pdens=500, fake_points=True)
        single_eqdata = _equilibrium(species, phase_records, cond_dict, grid)
        if np.all(np.isnan(single_eqdata.NP)):
            _log.debug('Calculation failure: all NaN phases with phases: %s, conditions: %s, parameters %s', current_phase, cond_dict, parameters)
            return np.inf
        select_energy = float(single_eqdata.GM)
        region_comps = []
        for comp in [c for c in sorted(comps) if c != 'VA']:
            region_comps.append(cond_dict.get(v.X(comp), np.nan))
        region_comps[region_comps.index(np.nan)] = 1 - np.nansum(region_comps)
        driving_force = np.multiply(target_hyperplane_chempots, region_comps).sum() - select_energy
        driving_force = float(driving_force)
    return driving_force


def _format_phase_compositions(phase_region):
    phase_comp_cond_pairs = zip(phase_region.region_phases, phase_region.comp_conds)
    phase_compositions = ', '.join(f'{ph}: {c}' for ph, c in phase_comp_cond_pairs)
    return f"conds: ({phase_region.potential_conds}), comps: ({phase_compositions})"

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
            eq_str = _format_phase_compositions(phase_region)
            target_hyperplane = estimate_hyperplane(phase_region, parameters, approximate_equilibrium=approximate_equilibrium)
            if np.any(np.isnan(target_hyperplane)):
                _log.debug('NaN target hyperplane. Equilibria: (%s), driving force: 0.0, reference: %s.', eq_str, dataset_ref)
                data_driving_forces.extend([0]*len(phase_region.comp_conds))
                data_weights.extend([weight]*len(phase_region.comp_conds))
                continue
            # 2. Calculate the driving force to that hyperplane for each vertex
            for vertex_idx in range(len(phase_region.comp_conds)):
                driving_force = driving_force_to_hyperplane(target_hyperplane, data_comps,
                                                            phase_region, vertex_idx, parameters,
                                                            approximate_equilibrium=approximate_equilibrium,
                                                            )
                if np.isinf(driving_force) and short_circuit:
                    _log.debug('Equilibria: (%s), current phase: %s, hyperplane: %s, driving force: %s, reference: %s. Short circuiting.', eq_str, phase_region.region_phases[vertex_idx], target_hyperplane, driving_force, dataset_ref)
                    return [[np.inf]], [[np.inf]]
                data_driving_forces.append(driving_force)
                data_weights.append(weight)
                _log.debug('Equilibria: (%s), current phase: %s, hyperplane: %s, driving force: %s, reference: %s', eq_str, phase_region.region_phases[vertex_idx], target_hyperplane, driving_force, dataset_ref)
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
