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
from collections import OrderedDict
from typing import Sequence, Dict, NamedTuple, Any

import numpy as np
from scipy.stats import norm
import tinydb

from pycalphad import Database, Model, variables as v
from pycalphad.codegen.callables import build_phase_records
from pycalphad.core.utils import instantiate_models, filter_phases, unpack_components
from pycalphad.core.phase_rec import PhaseRecord
from espei.utils import PickleableTinyDB
from espei.shadow_functions import equilibrium_, calculate_, no_op_equilibrium_

def _safe_index(items, index):
    try:
        return items[index]
    except IndexError:
        return None

def update_phase_record_parameters(phase_records: Dict[str, PhaseRecord], parameters: np.ndarray) -> None:
    if parameters.size > 0:
        for phase_name, phase_record in phase_records.items():
            phase_record.parameters[:] = parameters

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
            if len(phase_region) < 2:
                # Skip single-phase regions for fitting purposes
                continue
            # Extract the conditions for entire phase region
            region_potential_conds = extract_conditions(conditions, idx)
            region_potential_conds[v.N] = region_potential_conds.get(v.N) or 1.0  # Add v.N condition, if missing
            # Extract all the phases and compositions from the tie-line points
            region_phases, region_comp_conds, phase_flags = extract_phases_comps(phase_region)
            region_phase_records = [build_phase_records(dbf, species, data_phases, {**region_potential_conds, **comp_conds}, models, parameters=parameters, build_gradients=True, build_hessians=True)
                                    for comp_conds in region_comp_conds]
            phase_regions.append(PhaseRegion(region_phases, region_potential_conds, region_comp_conds, phase_flags, dbf, species, data_phases, models, region_phase_records))

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
    target_hyperplane_mean_chempots = np.nanmean(target_hyperplane_chempots, axis=0, dtype=np.float)
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
        desired_sitefracs = np.ones(num_dof, dtype=np.float)
        dof_idx = 0
        for c in dbf.phases[current_phase].constituents:
            dof = sorted(set(c).intersection(comps))
            if (len(dof) == 1) and (dof[0] == 'VA'):
                return 0
            # If it's disordered config of BCC_B2 with VA, disordered config is tiny vacancy count
            sitefracs_to_add = np.array([cond_dict.get(v.X(d)) for d in dof], dtype=np.float)
            # Fix composition of dependent component
            sitefracs_to_add[np.isnan(sitefracs_to_add)] = 1 - np.nansum(sitefracs_to_add)
            desired_sitefracs[dof_idx:dof_idx + len(dof)] = sitefracs_to_add
            dof_idx += len(dof)
        single_eqdata = calculate_(dbf, species, [current_phase], str_statevar_dict, models, phase_records, pdens=500)
        driving_force = np.multiply(target_hyperplane_chempots, single_eqdata.X).sum(axis=-1) - single_eqdata.GM
        driving_force = float(np.squeeze(driving_force))
    else:
        # Extract energies from single-phase calculations
        grid = calculate_(dbf, species, [current_phase], str_statevar_dict, models, phase_records, pdens=500, fake_points=True)
        single_eqdata = _equilibrium(species, phase_records, cond_dict, grid)
        if np.all(np.isnan(single_eqdata.NP)):
            logging.debug('Calculation failure: all NaN phases with phases: {}, conditions: {}, parameters {}'.format(current_phase, cond_dict, parameters))
            return np.inf
        select_energy = float(single_eqdata.GM)
        region_comps = []
        for comp in [c for c in sorted(comps) if c != 'VA']:
            region_comps.append(cond_dict.get(v.X(comp), np.nan))
        region_comps[region_comps.index(np.nan)] = 1 - np.nansum(region_comps)
        driving_force = np.multiply(target_hyperplane_chempots, region_comps).sum() - select_energy
        driving_force = float(driving_force)
    return driving_force


def calculate_zpf_error(zpf_data: Sequence[Dict[str, Any]],
                        parameters: np.ndarray = None,
                        data_weight: int = 1.0,
                        approximate_equilibrium: bool = False):
    """
    Calculate error due to phase equilibria data

    zpf_data : list
        Datasets that contain single phase data
    phase_models : dict
        Phase models to pass to pycalphad calculations
    parameters : np.ndarray
        Array of parameters to calculate the error with.
    callables : dict
        Callables to pass to pycalphad
    data_weight : float
        Scaling factor for the standard deviation of the measurement of a
        tieline which has units J/mol. The standard deviation is 1000 J/mol
        and the scaling factor defaults to 1.0.
    approximate_equilibrium : bool
        Whether or not to use an approximate version of equilibrium that does
        not refine the solution and uses ``starting_point`` instead.

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
        parameters = np.array([])
    prob_error = 0.0
    for data in zpf_data:
        data_comps = data['data_comps']
        weight = data['weight']
        dataset_ref = data['dataset_reference']
        # for the set of phases and corresponding tie-line verticies in equilibrium
        for phase_region in data['phase_regions']:
            # Calculate the average multiphase hyperplane
            eq_str = "conds: ({}), comps: ({})".format(phase_region.potential_conds, ', '.join(['{}: {}'.format(ph, c) for ph, c in zip(phase_region.region_phases, phase_region.comp_conds)]))
            target_hyperplane = estimate_hyperplane(phase_region, parameters, approximate_equilibrium=approximate_equilibrium)
            if np.any(np.isnan(target_hyperplane)):
                zero_probs = norm.logpdf(np.zeros(len(phase_region.comp_conds)), loc=0, scale=1000/data_weight/weight)
                total_zero_prob = np.sum(zero_probs)
                logging.debug('ZPF error - NaN target hyperplane. Equilibria: ({}), reference: {}. Treating all driving force: 0.0, probability: {}, probabilities: {}'.format(eq_str, dataset_ref, total_zero_prob, zero_probs))
                prob_error += total_zero_prob
                continue
            # Then calculate the driving force to that hyperplane for each individual vertex
            for vertex_idx in range(len(phase_region.comp_conds)):
                driving_force = driving_force_to_hyperplane(target_hyperplane, data_comps,
                                                            phase_region, vertex_idx, parameters,
                                                            approximate_equilibrium=approximate_equilibrium,
                                                            )
                vertex_prob = norm.logpdf(driving_force, loc=0, scale=1000/data_weight/weight)
                prob_error += vertex_prob
                logging.debug('ZPF error - Equilibria: ({}), current phase: {}, driving force: {}, probability: {}, reference: {}'.format(eq_str, phase_region.region_phases[vertex_idx], driving_force, vertex_prob, dataset_ref))
    if np.isnan(prob_error):
        return -np.inf
    return prob_error
