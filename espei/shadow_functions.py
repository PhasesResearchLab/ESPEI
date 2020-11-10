"""
Fast versions of equilibrium and calculate that "override" the equivalent
pycalphad functions for very fast performance.
"""

from collections import OrderedDict
from typing import Sequence, Dict, Optional
import numpy as np
from pycalphad import Database, Model, variables as v
from pycalphad.core.phase_rec import PhaseRecord
from pycalphad.core.starting_point import starting_point
from pycalphad.core.eqsolver import _solve_eq_at_conditions
from pycalphad.core.equilibrium import _adjust_conditions
from pycalphad.core.utils import get_state_variables, unpack_kwarg, point_sample, generate_dof
from pycalphad.core.light_dataset import LightDataset
from pycalphad.core.calculate import _sample_phase_constitution, _compute_phase_values


def update_phase_record_parameters(phase_records: Dict[str, PhaseRecord], parameters: np.ndarray) -> None:
    if parameters.size > 0:
        for phase_name, phase_record in phase_records.items():
            phase_record.parameters[:] = parameters


def calculate_(dbf: Database, species: Sequence[v.Species], phases: Sequence[str],
               str_statevar_dict: Dict[str, np.ndarray], models: Dict[str, Model],
               phase_records: Dict[str, PhaseRecord], output: Optional[str] = 'GM',
               points: Optional[Dict[str, np.ndarray]] = None,
               pdens: Optional[int] = 2000, broadcast: Optional[bool] = True,
               fake_points: Optional[bool] = False,
               ) -> LightDataset:
    """
    Quickly sample phase internal degree of freedom with virtually no overhead.
    """
    points_dict = unpack_kwarg(points, default_arg=None)
    pdens_dict = unpack_kwarg(pdens, default_arg=2000)
    nonvacant_components = [x for x in sorted(species) if x.number_of_atoms > 0]
    maximum_internal_dof = max(prx.phase_dof for prx in phase_records.values())
    all_phase_data = []
    for phase_name in sorted(phases):
        phase_obj = dbf.phases[phase_name]
        mod = models[phase_name]
        phase_record = phase_records[phase_name]
        points = points_dict[phase_name]
        variables, sublattice_dof = generate_dof(phase_obj, mod.components)
        if points is None:
            points = _sample_phase_constitution(phase_name, phase_obj.constituents, sublattice_dof, species,
                                                tuple(variables), point_sample, True, pdens_dict[phase_name])
        points = np.atleast_2d(points)

        fp = fake_points and (phase_name == sorted(phases)[0])
        phase_ds = _compute_phase_values(nonvacant_components, str_statevar_dict,
                                         points, phase_record, output,
                                         maximum_internal_dof, broadcast=broadcast,
                                         largest_energy=float(1e10), fake_points=fp,
                                         parameters={})
        all_phase_data.append(phase_ds)

    if len(all_phase_data) > 1:
        concatenated_coords = all_phase_data[0].coords

        data_vars = all_phase_data[0].data_vars
        concatenated_data_vars = {}
        for var in data_vars.keys():
            data_coords = data_vars[var][0]
            points_idx = data_coords.index('points')  # concatenation axis
            arrs = []
            for phase_data in all_phase_data:
                arrs.append(getattr(phase_data, var))
            concat_data = np.concatenate(arrs, axis=points_idx)
            concatenated_data_vars[var] = (data_coords, concat_data)
        final_ds = LightDataset(data_vars=concatenated_data_vars, coords=concatenated_coords)
    else:
        final_ds = all_phase_data[0]
    return final_ds


def equilibrium_(species: Sequence[v.Species], phase_records: Dict[str, PhaseRecord],
                 conditions: Dict[v.StateVariable, np.ndarray], grid: LightDataset
                 ) -> LightDataset:
    """
    Perform a fast equilibrium calculation with virtually no overhead.
    """
    statevars = get_state_variables(conds=conditions)
    conditions = _adjust_conditions(conditions)
    str_conds = OrderedDict([(str(ky), conditions[ky]) for ky in sorted(conditions.keys(), key=str)])
    start_point = starting_point(conditions, statevars, phase_records, grid)
    return _solve_eq_at_conditions(species, start_point, phase_records, grid, str_conds, statevars, False)


def no_op_equilibrium_(_, phase_records: Dict[str, PhaseRecord],
                       conditions: Dict[v.StateVariable, np.ndarray],
                       grid: LightDataset,
                       ) -> LightDataset:
    """
    Perform a fast "equilibrium" calculation with virtually no overhead that
    doesn't refine the solution or do global minimization, but just returns
    the starting point.

    Notes
    -----
    Uses a placeholder first argument for the same signature as
    ``_equilibrium``, but ``species`` are not needed.

    """
    statevars = get_state_variables(conds=conditions)
    conditions = _adjust_conditions(conditions)
    return starting_point(conditions, statevars, phase_records, grid)
