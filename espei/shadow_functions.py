"""
Fast versions of equilibrium and calculate that "override" the equivalent
pycalphad functions for very fast performance.
"""

from collections import OrderedDict
from typing import Sequence, Dict, Optional
from numpy.typing import ArrayLike
import numpy as np
from pycalphad import Model, variables as v
from pycalphad.codegen.phase_record_factory import PhaseRecordFactory
from pycalphad.core.phase_rec import PhaseRecord
from pycalphad.core.composition_set import CompositionSet
from pycalphad.core.starting_point import starting_point
from pycalphad.core.eqsolver import _solve_eq_at_conditions
from pycalphad.core.workspace import _adjust_conditions
from pycalphad.core.utils import get_state_variables, unpack_kwarg, point_sample
from pycalphad.core.light_dataset import LightDataset
from pycalphad.core.calculate import _sample_phase_constitution, _compute_phase_values
from pycalphad.core.solver import Solver


def update_phase_record_parameters(phase_record_factory: PhaseRecordFactory, parameters: ArrayLike) -> None:
    if parameters.size > 0:
        phase_record_factory.param_values[:] = np.asarray(parameters, dtype=np.float64)


def calculate_(species: Sequence[v.Species], phases: Sequence[str],
               str_statevar_dict: Dict[str, np.ndarray], models: Dict[str, Model],
               phase_records: Dict[str, PhaseRecord], output: Optional[str] = 'GM',
               points: Optional[Dict[str, np.ndarray]] = None,
               pdens: Optional[int] = 50, broadcast: Optional[bool] = True,
               fake_points: Optional[bool] = False,
               ) -> LightDataset:
    """
    Quickly sample phase internal degree of freedom with virtually no overhead.
    """
    points_dict = unpack_kwarg(points, default_arg=None)
    pdens_dict = unpack_kwarg(pdens, default_arg=50)
    nonvacant_components = [x for x in sorted(species) if x.number_of_atoms > 0]
    cur_phase_local_conditions = {} # XXX: Temporary hack to allow compatibility
    str_phase_local_conditions = {} # XXX: Temporary hack to allow compatibility
    maximum_internal_dof = max(prx.phase_dof for prx in phase_records.values())
    all_phase_data = []
    for phase_name in sorted(phases):
        mod = models[phase_name]
        phase_record = phase_records[phase_name]
        points = points_dict[phase_name]
        if points is None:
            points = _sample_phase_constitution(mod, point_sample, True, pdens_dict[phase_name], cur_phase_local_conditions)
        points = np.atleast_2d(points)
        fp = fake_points and (phase_name == sorted(phases)[0])
        phase_ds = _compute_phase_values(nonvacant_components, str_statevar_dict, str_phase_local_conditions,
                                         points, phase_record, output,
                                         maximum_internal_dof, broadcast=broadcast,
                                         largest_energy=float(1e10), fake_points=fp,
                                         parameters={})
        all_phase_data.append(phase_ds)

    # assumes phase_records all have the same nonvacant pure elements,
    # even if those elements are not present in this phase record
    fp_offset = len(tuple(phase_records.values())[0].nonvacant_elements) if fake_points else 0
    running_total = [fp_offset] + list(np.cumsum([phase_ds['X'].shape[-2] for phase_ds in all_phase_data]))
    islice_by_phase = {phase_name: slice(running_total[phase_idx], running_total[phase_idx+1], None)
                       for phase_idx, phase_name in enumerate(sorted(phases))}

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
    final_ds.attrs['phase_indices'] = islice_by_phase
    return final_ds
