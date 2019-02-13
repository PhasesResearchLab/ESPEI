"""
Tools used across parameter selection modules
"""

import numpy as np
import sympy
from pycalphad import variables as v

feature_transforms = {"CPM_FORM": lambda x: -v.T*sympy.diff(x, v.T, 2),
                      "CPM_MIX": lambda x: -v.T*sympy.diff(x, v.T, 2),
                      "CPM": lambda x: -v.T*sympy.diff(x, v.T, 2),
                      "SM_FORM": lambda x: -sympy.diff(x, v.T),
                      "SM_MIX": lambda x: -sympy.diff(x, v.T),
                      "SM": lambda x: -sympy.diff(x, v.T),
                      "HM_FORM": lambda x: x - v.T*sympy.diff(x, v.T),
                      "HM_MIX": lambda x: x - v.T*sympy.diff(x, v.T),
                      "HM": lambda x: x - v.T*sympy.diff(x, v.T)}


def shift_reference_state(desired_data, feature_transform, fixed_model, moles_per_formula_unit):
    """
    Shift data to a new common reference state.
    """
    total_response = []
    for dataset in desired_data:
        values = np.asarray(dataset['values'], dtype=np.object)
        if dataset['solver'].get('sublattice_occupancies', None) is not None:
            value_idx = 0
            for occupancy, config in zip(dataset['solver']['sublattice_occupancies'], dataset['solver']['sublattice_configurations']):
                if dataset['output'].endswith('_FORM'):
                    pass
                elif dataset['output'].endswith('_MIX'):
                    values[..., value_idx] += feature_transform(fixed_model.models['ref'])
                else:
                    raise ValueError('Unknown property to shift: {}'.format(dataset['output']))
                # These contributions are not present in the data, we need to add them here explicitly
                for excluded_contrib in dataset.get('excluded_model_contributions', []):
                    values[..., value_idx] += feature_transform(fixed_model.models[excluded_contrib])/moles_per_formula_unit
                value_idx += 1
        total_response.append(values.flatten())
    return total_response


