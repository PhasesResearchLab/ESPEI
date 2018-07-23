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


def interaction_test(configuration, order=None):
    """
    Returns True if the configuration has an interaction

    Parameters
    ----------
    order : int, optional
        Specific order to check for. E.g. a value of 3 checks for ternary interactions

    Returns
    -------
    bool
        True if there is an interaction.

    Examples
    --------
    >>> configuration = [['A'], ['A','B']]
    >>> interaction_test(configuration)
    True  # has an interaction
    >>> interaction_test(configuration, order=2)
    True  # has a binary interaction
    >>> interaction_test(configuration, order=3)
    False  # has no ternary interaction

    """
    interacting_species = [len(subl) for subl in configuration if isinstance(subl, tuple)]
    if order is None:  # checking for any interaction
        return any([subl_occupation > 1 for subl_occupation in interacting_species])
    else:
        return any([subl_occupation == order for subl_occupation in interacting_species])


def shift_reference_state(desired_data, feature_transform, fixed_model):
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
                    pass
                else:
                    raise ValueError('Unknown property to shift: {}'.format(dataset['output']))
                value_idx += 1
        total_response.append(values.flatten())
    return total_response


