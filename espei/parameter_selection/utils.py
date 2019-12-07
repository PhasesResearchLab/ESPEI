"""
Tools used across parameter selection modules
"""

import numpy as np
import sympy
from pycalphad import variables as v

feature_transforms = {"CPM_FORM": lambda GM: -v.T*sympy.diff(GM, v.T, 2),
                      "CPM_MIX": lambda GM: -v.T*sympy.diff(GM, v.T, 2),
                      "CPM": lambda GM: -v.T*sympy.diff(GM, v.T, 2),
                      "SM_FORM": lambda GM: -sympy.diff(GM, v.T),
                      "SM_MIX": lambda GM: -sympy.diff(GM, v.T),
                      "SM": lambda GM: -sympy.diff(GM, v.T),
                      "HM_FORM": lambda GM: GM - v.T*sympy.diff(GM, v.T),
                      "HM_MIX": lambda GM: GM - v.T*sympy.diff(GM, v.T),
                      "HM": lambda GM: GM - v.T*sympy.diff(GM, v.T)}


def shift_reference_state(desired_data, feature_transform, fixed_model, moles_per_formula_unit):
    """
    Shift _MIX or _FORM data to a common reference state in per mole-formula units.

    Parameters
    ----------
    desired_data : List[Dict[str, Any]]
        ESPEI single phase dataset
    feature_transform : Callable
        Function to transform an AST for the GM property to the property of
        interest, i.e. entropy would be ``lambda GM: -sympy.diff(GM, v.T)``
    fixed_model : pycalphad.Model
        Model with all lower order (in composition) terms already fit.
    moles_per_formula_unit : float
        Number of moles of atoms in every mole formula unit.

    Returns
    -------
    np.ndarray
        Data for this feature in [qty]/mole-formula in a common reference state.

    Raises
    ------
    ValueError

    """
    total_response = []
    for dataset in desired_data:
        # Transform data from J/mole-atom to J/mole-formula (or J/K-mole-atom to J/K-mole-formula etc.)
        # The pycalphad Model quantities are all per mole-formula
        values = np.asarray(dataset['values'], dtype=np.object)*moles_per_formula_unit
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
                    values[..., value_idx] += feature_transform(fixed_model.models[excluded_contrib])
                value_idx += 1
        total_response.append(values.flatten())
    return total_response
