"""
Build fittable models for ternary parameter selection
"""

import numpy as np
import sympy
from pycalphad import variables as v
from espei.core_utils import get_samples
from espei.parameter_selection.utils import feature_transforms


def build_ternary_feature_matrix(prop, candidate_models, desired_data):
    """
    Return an MxN matrix of M data sample and N features.

    Parameters
    ----------
    prop : str
        String name of the property, e.g. 'HM_MIX'
    candidate_models : list
        List of SymPy parameters that can be fit for this property.
    desired_data : dict
        Full dataset dictionary containing values, conditions, etc.

    Returns
    -------
    numpy.ndarray
        An MxN matrix of M samples (from desired data) and N features.

    """
    transformed_features = sympy.Matrix([feature_transforms[prop](i) for i in candidate_models])
    all_samples = get_samples(desired_data)
    feature_matrix = np.empty((len(all_samples), len(transformed_features)), dtype=np.float)
    feature_matrix[:, :] = [transformed_features.subs({v.T: temp, 'YS': ys, 'V_I': v_i, 'V_J': v_j, 'V_K': v_k}).evalf() for temp, (ys, (v_i, v_j, v_k)) in all_samples]
    return feature_matrix
