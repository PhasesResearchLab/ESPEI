"""
Build fittable models for ternary parameter selection
"""

import itertools
import operator
from functools import reduce

import numpy as np
import sympy
from pycalphad import variables as v

from espei.parameter_selection.utils import feature_transforms


def get_muggianu_samples(desired_data):
    """
    Return the data values from desired_data, transformed to interaction products.
    Specifically works for Muggianu extrapolation.

    Parameters
    ----------
    desired_data : list
        List of matched desired data, e.g. for a single property

    Returns
    -------
    list
        List of sample values that are properly transformed.

    Notes
    -----
    Transforms data to interaction products, e.g. YS*{}^{xs}G=YS*XS*DXS^{n} {}^{n}L
    Each tuple in the list is a tuple of (temperature, (site_fraction_product, interaction_product)) for each data sample
    Interaction product itself is a list that corresponds to the Mugiannu corrected interactions products for components [I, J, K]

    """
    # TODO: assumes no cross terms (I think)
    # TODO: does not ravel pressure conditions
    # TODO: could possibly combine with ravel_conditions if we do the math outside.
    all_samples = []
    for data in desired_data:
        temperatures = np.atleast_1d(data['conditions']['T'])
        num_configs = np.array(data['solver'].get('sublattice_configurations'), dtype=np.object).shape[0]
        site_fractions = data['solver'].get('sublattice_occupancies', [[1]] * num_configs)
        # product of site fractions for each dataset
        site_fraction_product = [reduce(operator.mul, list(itertools.chain(*[np.atleast_1d(f) for f in fracs])), 1)
                                 for fracs in site_fractions]
        # TODO: Subtle sorting bug here, if the interactions aren't already in sorted order...
        interaction_product = []
        for subl_fracs in site_fractions:
            # subl_fracs is the list of site fractions for each sublattice, e.g. [[0.25, 0.25, 0.5], 1] for an [[A,B,C], A] configuration
            # we need to generate v_i, v_j, or v_k for the ternary case, which v we are calculating depends on the parameter order
            prod = [1, 1, 1]  # product for V_I, V_J, V_K
            for f in subl_fracs:
                if isinstance(f, list) and (len(f) >= 3):
                    muggianu_correction = (1 - sum(f))/len(f)
                    for i in range(len(f)):
                        prod[i] *= f[i] + muggianu_correction
            interaction_product.append([float(p) for p in prod])
        comp_features = zip(site_fraction_product, interaction_product)
        all_samples.extend(list(itertools.product(temperatures, comp_features)))
    return all_samples


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
    all_samples = get_muggianu_samples(desired_data)
    feature_matrix = np.empty((len(all_samples), len(transformed_features)), dtype=np.float)
    feature_matrix[:, :] = [transformed_features.subs({v.T: temp, 'YS': ys, 'V_I': v_i, 'V_J': v_j, 'V_K': v_k}).evalf() for temp, (ys, (v_i, v_j, v_k)) in all_samples]
    return feature_matrix
