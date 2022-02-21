"""
Building candidate models
"""

import itertools
from pycalphad.core.cache import cacheit
import symengine
from espei.sublattice_tools import interaction_test

def make_successive(xs):
    """
    Return a list of successive combinations

    Parameters
    ----------
    xs : list
        List of elements, e.g. [X, Y, Z]

    Returns
    -------
    list
        List of combinations where each combination include all the preceding elements

    Examples
    --------
    >>> make_successive(['W', 'X', 'Y', 'Z'])
    [['W'], ['W', 'X'], ['W', 'X', 'Y'], ['W', 'X', 'Y', 'Z']]
    """
    return [xs[:(i+1)] for i in range(len(xs))]

@cacheit  # This can be expensive if run from an inner loop, so it is cached
def build_feature_sets(temperature_features, interaction_features):
    """
    Return a list of broadcasted features

    Parameters
    ----------
    temperature_features : list
        List of temperature features that will become a successive_list, such as [TlogT, T-1, T2]
    interaction_features : list
        List of interaction features that will become a successive_list, such as [YS, YS*Z, YS*Z**2]

    Returns
    -------
    list

    Notes
    -----
    This allows two sets of features, e.g. [TlogT, T-1, T2] and [YS, YS*Z, YS*Z**2]
    and generates a list of feature sets where the temperatures and interactions
    are broadcasted successively.

    Generates candidate feature sets like:
    L0: A + BT,  L1: A
    L0: A     ,  L1: A + BT

    but **not** lists that are not successive:
    L0: A + BT,  L1: Nothing, L2: A
    L0: Nothing, L1: A + BT

    There's still some debate whether it makes sense from an information theory
    perspective to add a L1 B term without an L0 B term. However this might be
    more representative of how people usually model thermodynamics.

    Does not distribute multiplication/sums or make assumptions about the elements
    of the feature lists. They can be strings, ints, objects, tuples, etc..

    The number of features (related to the complexity) is a geometric series.
    For :math:`N` temperature features and :math:`M` interaction features, the total
    number of feature sets should be :math:`N(1-N^M)/(1-N)`. If :math:`N=1`, then there
    are :math:`M` total feature sets.

    """
    # [[A], [A, B], [A, B, C], ...]
    temps = make_successive(temperature_features)
    # [ [temps for L0], [temps for L1], [temps for L2], ...]
    feats = [list(itertools.product(temps, [inter])) for inter in interaction_features]
    # [ [temps for L0], [temps for L0 and L1], [temps for L0, L1 and L2], ...
    model_sets = make_successive(feats)
    # models that are not distributed or summed
    candidate_feature_sets = list(itertools.chain(*[list(itertools.product(*model_set)) for model_set in model_sets]))
    candidate_models = []
    for feat_set in candidate_feature_sets:
        # multiply the interactions through and flatten the feature list
        candidate_models.append(list(itertools.chain(*[[param_order[1]*temp_feat for temp_feat in param_order[0]] for param_order in feat_set])))
    return candidate_models


def build_candidate_models(configuration, features):
    """
    Return a dictionary of features and candidate models

    Parameters
    ----------
    configuration : tuple
        Configuration tuple, e.g. (('A', 'B', 'C'), 'A')
    features : dict
        Dictionary of {str: list} of generic features for a model, not
        considering the configuration. For example:
        {'CPM_FORM': [symengine.S.One, v.T, v.T**2, v.T**3]}

    Returns
    -------
    dict
        Dictionary of {feature: [candidate_models])

    Notes
    -----
    Currently only works for binary and ternary interactions.

    Candidate models match the following spec:
    1. Candidates with multiple features specified will have
    2. orders of parameters (L0, L0 and L1, ...) have the same number of temperatures

    Note that high orders of parameters with multiple temperatures are not
    required to contain all the temperatures of the low order parameters. For
    example, the following parameters can be generated
    L0: A
    L1: A + BT
    """
    feature_candidate_models = {}
    if not interaction_test(configuration):  # endmembers only
        for feature_name, temperature_features in features.items():
            interaction_features = (symengine.S.One,)
            feature_candidate_models[feature_name] = build_feature_sets(temperature_features, interaction_features)
    elif interaction_test(configuration, 2):  # has a binary interaction
        YS = symengine.Symbol('YS')  # Product of all nonzero site fractions in all sublattices
        Z = symengine.Symbol('Z')
        for feature_name, temperature_features in features.items():
            # generate increasingly complex interactions (power of Z is Redlich-Kister order)
            interaction_features = (YS, YS*Z, YS*(Z**2), YS*(Z**3))  # L0, L1, L2, L3
            feature_candidate_models[feature_name] = build_feature_sets(temperature_features, interaction_features)
    elif interaction_test(configuration, 3):  # has a ternary interaction
        # Ternaries interactions should have exactly two interaction sets:
        # 1. a single symmetric ternary parameter (YS)
        YS = symengine.Symbol('YS')  # Product of all nonzero site fractions in all sublattices
        # 2. L0, L1, and L2 parameters
        V_I, V_J, V_K = symengine.Symbol('V_I'), symengine.Symbol('V_J'), symengine.Symbol('V_K')
        symmetric_interactions = (YS,) # symmetric L0
        for feature_name, temperature_features in features.items():
            # We are ignoring cases where we have L0 == L1 != L2 (and like
            # permutations) because these cases (where two elements exactly the
            # same behavior) don't exist in reality. Tthe symmetric case is
            # mainly for small corrections and dimensionality reduction.
            # Because we don't want our parameter interactions to be successive
            # (i.e. products of symmetric and asymmetric terms), we'll candidates in two steps
            tern_ix_cands = []
            tern_ix_cands += build_feature_sets(temperature_features, symmetric_interactions)
            # special handling for asymmetric features, we don't want a successive V_I, V_J, V_K, but all three should be present
            asym_feats = (
                build_feature_sets(temperature_features, (YS * V_I,)), # asymmetric L0
                build_feature_sets(temperature_features, (YS * V_J,)), # asymmetric L1
                build_feature_sets(temperature_features, (YS * V_K,)), # asymmetric L2
            )                
            for v_i_feats, v_j_feats, v_k_feats in zip(*asym_feats):
                tern_ix_cands.append(v_i_feats + v_j_feats + v_k_feats)
            feature_candidate_models[feature_name] = tern_ix_cands
    else:
        raise ValueError(f"Interaction order not known for configuration {configuration}")
    return feature_candidate_models
