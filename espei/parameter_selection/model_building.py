"""
Building candidate models
"""

import itertools
import sympy
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
    For $N$ temperature features and $M$ interaction features, the total number of
    feature sets should be $N*(1-N**M)/(1-N)$. If $N=1$, then there are $M$ total
    feature sets.

    """
    # [[A], [A, B], [A, B, C], ...]
    temps = make_successive(temperature_features)
    # [ [temps for L0], [temps for L1], [temps for L2], ...]
    feats = [list(itertools.product(temps, [inter])) for inter in interaction_features]
    # [ [temps for L0], [temps for L0 and L1], [temps for L0, L1 and L2], ...
    model_sets = make_successive(feats)
    # models that are not distributed or summed
    candidate_feature_sets = list(itertools.chain(*[list(itertools.product(*model_set)) for model_set in model_sets]))
    return candidate_feature_sets


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
        {'CPM_FORM': [sympy.S.One, v.T, v.T**2, v.T**3]}

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
    if not interaction_test(configuration):  # endmembers only
        for feature in features.keys():
            features[feature] = make_successive(features[feature])
        return features

    elif interaction_test(configuration, 2):  # has a binary interaction
        YS = sympy.Symbol('YS')  # Product of all nonzero site fractions in all sublattices
        Z = sympy.Symbol('Z')
        # generate increasingly complex interactions, # L0, L1, L2
        parameter_interactions = [YS, YS*Z, YS*(Z**2), YS*(Z**3)]
        for feature in features.keys():
            candidate_feature_sets = build_feature_sets(features[feature], parameter_interactions)
            # list of (for example): ((['TlogT'], 'YS'), (['TlogT', 'T**2'], 'YS*Z'))
            candidate_models = []
            for feat_set in candidate_feature_sets:
                # multiply the interactions through and flatten the feature list
                candidate_models.append(list(itertools.chain(*[[param_order[1]*temp_feat for temp_feat in param_order[0]] for param_order in feat_set])))
            features[feature] = candidate_models
        return features

    elif interaction_test(configuration, 3):  # has a ternary interaction
        # Ternaries interactions should have exactly two candidate models:
        # 1. a single symmetric ternary parameter (YS)
        # 2. L0, L1, and L2 parameters corresponding to Muggianu parameters
        # We are ignoring cases where we have L0 == L1, but L0 != L2 and all of the
        # combinations these cases don't exist in reality (where two elements have
        # exactly the same behavior) the symmetric case is mainly for small
        # corrections and dimensionality reduction.
        YS = sympy.Symbol('YS')  # Product of all nonzero site fractions in all sublattices
        # Muggianu ternary interaction product for components i, j, and k
        V_I, V_J, V_K = sympy.Symbol('V_I'), sympy.Symbol('V_J'), sympy.Symbol('V_K')
        # because we don't want our parameter interactions to go sequentially, we'll construct models in two steps
        symmetric_interactions = [(YS,)] # symmetric L0
        asymmetric_interactions = [(YS * V_I, YS * V_J, YS * V_K)] # asymmetric L0, L1, and L2
        for feature in features.keys():
            sym_candidate_feature_sets = build_feature_sets(features[feature], symmetric_interactions)
            asym_candidate_feature_sets = build_feature_sets(features[feature], asymmetric_interactions)
            # list of (for example): ((['TlogT'], ('YS',)), (['TlogT', 'T**2'], ('YS',))
            candidate_models = []
            for feat_set in itertools.chain(sym_candidate_feature_sets, asym_candidate_feature_sets):
                feat_set_params = []
                # multiply the interactions through with distributing
                for temp_feats, inter_feats in feat_set:
                    # temperature features and interaction features
                    feat_set_params.append([inter*feat for inter, feat in itertools.product(temp_feats, inter_feats)])
                candidate_models.append(list(itertools.chain(*feat_set_params)))
            features[feature] = candidate_models
        return features
