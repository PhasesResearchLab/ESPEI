"""
Building candidate models
"""

import itertools

import sympy

from espei.parameter_selection.utils import interaction_test

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
    features :

    Returns
    -------
    dict
        Dictionary of {feature: [candidate_models])

    Notes
    -----
    Currently only works for ternary interactions. The reason binaries don't
    work and ternaries are suboptimal is because we aren't generating all of the
    possible candidates. For example, we never generate candidates that are
    L0: A + BT
    L1: A
    L2: A

    All orders of parameters (L0, L0 and L1, ...) have the same number of temperatures
    """
    if not interaction_test(configuration):  # endmembers only
        raise NotImplementedError('Only ternary interactions supported')

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
        V_i, V_j, V_k = sympy.Symbol('V_i'), sympy.Symbol('V_j'), sympy.Symbol('V_k')
        parameter_interactions = [
            (YS,),  # symmetric L0
            (YS*V_i, YS*V_j, YS*V_k)  # asymmetric L0, L1, and L2
        ]

    # there was an interaction, so now we need to generate all the candidate models
    for feature in features.keys():
        feature_sets = make_successive(features[feature])
        # generate tuples of (parameter_interactions, feature_values), e.g. ('YS', (T*log(T), T**2))
        candidate_tuples = list(itertools.product(parameter_interactions, feature_sets))
        candidates = []
        for interactions, feature_values in candidate_tuples:
            # distribute the parameter interactions through the features
            candidates.append([inter*feat for inter, feat in itertools.product(interactions, feature_values)])
        # update the features dictionary with all the possible candidates for this feature
        features[feature] = candidates

    return features
