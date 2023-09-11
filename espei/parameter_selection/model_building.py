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


def build_redlich_kister_candidate_models(configuration, *noninteraction_features, max_binary_redlich_kister_order=3, ternary_symmetric_parameter=True, ternary_asymmetric_parameters=True):
    """
    Return a list of candidate symbolic models for a string configuration and
    set of symbolic features. The candidate models are a Cartesian product of
    the successive non-mixing features and successive Redlich-Kister-Muggianu
    mixing features.

    Here "successive" means that we take the features and (exhaustively)
    generate models of increasing complexity. For example, if we have non-mixing
    features [1, T, T**2, T**3], then we generate 4 candidates of increasing
    complexity: [1], [1, T], [1, T, T**2], and [1, T, T**2, T**3]. For a max
    Redlich-Kister order of 2, we have [L0, L1, L2] candidates and there will be
    3 candidate features sets of increasing complexity for mixing: [L0],
    [L0, L1], and [L0, L1, L2].

    Parameters
    ----------
    configuration : Tuple[str]
        Configuration tuple, e.g. (('A', 'B', 'C'), 'A')
    noninteraction_features : List[Symbol]
        Each entry is a list of non-mixing features, for example: temperature
        features [symengine.S.One, v.T, v.T**2, v.T**3] or pressure features
        [symengine.S.One, v.P, v.P**3, v.P**3]. Note that only one set of
        non-mixing features are currently allowed.
    max_binary_redlich_kister_order : int
        For binary mixing configurations: highest order Redlich-Kister
        interaction parameter to generate. 0 gives L0, 1 gives L0 and L1, etc.
    ternary_symmetric_parameter : bool
        For ternary mixing configurations: if true (the default), add a
        symmetric interaction parameter.
    ternary_asymmetric_parameters : bool
        For ternary mixing configurations: if true (the default), add asymmetric
        interaction parameters.

    Returns
    -------
    List[List[Symbol]]
        List of candidate models

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
    # TODO: generalize build_feature_sets to enable multiple sets of non-mixing features
    # suggested API: build_feature_sets(interaction_features, *noninteraction_features)
    assert len(noninteraction_features) == 1, f"Exactly one set of non-mixing features is allowed. Got {len(noninteraction_features)}: {noninteraction_features}"
    if not interaction_test(configuration):  # endmembers only
        interaction_features = (symengine.S.One,)
        return build_feature_sets(*noninteraction_features, interaction_features)
    elif interaction_test(configuration, 2):  # has a binary interaction
        YS = symengine.Symbol('YS')  # Product of all nonzero site fractions in all sublattices
        Z = symengine.Symbol('Z')
        if max_binary_redlich_kister_order >= 0:
            # generate increasingly complex interactions (power of Z is Redlich-Kister order)
            interaction_features = [YS*(Z**order) for order in range(0, max_binary_redlich_kister_order + 1)]
        else:
            interaction_features = []
        return build_feature_sets(*noninteraction_features, interaction_features)
    elif interaction_test(configuration, 3):  # has a ternary interaction
        # Ternaries interactions should have exactly two interaction sets:
        # 1. a single symmetric ternary parameter (YS)
        YS = symengine.Symbol('YS')  # Product of all nonzero site fractions in all sublattices
        # 2. L0, L1, and L2 parameters
        V_I, V_J, V_K = symengine.Symbol('V_I'), symengine.Symbol('V_J'), symengine.Symbol('V_K')
        ternary_feature_sets = []
        if ternary_symmetric_parameter:
            ternary_feature_sets += build_feature_sets(*noninteraction_features, (YS,))
        if ternary_asymmetric_parameters:
            # We are ignoring cases where we have L0 == L1 != L2 (and like
            # permutations) because these cases (where two elements exactly the
            # same behavior) don't exist in reality. The symmetric case is
            # mainly for small corrections and dimensionality reduction.
            # Because we don't want our parameter interactions to be successive
            # (i.e. products of symmetric and asymmetric terms), we'll candidates in two steps
            # special handling for asymmetric features, we don't want a successive V_I, V_J, V_K, but all three should be present
            asym_feats = (
                build_feature_sets(*noninteraction_features, (YS * V_I,)), # asymmetric L0
                build_feature_sets(*noninteraction_features, (YS * V_J,)), # asymmetric L1
                build_feature_sets(*noninteraction_features, (YS * V_K,)), # asymmetric L2
            )
            for v_i_feats, v_j_feats, v_k_feats in zip(*asym_feats):
                ternary_feature_sets.append(v_i_feats + v_j_feats + v_k_feats)
        return ternary_feature_sets
    else:
        raise ValueError(f"Interaction order not known for configuration {configuration}")
