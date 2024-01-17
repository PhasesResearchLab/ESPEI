"""
Tests for building models for parameter selection
"""

from collections import OrderedDict

import symengine
from pycalphad import variables as v

from espei.parameter_selection.model_building import build_candidate_models, build_redlich_kister_candidate_models, make_successive
from espei.sublattice_tools import generate_symmetric_group, sorted_interactions


def test_build_feature_sets_generates_desired_binary_features_for_cp_like():
    """Binary feature sets can be correctly generated for heat capacity-like features"""
    YS = symengine.Symbol("YS")
    Z = symengine.Symbol("Z")
    temp_feature_sets = make_successive([v.T, v.T**2, 1/v.T, v.T**3])
    excess_features= [YS, YS*Z, YS*Z**2, YS*Z**3]
    feat_sets = build_candidate_models(temp_feature_sets, excess_features)
    assert len(feat_sets) == 340
    assert feat_sets[0] == [v.T*YS]
    assert feat_sets[5] == [v.T*YS, v.T*YS*Z, v.T**2*YS*Z]
    assert feat_sets[-1] == [
        v.T * YS,        v.T**2 * YS,        1/v.T * YS,        v.T**3 * YS,
        v.T * YS * Z,    v.T**2 * YS * Z,    1/v.T * YS * Z,    v.T**3 * YS * Z,
        v.T * YS * Z**2, v.T**2 * YS * Z**2, 1/v.T * YS * Z**2, v.T**3 * YS * Z**2,
        v.T * YS * Z**3, v.T**2 * YS * Z**3, 1/v.T * YS * Z**3, v.T**3 * YS * Z**3,
        ]


def test_binary_candidate_models_are_constructed_correctly():
    """Candidate models should be generated for all valid combinations of possible models in the binary case"""
    YS = symengine.Symbol('YS')
    Z = symengine.Symbol('Z')
    CPM_FORM_feature_sets = make_successive([v.T*symengine.log(v.T), v.T**2])
    candidate_models = build_redlich_kister_candidate_models((('A', 'B'), 'A'), CPM_FORM_feature_sets)
    assert candidate_models == [
            [v.T*YS*symengine.log(v.T)],
            [v.T*YS*symengine.log(v.T), v.T**2*YS],
            [v.T*YS*symengine.log(v.T), v.T*YS*Z*symengine.log(v.T)],
            [v.T*YS*symengine.log(v.T), v.T*YS*Z*symengine.log(v.T), v.T**2*YS*Z],
            [v.T*YS*symengine.log(v.T), v.T**2*YS, v.T*YS*Z*symengine.log(v.T)],
            [v.T*YS*symengine.log(v.T), v.T**2*YS, v.T*YS*Z*symengine.log(v.T), v.T**2*YS*Z],
            [v.T*YS*symengine.log(v.T), v.T*YS*Z*symengine.log(v.T), v.T*YS*Z**2*symengine.log(v.T)],
            [v.T*YS*symengine.log(v.T), v.T*YS*Z*symengine.log(v.T), v.T*YS*Z**2*symengine.log(v.T), v.T**2*YS*Z**2],
            [v.T*YS*symengine.log(v.T), v.T*YS*Z*symengine.log(v.T), v.T**2*YS*Z, v.T*YS*Z**2*symengine.log(v.T)],
            [v.T*YS*symengine.log(v.T), v.T*YS*Z*symengine.log(v.T), v.T**2*YS*Z, v.T*YS*Z**2*symengine.log(v.T), v.T**2*YS*Z**2],
            [v.T*YS*symengine.log(v.T), v.T**2*YS, v.T*YS*Z*symengine.log(v.T), v.T*YS*Z**2*symengine.log(v.T)],
            [v.T*YS*symengine.log(v.T), v.T**2*YS, v.T*YS*Z*symengine.log(v.T), v.T*YS*Z**2*symengine.log(v.T), v.T**2*YS*Z**2],
            [v.T*YS*symengine.log(v.T), v.T**2*YS, v.T*YS*Z*symengine.log(v.T), v.T**2*YS*Z, v.T*YS*Z**2*symengine.log(v.T)],
            [v.T*YS*symengine.log(v.T), v.T**2*YS, v.T*YS*Z*symengine.log(v.T), v.T**2*YS*Z, v.T*YS*Z**2*symengine.log(v.T), v.T**2*YS*Z**2],
            [v.T*YS*symengine.log(v.T), v.T*YS*Z*symengine.log(v.T), v.T*YS*Z**2*symengine.log(v.T), v.T*YS*Z**3*symengine.log(v.T)],
            [v.T*YS*symengine.log(v.T), v.T*YS*Z*symengine.log(v.T), v.T*YS*Z**2*symengine.log(v.T), v.T*YS*Z**3*symengine.log(v.T), v.T**2*YS*Z**3],
            [v.T*YS*symengine.log(v.T), v.T*YS*Z*symengine.log(v.T), v.T*YS*Z**2*symengine.log(v.T), v.T**2*YS*Z**2, v.T*YS*Z**3*symengine.log(v.T)],
            [v.T*YS*symengine.log(v.T), v.T*YS*Z*symengine.log(v.T), v.T*YS*Z**2*symengine.log(v.T), v.T**2*YS*Z**2, v.T*YS*Z**3*symengine.log(v.T), v.T**2*YS*Z**3],
            [v.T*YS*symengine.log(v.T), v.T*YS*Z*symengine.log(v.T), v.T**2*YS*Z, v.T*YS*Z**2*symengine.log(v.T), v.T*YS*Z**3*symengine.log(v.T)],
            [v.T*YS*symengine.log(v.T), v.T*YS*Z*symengine.log(v.T), v.T**2*YS*Z, v.T*YS*Z**2*symengine.log(v.T), v.T*YS*Z**3*symengine.log(v.T), v.T**2*YS*Z**3],
            [v.T*YS*symengine.log(v.T), v.T*YS*Z*symengine.log(v.T), v.T**2*YS*Z, v.T*YS*Z**2*symengine.log(v.T), v.T**2*YS*Z**2, v.T*YS*Z**3*symengine.log(v.T)],
            [v.T*YS*symengine.log(v.T), v.T*YS*Z*symengine.log(v.T), v.T**2*YS*Z, v.T*YS*Z**2*symengine.log(v.T), v.T**2*YS*Z**2, v.T*YS*Z**3*symengine.log(v.T), v.T**2*YS*Z**3],
            [v.T*YS*symengine.log(v.T), v.T**2*YS, v.T*YS*Z*symengine.log(v.T), v.T*YS*Z**2*symengine.log(v.T), v.T*YS*Z**3*symengine.log(v.T)],
            [v.T*YS*symengine.log(v.T), v.T**2*YS, v.T*YS*Z*symengine.log(v.T), v.T*YS*Z**2*symengine.log(v.T), v.T*YS*Z**3*symengine.log(v.T), v.T**2*YS*Z**3],
            [v.T*YS*symengine.log(v.T), v.T**2*YS, v.T*YS*Z*symengine.log(v.T), v.T*YS*Z**2*symengine.log(v.T), v.T**2*YS*Z**2, v.T*YS*Z**3*symengine.log(v.T)],
            [v.T*YS*symengine.log(v.T), v.T**2*YS, v.T*YS*Z*symengine.log(v.T), v.T*YS*Z**2*symengine.log(v.T), v.T**2*YS*Z**2, v.T*YS*Z**3*symengine.log(v.T), v.T**2*YS*Z**3],
            [v.T*YS*symengine.log(v.T), v.T**2*YS, v.T*YS*Z*symengine.log(v.T), v.T**2*YS*Z, v.T*YS*Z**2*symengine.log(v.T), v.T*YS*Z**3*symengine.log(v.T)],
            [v.T*YS*symengine.log(v.T), v.T**2*YS, v.T*YS*Z*symengine.log(v.T), v.T**2*YS*Z, v.T*YS*Z**2*symengine.log(v.T), v.T*YS*Z**3*symengine.log(v.T), v.T**2*YS*Z**3],
            [v.T*YS*symengine.log(v.T), v.T**2*YS, v.T*YS*Z*symengine.log(v.T), v.T**2*YS*Z, v.T*YS*Z**2*symengine.log(v.T), v.T**2*YS*Z**2, v.T*YS*Z**3*symengine.log(v.T)],
            [v.T*YS*symengine.log(v.T), v.T**2*YS, v.T*YS*Z*symengine.log(v.T), v.T**2*YS*Z, v.T*YS*Z**2*symengine.log(v.T), v.T**2*YS*Z**2, v.T*YS*Z**3*symengine.log(v.T), v.T**2*YS*Z**3]
    ]
    SM_FORM_feature_sets = make_successive([v.T,])
    candidate_models = build_redlich_kister_candidate_models((('A', 'B'), 'A'), SM_FORM_feature_sets)
    assert candidate_models == [
            [v.T*YS],
            [v.T*YS, v.T*YS*Z],
            [v.T*YS, v.T*YS*Z, v.T*YS*Z**2],
            [v.T*YS, v.T*YS*Z, v.T*YS*Z**2, v.T*YS*Z**3]
    ]
    HM_FORM_feature_sets = make_successive([symengine.S.One,])
    candidate_models = build_redlich_kister_candidate_models((('A', 'B'), 'A'), HM_FORM_feature_sets)
    assert candidate_models == [
            [YS],
            [YS, YS*Z],
            [YS, YS*Z, YS*Z**2],
            [YS, YS*Z, YS*Z**2, YS*Z**3]
        ]

def test_simplified_candidate_model_generation():
    # this uses the feature sets as above, but
    YS = symengine.Symbol('YS')
    Z = symengine.Symbol('Z')
    CPM_FORM_feature_sets = make_successive([v.T*symengine.log(v.T), v.T**2])
    interaction_features = [YS*(Z**order) for order in range(0, 4)] # L0-L3
    candidate_models = build_candidate_models(CPM_FORM_feature_sets, interaction_features)
    assert len(candidate_models) == 30  # tested in detail in test_binary_candidate_models_are_constructed_correctly
    # now we limit the number of candidate models and test that fewer are generated
    candidate_models = build_candidate_models(CPM_FORM_feature_sets, interaction_features, complex_algorithm_candidate_limit=29)
    assert len(candidate_models) == len(CPM_FORM_feature_sets) * len(interaction_features)


def test_ternary_candidate_models_are_constructed_correctly():
    """Candidate models should be generated for all valid combinations of possible models in the ternary case"""
    YS = symengine.Symbol('YS')
    V_I, V_J, V_K = symengine.Symbol('V_I'), symengine.Symbol('V_J'), symengine.Symbol('V_K')

    CPM_FEATURE_SETS = make_successive([v.T*symengine.log(v.T), v.T**2])
    CPM_candidate_models = build_redlich_kister_candidate_models((('A', 'B', 'C'), 'A'), CPM_FEATURE_SETS)
    assert CPM_candidate_models == [
            [v.T*YS*symengine.log(v.T)],
            [v.T*YS*symengine.log(v.T), v.T**2*YS],
            [v.T*V_I*YS*symengine.log(v.T), v.T*V_J*YS*symengine.log(v.T), v.T*V_K*YS*symengine.log(v.T)],
            [v.T*V_I*YS*symengine.log(v.T), v.T**2*V_I*YS, v.T*V_J*YS*symengine.log(v.T), v.T**2*V_J*YS, v.T*V_K*YS*symengine.log(v.T), v.T**2*V_K*YS],
        ]
    SM_FEATURE_SETS = make_successive([v.T])
    SM_candidate_models = build_redlich_kister_candidate_models((('A', 'B', 'C'), 'A'), SM_FEATURE_SETS)
    assert SM_candidate_models == [
            [v.T*YS],
            [v.T*V_I*YS, v.T*V_J*YS, v.T*V_K*YS]
        ]
    HM_FEATURE_SETS = make_successive([symengine.S.One])
    HM_candidate_models = build_redlich_kister_candidate_models((('A', 'B', 'C'), 'A'), HM_FEATURE_SETS)
    assert HM_candidate_models == [
            [YS],
            [V_I*YS, V_J*YS, V_K*YS]
        ]

def test_symmetric_group_can_be_generated_for_2_sl_mixing_with_symmetry():
    """A phase with two sublattices that are mixing should generate a cross interaction"""
    symm_groups = generate_symmetric_group((('AL', 'CO'), ('AL', 'CO')), [[0, 1]])
    assert symm_groups == [(('AL', 'CO'), ('AL', 'CO'))]


def test_symmetric_group_can_be_generated_for_2_sl_endmembers_with_symmetry():
    """A phase with symmetric sublattices should find a symmetric endmember """
    symm_groups = generate_symmetric_group(('AL', 'CO'), [[0, 1]])
    assert symm_groups == [('AL', 'CO'), ('CO', 'AL')]


def test_generating_symmetric_group_works_without_symmetry():
    """generate_symmetric_group returns the passed configuration if symmetry=None"""

    config_D03_A3B = ["A", "A", "A", "B"]
    symm_groups = generate_symmetric_group(config_D03_A3B, None)
    assert symm_groups == [("A", "A", "A", "B")]

    symm_groups = generate_symmetric_group((("CR", "FE"), "VA"), None)
    assert symm_groups == [
        (("CR", "FE"), "VA")
    ]


def test_generating_symmetric_group_bcc_4sl():
    """Binary BCC 4SL ordered symmetric configurations can can be generated"""
    bcc_4sl_symmetry = [[0, 1], [2, 3]]

    config_D03_A3B = ["A", "A", "A", "B"]
    symm_groups = generate_symmetric_group(config_D03_A3B, bcc_4sl_symmetry)
    assert symm_groups == [
        ("A", "A", "A", "B"),
        ("A", "A", "B", "A"),
        ("A", "B", "A", "A"),
        ("B", "A", "A", "A"),
    ]

    config_B2_A2B2 = ["A", "A", "B", "B"]
    symm_groups = generate_symmetric_group(config_B2_A2B2, bcc_4sl_symmetry)
    assert symm_groups == [
        ("A", "A", "B", "B"),
        ("B", "B", "A", "A"),
    ]

    config_B32_A2B2 = ["A", "B", "A", "B"]
    symm_groups = generate_symmetric_group(config_B32_A2B2, bcc_4sl_symmetry)
    assert symm_groups == [
        ("A", "B", "A", "B"),
        ("A", "B", "B", "A"),
        ("B", "A", "A", "B"),
        ("B", "A", "B", "A"),
    ]


def test_generating_symmetric_group_fcc_4sl():
    """Binary FCC 4SL ordered symmetric configurations can can be generated"""
    fcc_4sl_symmetry = [[0, 1, 2, 3]]

    config_L1_2_A3B = ["A", "A", "A", "B"]
    symm_groups = generate_symmetric_group(config_L1_2_A3B, fcc_4sl_symmetry)
    assert symm_groups == [
        ("A", "A", "A", "B"),
        ("A", "A", "B", "A"),
        ("A", "B", "A", "A"),
        ("B", "A", "A", "A"),
    ]

    config_L1_0_A2B2 = ["A", "A", "B", "B"]
    symm_groups = generate_symmetric_group(config_L1_0_A2B2, fcc_4sl_symmetry)
    assert symm_groups == [
        ("A", "A", "B", "B"),
        ("A", "B", "A", "B"),
        ("A", "B", "B", "A"),
        ("B", "A", "A", "B"),
        ("B", "A", "B", "A"),
        ("B", "B", "A", "A"),
    ]


def test_generating_symmetric_group_works_with_interstitial_sublattice():
    """Symmetry groups for phases with an inequivalent vacancy sublattice are correctly generated"""
    bcc_4sl_symmetry = [[0, 1], [2, 3]]
    config_D03_A3B = ["A", "A", "A", "B", "VA"]
    symm_groups = generate_symmetric_group(config_D03_A3B, bcc_4sl_symmetry)
    assert symm_groups == [
        ("A", "A", "A", "B", "VA"),
        ("A", "A", "B", "A", "VA"),
        ("A", "B", "A", "A", "VA"),
        ("B", "A", "A", "A", "VA"),
    ]

    fcc_4sl_symmetry = [[0, 1, 2, 3]]
    config_L1_2_A3B = ["A", "A", "A", "B", "VA"]
    symm_groups = generate_symmetric_group(config_L1_2_A3B, fcc_4sl_symmetry)
    assert symm_groups == [
        ("A", "A", "A", "B", "VA"),
        ("A", "A", "B", "A", "VA"),
        ("A", "B", "A", "A", "VA"),
        ("B", "A", "A", "A", "VA"),
    ]

    # "Unrealistic" cases where the vacancy sublattice is in the middle at index 2
    bcc_4sl_symmetry = [[0, 1], [3, 4]]
    config_D03_A3B = ["A", "A", "VA", "A", "B"]
    symm_groups = generate_symmetric_group(config_D03_A3B, bcc_4sl_symmetry)
    assert symm_groups == [
        ("A", "A", "VA", "A", "B"),
        ("A", "A", "VA", "B", "A"),
        ("A", "B", "VA", "A", "A"),
        ("B", "A", "VA", "A", "A"),
    ]

    fcc_4sl_symmetry = [[0, 1, 3, 4]]
    config_L1_2_A3B = ["A", "A", "VA", "A", "B"]
    symm_groups = generate_symmetric_group(config_L1_2_A3B, fcc_4sl_symmetry)
    assert symm_groups == [
        ("A", "A", "VA", "A", "B"),
        ("A", "A", "VA", "B", "A"),
        ("A", "B", "VA", "A", "A"),
        ("B", "A", "VA", "A", "A"),
    ]



def test_interaction_sorting_is_correct():
    """High order (order >= 3) interactions should sort correctly"""
    # Correct sorting of n-order interactions should sort first by number of
    # interactions of order n, then n-1, then n-2... to 1
    unsorted_interactions = [
        ('AL', ('AL', 'CO', 'CR')),
        (('AL', 'CO'), ('AL', 'CO', 'CR')),
        (('AL', 'CO', 'CR'), ('AL', 'CO', 'CR')),
        (('AL', 'CO', 'CR'), 'AL'),
        (('AL', 'CO', 'CR'), ('AL', 'CO')),
        (('AL', 'CO', 'CR'), ('AL', 'CR')),
        (('AL', 'CO', 'CR'), 'CO'),
        (('AL', 'CO', 'CR'), ('CO', 'CR')),
        (('AL', 'CO', 'CR'), 'CR'),
        (('AL', 'CR'), ('AL', 'CO', 'CR')),
        ('CO', ('AL', 'CO', 'CR')),
        (('CO', 'CR'), ('AL', 'CO', 'CR')),
        ('CR', ('AL', 'CO', 'CR')),
    ]
    interactions = sorted_interactions(unsorted_interactions, max_interaction_order=3, symmetry=None)

    # the numbers are the different sort scores. Two of the same sort scores mean
    # the order doesn't matter
    assert interactions == [
        ('AL', ('AL', 'CO', 'CR')),                # (1, 0, 1)
        (('AL', 'CO', 'CR'), 'AL'),                # (1, 0, 1)
        (('AL', 'CO', 'CR'), 'CO'),                # (1, 0, 1)
        (('AL', 'CO', 'CR'), 'CR'),                # (1, 0, 1)
        ('CO', ('AL', 'CO', 'CR')),                # (1, 0, 1)
        ('CR', ('AL', 'CO', 'CR')),                # (1, 0, 1)
    ]
