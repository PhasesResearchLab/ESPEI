"""
Tests for building models for parameter selection
"""

from collections import OrderedDict

import sympy
from pycalphad import variables as v

from espei.parameter_selection.model_building import build_feature_sets, build_candidate_models
from espei.sublattice_tools import generate_symmetric_group, sorted_interactions


def test_build_feature_sets_generates_desired_binary_features_for_cp_like():
    """Binary feature sets can be correctly generated for heat capacity-like features"""
    binary_temp_features = ['TlogT', 'T**2', '1/T', 'T**3']
    binary_excess_features= ['YS', 'YS*Z', 'YS*Z**2', 'YS*Z**3']
    feat_sets = build_feature_sets(binary_temp_features, binary_excess_features)
    assert len(feat_sets) == 340
    assert feat_sets[0] == ((['TlogT'], 'YS'),)
    assert feat_sets[5] == ((['TlogT'], 'YS'), (['TlogT', 'T**2'], 'YS*Z'))
    assert feat_sets[-1] == ((['TlogT', 'T**2', '1/T', 'T**3'], 'YS'), (['TlogT', 'T**2', '1/T', 'T**3'], 'YS*Z'), (['TlogT', 'T**2', '1/T', 'T**3'], 'YS*Z**2'), (['TlogT', 'T**2', '1/T', 'T**3'], 'YS*Z**3'))


def test_build_feature_sets_generates_desired_binary_features_for_h_like():
    """Binary feature sets can be correctly generated for enthalpy-like models"""
    binary_temp_features = ['1']
    binary_excess_features= ['YS', 'YS*Z', 'YS*Z**2', 'YS*Z**3']
    feat_sets = build_feature_sets(binary_temp_features, binary_excess_features)
    assert len(feat_sets) == 4
    assert feat_sets[0] == ((['1'], 'YS'),)
    assert feat_sets[1] == ((['1'], 'YS'), (['1'], 'YS*Z'))
    assert feat_sets[2] == ((['1'], 'YS'), (['1'], 'YS*Z'), (['1'], 'YS*Z**2'))
    assert feat_sets[3] == ((['1'], 'YS'), (['1'], 'YS*Z'), (['1'], 'YS*Z**2'), (['1'], 'YS*Z**3'))


def test_build_feature_sets_generates_desired_ternary_features():
    """Ternary feature sets can be correctly generated"""
    ternary_temp_features = ['1']
    ternary_excess_features= [('YS',), ('YS*V_I', 'YS*V_J', 'YS*V_K')]
    feat_sets = build_feature_sets(ternary_temp_features, ternary_excess_features)
    assert len(feat_sets) == 2
    assert feat_sets[0] == ((['1'], ('YS',)),)
    assert feat_sets[1] == ((['1'], ('YS',)), (['1'], ('YS*V_I', 'YS*V_J', 'YS*V_K')))



def test_binary_candidate_models_are_constructed_correctly():
    """Candidate models should be generated for all valid combinations of possible models in the binary case"""
    features = OrderedDict([("CPM_FORM",
                 (v.T*sympy.log(v.T), v.T**2)),
                ("SM_FORM", (v.T,)),
                ("HM_FORM", (sympy.S.One,))
                ])
    YS = sympy.Symbol('YS')
    Z = sympy.Symbol('Z')
    candidate_models = build_candidate_models((('A', 'B'), 'A'), features)
    assert candidate_models == OrderedDict([
        ('CPM_FORM', [
            [v.T*YS*sympy.log(v.T)],
            [v.T*YS*sympy.log(v.T), v.T**2*YS],
            [v.T*YS*sympy.log(v.T), v.T*YS*Z*sympy.log(v.T)],
            [v.T*YS*sympy.log(v.T), v.T*YS*Z*sympy.log(v.T), v.T**2*YS*Z],
            [v.T*YS*sympy.log(v.T), v.T**2*YS, v.T*YS*Z*sympy.log(v.T)],
            [v.T*YS*sympy.log(v.T), v.T**2*YS, v.T*YS*Z*sympy.log(v.T), v.T**2*YS*Z],
            [v.T*YS*sympy.log(v.T), v.T*YS*Z*sympy.log(v.T), v.T*YS*Z**2*sympy.log(v.T)],
            [v.T*YS*sympy.log(v.T), v.T*YS*Z*sympy.log(v.T), v.T*YS*Z**2*sympy.log(v.T), v.T**2*YS*Z**2],
            [v.T*YS*sympy.log(v.T), v.T*YS*Z*sympy.log(v.T), v.T**2*YS*Z, v.T*YS*Z**2*sympy.log(v.T)],
            [v.T*YS*sympy.log(v.T), v.T*YS*Z*sympy.log(v.T), v.T**2*YS*Z, v.T*YS*Z**2*sympy.log(v.T), v.T**2*YS*Z**2],
            [v.T*YS*sympy.log(v.T), v.T**2*YS, v.T*YS*Z*sympy.log(v.T), v.T*YS*Z**2*sympy.log(v.T)],
            [v.T*YS*sympy.log(v.T), v.T**2*YS, v.T*YS*Z*sympy.log(v.T), v.T*YS*Z**2*sympy.log(v.T), v.T**2*YS*Z**2],
            [v.T*YS*sympy.log(v.T), v.T**2*YS, v.T*YS*Z*sympy.log(v.T), v.T**2*YS*Z, v.T*YS*Z**2*sympy.log(v.T)],
            [v.T*YS*sympy.log(v.T), v.T**2*YS, v.T*YS*Z*sympy.log(v.T), v.T**2*YS*Z, v.T*YS*Z**2*sympy.log(v.T), v.T**2*YS*Z**2],
            [v.T*YS*sympy.log(v.T), v.T*YS*Z*sympy.log(v.T), v.T*YS*Z**2*sympy.log(v.T), v.T*YS*Z**3*sympy.log(v.T)],
            [v.T*YS*sympy.log(v.T), v.T*YS*Z*sympy.log(v.T), v.T*YS*Z**2*sympy.log(v.T), v.T*YS*Z**3*sympy.log(v.T), v.T**2*YS*Z**3],
            [v.T*YS*sympy.log(v.T), v.T*YS*Z*sympy.log(v.T), v.T*YS*Z**2*sympy.log(v.T), v.T**2*YS*Z**2, v.T*YS*Z**3*sympy.log(v.T)],
            [v.T*YS*sympy.log(v.T), v.T*YS*Z*sympy.log(v.T), v.T*YS*Z**2*sympy.log(v.T), v.T**2*YS*Z**2, v.T*YS*Z**3*sympy.log(v.T), v.T**2*YS*Z**3],
            [v.T*YS*sympy.log(v.T), v.T*YS*Z*sympy.log(v.T), v.T**2*YS*Z, v.T*YS*Z**2*sympy.log(v.T), v.T*YS*Z**3*sympy.log(v.T)],
            [v.T*YS*sympy.log(v.T), v.T*YS*Z*sympy.log(v.T), v.T**2*YS*Z, v.T*YS*Z**2*sympy.log(v.T), v.T*YS*Z**3*sympy.log(v.T), v.T**2*YS*Z**3],
            [v.T*YS*sympy.log(v.T), v.T*YS*Z*sympy.log(v.T), v.T**2*YS*Z, v.T*YS*Z**2*sympy.log(v.T), v.T**2*YS*Z**2, v.T*YS*Z**3*sympy.log(v.T)],
            [v.T*YS*sympy.log(v.T), v.T*YS*Z*sympy.log(v.T), v.T**2*YS*Z, v.T*YS*Z**2*sympy.log(v.T), v.T**2*YS*Z**2, v.T*YS*Z**3*sympy.log(v.T), v.T**2*YS*Z**3],
            [v.T*YS*sympy.log(v.T), v.T**2*YS, v.T*YS*Z*sympy.log(v.T), v.T*YS*Z**2*sympy.log(v.T), v.T*YS*Z**3*sympy.log(v.T)],
            [v.T*YS*sympy.log(v.T), v.T**2*YS, v.T*YS*Z*sympy.log(v.T), v.T*YS*Z**2*sympy.log(v.T), v.T*YS*Z**3*sympy.log(v.T), v.T**2*YS*Z**3],
            [v.T*YS*sympy.log(v.T), v.T**2*YS, v.T*YS*Z*sympy.log(v.T), v.T*YS*Z**2*sympy.log(v.T), v.T**2*YS*Z**2, v.T*YS*Z**3*sympy.log(v.T)],
            [v.T*YS*sympy.log(v.T), v.T**2*YS, v.T*YS*Z*sympy.log(v.T), v.T*YS*Z**2*sympy.log(v.T), v.T**2*YS*Z**2, v.T*YS*Z**3*sympy.log(v.T), v.T**2*YS*Z**3],
            [v.T*YS*sympy.log(v.T), v.T**2*YS, v.T*YS*Z*sympy.log(v.T), v.T**2*YS*Z, v.T*YS*Z**2*sympy.log(v.T), v.T*YS*Z**3*sympy.log(v.T)],
            [v.T*YS*sympy.log(v.T), v.T**2*YS, v.T*YS*Z*sympy.log(v.T), v.T**2*YS*Z, v.T*YS*Z**2*sympy.log(v.T), v.T*YS*Z**3*sympy.log(v.T), v.T**2*YS*Z**3],
            [v.T*YS*sympy.log(v.T), v.T**2*YS, v.T*YS*Z*sympy.log(v.T), v.T**2*YS*Z, v.T*YS*Z**2*sympy.log(v.T), v.T**2*YS*Z**2, v.T*YS*Z**3*sympy.log(v.T)],
            [v.T*YS*sympy.log(v.T), v.T**2*YS, v.T*YS*Z*sympy.log(v.T), v.T**2*YS*Z, v.T*YS*Z**2*sympy.log(v.T), v.T**2*YS*Z**2, v.T*YS*Z**3*sympy.log(v.T), v.T**2*YS*Z**3]
        ]),
        ('SM_FORM', [
            [v.T*YS],
            [v.T*YS, v.T*YS*Z],
            [v.T*YS, v.T*YS*Z, v.T*YS*Z**2],
            [v.T*YS, v.T*YS*Z, v.T*YS*Z**2, v.T*YS*Z**3]
        ]),
        ('HM_FORM', [
            [YS],
            [YS, YS*Z],
            [YS, YS*Z, YS*Z**2],
            [YS, YS*Z, YS*Z**2, YS*Z**3]
        ])
    ])


def test_ternary_candidate_models_are_constructed_correctly():
    """Candidate models should be generated for all valid combinations of possible models in the ternary case"""
    features = OrderedDict([("CPM_FORM",
                 (v.T*sympy.log(v.T), v.T**2)),
                ("SM_FORM", (v.T,)),
                ("HM_FORM", (sympy.S.One,))
                ])
    YS = sympy.Symbol('YS')
    V_I, V_J, V_K = sympy.Symbol('V_I'), sympy.Symbol('V_J'), sympy.Symbol('V_K')
    candidate_models = build_candidate_models((('A', 'B', 'C'), 'A'), features)
    assert candidate_models == OrderedDict([
        ('CPM_FORM', [
            [v.T*YS*sympy.log(v.T)],
            [v.T*YS*sympy.log(v.T), v.T**2*YS],
            [v.T*V_I*YS*sympy.log(v.T), v.T*V_J*YS*sympy.log(v.T), v.T*V_K*YS*sympy.log(v.T)],
            [v.T*V_I*YS*sympy.log(v.T), v.T*V_J*YS*sympy.log(v.T), v.T*V_K*YS*sympy.log(v.T), v.T**2*V_I*YS, v.T**2*V_J*YS, v.T**2*V_K*YS],
        ]),
        ('SM_FORM', [
            [v.T*YS],
            [v.T*V_I*YS, v.T*V_J*YS, v.T*V_K*YS]
        ]),
        ('HM_FORM', [
            [YS],
            [V_I*YS, V_J*YS, V_K*YS]
        ])
    ])

def test_symmetric_group_can_be_generated_for_2_sl_mixing_with_symmetry():
    """A phase with two sublattices that are mixing should generate a cross interaction"""
    symm_groups = generate_symmetric_group((('AL', 'CO'), ('AL', 'CO')), [[0, 1]])
    assert symm_groups == [(('AL', 'CO'), ('AL', 'CO'))]


def test_symmetric_group_can_be_generated_for_2_sl_endmembers_with_symmetry():
    """A phase with symmetric sublattices should find a symmetric endmember """
    symm_groups = generate_symmetric_group(('AL', 'CO'), [[0, 1]])
    assert symm_groups == [('AL', 'CO'), ('CO', 'AL')]


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

