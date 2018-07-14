"""
Tests for building models for parameter selection
"""

from collections import OrderedDict

import sympy
from pycalphad import variables as v

from espei.parameter_selection.model_building import build_feature_sets, build_candidate_models

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
