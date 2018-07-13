"""
Tests for building models for parameter selection
"""

from espei.parameter_selection.model_building import build_feature_sets

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



