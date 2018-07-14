"""
Test the utilities for parameter generation
"""

from espei.parameter_selection.utils import interaction_test

def test_interaction_detection():
    """interaction_test should correctly calculate interactions for different candidate configurations"""

    no_interaction_configurations = [
        ('A', 'B', 'AL', 'ABKEJF'),
        ('A', 'A', 'A',),
        ('A',),
        ('ABCDEDFG',),
        ('AL',),
        ('AL', 'ZN'),
    ]
    for config in no_interaction_configurations:
        assert interaction_test(config) == False

    binary_configurations = [
        (('A', 'B'),),
        (('A', 'B'), ('A', 'B')),
        (('AL', 'ZN'), 'A'),
        (('AL', 'ZN'), 'AB'),
        (('AL', 'ZN'), 'ABC'),
        (('AL', 'ZN'), 'NI'),
        (('AL', 'ZN'), 'NI', 'NI'),
        (('AL', 'NI'), ('AL', 'NI')),
    ]
    for config in binary_configurations:
        assert interaction_test(config) == True
        assert interaction_test(config, 2) == True
        assert interaction_test(config, 3) == False

    ternary_configurations = [
        (('A', 'B', 'C'),),
        (('AL', 'BR', 'CL'),),
        (('ALAEF', 'BREFAEF', 'CFEFAL'),),
        (('A', 'B', 'C'), ('A', 'B', 'ZN')),
        (('AL', 'CR', 'ZN'), 'A'),
        (('AL', 'CR', 'ZN'), 'AB'),
        (('AL', 'CR', 'ZN'), 'ABC'),
        (('AL', 'CR', 'ZN'), 'NI',),
        (('AL', 'CR', 'ZN'), 'NI', 'NI'),
        (('AL', 'CR', 'NI'), ('AL', 'CR', 'NI')),
    ]
    for config in ternary_configurations:
        assert interaction_test(config) == True
        assert interaction_test(config, 2) == False
        assert interaction_test(config, 3) == True
