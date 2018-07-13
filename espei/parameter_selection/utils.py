"""
Tools used across parameter selection modules
"""

import itertools
import sympy

from pycalphad import variables as v

feature_transforms = {"CPM_FORM": lambda x: -v.T*sympy.diff(x, v.T, 2),
                      "CPM_MIX": lambda x: -v.T*sympy.diff(x, v.T, 2),
                      "CPM": lambda x: -v.T*sympy.diff(x, v.T, 2),
                      "SM_FORM": lambda x: -sympy.diff(x, v.T),
                      "SM_MIX": lambda x: -sympy.diff(x, v.T),
                      "SM": lambda x: -sympy.diff(x, v.T),
                      "HM_FORM": lambda x: x - v.T*sympy.diff(x, v.T),
                      "HM_MIX": lambda x: x - v.T*sympy.diff(x, v.T),
                      "HM": lambda x: x - v.T*sympy.diff(x, v.T)}


def endmembers_from_interaction(configuration):
    """For a given configuration with possible interactions, return all the endmembers"""
    config = []
    for c in configuration:
        if isinstance(c, (list, tuple)):
            config.append(c)
        else:
            config.append([c])
    return list(itertools.product(*[tuple(c) for c in config]))


def interaction_test(configuration, order=None):
    """
    Returns True if the configuration has an interaction

    Parameters
    ----------
    order : int, optional
        Specific order to check for. E.g. a value of 3 checks for ternary interactions

    Returns
    -------
    bool
        True if there is an interaction.

    Examples
    --------
    >>> configuration = [['A'], ['A','B']]
    >>> interaction_test(configuration)
    True  # has an interaction
    >>> interaction_test(configuration, order=2)
    True  # has a binary interaction
    >>> interaction_test(configuration, order=3)
    False  # has no ternary interaction

    """
    interacting_species = [len(subl) for subl in configuration]
    if order is None:  # checking for any interaction
        return any([subl_occupation > 1 for subl_occupation in interacting_species])
    else:
        return any([subl_occupation == order for subl_occupation in interacting_species])
