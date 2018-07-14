"""
Tools used across parameter selection modules
"""

import itertools

import numpy as np
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
    interacting_species = [len(subl) for subl in configuration if isinstance(subl, tuple)]
    if order is None:  # checking for any interaction
        return any([subl_occupation > 1 for subl_occupation in interacting_species])
    else:
        return any([subl_occupation == order for subl_occupation in interacting_species])


def shift_reference_state(desired_data, feature_transform, fixed_model):
    """
    Shift data to a new common reference state.
    """
    total_response = []
    for dataset in desired_data:
        values = np.asarray(dataset['values'], dtype=np.object)
        if dataset['solver'].get('sublattice_occupancies', None) is not None:
            value_idx = 0
            for occupancy, config in zip(dataset['solver']['sublattice_occupancies'], dataset['solver']['sublattice_configurations']):
                if dataset['output'].endswith('_FORM'):
                    pass
                elif dataset['output'].endswith('_MIX'):
                    values[..., value_idx] += feature_transform(fixed_model.models['ref'])
                    pass
                else:
                    raise ValueError('Unknown property to shift: {}'.format(dataset['output']))
                value_idx += 1
        total_response.append(values.flatten())
    return total_response


def build_sitefractions(phase_name, sublattice_configurations, sublattice_occupancies):
    """Convert nested lists of sublattice configurations and occupancies to a list
    of dictionaries. The dictionaries map SiteFraction symbols to occupancy
    values. Note that zero occupancy site fractions will need to be added
    separately since the total degrees of freedom aren't known in this function.

    Parameters
    ----------
    phase_name : str
        Name of the phase
    sublattice_configurations : [[str]]
        sublattice configuration
    sublattice_occupancies : [[float]]
        occupancy of each sublattice

    Returns
    -------
    [[float]]
        a list of site fractions over sublattices

    """
    result = []
    for config, occ in zip(sublattice_configurations, sublattice_occupancies):
        sitefracs = {}
        config = [[c] if not isinstance(c, (list, tuple)) else c for c in config]
        occ = [[o] if not isinstance(o, (list, tuple)) else o for o in occ]
        if len(config) != len(occ):
            raise ValueError('Sublattice configuration length differs from occupancies')
        for sublattice_idx in range(len(config)):
            if isinstance(config[sublattice_idx], (list, tuple)) != isinstance(occ[sublattice_idx], (list, tuple)):
                raise ValueError('Sublattice configuration type differs from occupancies')
            if not isinstance(config[sublattice_idx], (list, tuple)):
                # This sublattice is fully occupied by one component
                sitefracs[v.SiteFraction(phase_name, sublattice_idx, config[sublattice_idx])] = occ[sublattice_idx]
            else:
                # This sublattice is occupied by multiple elements
                if len(config[sublattice_idx]) != len(occ[sublattice_idx]):
                    raise ValueError('Length mismatch in sublattice configuration')
                for comp, val in zip(config[sublattice_idx], occ[sublattice_idx]):
                    sitefracs[v.SiteFraction(phase_name, sublattice_idx, comp)] = val
        result.append(sitefracs)
    return result
