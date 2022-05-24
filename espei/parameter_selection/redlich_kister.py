"""
Tools for construction Redlich-Kister polynomials used in parameter selection.
"""

import itertools
import numpy as np
from espei.sublattice_tools import interaction_test


def calc_site_fraction_product(site_fractions):
    """Calculate the site fraction product for sublattice configurations

    Parameters
    ----------
    site_fractions : list
        List of sublattice configurations. The list should be 3d of (configurations, sublattices, values)

    Returns
    -------
    list
        List of site fraction products, YS, for each sublattice

    Examples
    --------
    >>> # site fraction product for an (A,B)(A) configuration
    >>> calc_site_fraction_product([[[0.2, 0.8], 1.0]])  # doctest: +ELLIPSIS
    [0.16...]
    >>> # site fraction product for [(A,B)(A), (A,B)(A)] configurations
    >>> calc_site_fraction_product([[[0.2, 0.8], 1.0], [[0.3, 0.7], 1.0]])  # doctest: +ELLIPSIS
    [0.16..., 0.21]
    >>> # site fraction product for [(A,B)(A,B)] configurations
    >>> calc_site_fraction_product([[[0.2, 0.8], [0.4, 0.6]]])  # doctest: +ELLIPSIS
    [0.0384...]
    >>> # ternary case, (A,B,C) interaction
    >>> calc_site_fraction_product([[[0.25, 0.25, 0.5]]])
    [0.03125]

    """
    # we use itertools.chain to flatten out the site fractions so [[A, B], [C]]
    # will become [A, B, C], for which we can take the product
    return [np.prod(list(itertools.chain(*[np.atleast_1d(c) for c in config]))) for config in site_fractions]


def calc_interaction_product(site_fractions):
    """Calculate the interaction product for sublattice configurations

    Parameters
    ----------
    site_fractions : list
        List of sublattice configurations. *Sites on each sublattice be in order with respect to
        the elements in the sublattice.* The list should be 3d of (configurations, sublattices, values)

    Returns
    -------
    list
        List of interaction products, Z, for each sublattice

    Examples
    --------
    >>> # interaction product for an (A) configuration
    >>> calc_interaction_product([[1.0]])  # doctest: +ELLIPSIS
    [1.0]
    >>> # interaction product for [(A,B), (A,B)(A)] configurations that are equal
    >>> calc_interaction_product([[[0.5, 0.5]], [[0.5, 0.5], 1]])  # doctest: +ELLIPSIS
    [0.0, 0.0]
    >>> # interaction product for an [(A,B)] configuration
    >>> calc_interaction_product([[[0.1, 0.9]]])  # doctest: +ELLIPSIS
    [-0.8]
    >>> # interaction product for an [(A,B)(A,B)] configuration
    >>> calc_interaction_product([[[0.2, 0.8], [0.4, 0.6]]])  # doctest: +ELLIPSIS
    [0.12]
    >>> # ternary case, (A,B,C) interaction
    >>> calc_interaction_product([[[0.333, 0.333, 0.334]]])
    [[0.333, 0.333, 0.334]]
    >>> # ternary 2SL case, (A,B,C)(A) interaction
    >>> calc_interaction_product([[[0.333, 0.333, 0.334], 1.0]])
    [[0.333, 0.333, 0.334]]

    """
    interaction_product = []
    # config is the list of site fractions for each sublattice, e.g. [[0.25, 0.25, 0.5], 1] for an [[A,B,C], A] configuration
    for config in site_fractions:
        is_ternary = interaction_test(config, 3)
        if not is_ternary:
            prod = 1.0
            for subl in config:
                if isinstance(subl, list) and len(subl) == 2:
                    # must be in sorted order!!
                    prod *= subl[0] - subl[1]
            interaction_product.append(prod)
        else:
            # we need to generate v_i, v_j, or v_k for the ternary case, which v we are calculating depends on the parameter order
            prod = [1, 1, 1]  # product for V_I, V_J, V_K
            for subl in config:
                if isinstance(subl, list) and (len(subl) >= 3):
                    muggianu_correction = (1 - sum(subl)) / len(subl)
                    for i in range(len(subl)):
                        prod[i] *= subl[i] + muggianu_correction
            interaction_product.append([float(p) for p in prod])
    return interaction_product
