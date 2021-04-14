"""
Tools for construction Redlich-Kister polynomials used in parameter selection.
"""

import itertools
import numpy as np
from espei.sublattice_tools import interaction_test


def calc_interaction_product(site_fractions):
    """Calculate the interaction product for sublattice site fractions

    **Callers should take care that the site fractions correspond to constituents in
    sorted order, since there's an order-dependent subtraction.**

    Parameters
    ----------
    site_fractions : List[List[float]]
        List of site fractions for each sublattice. The list should a ragged 2d list of
        shape (sublattices, site fractions).

    Returns
    -------
    Union[float, List[float]]
        A scalar for binary interactions and a list of 3 floats for ternary interactions

    Examples
    --------
    >>> # interaction product for an (A) site_fractions
    >>> calc_interaction_product([[1.0]])  # doctest: +ELLIPSIS
    1.0
    >>> # interaction product for [(A,B), (A,B)(A)] site fractions that are equal
    >>> calc_interaction_product([[0.5, 0.5]])  # doctest: +ELLIPSIS
    0.0
    >>> calc_interaction_product([[0.5, 0.5], 1])  # doctest: +ELLIPSIS
    0.0
    >>> # interaction product for an [(A,B)] site_fractions
    >>> calc_interaction_product([[0.1, 0.9]])  # doctest: +ELLIPSIS
    -0.8
    >>> # interaction product for an [(A,B)(A,B)] site_fractions
    >>> calc_interaction_product([[0.2, 0.8], [0.4, 0.6]])  # doctest: +ELLIPSIS
    0.12
    >>> # ternary case, (A,B,C) interaction
    >>> calc_interaction_product([[0.333, 0.333, 0.334]])
    [0.333, 0.333, 0.334]
    >>> # ternary 2SL case, (A,B,C)(A) interaction
    >>> calc_interaction_product([[0.333, 0.333, 0.334], 1.0])
    [0.333, 0.333, 0.334]

    """
    # config is the list of site fractions for each sublattice, e.g. [[0.25, 0.25, 0.5], 1] for an [[A,B,C], A] site_fractions
    is_ternary = interaction_test(site_fractions, 3)
    if not is_ternary:
        prod = 1.0
        for subl in site_fractions:
            if isinstance(subl, list) and len(subl) == 2:
                # must be in sorted order!!
                prod *= subl[0] - subl[1]
        return prod
    else:
        # we need to generate v_i, v_j, or v_k for the ternary case, which v we are calculating depends on the parameter order
        prod = [1, 1, 1]  # product for V_I, V_J, V_K
        for subl in site_fractions:
            if isinstance(subl, list) and (len(subl) >= 3):
                muggianu_correction = (1 - sum(subl)) / len(subl)
                for i in range(len(subl)):
                    prod[i] *= subl[i] + muggianu_correction
        return [float(p) for p in prod]
