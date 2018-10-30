"""
Tools for construction Redlich-Kister polynomials used in parameter selection.
"""

import itertools
import numpy as np

def calc_site_fraction_product(site_fractions):
    """Calculate the site fraction product for a sublattice configurations

    Parameters
    ----------
    site_fractions : list
        List of sublattice configurations. Sites on each sublattice be in order with respect to
        the elements in the sublattice. The list should be 3d of (configurations, sublattices, values)

    Returns
    -------
    list
        List of interaction products for each sublattice

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
