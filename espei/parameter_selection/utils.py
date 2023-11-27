"""
Tools used across parameter selection modules
"""

# TODO: small refactor so this file can be deleted :) - down with utils files!

from typing import Any, List, Dict, Tuple, Union
import itertools
import numpy as np
import symengine
from symengine import Symbol
from pycalphad import variables as v
from espei.utils import build_sitefractions
from espei.parameter_selection.redlich_kister import calc_interaction_product

feature_transforms = {"CPM_FORM": lambda GM: -v.T*symengine.diff(GM, v.T, 2),
                      "CPM_MIX": lambda GM: -v.T*symengine.diff(GM, v.T, 2),
                      "CPM": lambda GM: -v.T*symengine.diff(GM, v.T, 2),
                      "SM_FORM": lambda GM: -symengine.diff(GM, v.T),
                      "SM_MIX": lambda GM: -symengine.diff(GM, v.T),
                      "SM": lambda GM: -symengine.diff(GM, v.T),
                      "HM_FORM": lambda GM: GM - v.T*symengine.diff(GM, v.T),
                      "HM_MIX": lambda GM: GM - v.T*symengine.diff(GM, v.T),
                      "HM": lambda GM: GM - v.T*symengine.diff(GM, v.T)}


def _get_sample_condition_dicts(calculate_dict: Dict[Any, Any], configuration_tuple: Tuple[Union[str, Tuple[str]]], phase_name: str) -> List[Dict[Symbol, float]]:
    sublattice_dof = list(map(len, configuration_tuple))
    sample_condition_dicts = []
    for sample_idx in range(calculate_dict["values"].size):
        cond_dict = {}
        points = calculate_dict["points"][sample_idx, :]

        # T and P
        cond_dict[v.T] = calculate_dict["T"][sample_idx]
        cond_dict[v.P] = calculate_dict["P"][sample_idx]

        # YS site fraction product
        site_fraction_product = np.prod(points)
        cond_dict[Symbol("YS")] = site_fraction_product

        # Reconstruct site fractions in sublattice form from points
        # Required so we can identify which sublattices have interactions
        points_idxs = [0] + np.cumsum(sublattice_dof).tolist()
        site_fractions = []
        for subl_idx in range(len(points_idxs)-1):
            subl_site_fractions = points[points_idxs[subl_idx]:points_idxs[subl_idx+1]]
            for species_name, site_frac in zip(configuration_tuple[subl_idx], subl_site_fractions):
                cond_dict[v.Y(phase_name, subl_idx, species_name)] = site_frac
            site_fractions.append(subl_site_fractions.tolist())

        # Z (binary) or V_I, V_J, V_K (ternary) interaction products
        interaction_product = calc_interaction_product(site_fractions)
        if hasattr(interaction_product, "__len__"):
            # Ternary interaction
            assert len(interaction_product) == 3
            cond_dict[Symbol("V_I")] = interaction_product[0]
            cond_dict[Symbol("V_J")] = interaction_product[1]
            cond_dict[Symbol("V_K")] = interaction_product[2]
        else:
            cond_dict[Symbol("Z")] = interaction_product

        sample_condition_dicts.append(cond_dict)
    return sample_condition_dicts
