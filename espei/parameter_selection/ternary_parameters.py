"""
Build fittable models for ternary parameter selection
"""

import itertools
import logging
import operator
from collections import OrderedDict
from functools import reduce

import numpy as np
import sympy
from pycalphad import Model, variables as v

from espei.parameter_selection.utils import feature_transforms, endmembers_from_interaction, interaction_test
from espei.core_utils import get_data, get_samples

def get_muggianu_samples(desired_data, interaction_index):
    """
    Return the data values from desired_data, transformed to interaction products.
    Specifically works for Muggianu extrapolation.

    Parameters
    ----------
    desired_data : list
        List of matched desired data, e.g. for a single property
    interaction_index : int
        Which ternary interaction index to the parameter corresponds to,
        e.g. a 1 corresponds to an L1 parameter, which gives a Muggianu correction for
        v_B = y_B + (1 - y_A - y_B - y_C). If None is passed, then samples for a
        symmetric parameter will be returned.

    Returns
    -------
    list
        List of sample values that are properly transformed.

    Notes
    -----
    Transforms data to interaction products, e.g. YS*{}^{xs}G=YS*XS*DXS^{n} {}^{n}L
    Each tuple in the list is a tuple of (temperature, (site_fraction_product, interaction_product)) for each data sample

    """
    # TODO does not ravel pressure conditions
    # TODO: could possibly combine with ravel_conditions if we do the math outside.
    all_samples = []
    for data in desired_data:
        temperatures = np.atleast_1d(data['conditions']['T'])
        num_configs = np.array(data['solver'].get('sublattice_configurations'), dtype=np.object).shape[0]
        site_fractions = data['solver'].get('sublattice_occupancies', [[1]] * num_configs)
        print(site_fractions)
        # product of site fractions for each dataset
        site_fraction_product = [reduce(operator.mul, list(itertools.chain(*[np.atleast_1d(f) for f in fracs])), 1)
                                 for fracs in site_fractions]
        print(site_fraction_product)
        # TODO: Subtle sorting bug here, if the interactions aren't already in sorted order...
        interaction_product = []
        for fracs in site_fractions:
            # fracs is the list of site fractions for each sublattice, e.g. [[0.25, 0.25, 0.5], 1] for an [[A,B,C], A] configuration
            prod = 1
            # None interaction_index means we are only concerned with the symmetric case. The interaction product is 1.
            if interaction_index is not None:
                # we need to generate v_i, v_j, or v_k for the ternary case, which v we are calculating depends on the parameter order
                for f in fracs:
                    if isinstance(f, list) and (len(f) >= 3):
                        prod *= f[interaction_index] + (1 - sum(f))/len(f)
            interaction_product.append(float(prod))
        if len(interaction_product) == 0:
            interaction_product = [0]
        comp_features = zip(site_fraction_product, interaction_product)
        all_samples.extend(list(itertools.product(temperatures, comp_features)))
    return all_samples


def build_ternary_feature_matrix(prop, features, desired_data, parameter_order=None):
    """
    Return an MxN matrix of M data sample and N features.

    Parameters
    ----------
    prop : str
        String name of the property, e.g. 'HM_MIX'
    features : tuple
        Tuple of SymPy parameters that can be fit for this property.
    desired_data : dict
        Full dataset dictionary containing values, conditions, etc.
    parameter_order : int
        Order of the index to build a feature matrix for. Corresponds to the Muggianu
        parameter index, e.g. 0 is an L0 parameter.

    Returns
    -------
    numpy.ndarray
        An MxN matrix of M samples (from desired data) and N features.

    """
    print('prop')
    print(prop)
    print('features')
    print(features)
    print("desired_data")
    print(desired_data)
    transformed_features = sympy.Matrix([feature_transforms[prop](i) for i in features])
    all_samples = get_muggianu_samples(desired_data, interaction_index=parameter_order)
    print('all_samples')
    print(all_samples)
    feature_matrix = np.empty((len(all_samples), len(transformed_features)), dtype=np.float)
    feature_matrix[:, :] = [transformed_features.subs({v.T: temp, 'YS': compf[0], 'Z': compf[1]}).evalf() for temp, compf in all_samples]
    return feature_matrix


def generate_feature_sets(features):
    """
    Return a set of distinct feature combinations

    Parameters
    ----------
    features : list
        List of features, e.g. [X, Y, Z]

    Returns
    -------
    list
        List of feature combinations

    Examples
    --------
    >>> generate_feature_sets(['W', 'X', 'Y', 'Z'])
    [['W'], ['W', 'X'], ['W', 'X', 'Y'], ['W', 'X', 'Y', 'Z']]
    """
    return [features[:(i+1)] for i in range(len(features))]

def build_candidate_models(configuration, features):
    if not interaction_test(configuration):  # endmembers only
        raise NotImplementedError('Only ternary interactions supported')

    elif interaction_test(configuration, 2):  # has a binary interaction
        YS = sympy.Symbol('YS')  # Product of all nonzero site fractions in all sublattices
        Z = sympy.Symbol('Z')
        # generate increasingly complex interactions
        parameter_interactions = [
            (YS,),  # L0
            (YS, YS*Z),  # L0 and L1
            (YS, YS*Z, YS*(Z**2)),  # L0, L1, and L2
            (YS, YS*Z, YS*(Z**2), YS*(Z**3)),  # L0, L1, L2, and L3
        ]
        raise NotImplementedError('Binary candidates not fully implemented. Doesn\'t give enough granularity in candidates ')

    elif interaction_test(configuration, 3):  # has a ternary interaction
        # Ternaries interactions should have exactly two candidate models:
        # 1. a single symmetric ternary parameter (YS)
        # 2. L0, L1, and L2 parameters corresponding to Muggianu parameters
        # We are ignoring cases where we have L0 == L1, but L0 != L2 and all of the
        # combinations these cases don't exist in reality (where two elements have
        # exactly the same behavior) the symmetric case is mainly for small
        # corrections and dimensionality reduction.
        YS = sympy.Symbol('YS')  # Product of all nonzero site fractions in all sublattices
        # Muggianu ternary interaction product for components i, j, and k
        V_i, V_j, V_k = sympy.Symbol('V_i'), sympy.Symbol('V_j'), sympy.Symbol('V_k')
        parameter_interactions = [
            (YS,),  # symmetric L0
            (YS*V_i, YS*V_j, YS*V_k)  # asymmetric L0, L1, and L2
        ]

    # there was an interaction, so now we need to generate all the candidate models
    for feature in features.keys():
        feature_sets = generate_feature_sets(features[feature])
        # generate tuples of (parameter_interactions, feature_values), e.g. ('YS', (T*log(T), T**2))
        candidate_tuples = list(itertools.product(parameter_interactions, feature_sets))
        print('candidate tuples')
        print(candidate_tuples)
        candidates = []
        for interactions, feature_values in candidate_tuples:
            # distribute the parameter interactions through the features
            candidates.append([inter*feat for inter, feat in itertools.product(interactions, feature_values)])
            print(candidates[-1])
        # candidates = [[inter*feature_value for inter in interactions] for interactions, feature_value in candidate_tuples]
        # update the features dictionary with all the possible candidates for this feature
        features[feature] = candidates

    return features

def fit_ternary_formation_energy(dbf, comps, phase_name, configuration, symmetry, datasets, features=None):
    """
    Find suitable linear model parameters for the given phase.
    We do this by successively fitting heat capacities, entropies and
    enthalpies of formation, and selecting against criteria to prevent
    overfitting. The "best" set of parameters minimizes the error
    without overfitting.

    Parameters
    ----------
    dbf : Database
        pycalphad Database. Partially complete, so we know what degrees of freedom to fix.
    comps : [str]
        Names of the relevant components.
    phase_name : str
        Name of the desired phase for which the parameters will be found.
    configuration : ndarray
        Configuration of the sublattices for the fitting procedure.
    symmetry : [[int]]
        Symmetry of the sublattice configuration.
    datasets : PickleableTinyDB
        All the datasets desired to fit to.
    features : dict
        Maps "property" to a list of features for the linear model.
        These will be transformed from "GM" coefficients
        e.g., {"CPM_FORM": (v.T*sympy.log(v.T), v.T**2, v.T**-1, v.T**3)} (Default value = None)

    Returns
    -------
    dict
        {feature: estimated_value}

    """
    # create the candidate models and fitting steps
    if features is None:
        features = [("CPM_FORM", (v.T * sympy.log(v.T), v.T**2, v.T**-1, v.T**3)),
                    ("SM_FORM", (v.T,)),
                    ("HM_FORM", (sympy.S.One,))
                    ]
        features = OrderedDict(features)
    fitting_steps = (["CPM_FORM", "CPM_MIX"], ["SM_FORM", "SM_MIX"], ["HM_FORM", "HM_MIX"])
    # Product of all nonzero site fractions in all sublattices
    YS = sympy.Symbol('YS')
    # Product of all binary interaction terms
    Z = sympy.Symbol('Z')
    redlich_kister_features = (YS,)  # gives the symmetric if build_feature_matrix
    for feature in features.keys():
        all_features = list(itertools.product(redlich_kister_features, features[feature]))
        features[feature] = [i[0]*i[1] for i in all_features]
    logging.debug('ENDMEMBERS FROM INTERACTION: {}'.format(endmembers_from_interaction(configuration)))
    print('features')
    print(features)

    parameters = {}
    for feature in features.values():
        for coef in feature:
            parameters[coef] = 0

    # These is our previously fit partial model
    # Subtract out all of these contributions (zero out reference state because these are formation properties)
    fixed_model = Model(dbf, comps, phase_name, parameters={'GHSER'+(c.upper()*2)[:2]: 0 for c in comps})
    fixed_model.models['idmix'] = 0
    fixed_portions = [0]

    moles_per_formula_unit = sympy.S(0)
    subl_idx = 0
    for num_sites, const in zip(dbf.phases[phase_name].sublattices, dbf.phases[phase_name].constituents):
        if v.Species('VA') in const:
            moles_per_formula_unit += num_sites * (1 - v.SiteFraction(phase_name, subl_idx, v.Species('VA')))
        else:
            moles_per_formula_unit += num_sites
        subl_idx += 1

    for desired_props in fitting_steps:
        desired_data = get_data(comps, phase_name, configuration, symmetry, datasets, desired_props)
        logging.debug('{}: datasets found: {}'.format(desired_props, len(desired_data)))
        if len(desired_data) > 0:
            # We assume all properties in the same fitting step have the same features (but different ref states)
            feature_matrix = build_ternary_feature_matrix(desired_props[0], features[desired_props[0]], desired_data)
            all_samples = get_samples(desired_data)
            data_quantities = np.concatenate(_shift_reference_state(desired_data,
                                                                    feature_transforms[desired_props[0]],
                                                                    fixed_model), axis=-1)
            site_fractions = [build_sitefractions(phase_name, ds['solver']['sublattice_configurations'],
                                                  ds['solver'].get('sublattice_occupancies',
                                 np.ones((len(ds['solver']['sublattice_configurations']),
                                          len(ds['solver']['sublattice_configurations'][0])), dtype=np.float)))
                              for ds in desired_data for _ in ds['conditions']['T']]
            # Flatten list
            site_fractions = list(itertools.chain(*site_fractions))
            # Remove existing partial model contributions from the data
            data_quantities = data_quantities - feature_transforms[desired_props[0]](fixed_model.ast)
            # Subtract out high-order (in T) parameters we've already fit
            data_quantities = data_quantities - \
                feature_transforms[desired_props[0]](sum(fixed_portions)) / moles_per_formula_unit
            for sf, i in zip(site_fractions, data_quantities):
                missing_variables = sympy.S(i * moles_per_formula_unit).atoms(v.SiteFraction) - set(sf.keys())
                sf.update({x: 0. for x in missing_variables})
            # moles_per_formula_unit factor is here because our data is stored per-atom
            # but all of our fits are per-formula-unit
            data_quantities = [sympy.S(i * moles_per_formula_unit).xreplace(sf).xreplace({v.T: ixx[0]}).evalf()
                               for i, sf, ixx in zip(data_quantities, site_fractions, all_samples)]
            data_quantities = np.asarray(data_quantities, dtype=np.float)
            # provide candidate models and get back a selected model. candidate models come from
            parameters.update(_fit_parameters(feature_matrix, data_quantities, features[desired_props[0]]))
            # Add these parameters to be fixed for the next fitting step
            fixed_portion = np.array(features[desired_props[0]], dtype=np.object)
            fixed_portion = np.dot(fixed_portion, [parameters[feature] for feature in features[desired_props[0]]])
            fixed_portions.append(fixed_portion)
    return parameters
