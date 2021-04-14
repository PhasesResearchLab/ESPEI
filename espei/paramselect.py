"""
The paramselect module handles automated parameter selection for linear models.

Automated Parameter Selection
End-members

Note: All magnetic parameters from literature for now.
Note: No fitting below 298 K (so neglect third law issues for now).

For each step, add one parameter at a time and compute AICc with max likelihood.

Cp - TlnT, T**2, T**-1, T**3 - 4 candidate models
(S and H only have one required parameter each. Will fit in full MCMC procedure)

Choose parameter set with best AICc score.

"""

import sys
from typing import List, Dict
import logging
import operator
from collections import OrderedDict

import numpy as np
import sympy
from sympy import Symbol
from tinydb import where
from pycalphad import Database, Model, variables as v

import espei.refdata
from espei.database_utils import initialize_database
from espei.core_utils import get_data
from espei.error_functions.non_equilibrium_thermochemical_error import get_prop_samples
from espei.parameter_selection.model_building import build_candidate_models
from espei.parameter_selection.selection import select_model
from espei.parameter_selection.utils import get_data_quantities, feature_transforms, _get_sample_condition_dicts
from espei.sublattice_tools import generate_symmetric_group, generate_interactions, \
    tuplify, interaction_test, endmembers_from_interaction, generate_endmembers
from espei.utils import PickleableTinyDB, sigfigs, extract_aliases

_log = logging.getLogger(__name__)


def _param_present_in_database(dbf, phase_name, configuration, param_type):
    const_arr = tuple([tuple(map(lambda x: v.Species(x), subl)) for subl in map(tuplify, configuration)])
    # parameter order doesn't matter here, since the generated might not exactly match. Always override.
    query = (where('phase_name') == phase_name) & \
            (where('parameter_type') == param_type) & \
            (where('constituent_array') == const_arr)
    search_result = dbf._parameters.search(query)
    if len(search_result) > 0:
        return True


def _build_feature_matrix(sample_condition_dicts: List[Dict[Symbol, float]], symbolic_coefficients: List[Symbol]):
    """
    Builds A for solving x = A\\b. A is an MxN matrix of M sampled data points and N is the symbolic coefficients.

    Parameters
    ----------
    sample_condition_dicts : List[Dict[Symbol, float]]
        List of length ``M`` containing the conditions (T, P, YS, Z, V_I, V_J, V_K) for
        each sampled point.
    symbolic_coefficients : List[Symbol]
        Symbolic coefficients of length ```N`` (e.g. ``v.T``, ``YS``) of the features
        corresponding to the variables that will be fit.

    Returns
    -------
    ArrayLike
        MxN array of coefficients with sampled data conditions plugged in.
    """
    M = len(sample_condition_dicts)
    N = len(symbolic_coefficients)
    feature_matrix = np.empty((M, N))
    for i in range(M):
        for j in range(N):
            feature_matrix[i, j] = symbolic_coefficients[j].subs(sample_condition_dicts[i])
    return feature_matrix


def fit_formation_energy(dbf, comps, phase_name, configuration, symmetry, datasets, ridge_alpha=None, aicc_phase_penalty=None, features=None):
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
    ridge_alpha : float
        Value of the $alpha$ hyperparameter used in ridge regression. Defaults to 1.0e-100, which should be degenerate
        with ordinary least squares regression. For now, the parameter is applied to all features.
    aicc_feature_factors : dict
        Map of phase name to feature to a multiplication factor for the AICc's parameter penalty.
    features : dict
        Maps "property" to a list of features for the linear model.
        These will be transformed from "GM" coefficients
        e.g., {"CPM_FORM": (v.T*sympy.log(v.T), v.T**2, v.T**-1, v.T**3)} (Default value = None)

    Returns
    -------
    dict
        {feature: estimated_value}

    """
    aicc_feature_factors = aicc_phase_penalty if aicc_phase_penalty is not None else {}
    if interaction_test(configuration):
        _log.debug('ENDMEMBERS FROM INTERACTION: %s', endmembers_from_interaction(configuration))
        fitting_steps = (["CPM_FORM", "CPM_MIX"], ["SM_FORM", "SM_MIX"], ["HM_FORM", "HM_MIX"])

    else:
        # We are only fitting an endmember; no mixing data needed
        fitting_steps = (["CPM_FORM"], ["SM_FORM"], ["HM_FORM"])

    # create the candidate models and fitting steps
    if features is None:
        features = OrderedDict([("CPM_FORM", (v.T * sympy.log(v.T), v.T**2, v.T**-1, v.T**3)),
                                ("SM_FORM", (v.T,)),
                                ("HM_FORM", (sympy.S.One,)),
                                ])
    # dict of {feature, [candidate_models]}
    candidate_models_features = build_candidate_models(configuration, features)

    # All possible parameter values that could be taken on. This is some legacy
    # code from before there were many candidate models built. For very large
    # sets of candidate models, this could be quite slow.
    # TODO: we might be able to remove this initialization for clarity, depends on fixed poritions
    parameters = {}
    for candidate_models in candidate_models_features.values():
        for model in candidate_models:
            for coef in model:
                parameters[coef] = 0

    # These is our previously fit partial model from previous steps
    # Subtract out all of these contributions (zero out reference state because these are formation properties)
    fixed_model = Model(dbf, comps, phase_name, parameters={'GHSER'+(c.upper()*2)[:2]: 0 for c in comps})
    fixed_portions = [0]

    for desired_props in fitting_steps:
        feature_type = desired_props[0].split('_')[0]  # HM_FORM -> HM
        aicc_factor = aicc_feature_factors.get(feature_type, 1.0)
        desired_data = get_data(comps, phase_name, configuration, symmetry, datasets, desired_props)
        _log.trace('%s: datasets found: %s', desired_props, len(desired_data))
        if len(desired_data) > 0:
            config_tup = tuple(map(tuplify, configuration))
            calculate_dict = get_prop_samples(desired_data, config_tup)
            sample_condition_dicts = _get_sample_condition_dicts(calculate_dict, list(map(len, config_tup)))
            weights = calculate_dict['weights']
            assert len(sample_condition_dicts) == len(weights)

            # We assume all properties in the same fitting step have the same
            # features (all CPM, all HM, etc., but different ref states).
            # data quantities are the same for each candidate model and can be computed up front
            data_qtys = get_data_quantities(feature_type, fixed_model, fixed_portions, desired_data, sample_condition_dicts)

            # build the candidate model transformation matrix and response vector (A, b in Ax=b)
            feature_matricies = []
            data_quantities = []
            for candidate_coefficients in candidate_models_features[desired_props[0]]:
                # Map coeffiecients in G to coefficients in the feature_type (H, S, CP)
                transformed_coefficients = list(map(feature_transforms[feature_type], candidate_coefficients))
                if interaction_test(configuration, 3):
                    feature_matricies.append(_build_feature_matrix(sample_condition_dicts, transformed_coefficients))
                else:
                    feature_matricies.append(_build_feature_matrix(sample_condition_dicts, transformed_coefficients))
                data_quantities.append(data_qtys)

            # provide candidate models and get back a selected model.
            selected_model = select_model(zip(candidate_models_features[desired_props[0]], feature_matricies, data_quantities), ridge_alpha, weights=weights, aicc_factor=aicc_factor)
            selected_features, selected_values = selected_model
            parameters.update(zip(*(selected_features, selected_values)))
            # Add these parameters to be fixed for the next fitting step
            fixed_portion = np.array(selected_features, dtype=np.object_)
            fixed_portion = np.dot(fixed_portion, selected_values)
            fixed_portions.append(fixed_portion)
    return parameters


def get_next_symbol(dbf):
    """
    Return a string name of the next free symbol to set

    Parameters
    ----------
    dbf : Database
        pycalphad Database. Must have the ``varcounter`` attribute set to an integer.

    Returns
    -------
    str
    """
    # TODO: PEP-572 optimization
    symbol_name = 'VV' + str(dbf.varcounter).zfill(4)
    while dbf.symbols.get(symbol_name, None) is not None:
        dbf.varcounter += 1
        symbol_name = 'VV' + str(dbf.varcounter).zfill(4)
    return symbol_name


def fit_ternary_interactions(dbf, phase_name, symmetry, endmembers, datasets, ridge_alpha=None, aicc_phase_penalty=None):
    """
    Fit ternary interactions for a database in place

    Parameters
    ----------
    dbf : Database
        pycalphad Database to add parameters to
    phase_name : str
        Name of the phase to fit
    symmetry : list
        List of symmetric sublattices, e.g. [[0, 1, 2], [3, 4]]
    endmembers : list
        List of endmember tuples, e.g. [('CU', 'MG')]
    datasets : PickleableTinyDB
        TinyDB database of datasets
    ridge_alpha : float
        Value of the $alpha$ hyperparameter used in ridge regression. Defaults to 1.0e-100, which should be degenerate
        with ordinary least squares regression. For now, the parameter is applied to all features.

    Returns
    -------
    None
        Modified the Database in place
    """
    numdigits = 6  # number of significant figures, might cause rounding errors
    interactions = generate_interactions(endmembers, order=3, symmetry=symmetry)
    _log.trace('%s distinct ternary interactions', len(interactions))
    for interaction in interactions:
        ixx = interaction
        config = tuple(map(tuplify, ixx))
        if _param_present_in_database(dbf, phase_name, config, 'L'):
            _log.warning('INTERACTION: %s already in Database. Skipping.', ixx)
            continue
        else:
            _log.trace('INTERACTION: %s', ixx)
        parameters = fit_formation_energy(dbf, sorted(dbf.elements), phase_name, ixx, symmetry, datasets, ridge_alpha, aicc_phase_penalty=aicc_phase_penalty)
        # Organize parameters by polynomial degree
        degree_polys = np.zeros(3, dtype=np.object_)
        YS = Symbol('YS')
        # asymmetric parameters should have Mugiannu V_I/V_J/V_K, while symmetric just has YS
        is_asymmetric = any([(k.has(Symbol('V_I'))) and (v != 0) for k, v in parameters.items()])
        if is_asymmetric:
            params = [(2, YS*Symbol('V_K')), (1, YS*Symbol('V_J')), (0, YS*Symbol('V_I'))]  # (excess parameter degree, symbol) tuples
        else:
            params = [(0, YS)]  # (excess parameter degree, symbol) tuples
        for degree, check_symbol in params:
            keys_to_remove = []
            for key, value in sorted(parameters.items(), key=str):
                if key.has(check_symbol):
                    if value != 0:
                        symbol_name = get_next_symbol(dbf)
                        dbf.symbols[symbol_name] = sigfigs(parameters[key], numdigits)
                        parameters[key] = Symbol(symbol_name)
                    coef = parameters[key] * (key / check_symbol)
                    try:
                        coef = float(coef)
                    except TypeError:
                        pass
                    degree_polys[degree] += coef
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                parameters.pop(key)
        _log.trace('Polynomial coefs: %s', degree_polys)
        # Insert into database
        symmetric_interactions = generate_symmetric_group(interaction, symmetry)
        for degree in np.arange(degree_polys.shape[0]):
            if degree_polys[degree] != 0:
                for syminter in symmetric_interactions:
                    dbf.add_parameter('L', phase_name, tuple(map(tuplify, syminter)), degree, degree_polys[degree])


def phase_fit(dbf, phase_name, symmetry, datasets, refdata, ridge_alpha, aicc_penalty=None, aliases=None):
    """Generate an initial CALPHAD model for a given phase and sublattice model.

    Parameters
    ----------
    dbf : Database
        pycalphad Database to add parameters to.
    phase_name : str
        Name of the phase.
    symmetry : [[int]]
        Sublattice model symmetry.
    datasets : PickleableTinyDB
        All datasets to consider for the calculation.
    refdata : dict
        Maps tuple(element, phase_name) -> SymPy object defining
        energy relative to SER
    ridge_alpha : float
        Value of the $alpha$ hyperparameter used in ridge regression. Defaults to 1.0e-100, which should be degenerate
        with ordinary least squares regression. For now, the parameter is applied to all features.
    aicc_penalty : dict
        Map of phase name to feature to a multiplication factor for the AICc's parameter penalty.
    aliases : Dict[str, str]
        Mapping of possible aliases to the Database phase names.

    Returns
    -------
    None
        Modifies the dbf.

    """
    aicc_penalty = aicc_penalty if aicc_penalty is not None else {}
    aicc_phase_penalty = aicc_penalty.get(phase_name, {})
    if not hasattr(dbf, 'varcounter'):
        dbf.varcounter = 0
    phase_obj = dbf.phases[phase_name]
    # TODO: assumed pure elements - add proper support for Species objects
    subl_model = [sorted([sp.name for sp in subl]) for subl in phase_obj.constituents]
    site_ratios = phase_obj.sublattices
    # First fit endmembers
    all_em_count = len(generate_endmembers(subl_model))  # number of total endmembers
    endmembers = generate_endmembers(subl_model, symmetry)
    # Number of significant figures in parameters, might cause rounding errors
    numdigits = 6
    em_dict = {}
    # TODO: use the global aliases dictionary passed in as-is instead of converting it to a phase-local dict
    # TODO: use the aliases dictionary in dataset queries to find relevant data
    if aliases is None:
        aliases = [phase_name]
    else:
        aliases = sorted([alias for alias, database_phase in aliases.items() if database_phase == phase_name])
    _log.info('FITTING: %s', phase_name)
    _log.trace('%s endmembers (%s distinct by symmetry)', all_em_count, len(endmembers))

    all_endmembers = []
    for endmember in endmembers:
        symmetric_endmembers = generate_symmetric_group(endmember, symmetry)
        all_endmembers.extend(symmetric_endmembers)
        if _param_present_in_database(dbf, phase_name, endmember, 'G'):
            _log.trace('ENDMEMBER: %s already in Database. Skipping.', endmember)
            continue
        else:
            _log.trace('ENDMEMBER: %s', endmember)
        # Some endmembers are fixed by our choice of standard lattice stabilities, e.g., SGTE91
        # If a (phase, pure component endmember) tuple is fixed, we should use that value instead of fitting
        endmember_comps = list(set(endmember))
        fit_eq = None
        # only one non-VA component, or two components but the other is VA and its only the last sublattice
        if ((len(endmember_comps) == 1) and (endmember_comps[0] != 'VA')) or\
                ((len(endmember_comps) == 2) and (endmember[-1] == 'VA') and (len(set(endmember[:-1])) == 1)):
            # this is a "pure component endmember"
            # try all phase name aliases until we get run out or get a hit
            em_comp = list(set(endmember_comps) - {'VA'})[0]
            sym_name = None
            for name in aliases:
                sym_name = 'G'+name[:3].upper()+em_comp.upper()
                stability = refdata.get((em_comp.upper(), name.upper()), None)
                if stability is not None:
                    if isinstance(stability, sympy.Piecewise):
                        # Default zero required for the compiled backend
                        if (0, True) not in stability.args:
                            new_args = stability.args + ((0, True),)
                            stability = sympy.Piecewise(*new_args)
                    dbf.symbols[sym_name] = stability
                    break
            if dbf.symbols.get(sym_name, None) is not None:
                num_moles = sum([sites for elem, sites in zip(endmember, site_ratios) if elem != 'VA'])
                fit_eq = num_moles * Symbol(sym_name)
        if fit_eq is None:
            # No reference lattice stability data -- we have to fit it
            parameters = fit_formation_energy(dbf, sorted(dbf.elements), phase_name, endmember, symmetry, datasets, ridge_alpha, aicc_phase_penalty=aicc_phase_penalty)
            for key, value in sorted(parameters.items(), key=str):
                if value == 0:
                    continue
                symbol_name = get_next_symbol(dbf)
                dbf.symbols[symbol_name] = sigfigs(value, numdigits)
                parameters[key] = Symbol(symbol_name)
            fit_eq = sympy.Add(*[value * key for key, value in parameters.items()])
            ref = 0
            for subl, ratio in zip(endmember, site_ratios):
                if subl == 'VA':
                    continue
                subl = (subl.upper()*2)[:2]
                ref = ref + ratio * Symbol('GHSER'+subl)
            fit_eq += ref
        _log.trace('SYMMETRIC_ENDMEMBERS: %s', symmetric_endmembers)
        for em in symmetric_endmembers:
            em_dict[em] = fit_eq
            dbf.add_parameter('G', phase_name, tuple(map(tuplify, em)), 0, fit_eq)

    _log.trace('FITTING BINARY INTERACTIONS')
    bin_interactions = generate_interactions(all_endmembers, order=2, symmetry=symmetry)
    _log.trace('%s distinct binary interactions', len(bin_interactions))
    for interaction in bin_interactions:
        ixx = []
        for i in interaction:
            if isinstance(i, (tuple, list)):
                ixx.append(tuple(i))
            else:
                ixx.append(i)
        ixx = tuple(ixx)
        config = tuple(map(tuplify, ixx))
        if _param_present_in_database(dbf, phase_name, config, 'L'):
            _log.trace('INTERACTION: %s already in Database', ixx)
            continue
        else:
            _log.trace('INTERACTION: %s', ixx)
        parameters = fit_formation_energy(dbf, sorted(dbf.elements), phase_name, ixx, symmetry, datasets, ridge_alpha, aicc_phase_penalty=aicc_phase_penalty)
        # Organize parameters by polynomial degree
        degree_polys = np.zeros(10, dtype=np.object_)
        for degree in reversed(range(10)):
            check_symbol = Symbol('YS') * Symbol('Z')**degree
            keys_to_remove = []
            for key, value in sorted(parameters.items(), key=str):
                if key.has(check_symbol):
                    if value != 0:
                        symbol_name = get_next_symbol(dbf)
                        dbf.symbols[symbol_name] = sigfigs(parameters[key], numdigits)
                        parameters[key] = Symbol(symbol_name)
                    coef = parameters[key] * (key / check_symbol)
                    try:
                        coef = float(coef)
                    except TypeError:
                        pass
                    degree_polys[degree] += coef
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                parameters.pop(key)
        _log.trace('Polynomial coefs: %s', degree_polys)
        # Insert into database
        symmetric_interactions = generate_symmetric_group(interaction, symmetry)
        for degree in np.arange(degree_polys.shape[0]):
            if degree_polys[degree] != 0:
                for syminter in symmetric_interactions:
                    dbf.add_parameter('L', phase_name, tuple(map(tuplify, syminter)), degree, degree_polys[degree])

    _log.trace('FITTING TERNARY INTERACTIONS')
    fit_ternary_interactions(dbf, phase_name, symmetry, all_endmembers, datasets, aicc_phase_penalty=aicc_phase_penalty)
    if hasattr(dbf, 'varcounter'):
        del dbf.varcounter


def generate_parameters(phase_models, datasets, ref_state, excess_model, ridge_alpha=None, aicc_penalty_factor=None, dbf=None):
    """Generate parameters from given phase models and datasets

    Parameters
    ----------
    phase_models : dict
        Dictionary of components and phases to fit.
    datasets : PickleableTinyDB
        database of single- and multi-phase to fit.
    ref_state : str
        String of the reference data to use, e.g. 'SGTE91' or 'SR2016'
    excess_model : str
        String of the type of excess model to fit to, e.g. 'linear'
    ridge_alpha : float
        Value of the $alpha$ hyperparameter used in ridge regression. Defaults
        to None, which falls back to ordinary least squares regression.
        For now, the parameter is applied to all features.
    aicc_penalty_factor : dict
        Map of phase name to feature to a multiplication factor for the AICc's parameter penalty.
    dbf : Database
        Initial pycalphad Database that can have parameters that would not be fit by ESPEI

    Returns
    -------
    pycalphad.Database

    """
    # Set NumPy print options so logged arrays print on one line. Reset at the end.
    np.set_printoptions(linewidth=sys.maxsize)
    _log.info('Generating parameters.')
    _log.trace('Found the following user reference states: %s', espei.refdata.INSERTED_USER_REFERENCE_STATES)
    refdata = getattr(espei.refdata, ref_state)
    aliases = extract_aliases(phase_models)
    dbf = initialize_database(phase_models, ref_state, dbf)
    # Fit phases in alphabetic order so the VV#### counter is constistent between runs
    for phase_name, phase_data in sorted(phase_models['phases'].items(), key=operator.itemgetter(0)):
        symmetry = phase_data.get('equivalent_sublattices', None)
        phase_fit(dbf, phase_name, symmetry, datasets, refdata, ridge_alpha, aicc_penalty=aicc_penalty_factor, aliases=aliases)
    _log.info('Finished generating parameters.')
    np.set_printoptions(linewidth=75)
    return dbf
