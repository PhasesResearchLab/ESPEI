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
import itertools, operator, logging

import numpy as np
import sympy
from collections import OrderedDict
from pycalphad import Database, Model, variables as v
from sklearn.linear_model import LinearRegression

from espei.core_utils import get_data, get_samples, canonicalize
from espei.parameter_selection.utils import (
    feature_transforms, shift_reference_state, interaction_test
)
from espei.parameter_selection.selection import select_model
from espei.parameter_selection.model_building import build_candidate_models, generate_interactions, generate_symmetric_group
from espei.parameter_selection.ternary_parameters import build_ternary_feature_matrix
from espei.utils import PickleableTinyDB, sigfigs, endmembers_from_interaction, \
    build_sitefractions
import espei.refdata

# backwards compatibility:
from pycalphad.io.database import Species

def _to_tuple(x):
    if isinstance(x, list) or isinstance(x, tuple):
        return tuple(x)
    else:
        return tuple([x])


def _build_feature_matrix(prop, features, desired_data):
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

    Returns
    -------
    numpy.ndarray
        An MxN matrix of M samples (from desired data) and N features.

    """
    transformed_features = sympy.Matrix([feature_transforms[prop](i) for i in features])
    all_samples = get_samples(desired_data)
    feature_matrix = np.empty((len(all_samples), len(transformed_features)), dtype=np.float)
    feature_matrix[:, :] = [transformed_features.subs({v.T: temp, 'YS': compf[0], 'Z': compf[1]}).evalf() for temp, compf in all_samples]
    return feature_matrix


def fit_formation_energy(dbf, comps, phase_name, configuration, symmetry, datasets, ridge_alpha=1.0e-100, features=None):
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
    features : dict
        Maps "property" to a list of features for the linear model.
        These will be transformed from "GM" coefficients
        e.g., {"CPM_FORM": (v.T*sympy.log(v.T), v.T**2, v.T**-1, v.T**3)} (Default value = None)

    Returns
    -------
    dict
        {feature: estimated_value}

    """
    if interaction_test(configuration):
        logging.debug('ENDMEMBERS FROM INTERACTION: {}'.format(endmembers_from_interaction(configuration)))
        # fitting heat capacity in excess parameters is not usually encouraged in the CALPHAD community. We don't fit it here.
        fitting_steps = (["SM_FORM", "SM_MIX"], ["HM_FORM", "HM_MIX"])

    else:
        # We are only fitting an endmember; no mixing data needed
        fitting_steps = (["CPM_FORM"], ["SM_FORM"], ["HM_FORM"])

    # create the candidate models and fitting steps
    if features is None:
        features = OrderedDict([("CPM_FORM", (v.T * sympy.log(v.T), v.T**2, v.T**-1, v.T**3)),
                    ("SM_FORM", (v.T,)),
                    ("HM_FORM", (sympy.S.One,))
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
            # We assume all properties in the same fitting step have the same features (all CPM, all HM, etc.) (but different ref states)
            all_samples = get_samples(desired_data)
            site_fractions = [build_sitefractions(phase_name, ds['solver']['sublattice_configurations'], ds['solver'].get('sublattice_occupancies',
                                 np.ones((len(ds['solver']['sublattice_configurations']), len(ds['solver']['sublattice_configurations'][0])), dtype=np.float)))
                              for ds in desired_data for _ in ds['conditions']['T']]
            # Flatten list
            site_fractions = list(itertools.chain(*site_fractions))

            # build the candidate model transformation matrix and response vector (A, b in Ax=b)
            feature_matricies = []
            data_quantities = []
            for candidate_model in candidate_models_features[desired_props[0]]:
                if interaction_test(configuration, 3):
                    feature_matricies.append(build_ternary_feature_matrix(desired_props[0], candidate_model, desired_data))
                else:
                    feature_matricies.append(_build_feature_matrix(desired_props[0], candidate_model, desired_data))

                data_qtys = np.concatenate(shift_reference_state(desired_data, feature_transforms[desired_props[0]], fixed_model), axis=-1)

                # Remove existing partial model contributions from the data
                data_qtys = data_qtys - feature_transforms[desired_props[0]](fixed_model.ast)
                # Subtract out high-order (in T) parameters we've already fit
                data_qtys = data_qtys - feature_transforms[desired_props[0]](sum(fixed_portions)) / moles_per_formula_unit

                for sf, i in zip(site_fractions, data_qtys):
                    missing_variables = sympy.S(i * moles_per_formula_unit).atoms(v.SiteFraction) - set(sf.keys())
                    sf.update({x: 0. for x in missing_variables})

                # moles_per_formula_unit factor is here because our data is stored per-atom
                # but all of our fits are per-formula-unit
                data_qtys = [sympy.S(i * moles_per_formula_unit).xreplace(sf).xreplace({v.T: ixx[0]}).evalf()
                                   for i, sf, ixx in zip(data_qtys, site_fractions, all_samples)]
                data_qtys = np.asarray(data_qtys, dtype=np.float)
                data_quantities.append(data_qtys)

            # provide candidate models and get back a selected model.
            selected_model = select_model(zip(candidate_models_features[desired_props[0]], feature_matricies, data_quantities), ridge_alpha)
            selected_features, selected_values = selected_model
            parameters.update(zip(*(selected_features, selected_values)))
            # Add these parameters to be fixed for the next fitting step
            fixed_portion = np.array(selected_features, dtype=np.object)
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


def fit_ternary_interactions(dbf, phase_name, symmetry, endmembers, datasets, ridge_alpha=1.0e-1000000):
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
    logging.debug('{0} distinct ternary interactions'.format(len(interactions)))
    for interaction in interactions:
        ixx = interaction
        logging.debug('INTERACTION: {}'.format(ixx))
        parameters = fit_formation_energy(dbf, sorted(dbf.elements), phase_name, ixx, symmetry, datasets, ridge_alpha)
        # Organize parameters by polynomial degree
        degree_polys = np.zeros(3, dtype=np.object)
        YS = sympy.Symbol('YS')
        # asymmetric parameters should have Mugiannu V_I/V_J/V_K, while symmetric just has YS
        is_asymmetric = any([(k.has(sympy.Symbol('V_I'))) and (v != 0) for k, v in parameters.items()])
        if is_asymmetric:
            params = [(2, YS*sympy.Symbol('V_K')), (1, YS*sympy.Symbol('V_J')), (0, YS*sympy.Symbol('V_I'))]  # (excess parameter degree, symbol) tuples
        else:
            params = [(0, YS)]  # (excess parameter degree, symbol) tuples
        for degree, check_symbol in params:
            keys_to_remove = []
            for key, value in sorted(parameters.items(), key=str):
                if key.has(check_symbol):
                    if value != 0:
                        symbol_name = get_next_symbol(dbf)
                        dbf.symbols[symbol_name] = sigfigs(parameters[key], numdigits)
                        parameters[key] = sympy.Symbol(symbol_name)
                    coef = parameters[key] * (key / check_symbol)
                    try:
                        coef = float(coef)
                    except TypeError:
                        pass
                    degree_polys[degree] += coef
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                parameters.pop(key)
        logging.debug('Polynomial coefs: {}'.format(degree_polys))
        # Insert into database
        symmetric_interactions = generate_symmetric_group(interaction, symmetry)
        for degree in np.arange(degree_polys.shape[0]):
            if degree_polys[degree] != 0:
                for syminter in symmetric_interactions:
                    dbf.add_parameter('L', phase_name, tuple(map(_to_tuple, syminter)), degree, degree_polys[degree])


def phase_fit(dbf, phase_name, symmetry, subl_model, site_ratios, datasets, refdata, ridge_alpha, aliases=None):
    """Generate an initial CALPHAD model for a given phase and sublattice model.

    Parameters
    ----------
    dbf : Database
        pycalphad Database to add parameters to.
    phase_name : str
        Name of the phase.
    symmetry : [[int]]
        Sublattice model symmetry.
    subl_model : [[str]]
        Sublattice model for the phase of interest.
    site_ratios : [float]
        Number of sites in each sublattice, normalized to one atom.
    datasets : PickleableTinyDB
        All datasets to consider for the calculation.
    refdata : dict
        Maps tuple(element, phase_name) -> SymPy object defining
        energy relative to SER
    ridge_alpha : float
        Value of the $alpha$ hyperparameter used in ridge regression. Defaults to 1.0e-100, which should be degenerate
        with ordinary least squares regression. For now, the parameter is applied to all features.
    aliases : [str]
        Alternative phase names. Useful for matching against
        reference data or other datasets. (Default value = None)

    Returns
    -------
    None
        Modifies the dbf.

    """
    if not hasattr(dbf, 'varcounter'):
        dbf.varcounter = 0
    # First fit endmembers
    all_em_count = len(list(itertools.product(*subl_model)))
    endmembers = sorted(set(canonicalize(i, symmetry) for i in itertools.product(*subl_model)))
    # Number of significant figures in parameters, might cause rounding errors
    numdigits = 6
    em_dict = {}
    aliases = [] if aliases is None else aliases
    aliases = sorted(set(aliases + [phase_name]))
    logging.info('FITTING: {}'.format(phase_name))
    logging.debug('{0} endmembers ({1} distinct by symmetry)'.format(all_em_count, len(endmembers)))

    all_endmembers = []
    for endmember in endmembers:
        logging.debug('ENDMEMBER: {}'.format(endmember))
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
                fit_eq = num_moles * sympy.Symbol(sym_name)
        if fit_eq is None:
            # No reference lattice stability data -- we have to fit it
            parameters = fit_formation_energy(dbf, sorted(dbf.elements), phase_name, endmember, symmetry, datasets, ridge_alpha)
            for key, value in sorted(parameters.items(), key=str):
                if value == 0:
                    continue
                symbol_name = get_next_symbol(dbf)
                dbf.symbols[symbol_name] = sigfigs(value, numdigits)
                parameters[key] = sympy.Symbol(symbol_name)
            fit_eq = sympy.Add(*[value * key for key, value in parameters.items()])
            ref = 0
            for subl, ratio in zip(endmember, site_ratios):
                if subl == 'VA':
                    continue
                subl = (subl.upper()*2)[:2]
                ref = ref + ratio * sympy.Symbol('GHSER'+subl)
            fit_eq += ref
        symmetric_endmembers = generate_symmetric_group(endmember, symmetry)
        logging.debug('SYMMETRIC_ENDMEMBERS: {}'.format(symmetric_endmembers))
        all_endmembers.extend(symmetric_endmembers)
        for em in symmetric_endmembers:
            em_dict[em] = fit_eq
            dbf.add_parameter('G', phase_name, tuple(map(_to_tuple, em)), 0, fit_eq)

    logging.debug('FITTING BINARY INTERACTIONS')
    bin_interactions = generate_interactions(all_endmembers, order=2, symmetry=symmetry)
    logging.debug('{0} distinct binary interactions'.format(len(bin_interactions)))
    for interaction in bin_interactions:
        ixx = []
        for i in interaction:
            if isinstance(i, (tuple, list)):
                ixx.append(tuple(i))
            else:
                ixx.append(i)
        ixx = tuple(ixx)
        logging.debug('INTERACTION: {}'.format(ixx))
        parameters = fit_formation_energy(dbf, sorted(dbf.elements), phase_name, ixx, symmetry, datasets, ridge_alpha)
        # Organize parameters by polynomial degree
        degree_polys = np.zeros(10, dtype=np.object)
        for degree in reversed(range(10)):
            check_symbol = sympy.Symbol('YS') * sympy.Symbol('Z')**degree
            keys_to_remove = []
            for key, value in sorted(parameters.items(), key=str):
                if key.has(check_symbol):
                    if value != 0:
                        symbol_name = get_next_symbol(dbf)
                        dbf.symbols[symbol_name] = sigfigs(parameters[key], numdigits)
                        parameters[key] = sympy.Symbol(symbol_name)
                    coef = parameters[key] * (key / check_symbol)
                    try:
                        coef = float(coef)
                    except TypeError:
                        pass
                    degree_polys[degree] += coef
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                parameters.pop(key)
        logging.debug('Polynomial coefs: {}'.format(degree_polys))
        # Insert into database
        symmetric_interactions = generate_symmetric_group(interaction, symmetry)
        for degree in np.arange(degree_polys.shape[0]):
            if degree_polys[degree] != 0:
                for syminter in symmetric_interactions:
                    dbf.add_parameter('L', phase_name, tuple(map(_to_tuple, syminter)), degree, degree_polys[degree])

    logging.debug('FITTING TERNARY INTERACTIONS')
    fit_ternary_interactions(dbf, phase_name, symmetry, all_endmembers, datasets)
    if hasattr(dbf, 'varcounter'):
        del dbf.varcounter


def generate_parameters(phase_models, datasets, ref_state, excess_model, ridge_alpha=1.0e-100):
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
        Value of the $alpha$ hyperparameter used in ridge regression. Defaults to 1.0e-100, which should be degenerate
        with ordinary least squares regression. For now, the parameter is applied to all features.

    Returns
    -------
    pycalphad.Database

    """
    logging.info('Generating parameters.')
    dbf = Database()
    dbf.elements = set(phase_models['components'])
    for el in dbf.elements:
        dbf.species.add(Species(el, {el: 1}, 0))
    # Write reference state to Database
    refdata = getattr(espei.refdata, ref_state)
    stabledata = getattr(espei.refdata, ref_state + 'Stable')
    for key, element in refdata.items():
        if isinstance(element, sympy.Piecewise):
            newargs = element.args + ((0, True),)
            refdata[key] = sympy.Piecewise(*newargs)
    for key, element in stabledata.items():
        if isinstance(element, sympy.Piecewise):
            newargs = element.args + ((0, True),)
            stabledata[key] = sympy.Piecewise(*newargs)
    comp_refs = {c.upper(): stabledata[c.upper()] for c in dbf.elements if c.upper() != 'VA'}
    comp_refs['VA'] = 0
    # note that the `c.upper()*2)[:2]` returns 'AL' for c.upper()=='AL' and 'VV' for c.upper()=='V'
    dbf.symbols.update({'GHSER' + (c.upper()*2)[:2]: data for c, data in comp_refs.items()})
    for phase_name, phase_obj in sorted(phase_models['phases'].items(), key=operator.itemgetter(0)):
        # Perform parameter selection and single-phase fitting based on input
        # TODO: Need to pass particular models to include: magnetic, order-disorder, etc.
        symmetry = phase_obj.get('equivalent_sublattices', None)
        aliases = phase_obj.get('aliases', None)
        # TODO: More advanced phase data searching
        site_ratios = phase_obj['sublattice_site_ratios']
        subl_model = phase_obj['sublattice_model']
        dbf.add_phase(phase_name, dict(), site_ratios)
        dbf.add_phase_constituents(phase_name, subl_model)
        dbf.add_structure_entry(phase_name, phase_name)
        phase_fit(dbf, phase_name, symmetry, subl_model, site_ratios, datasets, refdata, ridge_alpha, aliases=aliases)
    logging.info('Finished generating parameters.')
    return dbf
