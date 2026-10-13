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
import symengine
from symengine import Symbol
from tinydb import where
import tinydb
from pycalphad import Database, variables as v

import espei.refdata
from espei.database_utils import initialize_database
from espei.parameter_selection.model_building import build_feature_matrix as _build_feature_matrix  # backwards compatibility
from espei.sublattice_tools import generate_symmetric_group, generate_interactions, \
    tuplify, interaction_test, endmembers_from_interaction, generate_endmembers
from espei.utils import PickleableTinyDB, sigfigs, extract_aliases
from espei.parameter_selection.fitting_descriptions import gibbs_energy_fitting_description

_log = logging.getLogger(__name__)


def _param_present_in_database(dbf, phase_name, configuration, param_type):
    species = {sp.name: sp for sp in dbf.species}
    const_arr = tuple([tuple(species[name] for name in subl) for subl in map(tuplify, configuration)])
    # parameter order doesn't matter here, since the generated might not exactly match. Always override.
    query = (where('phase_name') == phase_name) & \
            (where('parameter_type') == param_type) & \
            (where('constituent_array') == const_arr)
    search_result = dbf._parameters.search(query)
    if len(search_result) > 0:
        return True


def _stable_sort_key(x):
    return str(sorted(x[0].args, key=str))


def _poly_degrees(expr):
    poly_dict = {}
    for at in expr.atoms(symengine.Symbol):
        poly_dict[at] = 1
    for at in expr.atoms(symengine.log):
        poly_dict[at] = 1
    for at in expr.atoms(symengine.Pow):
        poly_dict[at.args[0]] = at.args[1]
    return poly_dict


def has_symbol(expr, check_symbol):
    """
    Workaround for SymEngine not supporting Basic.has() with non-Symbol arguments.
    Only works for detecting subsets of multiplication of variables.
    """
    if isinstance(check_symbol, Symbol):
        return expr.has(check_symbol)
    expr_poly = _poly_degrees(expr)
    check_poly = _poly_degrees(check_symbol)
    for cs, check_degree in check_poly.items():
        eps = expr_poly.get(cs, 0)
        if check_degree > eps:
            return False
    return True


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


def _endmember_reference(dbf, phase_name, configuration):
    """Return the GHSER reference energy of an endmember per mole of formula."""
    # TODO: generalize for models with composition dependent site ratios, e.g. ionic liquids
    species = {sp.name: sp for sp in dbf.species}
    reference = 0
    for name, ratio in zip(configuration, dbf.phases[phase_name].sublattices):
        for el, amount in species[name].constituents.items():
            if el not in ("VA", "/-"):
                # e.g. GHSERAL for AL and GHSERBB for B
                reference += ratio * amount * Symbol("GHSER" + (el.upper()*2)[:2])
    return reference


def get_parameter_reference(dbf, phase_name, configuration, parameter_name):
    """
    Return the reference part of a parameter that is not fit.

    Endmember Gibbs energy parameters are the sum of the pure element (SER)
    Gibbs energies, i.e. the GHSER functions weighted by the site ratios,
    plus the fitted terms. All other parameters have no reference part.

    Parameters
    ----------
    dbf : Database
        Database with the phase.
    phase_name : str
        Name of the phase.
    configuration : Tuple[Union[str, Tuple[str]]]
        Sublattice configuration of the parameter.
    parameter_name : str
        Type of the parameter, e.g. "G".

    Returns
    -------
    symengine.Expr

    """
    if interaction_test(configuration) or parameter_name != "G":
        return symengine.S.Zero
    return symengine.S(_endmember_reference(dbf, phase_name, configuration))


def insert_parameter(dbf, phase_name, configuration, parameter_name, parameters, symmetry):
    numdigits = 6  # number of significant figures, might cause rounding errors
    # Organize parameters by polynomial degree
    degree_polys = np.zeros(10, dtype=np.object_)
    YS = Symbol('YS')
    is_endmember = not interaction_test(configuration)
    if parameter_name == "G" and not is_endmember:
        # Special case for Gibbs parameters. "G" would still work
        # for databases, but "L" is backwards compatible and
        # slightly more readable.
        parameter_name = "L"
    # asymmetric parameters should have Mugiannu V_I/V_J/V_K, while symmetric just has YS
    is_asymmetric_ternary = any([(ky.has(Symbol('V_I'))) and (vl != 0) for ky, vl in parameters.items()])
    for degree in reversed(range(10)):
        if is_endmember:
            # shortcut because we know higher order degrees are impossible.
            # must be paired with a forced break at the end of the first
            # iteration to prevent duplicated work.
            degree = 0
            check_symbol = symengine.S.One
        elif is_asymmetric_ternary:
            if degree == 0:
                check_symbol = YS*Symbol('V_I')
            elif degree == 1:
                check_symbol = YS*Symbol('V_J')
            elif degree == 2:
                check_symbol = YS*Symbol('V_K')
            else:
                # don't do anything unless degree is 0, 1, or 2.
                continue
        else:
            check_symbol = Symbol('YS') * Symbol('Z')**degree
        keys_to_remove = []
        for key, value in parameters.items():
            if value == 0:
                continue
            if has_symbol(key, check_symbol) or is_endmember:
                symbol_name = get_next_symbol(dbf)
                dbf.symbols[symbol_name] = sigfigs(value, numdigits)
                parameters[key] = Symbol(symbol_name)
                coef = parameters[key] * (key / check_symbol)
                try:
                    coef = float(coef)
                except RuntimeError:
                    pass
                degree_polys[degree] += coef
                keys_to_remove.append(key)
        for key in keys_to_remove:
            parameters.pop(key)
        if is_endmember:
            break  # we forced a degree=0, any continued iterations is duplicated work.
    _log.trace('Polynomial coefs: %s', degree_polys)
    # Add the reference part (e.g. GHSER functions for endmember Gibbs energies).
    # Always degree 0 because of the short-circuit for endmembers in the previous loop
    degree_polys[0] += get_parameter_reference(dbf, phase_name, configuration, parameter_name)
    # Insert into database
    symmetric_configurations = generate_symmetric_group(configuration, symmetry)
    for degree in np.arange(degree_polys.shape[0]):
        if degree_polys[degree] != 0:
            for symmetric_config in symmetric_configurations:
                dbf.add_parameter(parameter_name, phase_name, tuple(map(tuplify, symmetric_config)), degree, degree_polys[degree])


def fit_parameters(dbf, comps, phase_name, configuration, symmetry, datasets, ridge_alpha=None, aicc_phase_penalty=None, fitting_description=gibbs_energy_fitting_description):
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
        Value of the :math:`\\alpha` hyperparameter used in ridge regression. Defaults to 1.0e-100, which should be degenerate
        with ordinary least squares regression. For now, the parameter is applied to all features.
    aicc_feature_factors : dict
        Map of phase name to feature to a multiplication factor for the AICc's parameter penalty.
    features : dict
        Maps "property" to a list of features for the linear model.
        These will be transformed from "GM" coefficients
        e.g., {"CPM_FORM": (v.T*symengine.log(v.T), v.T**2, v.T**-1, v.T**3)} (Default value = None)
    fitting_description : Type[ModelFittingDescription]
        ModelFittingDescription object describing the fitting steps and model

    Returns
    -------
    dict
        {feature: estimated_value}

    """
    aicc_feature_factors = aicc_phase_penalty if aicc_phase_penalty is not None else {}
    if interaction_test(configuration):
        _log.debug('ENDMEMBERS FROM INTERACTION: %s', endmembers_from_interaction(configuration))
    # fixed_model is our previously fit partial model from previous steps.
    # Its properties are absolute (i.e. include the pure element reference
    # contributions), and fitting steps shift data into absolute values.
    # fixed_portions are the parts of the parameter being fit that are already
    # known: the reference part, then the terms fit by each previous step.
    fixed_model = None  # Profiling suggests we delay instantiation
    fixed_portions = [0]
    parameters = {}
    # non-idiomatic loop so we can look ahead and see if we should write parameters or not
    for i in range(len(fitting_description.fitting_steps)):
        fitting_step = fitting_description.fitting_steps[i]
        _log.debug('Fitting step: %s', fitting_step)
        if _param_present_in_database(dbf, phase_name, configuration, fitting_step.parameter_name):
            _log.trace('Parameter %s already in the database for configuration %s. Skipping.', fitting_step.parameter_name, configuration)
            continue
        elif fitting_step.parameter_name == "G" and _param_present_in_database(dbf, phase_name, configuration, "L"):
            # special case for Gibbs energy parameters as L is valid for interactions
            _log.trace('Parameter L already in the database for configuration %s. Skipping.', configuration)
            continue
        # Search for relevant data
        desired_data = fitting_step.get_data(comps, phase_name, configuration, symmetry, datasets)
        if len(desired_data) > 0:
            if fixed_model is None:
                fixed_model = fitting_description.model(dbf, comps, phase_name)
                parameter_reference = get_parameter_reference(dbf, phase_name, configuration, fitting_step.parameter_name)
                fixed_portions = [fixed_model.symbol_replace(parameter_reference, fixed_model._symbols)]
            # Build candidate models, then fit and select a model from the candidates
            selected_model = fitting_step.select_parameters(fixed_model, fixed_portions, desired_data, configuration, ridge_alpha, aicc_feature_factors)
            if selected_model is not None:
                selected_features, selected_values = selected_model
                parameters.update(zip(*(selected_features, selected_values)))
                fixed_portion = np.array(selected_features, dtype=np.object_)
                fixed_portion = np.dot(fixed_portion, selected_values)
                fixed_portions.append(fixed_portion)

        # If we have a Gibbs energy type model, each step contributes to the
        # same parameter type, so we can't insert_parameter during each step.
        # Well... we could, but then we'd have duplicate parameters (e.g.
        # `L(PHASE,A,B;0)`) with different order interactions. That's okay by
        # pycalphad, but not other software so we preserve the legacy fixed
        # portions behavior to allow the values to accumulate. There may be
        # other alternatives to explore.
        if (i == len(fitting_description.fitting_steps) - 1) or (fitting_step.parameter_name != fitting_description.fitting_steps[i+1].parameter_name):
            # we're either on the last fitting step or the next step is a
            # different parameter type, so we insert and reset the state.
            parameters = OrderedDict([(ky, vl) for ky, vl in sorted(parameters.items(), key=_stable_sort_key)])
            insert_parameter(dbf, phase_name, configuration, fitting_step.parameter_name, parameters, symmetry)
            fixed_model = None
            fixed_portions = [0]


def phase_fit(dbf, phase_name, symmetry, datasets, refdata, ridge_alpha, aicc_penalty=None, aliases=None, fitting_description=gibbs_energy_fitting_description):
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
        Maps tuple(element, phase_name) -> SymEngine object defining
        energy relative to SER
    ridge_alpha : float
        Value of the :math:`\\alpha` hyperparameter used in ridge regression. Defaults to 1.0e-100, which should be degenerate
        with ordinary least squares regression. For now, the parameter is applied to all features.
    aicc_penalty : dict
        Map of phase name to feature to a multiplication factor for the AICc's parameter penalty.
    aliases : Dict[str, str]
        Mapping of possible aliases to the Database phase names.
    fitting_description : Type[ModelFittingDescription]
        ModelFittingDescription object describing the fitting steps and model

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
    subl_model = [sorted([sp.name for sp in subl]) for subl in phase_obj.constituents]
    site_ratios = phase_obj.sublattices
    # First fit endmembers
    all_em_count = len(generate_endmembers(subl_model))  # number of total endmembers
    endmembers = generate_endmembers(subl_model, symmetry)
    # TODO: use the global aliases dictionary passed in as-is instead of converting it to a phase-local dict
    # TODO: use the aliases dictionary in dataset queries to find relevant data
    if aliases is None:
        aliases = [phase_name]
    else:
        aliases = sorted([alias for alias, database_phase in aliases.items() if database_phase == phase_name])
    _log.info('FITTING: %s', phase_name)
    _log.trace('%s endmembers (%s distinct by symmetry)', all_em_count, len(endmembers))

    parameter_types_to_fit = sorted(set([step.parameter_name for step in fitting_description.fitting_steps]))
    # Special case Gibbs energy fitting as we need to do things like add lattice stabilities
    if "G" in parameter_types_to_fit:
        parameter_types_to_fit.append("L")  # G and L are both valid for interactions
        has_gibbs_fitting_step = True
    else:
        has_gibbs_fitting_step = False

    all_endmembers = []
    for endmember in endmembers:
        symmetric_endmembers = generate_symmetric_group(endmember, symmetry)
        all_endmembers.extend(symmetric_endmembers)
        _log.trace('ENDMEMBER: %s', endmember)
        if has_gibbs_fitting_step:
            # Special case to add pure element lattice stabilities IFF we fit G parameters
            if _param_present_in_database(dbf, phase_name, endmember, "G"):
                _log.trace('Parameter G already in the database for configuration %s. Skipping.', endmember)
                continue
            # Some endmembers are fixed by our choice of standard lattice stabilities, e.g., SGTE91
            # If a (phase, pure component endmember) tuple is fixed, we should use that value instead of fitting
            endmember_comps = list(set(endmember))
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
                        dbf.symbols[sym_name] = stability
                        break
                if dbf.symbols.get(sym_name, None) is not None:
                    num_moles = sum([sites for elem, sites in zip(endmember, site_ratios) if elem != 'VA'])
                    _log.trace("Found lattice stability: %s", sym_name)
                    _log.debug("%s = %s", sym_name, dbf.symbols[sym_name])
                    for em in symmetric_endmembers:
                        dbf.add_parameter('G', phase_name, tuple(map(tuplify, em)), 0, num_moles * Symbol(sym_name))
        # fit_parameters knows how to skip endmember Gibbs energies if we added a lattice stability parameter
        fit_parameters(dbf, sorted(dbf.elements), phase_name, endmember, symmetry, datasets, ridge_alpha, aicc_phase_penalty=aicc_phase_penalty, fitting_description=fitting_description)

    for interaction_order, order_name in ((2, "BINARY"), (3, "TERNARY")):
        _log.trace('FITTING %s INTERACTIONS', order_name)
        interactions = generate_interactions(all_endmembers, order=interaction_order, symmetry=symmetry)
        _log.trace('%s distinct order %s interactions', len(interactions), interaction_order)
        for interaction in interactions:
            _log.trace('INTERACTION: %s', interaction)
            fit_parameters(dbf, sorted(dbf.elements), phase_name, interaction, symmetry, datasets, ridge_alpha, aicc_phase_penalty=aicc_phase_penalty, fitting_description=fitting_description)

    if hasattr(dbf, 'varcounter'):
        del dbf.varcounter


def generate_parameters(phase_models, datasets, ref_state, excess_model, ridge_alpha=None, aicc_penalty_factor=None, dbf=None, fitting_description=gibbs_energy_fitting_description):
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
        Value of the :math:`\\alpha` hyperparameter used in ridge regression. Defaults
        to None, which falls back to ordinary least squares regression.
        For now, the parameter is applied to all features.
    aicc_penalty_factor : dict
        Map of phase name to feature to a multiplication factor for the AICc's parameter penalty.
    dbf : Database
        Initial pycalphad Database that can have parameters that would not be fit by ESPEI
    fitting_description : Type[ModelFittingDescription]
        ModelFittingDescription object describing the fitting steps and model

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
        if phase_name in dbf.phases:
            symmetry = phase_data.get('equivalent_sublattices', None)
            # Filter datasets by thermochemical data for this phase
            dataset = tinydb.Query()
            phase_filtered_datasets = PickleableTinyDB(storage=tinydb.storages.MemoryStorage)
            single_phase_thermochemical_query = (
                (dataset.phases == [phase_name])  # TODO: aliases support
                & dataset.solver.exists()
            )
            phase_filtered_datasets.insert_multiple(datasets.search(single_phase_thermochemical_query))
            phase_fit(dbf, phase_name, symmetry, phase_filtered_datasets, refdata, ridge_alpha, aicc_penalty=aicc_penalty_factor, aliases=aliases, fitting_description=fitting_description)
    _log.info('Finished generating parameters.')
    np.set_printoptions(linewidth=75)
    return dbf
