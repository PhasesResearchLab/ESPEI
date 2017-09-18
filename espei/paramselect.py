"""
The paramselect module handles automated parameter selection for linear models.

Automated Parameter Selection
End-members

Note: All magnetic parameters from literature for now.
Note: No fitting below 298 K (so neglect third law issues for now).

For each step, add one parameter at a time and compute AIC with max likelihood.

Cp - TlnT, T**2, T**-1, T**3 - 4 candidate models
(S and H only have one required parameter each. Will fit in full MCMC procedure)

Choose parameter set with best AIC score.

4. G (full MCMC) - all parameters selected at least once by above procedure

MCMC uses an EnsembleSampler based on Goodman and Weare, Ensemble Samplers with
Affine Invariance. Commun. Appl. Math. Comput. Sci. 5, 65-80 (2010).
"""
import itertools
import json
import textwrap
import time
import logging
import sys

import dask
import numpy as np
import operator
import pycalphad.refdata
import re
import sympy
import tinydb
import emcee
from collections import OrderedDict, defaultdict
from pycalphad import calculate, equilibrium, Database, Model, CompiledModel, \
    variables as v
from sklearn.linear_model import LinearRegression
from numpy.linalg import LinAlgError
from emcee.utils import MPIPool

from espei.core_utils import get_data, get_samples, canonicalize, canonical_sort_key, \
    list_to_tuple, endmembers_from_interaction, build_sitefractions
from espei.utils import PickleableTinyDB, sigfigs

feature_transforms = {"CPM_FORM": lambda x: -v.T*sympy.diff(x, v.T, 2),
                      "CPM_MIX": lambda x: -v.T*sympy.diff(x, v.T, 2),
                      "CPM": lambda x: -v.T*sympy.diff(x, v.T, 2),
                      "SM_FORM": lambda x: -sympy.diff(x, v.T),
                      "SM_MIX": lambda x: -sympy.diff(x, v.T),
                      "SM": lambda x: -sympy.diff(x, v.T),
                      "HM_FORM": lambda x: x - v.T*sympy.diff(x, v.T),
                      "HM_MIX": lambda x: x - v.T*sympy.diff(x, v.T),
                      "HM": lambda x: x - v.T*sympy.diff(x, v.T)}


def _fit_parameters(feature_matrix, data_quantities, feature_tuple):
    """Solve Ax = b, where 'feature_matrix' is A and 'data_quantities' is b.

    Parameters
    ----------
    feature_matrix : ndarray
        (M*N) regressor matrix.
    data_quantities : ndarray
        (M,) response vector
    feature_tuple : (float
        Polynomial coefficient corresponding to each column of 'feature_matrix'

    Returns
    -------
    OrderedDict
        {featured_tuple: fitted_parameter}. Maps 'feature_tuple'
        to fitted parameter value. If a coefficient is not used, it maps to zero.

    """
    # Now generate candidate models; add parameters one at a time
    model_scores = []
    results = np.zeros((len(feature_tuple), len(feature_tuple)))
    clf = LinearRegression(fit_intercept=False, normalize=True)
    for num_params in range(1, feature_matrix.shape[-1] + 1):
        current_matrix = feature_matrix[:, :num_params]
        clf.fit(current_matrix, data_quantities)
        # This may not exactly be the correct form for the likelihood
        # We're missing the "ridge" contribution here which could become relevant for sparse data
        rss = np.square(np.dot(current_matrix, clf.coef_) - data_quantities.astype(np.float)).sum()
        # Compute Aikaike Information Criterion
        # Form valid under assumption all sample variances are equal and unknown
        score = 2*num_params + current_matrix.shape[-2] * np.log(rss)
        model_scores.append(score)
        results[num_params - 1, :num_params] = clf.coef_
        logging.debug('{} rss: {}, AIC: {}'.format(feature_tuple[:num_params], rss, score))
    return OrderedDict(zip(feature_tuple, results[np.argmin(model_scores), :]))


def _build_feature_matrix(prop, features, desired_data):
    transformed_features = sympy.Matrix([feature_transforms[prop](i) for i in features])
    all_samples = get_samples(desired_data)
    feature_matrix = np.empty((len(all_samples), len(transformed_features)), dtype=np.float)
    feature_matrix[:, :] = [transformed_features.subs({v.T: temp, 'YS': compf[0],
                                                       'Z': compf[1]}).evalf()
                            for temp, compf in all_samples]
    return feature_matrix


def _shift_reference_state(desired_data, feature_transform, fixed_model):
    """
    Shift data to a new common reference state.
    """
    total_response = []
    for dataset in desired_data:
        values = np.asarray(dataset['values'], dtype=np.object)
        if dataset['solver'].get('sublattice_occupancies', None) is not None:
            value_idx = 0
            for occupancy, config in zip(dataset['solver']['sublattice_occupancies'],
                                         dataset['solver']['sublattice_configurations']):
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


def fit_formation_energy(dbf, comps, phase_name, configuration, symmetry, datasets, features=None):
    """Find suitable linear model parameters for the given phase.
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
    if features is None:
        features = [("CPM_FORM", (v.T * sympy.log(v.T), v.T**2, v.T**-1, v.T**3)),
                    ("SM_FORM", (v.T,)),
                    ("HM_FORM", (sympy.S.One,))
                    ]
        features = OrderedDict(features)
    if any([isinstance(conf, (list, tuple)) for conf in configuration]):
        fitting_steps = (["CPM_FORM", "CPM_MIX"], ["SM_FORM", "SM_MIX"], ["HM_FORM", "HM_MIX"])
        # Product of all nonzero site fractions in all sublattices
        YS = sympy.Symbol('YS')
        # Product of all binary interaction terms
        Z = sympy.Symbol('Z')
        redlich_kister_features = (YS, YS*Z, YS*(Z**2), YS*(Z**3))
        for feature in features.keys():
            all_features = list(itertools.product(redlich_kister_features, features[feature]))
            features[feature] = [i[0]*i[1] for i in all_features]
        logging.debug('ENDMEMBERS FROM INTERACTION: {}'.format(endmembers_from_interaction(configuration)))
    else:
        # We are only fitting an endmember; no mixing data needed
        fitting_steps = (["CPM_FORM"], ["SM_FORM"], ["HM_FORM"])

    parameters = {}
    for feature in features.values():
        for coef in feature:
            parameters[coef] = 0

    # These is our previously fit partial model
    # Subtract out all of these contributions (zero out reference state because these are formation properties)
    fixed_model = Model(dbf, comps, phase_name, parameters={'GHSER'+c.upper(): 0 for c in comps})
    fixed_model.models['idmix'] = 0
    fixed_portions = [0]

    moles_per_formula_unit = sympy.S(0)
    subl_idx = 0
    for num_sites, const in zip(dbf.phases[phase_name].sublattices, dbf.phases[phase_name].constituents):
        if 'VA' in const:
            moles_per_formula_unit += num_sites * (1 - v.SiteFraction(phase_name, subl_idx, 'VA'))
        else:
            moles_per_formula_unit += num_sites
        subl_idx += 1

    for desired_props in fitting_steps:
        desired_data = get_data(comps, phase_name, configuration, symmetry, datasets, desired_props)
        logging.debug('{}: datasets found: {}'.format(desired_props, len(desired_data)))
        if len(desired_data) > 0:
            # We assume all properties in the same fitting step have the same features (but different ref states)
            feature_matrix = _build_feature_matrix(desired_props[0], features[desired_props[0]], desired_data)
            all_samples = get_samples(desired_data)
            data_quantities = np.concatenate(_shift_reference_state(desired_data,
                                                                    feature_transforms[desired_props[0]],
                                                                    fixed_model),
                                             axis=-1)
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
            parameters.update(_fit_parameters(feature_matrix, data_quantities, features[desired_props[0]]))
            # Add these parameters to be fixed for the next fitting step
            fixed_portion = np.array(features[desired_props[0]], dtype=np.object)
            fixed_portion = np.dot(fixed_portion, [parameters[feature] for feature in features[desired_props[0]]])
            fixed_portions.append(fixed_portion)
    return parameters


def _generate_symmetric_group(configuration, symmetry):
    configurations = [list_to_tuple(configuration)]
    permutation = np.array(symmetry, dtype=np.object)

    def permute(x):
        if len(x) == 0:
            return x
        x[0] = np.roll(x[0], 1)
        x[:] = np.roll(x, 1, axis=0)
        return x

    if symmetry is not None:
        while np.any(np.array(symmetry, dtype=np.object) != permute(permutation)):
            new_conf = np.array(configurations[0], dtype=np.object)
            subgroups = []
            # There is probably a more efficient way to do this
            for subl in permutation:
                subgroups.append([configuration[idx] for idx in subl])
            # subgroup is ordered according to current permutation
            # but we'll index it based on the original symmetry
            # This should permute the configurations
            for subl, subgroup in zip(symmetry, subgroups):
                for subl_idx, conf_idx in enumerate(subl):
                    new_conf[conf_idx] = subgroup[subl_idx]
            configurations.append(tuple(new_conf))

    return sorted(set(configurations), key=canonical_sort_key)


def phase_fit(dbf, phase_name, symmetry, subl_model, site_ratios, datasets, refdata, aliases=None):
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
    endmembers = sorted(set(
        canonicalize(i, symmetry) for i in itertools.product(*subl_model)))
    # Number of significant figures in parameters
    numdigits = 6
    em_dict = {}
    aliases = [] if aliases is None else aliases
    aliases = sorted(set(aliases + [phase_name]))
    logging.info('FITTING: {}'.format(phase_name))
    logging.info('{0} endmembers ({1} distinct by symmetry)'.format(all_em_count, len(endmembers)))

    def _to_tuple(x):
        if isinstance(x, list) or isinstance(x, tuple):
            return tuple(x)
        else:
            return tuple([x])

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
            parameters = fit_formation_energy(dbf, sorted(dbf.elements), phase_name, endmember, symmetry, datasets)
            for key, value in sorted(parameters.items(), key=str):
                if value == 0:
                    continue
                symbol_name = 'VV'+str(dbf.varcounter).zfill(4)
                while dbf.symbols.get(symbol_name, None) is not None:
                    dbf.varcounter += 1
                    symbol_name = 'VV' + str(dbf.varcounter).zfill(4)
                dbf.symbols[symbol_name] = sigfigs(value, numdigits)
                parameters[key] = sympy.Symbol(symbol_name)
            fit_eq = sympy.Add(*[value * key for key, value in parameters.items()])
            ref = 0
            for subl, ratio in zip(endmember, site_ratios):
                if subl == 'VA':
                    continue
                ref = ref + ratio * sympy.Symbol('GHSER'+subl)
            fit_eq += ref
        symmetric_endmembers = _generate_symmetric_group(endmember, symmetry)
        logging.debug('SYMMETRIC_ENDMEMBERS: {}'.format(symmetric_endmembers))
        all_endmembers.extend(symmetric_endmembers)
        for em in symmetric_endmembers:
            em_dict[em] = fit_eq
            dbf.add_parameter('G', phase_name, tuple(map(_to_tuple, em)), 0, fit_eq)
    # Now fit all binary interactions
    # Need to use 'all_endmembers' instead of 'endmembers' because you need to generate combinations
    # of ALL endmembers, not just symmetry equivalent ones
    bin_interactions = list(itertools.combinations(all_endmembers, 2))
    transformed_bin_interactions = []
    for first_endmember, second_endmember in bin_interactions:
        interaction = []
        for first_occupant, second_occupant in zip(first_endmember, second_endmember):
            if first_occupant == second_occupant:
                interaction.append(first_occupant)
            else:
                interaction.append(tuple(sorted([first_occupant, second_occupant])))
        transformed_bin_interactions.append(interaction)

    def bin_int_sort_key(x):
        interacting_sublattices = sum((isinstance(n, (list, tuple)) and len(n) == 2) for n in x)
        return canonical_sort_key((interacting_sublattices,) + x)

    bin_interactions = sorted(set(
        canonicalize(i, symmetry) for i in transformed_bin_interactions),
                              key=bin_int_sort_key)
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
        parameters = fit_formation_energy(dbf, sorted(dbf.elements), phase_name, ixx, symmetry, datasets)
        # Organize parameters by polynomial degree
        degree_polys = np.zeros(10, dtype=np.object)
        for degree in reversed(range(10)):
            check_symbol = sympy.Symbol('YS') * sympy.Symbol('Z')**degree
            keys_to_remove = []
            for key, value in sorted(parameters.items(), key=str):
                if key.has(check_symbol):
                    if value != 0:
                        symbol_name = 'VV' + str(dbf.varcounter).zfill(4)
                        while dbf.symbols.get(symbol_name, None) is not None:
                            dbf.varcounter += 1
                            symbol_name = 'VV' + str(dbf.varcounter).zfill(4)
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
        symmetric_interactions = _generate_symmetric_group(interaction, symmetry)
        for degree in np.arange(degree_polys.shape[0]):
            if degree_polys[degree] != 0:
                for syminter in symmetric_interactions:
                    dbf.add_parameter('L', phase_name, tuple(map(_to_tuple, syminter)), degree, degree_polys[degree])
    # TODO: fit ternary interactions
    if hasattr(dbf, 'varcounter'):
        del dbf.varcounter


def estimate_hyperplane(dbf, comps, phases, current_statevars, comp_dicts, phase_models, parameters):
    region_chemical_potentials = []
    parameters = OrderedDict(sorted(parameters.items(), key=str))
    for cond_dict, phase_flag in comp_dicts:
        # We are now considering a particular tie vertex
        for key, val in cond_dict.items():
            if val is None:
                cond_dict[key] = np.nan
        cond_dict.update(current_statevars)
        if np.any(np.isnan(list(cond_dict.values()))):
            # This composition is unknown -- it doesn't contribute to hyperplane estimation
            pass
        else:
            # Extract chemical potential hyperplane from multi-phase calculation
            # Note that we consider all phases in the system, not just ones in this tie region
            multi_eqdata = equilibrium(dbf, comps, phases, cond_dict, verbose=False,
                                       model=phase_models, scheduler=dask.local.get_sync, parameters=parameters)
            # Does there exist only a single phase in the result with zero internal degrees of freedom?
            # We should exclude those chemical potentials from the average because they are meaningless.
            num_phases = len(np.squeeze(multi_eqdata['Phase'].values != ''))
            zero_dof = np.all((multi_eqdata['Y'].values == 1.) | np.isnan(multi_eqdata['Y'].values))
            if (num_phases == 1) and zero_dof:
                region_chemical_potentials.append(np.full_like(np.squeeze(multi_eqdata['MU'].values), np.nan))
            else:
                region_chemical_potentials.append(np.squeeze(multi_eqdata['MU'].values))
    region_chemical_potentials = np.nanmean(region_chemical_potentials, axis=0, dtype=np.float)
    return region_chemical_potentials


def tieline_error(dbf, comps, current_phase, cond_dict, region_chemical_potentials, phase_flag,
                  phase_models, parameters, debug_mode=False):
    if np.any(np.isnan(list(cond_dict.values()))):
        # We don't actually know the phase composition here, so we estimate it
        single_eqdata = calculate(dbf, comps, [current_phase],
                                  T=cond_dict[v.T], P=cond_dict[v.P],
                                  model=phase_models, parameters=parameters, pdens=100)
        driving_force = np.multiply(region_chemical_potentials,
                                    single_eqdata['X'].values).sum(axis=-1) - single_eqdata['GM'].values
        error = float(driving_force.max())
    elif phase_flag == 'disordered':
        # Construct disordered sublattice configuration from composition dict
        # Compute energy
        # Compute residual driving force
        # TODO: Check that it actually makes sense to declare this phase 'disordered'
        num_dof = sum([len(set(c).intersection(comps)) for c in dbf.phases[current_phase].constituents])
        desired_sitefracs = np.ones(num_dof, dtype=np.float)
        dof_idx = 0
        for c in dbf.phases[current_phase].constituents:
            dof = sorted(set(c).intersection(comps))
            if (len(dof) == 1) and (dof[0] == 'VA'):
                return 0
            # If it's disordered config of BCC_B2 with VA, disordered config is tiny vacancy count
            sitefracs_to_add = np.array([cond_dict.get(v.X(d)) for d in dof],
                                        dtype=np.float)
            # Fix composition of dependent component
            sitefracs_to_add[np.isnan(sitefracs_to_add)] = 1 - np.nansum(sitefracs_to_add)
            desired_sitefracs[dof_idx:dof_idx + len(dof)] = sitefracs_to_add
            dof_idx += len(dof)
        single_eqdata = calculate(dbf, comps, [current_phase],
                                  T=cond_dict[v.T], P=cond_dict[v.P], points=desired_sitefracs,
                                  model=phase_models, parameters=parameters)
        driving_force = np.multiply(region_chemical_potentials,
                                    single_eqdata['X'].values).sum(axis=-1) - single_eqdata['GM'].values
        error = float(np.squeeze(driving_force))
    else:
        # Extract energies from single-phase calculations
        single_eqdata = equilibrium(dbf, comps, [current_phase], cond_dict, verbose=False,
                                    model=phase_models,
                                    scheduler=dask.local.get_sync, parameters=parameters)
        if np.all(np.isnan(single_eqdata['NP'].values)):
            error_time = time.time()
            template_error = """
            from pycalphad import Database, equilibrium
            from pycalphad.variables import T, P, X
            import dask
            dbf_string = \"\"\"
            {0}
            \"\"\"
            dbf = Database(dbf_string)
            comps = {1}
            phases = {2}
            cond_dict = {3}
            parameters = {4}
            equilibrium(dbf, comps, phases, cond_dict, scheduler=dask.local.get_sync, parameters=parameters)
            """
            template_error = textwrap.dedent(template_error)
            if debug_mode:
                logging.warning('Dumping', 'error-'+str(error_time)+'.py')
                with open('error-'+str(error_time)+'.py', 'w') as f:
                    f.write(template_error.format(dbf.to_string(fmt='tdb'), comps, [current_phase], cond_dict, {key: float(x) for key, x in parameters.items()}))
        # Sometimes we can get a miscibility gap in our "single-phase" calculation
        # Choose the weighted mixture of site fractions
            logging.warning('Dropping condition due to calculation failure: {}'.format(cond_dict))
            return 0
        select_energy = float(single_eqdata['GM'].values)
        region_comps = []
        for comp in [c for c in sorted(comps) if c != 'VA']:
            region_comps.append(cond_dict.get(v.X(comp), np.nan))
        region_comps[region_comps.index(np.nan)] = 1 - np.nansum(region_comps)
        error = np.multiply(region_chemical_potentials, region_comps).sum() - select_energy
        error = float(error)
    return error


def multi_phase_fit(dbf, comps, phases, datasets, phase_models, parameters=None, scheduler=None):
    scheduler = scheduler or dask.local
    # TODO: support distributed schedulers for multi_phase_fit.
    # This can be done if the scheduler passed is a distributed.worker_client
    if scheduler is not dask.local:
        raise NotImplementedError('Schedulers other than dask.local are not currently supported for multiphase fitting.')
    desired_data = datasets.search((tinydb.where('output') == 'ZPF') &
                                   (tinydb.where('components').test(lambda x: set(x).issubset(comps))) &
                                   (tinydb.where('phases').test(lambda x: len(set(phases).intersection(x)) > 0)))

    def safe_get(itms, idxx):
        try:
            return itms[idxx]
        except IndexError:
            return None

    fit_jobs = []
    for data in desired_data:
        payload = data['values']
        conditions = data['conditions']
        data_comps = list(set(data['components']).union({'VA'}))
        phase_regions = defaultdict(lambda: list())
        # TODO: Fix to only include equilibria listed in 'phases'
        for idx, p in enumerate(payload):
            phase_key = tuple(sorted(rp[0] for rp in p))
            if len(phase_key) < 2:
                # Skip single-phase regions for fitting purposes
                continue
            # Need to sort 'p' here so we have the sorted ordering used in 'phase_key'
            # rp[3] optionally contains additional flags, e.g., "disordered", to help the solver
            comp_dicts = [(dict(zip([v.X(x.upper()) for x in rp[1]], rp[2])), safe_get(rp, 3))
                          for rp in sorted(p, key=operator.itemgetter(0))]
            cur_conds = {}
            for key, value in conditions.items():
                value = np.atleast_1d(np.asarray(value))
                if len(value) > 1:
                    value = value[idx]
                cur_conds[getattr(v, key)] = float(value)
            phase_regions[phase_key].append((cur_conds, comp_dicts))
        for region, region_eq in phase_regions.items():
            for req in region_eq:
                # We are now considering a particular tie region
                current_statevars, comp_dicts = req
                region_chemical_potentials = \
                    dask.delayed(estimate_hyperplane)(dbf, data_comps, phases, current_statevars, comp_dicts,
                                                      phase_models, parameters)
                # Now perform the equilibrium calculation for the isolated phases and add the result to the error record
                for current_phase, cond_dict in zip(region, comp_dicts):
                    # TODO: Messy unpacking
                    cond_dict, phase_flag = cond_dict
                    # We are now considering a particular tie vertex
                    for key, val in cond_dict.items():
                        if val is None:
                            cond_dict[key] = np.nan
                    cond_dict.update(current_statevars)
                    error = dask.delayed(tieline_error)(dbf, data_comps, current_phase, cond_dict, region_chemical_potentials, phase_flag,
                                                        phase_models, parameters)
                    fit_jobs.append(error)
    errors = dask.compute(*fit_jobs, get=scheduler.get_sync)
    return errors


def lnprob(params, data=None, comps=None, dbf=None, phases=None, datasets=None,
           symbols_to_fit=None, phase_models=None, scheduler=None):
    """
    Returns the error from multiphase fitting as a log probability.
    """
    parameters = {param_name: param for param_name, param in zip(symbols_to_fit, params)}
    try:
        iter_error = multi_phase_fit(dbf, comps, phases, datasets, phase_models,
                                     parameters=parameters, scheduler=scheduler)
    except (ValueError, LinAlgError) as e:
        iter_error = [np.inf]
    iter_error = [np.inf if np.isnan(x) else x ** 2 for x in iter_error]
    iter_error = -np.sum(iter_error)
    return np.array(iter_error, dtype=np.float64)


def fit(input_fname, datasets, resume=None, scheduler=None, run_mcmc=True,
        tracefile=None, probfile=None, restart_chain=None, mcmc_steps=1000,
        save_interval=100, chains_per_parameter=2, chain_std_deviation=0.1):
    """Fit thermodynamic and phase equilibria data to a model.
    
    Parameters
    ----------
    input_fname : str
        name of the input file containing the sublattice models.
    datasets : PickleableTinyDB
        database of single- and multi-phase to fit.
    resume : Database
        pycalphad Database of a file to start from. Using this parameter causes
        single phase fitting to be skipped (multi-phase only).
    scheduler : callable
        Scheduler to use with emcee. Must implement a map method.
    run_mcmc : bool
        Controls if MCMC should be run. Default is True. Useful for
        first-principles (single-phase only) runs.
    tracefile : str
        filename to store the flattened chain with NumPy.save. Array has shape
        (nwalkers, iterations, nparams)
    probfile : str
        filename to store the flattened ln probability with NumPy.save
    restart_chain : np.ndarray
        ndarray of the previous chain. Should have shape (nwalkers, iterations, nparams)
    mcmc_steps : int
        number of chain steps to calculate in MCMC. Note the flattened chain will
        have (mcmc_steps*DOF) values.
        int (Default value = 1000)
    save_interval : int
        interval of steps to save the chain to the tracefile.
    chains_per_parameter : int
        number of chains for each parameter. Must be an even integer greater or
        equal to 2. Defaults to 2.
    chain_std_deviation : float
        standard deviation of normal for parameter initialization as a fraction
        of each parameter. Must be greater than 0. Default is 0.1, which is 10%.

    Returns
    -------
    dbf : Database
        Resulting pycalphad database of optimized parameters
    sampler : EnsembleSampler, ndarray)
        emcee sampler for further data wrangling
    parameters_dict : dict
        Optimized parameters

    """
    # TODO: Validate input JSON
    data = json.load(open(input_fname))
    if resume is None:
        dbf = Database()
        dbf.elements = set(data['components'])
        # Write reference state to Database
        refdata = getattr(pycalphad.refdata, data['refdata'])
        stabledata = getattr(pycalphad.refdata, data['refdata']+'Stable')
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
        dbf.symbols.update({'GHSER'+c.upper(): data for c, data in comp_refs.items()})
        for phase_name, phase_obj in sorted(data['phases'].items(), key=operator.itemgetter(0)):
            # Perform parameter selection and single-phase fitting based on input
            # TODO: Need to pass particular models to include: magnetic, order-disorder, etc.
            symmetry = phase_obj.get('equivalent_sublattices', None)
            aliases = phase_obj.get('aliases', None)
            # TODO: More advanced phase data searching
            site_ratios = phase_obj['sublattice_site_ratios']
            subl_model = phase_obj['sublattice_model']
            dbf.add_phase(phase_name, {}, site_ratios)
            dbf.add_phase_constituents(phase_name, subl_model)
            dbf.add_structure_entry(phase_name, phase_name)
            phase_fit(dbf, phase_name, symmetry, subl_model, site_ratios, datasets, refdata, aliases=aliases)
    else:
        logging.info('STARTING FROM USER-SPECIFIED DATABASE')
        dbf = resume

    if run_mcmc:
        comps = sorted(data['components'])
        pattern = re.compile("^V[V]?([0-9]+)$")
        symbols_to_fit = sorted([x for x in sorted(dbf.symbols.keys()) if pattern.match(x)])

        if len(symbols_to_fit) == 0:
            raise ValueError('No degrees of freedom. Database must contain symbols starting with \'V\' or \'VV\', followed by a number.')
        else:
            logging.info('Fitting {} degrees of freedom.'.format(len(symbols_to_fit)))

        for x in symbols_to_fit:
            if isinstance(dbf.symbols[x], sympy.Piecewise):
                logging.debug('Replacing {} in database'.format(x))
                dbf.symbols[x] = dbf.symbols[x].args[0].expr

        # get initial parameters and remove these from the database
        # we'll replace them with SymPy symbols initialized to 0 in the phase models
        initial_parameters = [np.array(float(dbf.symbols[x])) for x in symbols_to_fit]
        for x in symbols_to_fit:
            del dbf.symbols[x]

        # construct the models for each phase, substituting in the SymPy symbol to fit.
        phase_models = dict()
        logging.info('Building phase models')
        # 0 is placeholder value
        for phase_name in sorted(data['phases'].keys()):
            mod = CompiledModel(dbf, comps, phase_name, parameters=OrderedDict([(sympy.Symbol(s), 0) for s in symbols_to_fit]))
            phase_models[phase_name] = mod
        logging.info('Finished building phase models')
        dbf = dask.delayed(dbf, pure=True)
        phase_models = dask.delayed(phase_models, pure=True)

        # contect for the log probability function
        error_context = {'data': data, 'comps': comps, 'dbf': dbf, 'phases': sorted(data['phases'].keys()),
                         'datasets': datasets, 'symbols_to_fit': symbols_to_fit,
                         'phase_models': phase_models}


        def save_sampler_state(sampler):
            if tracefile:
                logging.debug('Writing chain to {}'.format(tracefile))
                np.save(tracefile, sampler.chain)
            if probfile:
                logging.debug('Writing lnprob to {}'.format(probfile))
                np.save(probfile, sampler.lnprobability)

        # initialize the walkers either fresh or from the restart
        if restart_chain is not None:
            walkers = restart_chain[np.nonzero(restart_chain)].reshape((restart_chain.shape[0], -1, restart_chain.shape[2]))[:, -1, :]
            nwalkers = walkers.shape[0]
            ndim = walkers.shape[1]
            logging.debug('Restarting from previous calculation. Parameters on restart: {}'.format(initial_parameters))
        else:
            # initialize the RNG
            rng = np.random.RandomState(1769)
            # set up the MCMC run
            # set up the initial parameters
            # apply a Gaussian random to each parameter with std dev of 0.10*parameter
            initial_parameters = np.array(initial_parameters)
            logging.debug('Initial parameters: {}'.format(initial_parameters))
            ndim = len(initial_parameters)
            nwalkers = chains_per_parameter*ndim # walkers must be of size (2n*ndim)
            initial_walkers = np.tile(initial_parameters, (nwalkers, 1))
            walkers = rng.normal(initial_walkers, np.abs(initial_walkers*chain_std_deviation))

        # set up with emcee
        import emcee
        import sys
        # the pool must implement a map function
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, kwargs=error_context, pool=scheduler)
        progbar_width = 30
        logging.info('Running MCMC with {} steps.'.format(mcmc_steps))
        try:
            for i, result in enumerate(sampler.sample(walkers, iterations=mcmc_steps)):
                # progress bar
                if (i+1) % save_interval == 0:
                    save_sampler_state(sampler)
                    logging.debug('Acceptance ratios for parameters: {}'.format(sampler.acceptance_fraction))
                n = int((progbar_width + 1) * float(i) / mcmc_steps)
                sys.stdout.write("\r[{0}{1}] ({2} of {3})\n".format('#' * n, ' ' * (progbar_width - n), i+1, mcmc_steps))
            n = int((progbar_width + 1) * float(i+1) / mcmc_steps)
            sys.stdout.write("\r[{0}{1}] ({2} of {3})\n".format('#' * n, ' ' * (progbar_width - n), i + 1, mcmc_steps))
        except KeyboardInterrupt:
            pass
        # close the pool if it is an MPIPool.
        if isinstance(scheduler, MPIPool):
            scheduler.close()
        # final processing
        save_sampler_state(sampler)
        flatchain = sampler.flatchain
        optimal_parameters = flatchain[np.nanargmin(-sampler.flatlnprobability)]
        logging.debug('Intial parameters: {}'.format(initial_parameters))
        logging.debug('Optimal parameters: {}'.format(optimal_parameters))
        logging.debug('Change in parameters: {}'.format(np.abs(initial_parameters - optimal_parameters) / initial_parameters))
        parameters_dict = {param_name: value for param_name, value in zip(symbols_to_fit, optimal_parameters)}
        dbf = dbf.compute()
        for param_name, value in parameters_dict.items():
            dbf.symbols[param_name] = value

        return dbf, sampler, parameters_dict

    else:
        logging.info('Not running MCMC. Returning the single-phase fit database.')
        return dbf, None, None
