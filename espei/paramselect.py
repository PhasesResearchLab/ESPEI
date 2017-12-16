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
import itertools, operator, logging

import numpy as np
import sympy
from collections import OrderedDict
from pycalphad import Database, Model, variables as v
from sklearn.linear_model import LinearRegression

from espei.core_utils import get_data, get_samples, canonicalize, canonical_sort_key, \
    list_to_tuple, endmembers_from_interaction, build_sitefractions
from espei.utils import PickleableTinyDB, sigfigs
import espei.refdata

# backwards compatibility:
# TODO: drop support on release pycalphad 0.7
try:
    from pycalphad.io.database import Species
except ImportError:
    Species = None

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
    fixed_model = Model(dbf, comps, phase_name, parameters={'GHSER'+(c.upper()*2)[:2]: 0 for c in comps})
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
    logging.debug('{0} endmembers ({1} distinct by symmetry)'.format(all_em_count, len(endmembers)))

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
                subl = (subl.upper()*2)[:2]
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

def generate_parameters(phase_models, datasets, ref_state, excess_model):
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

    Returns
    -------
    Database
    """
    logging.info('Generating parameters.')
    dbf = Database()
    dbf.elements = set(phase_models['components'])
    for el in dbf.elements:
        if Species is not None:  # TODO: drop this on release of pycalphad 0.7
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
        phase_fit(dbf, phase_name, symmetry, subl_model, site_ratios, datasets, refdata, aliases=aliases)
    logging.info('Finished generating parameters.')
    return dbf
