"""
The paramselect module handles automated parameter selection for linear models.


Automated Parameter Selection
End-members

Note: All magnetic parameters from literature for now.
Note: No fitting below 298 K (so neglect third law issues for now).
Note: Should use calculate(..., mode='numpy') for this step for performance reasons.

For each step, add one parameter at a time and compute AIC with max likelihood.

Cp - TlnT, T**2, T**-1, T**3 - 4 candidate models
(S and H only have one required parameter each -- will fit in full MCMC procedure)

Choose parameter set with best AIC score.

4. G (full MCMC) - all parameters selected at least once by above procedure

Choice of priors:

(const.): Normal(mean=0, sd=1e6)
T:        Normal(mean=0, sd=50)
TlnT:     Normal(mean=0, sd=20)
T**2:     Normal(mean=0, sd=10)
T**-1:    Normal(mean=0, sd=1e6)
T**3:     Normal(mean=0, sd=5)

Should we use the ridge regression method instead of MCMC for selection?
If we use zero-mean normally-distributed priors, we just construct
a diagonal weighting matrix like 1/variance, and it's actually
equivalent. (But much, much faster to calculate.)
But then how do we compute a model score?
We compute AIC from 2k - 2ln(L_max), where L_max is max likelihood and k is
the number of parameters. All we need, then, is our likelihood estimate.
How do we get that from the regression result?
1. Use lstsq (or ridge) to get parameter values at max likelihood (min error).
2. Plug parameter values into likelihood function to get L_max.
3. Compute AIC.
4. Choose model with minimum AIC.

Looks straightforward: see Tikhonov regularization on Wikipedia
I think we're okay on parameter correlations if we build the matrix correctly

If we do pinv method like ZKL suggested, it's like doing Bayesian regression
with uninformative priors. With bias of AIC toward complex models, I think doing regularization
with ridge regression is advisible.
"""
import pycalphad.variables as v
from pycalphad import binplot, calculate, equilibrium, Database, Model
from pycalphad.plot.utils import phase_legend
from pycalphad.core.autograd_utils import build_functions
import pycalphad.refdata
from sklearn.linear_model import LinearRegression, Lasso, LassoCV
from sklearn.covariance import EmpiricalCovariance
from sklearn.preprocessing import Normalizer, MaxAbsScaler, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
import tinydb
import sympy
import numpy as np
import json
from collections import Counter, OrderedDict, defaultdict
import itertools
import operator
import copy
from functools import reduce
from datetime import datetime


# Mapping of energy polynomial coefficients to corresponding property coefficients
feature_transforms = {"CPM_FORM": lambda x: -v.T*sympy.diff(x, v.T, 2),
                      "CPM_MIX": lambda x: -v.T*sympy.diff(x, v.T, 2),
                      "CPM": lambda x: -v.T*sympy.diff(x, v.T, 2),
                      "SM_FORM": lambda x: -sympy.diff(x, v.T),
                      "SM_MIX": lambda x: -sympy.diff(x, v.T),
                      "SM": lambda x: -sympy.diff(x, v.T),
                      "HM_FORM": lambda x: x - v.T*sympy.diff(x, v.T),
                      "HM_MIX": lambda x: x - v.T*sympy.diff(x, v.T),
                      "HM": lambda x: x - v.T*sympy.diff(x, v.T)}

plot_mapping = {
    'T': 'Temperature (K)',
    'CPM': 'Heat Capacity (J/K-mol-atom)',
    'HM': 'Enthalpy (J/mol-atom)',
    'SM': 'Entropy (J/K-mol-atom)',
    'CPM_FORM': 'Heat Capacity of Formation (J/K-mol-atom)',
    'HM_FORM': 'Enthalpy of Formation (J/mol-atom)',
    'SM_FORM': 'Entropy of Formation (J/K-mol-atom)',
    'CPM_MIX': 'Heat Capacity of Mixing (J/K-mol-atom)',
    'HM_MIX': 'Enthalpy of Mixing (J/mol-atom)',
    'SM_MIX': 'Entropy of Mixing (J/K-mol-atom)'
}


def load_datasets(dataset_filenames):
    ds_database = tinydb.TinyDB(storage=tinydb.storages.MemoryStorage)
    for fname in dataset_filenames:
        with open(fname) as file_:
            try:
                ds_database.insert(json.load(file_))
            except ValueError as e:
                print('JSON Error in {}: {}'.format(fname, e))
    return ds_database


def canonical_sort_key(x):
    """
    Wrap strings in tuples so they'll sort.

    Parameters
    ==========
    x : sequence
    """
    return [tuple(i) if isinstance(i, (tuple, list)) else (i,) for i in x]


def canonicalize(configuration, equivalent_sublattices):
    """
    Sort a sequence with symmetry. This routine gives the sequence
    a deterministic ordering while respecting symmetry.

    Parameters
    ==========
    configuration : list
        Sublattice configuration to sort.
    equivalent_sublattices : set of set of int
        Indices of 'configuration' which should be equivalent by symmetry, i.e.,
        [[0, 4], [1, 2, 3]] means permuting elements 0 and 4, or 1, 2 and 3, respectively,
        has no effect on the equivalence of the sequence.

    Returns
    =======
    canonicalized : tuple
    """
    canonicalized = list(configuration)
    if equivalent_sublattices is not None:
        for subl in equivalent_sublattices:
            subgroup = sorted([configuration[idx] for idx in sorted(subl)], key=canonical_sort_key)
            for subl_idx, conf_idx in enumerate(sorted(subl)):
                if isinstance(subgroup[subl_idx], list):
                    canonicalized[conf_idx] = tuple(subgroup[subl_idx])
                else:
                    canonicalized[conf_idx] = subgroup[subl_idx]

    return _list_to_tuple(canonicalized)


def _symmetry_filter(x, config, symmetry):
    if x['mode'] == 'manual':
        if len(config) != len(x['sublattice_configurations'][0]):
            return False
        # If even one matches, it's a match
        # We do more filtering downstream
        for data_config in x['sublattice_configurations']:
            if canonicalize(config, symmetry) == canonicalize(data_config, symmetry):
                return True
    return False


def _get_data(comps, phase_name, configuration, symmetry, datasets, prop):
    configuration = list(configuration)
    desired_data = datasets.search((tinydb.where('output').test(lambda x: x in prop)) &
                                   (tinydb.where('components').test(lambda x: set(x).issubset(comps))) &
                                   (tinydb.where('solver').test(_symmetry_filter, configuration, symmetry)) &
                                   (tinydb.where('phases') == [phase_name]))
    # This seems to be necessary because the 'values' member does not modify 'datasets'
    # But everything else does!
    desired_data = copy.deepcopy(desired_data)
    #if len(desired_data) == 0:
    #    raise ValueError('No datasets for the system of interest containing {} were in \'datasets\''.format(prop))

    def recursive_zip(a, b):
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            return list(recursive_zip(x, y) for x, y in zip(a, b))
        else:
            return list(zip(a, b))

    for idx, data in enumerate(desired_data):
        # Filter output values to only contain data for matching sublattice configurations
        matching_configs = np.array([(canonicalize(sblconf, symmetry) == canonicalize(configuration, symmetry))
                                     for sblconf in data['solver']['sublattice_configurations']])
        matching_configs = np.arange(len(data['solver']['sublattice_configurations']))[matching_configs]
        # Rewrite output values with filtered data
        desired_data[idx]['values'] = np.array(data['values'], dtype=np.float)[..., matching_configs]
        desired_data[idx]['solver']['sublattice_configurations'] = _list_to_tuple(np.array(data['solver']['sublattice_configurations'],
                                                                            dtype=np.object)[matching_configs].tolist())
        try:
            desired_data[idx]['solver']['sublattice_occupancies'] = np.array(data['solver']['sublattice_occupancies'],
                                                                             dtype=np.object)[matching_configs].tolist()
        except KeyError:
            pass
        # Filter out temperatures below 298.15 K (for now, until better refstates exist)
        temp_filter = np.atleast_1d(data['conditions']['T']) >= 298.15
        desired_data[idx]['conditions']['T'] = np.atleast_1d(data['conditions']['T'])[temp_filter]
        # Don't use data['values'] because we rewrote it above; not sure what 'data' references now
        desired_data[idx]['values'] = desired_data[idx]['values'][..., temp_filter, :]
    return desired_data


def _fit_parameters(feature_matrix, data_quantities, feature_tuple):
    """
    Solve Ax = b, where 'feature_matrix' is A and 'data_quantities' is b.

    Parameters
    ==========
    feature_matrix : ndarray (M*N)
        Regressor matrix
    data_quantities : ndarray (M,)
        Response vector
    feature_tuple : tuple
        Polynomial coefficient corresponding to each column of 'feature_matrix'

    Returns
    =======
    parameters : OrderedDict
       Maps 'feature_tuple' to fitted parameter value.
       If a coefficient is not used, it maps to zero.
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
        print(feature_tuple[:num_params], 'rss:', rss, 'AIC:', score)
    #cov = EmpiricalCovariance(store_precision=False, assume_centered=False)
    #cov.fit(feature_matrix[:, :np.argmin(model_scores)+1], data_quantities)
    #print(cov.covariance_)
    return OrderedDict(zip(feature_tuple, results[np.argmin(model_scores), :]))


def _get_samples(desired_data):
    all_samples = []
    for data in desired_data:
        temperatures = np.atleast_1d(data['conditions']['T'])
        num_configs = np.array(data['solver'].get('sublattice_configurations'), dtype=np.object).shape[0]
        site_fractions = data['solver'].get('sublattice_occupancies', [[1]] * num_configs)
        site_fraction_product = [reduce(operator.mul, list(itertools.chain(*[np.atleast_1d(f) for f in fracs])), 1)
                                 for fracs in site_fractions]
        # TODO: Subtle sorting bug here, if the interactions aren't already in sorted order...
        interaction_product = []
        for fracs in site_fractions:
            interaction_product.append(float(reduce(operator.mul,
                                                    [f[0] - f[1] for f in fracs if isinstance(f, list) and len(f) == 2],
                                                    1)))
        if len(interaction_product) == 0:
            interaction_product = [0]
        comp_features = zip(site_fraction_product, interaction_product)
        all_samples.extend(list(itertools.product(temperatures, comp_features)))
    return all_samples


def _build_feature_matrix(prop, features, desired_data):
    transformed_features = sympy.Matrix([feature_transforms[prop](i) for i in features])
    all_samples = _get_samples(desired_data)
    feature_matrix = np.empty((len(all_samples), len(transformed_features)), dtype=np.float)
    feature_matrix[:, :] = [transformed_features.subs({v.T: temp, 'YS': compf[0],
                                                       'Z': compf[1]}).evalf()
                            for temp, compf in all_samples]
    return feature_matrix


def _endmembers_from_interaction(configuration):
    config = []
    for c in configuration:
        if isinstance(c, (list, tuple)):
            config.append(c)
        else:
            config.append([c])
    return list(itertools.product(*[tuple(c) for c in config]))


def _shift_reference_state(desired_data, feature_transform, fixed_model):
    "Shift data to a new common reference state."
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


def sigfigs(x, n):
    if x != 0:
        return np.around(x, -(np.floor(np.log10(np.abs(x)))).astype(np.int) + (n - 1))
    else:
        return x


def fit_formation_energy(dbf, comps, phase_name, configuration, symmetry,
                         datasets, features=None):
    """
    Find suitable linear model parameters for the given phase.
    We do this by successively fitting heat capacities, entropies and
    enthalpies of formation, and selecting against criteria to prevent
    overfitting. The "best" set of parameters minimizes the error
    without overfitting.

    Parameters
    ==========
    dbf : Database
        Partially complete, so we know what degrees of freedom to fix.
    comps : list of str
        Names of the relevant components.
    phase_name : str
        Name of the desired phase for which the parameters will be found.
    configuration : ndarray
        Configuration of the sublattices for the fitting procedure.
    symmetry : set of set of int or None
        Symmetry of the sublattice configuration.
    datasets : tinydb of Dataset
        All the datasets desired to fit to.
    features : dict (optional)
        Maps "property" to a list of features for the linear model.
        These will be transformed from "GM" coefficients
        e.g., {"CPM_FORM": (v.T*sympy.log(v.T), v.T**2, v.T**-1, v.T**3)}

    Returns
    =======
    dict of feature: estimated value
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
        print('ENDMEMBERS FROM INTERACTION: '+str(_endmembers_from_interaction(configuration)))
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
        desired_data = _get_data(comps, phase_name, configuration, symmetry, datasets, desired_props)
        print('{}: datasets found: {}'.format(desired_props, len(desired_data)))
        if len(desired_data) > 0:
            # We assume all properties in the same fitting step have the same features (but different ref states)
            feature_matrix = _build_feature_matrix(desired_props[0], features[desired_props[0]], desired_data)
            all_samples = _get_samples(desired_data)
            data_quantities = np.concatenate(_shift_reference_state(desired_data,
                                                                    feature_transforms[desired_props[0]],
                                                                    fixed_model),
                                             axis=-1)
            site_fractions = [_build_sitefractions(phase_name, ds['solver']['sublattice_configurations'],
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


def _translate_endmember_to_array(endmember, variables):
    site_fractions = sorted(variables, key=str)
    frac_array = np.zeros(len(site_fractions))
    for idx, component in enumerate(endmember):
        frac_array[site_fractions.index(v.SiteFraction(site_fractions[0].phase_name, idx, component))] = 1
    return frac_array


def _build_sitefractions(phase_name, sublattice_configurations, sublattice_occupancies):
    """
    Convert nested lists of sublattice configurations and occupancies to a list of dictionaries.
    The dictionaries map SiteFraction symbols to occupancy values. Note that zero occupancy
    site fractions will need to be added separately since the total degrees of freedom aren't
    known in this function.
    :param phase_name:
    :param sublattice_configurations:
    :param sublattice_occupancies:
    :return:
    """
    result = []
    for config, occ in zip(sublattice_configurations, sublattice_occupancies):
        sitefracs = {}
        config = [[c] if not isinstance(c, (list, tuple)) else c for c in config]
        occ = [[o] if not isinstance(o, (list, tuple)) else o for o in occ]
        if len(config) != len(occ):
            raise ValueError('Sublattice configuration length differs from occupancies')
        for sublattice_idx in range(len(config)):
            if isinstance(config[sublattice_idx], (list, tuple)) != isinstance(occ[sublattice_idx], (list, tuple)):
                raise ValueError('Sublattice configuration type differs from occupancies')
            if not isinstance(config[sublattice_idx], (list, tuple)):
                # This sublattice is fully occupied by one component
                sitefracs[v.SiteFraction(phase_name, sublattice_idx, config[sublattice_idx])] = occ[sublattice_idx]
            else:
                # This sublattice is occupied by multiple elements
                if len(config[sublattice_idx]) != len(occ[sublattice_idx]):
                    raise ValueError('Length mismatch in sublattice configuration')
                for comp, val in zip(config[sublattice_idx], occ[sublattice_idx]):
                    sitefracs[v.SiteFraction(phase_name, sublattice_idx, comp)] = val
        result.append(sitefracs)
    return result


def _compare_data_to_parameters(dbf, comps, phase_name, desired_data, mod, configuration, x, y):
    import matplotlib.pyplot as plt
    all_samples = np.array(_get_samples(desired_data), dtype=np.object)
    endpoints = _endmembers_from_interaction(configuration)
    interacting_subls = [c for c in _list_to_tuple(configuration) if isinstance(c, tuple)]
    disordered_config = False
    if (len(set(interacting_subls)) == 1) and (len(interacting_subls[0]) == 2):
        # This configuration describes all sublattices with the same two elements interacting
        # In general this is a high-dimensional space; just plot the diagonal to see the disordered mixing
        endpoints = [endpoints[0], endpoints[-1]]
        disordered_config = True
    fig = plt.figure(figsize=(9, 9))
    bar_chart = False
    bar_labels = []
    bar_data = []
    if y.endswith('_FORM'):
        # We were passed a Model object with zeroed out reference states
        yattr = y[:-5]
    else:
        yattr = y
    if len(endpoints) == 1:
        # This is an endmember so we can just compute T-dependent stuff
        temperatures = np.array([i[0] for i in all_samples], dtype=np.float)
        if temperatures.min() != temperatures.max():
            temperatures = np.linspace(temperatures.min(), temperatures.max(), num=100)
        else:
            # We only have one temperature: let's do a bar chart instead
            bar_chart = True
            temperatures = temperatures.min()
        endmember = _translate_endmember_to_array(endpoints[0], mod.ast.atoms(v.SiteFraction))[None, None]
        predicted_quantities = calculate(dbf, comps, [phase_name], output=yattr,
                                         T=temperatures, P=101325, points=endmember, model=mod, mode='numpy')
        if y == 'HM' and x == 'T':
            # Shift enthalpy data so that value at minimum T is zero
            predicted_quantities[yattr] -= predicted_quantities[yattr].sel(T=temperatures[0]).values.flatten()
        response_data = predicted_quantities[yattr].values.flatten()
        if not bar_chart:
            extra_kwargs = {}
            if len(response_data) < 10:
                extra_kwargs['markersize'] = 20
                extra_kwargs['marker'] = '.'
                extra_kwargs['linestyle'] = 'none'
                extra_kwargs['clip_on'] = False
            fig.gca().plot(temperatures, response_data,
                           label='This work', color='k', **extra_kwargs)
            fig.gca().set_xlabel(plot_mapping.get(x, x))
            fig.gca().set_ylabel(plot_mapping.get(y, y))
        else:
            bar_labels.append('This work')
            bar_data.append(response_data[0])
    elif len(endpoints) == 2:
        # Binary interaction parameter
        first_endpoint = _translate_endmember_to_array(endpoints[0], mod.ast.atoms(v.SiteFraction))
        second_endpoint = _translate_endmember_to_array(endpoints[1], mod.ast.atoms(v.SiteFraction))
        point_matrix = np.linspace(0, 1, num=100)[None].T * second_endpoint + \
            (1 - np.linspace(0, 1, num=100))[None].T * first_endpoint
        # TODO: Real temperature support
        point_matrix = point_matrix[None, None]
        predicted_quantities = calculate(dbf, comps, [phase_name], output=yattr,
                                         T=300, P=101325, points=point_matrix, model=mod, mode='numpy')
        response_data = predicted_quantities[yattr].values.flatten()
        if not bar_chart:
            extra_kwargs = {}
            if len(response_data) < 10:
                extra_kwargs['markersize'] = 20
                extra_kwargs['marker'] = '.'
                extra_kwargs['linestyle'] = 'none'
                extra_kwargs['clip_on'] = False
            fig.gca().plot(np.linspace(0, 1, num=100), response_data,
                           label='This work', color='k', **extra_kwargs)
            fig.gca().set_xlim((0, 1))
            fig.gca().set_xlabel(str(':'.join(endpoints[0])) + ' to ' + str(':'.join(endpoints[1])))
            fig.gca().set_ylabel(plot_mapping.get(y, y))
        else:
            bar_labels.append('This work')
            bar_data.append(response_data[0])
    else:
        raise NotImplementedError('No support for plotting configuration {}'.format(configuration))

    for data in desired_data:
        indep_var_data = None
        response_data = np.zeros_like(data['values'], dtype=np.float)
        if x == 'T' or x == 'P':
            indep_var_data = np.array(data['conditions'][x], dtype=np.float).flatten()
        elif x == 'Z':
            if disordered_config:
                # Take the second element of the first interacting sublattice as the coordinate
                # Because it's disordered all sublattices should be equivalent
                # TODO: Fix this to filter because we need to guarantee the plot points are disordered
                occ = data['solver']['sublattice_occupancies']
                subl_idx = np.nonzero([isinstance(c, (list, tuple)) for c in occ[0]])[0]
                if len(subl_idx) > 1:
                    subl_idx = int(subl_idx[0])
                else:
                    subl_idx = int(subl_idx)
                indep_var_data = [c[subl_idx][1] for c in occ]
            else:
                interactions = np.array([i[1][1] for i in _get_samples([data])], dtype=np.float)
                indep_var_data = 1 - (interactions+1)/2
            if y.endswith('_MIX') and data['output'].endswith('_FORM'):
                # All the _FORM data we have still has the lattice stability contribution
                # Need to zero it out to shift formation data to mixing
                mod_latticeonly = Model(dbf, comps, phase_name, parameters={'GHSER'+c.upper(): 0 for c in comps})
                mod_latticeonly.models = {key: value for key, value in mod_latticeonly.models.items()
                                          if key == 'ref'}
                temps = data['conditions'].get('T', 300)
                pressures = data['conditions'].get('P', 101325)
                points = _build_sitefractions(phase_name, data['solver']['sublattice_configurations'],
                                              data['solver']['sublattice_occupancies'])
                for point_idx in range(len(points)):
                    missing_variables = mod_latticeonly.ast.atoms(v.SiteFraction) - set(points[point_idx].keys())
                    # Set unoccupied values to zero
                    points[point_idx].update({key: 0 for key in missing_variables})
                    # Change entry to a sorted array of site fractions
                    points[point_idx] = list(OrderedDict(sorted(points[point_idx].items(), key=str)).values())
                points = np.array(points, dtype=np.float)
                # TODO: Real temperature support
                points = points[None, None]
                stability = calculate(dbf, comps, [phase_name], output=data['output'][:-5],
                                      T=temps, P=pressures, points=points,
                                      model=mod_latticeonly, mode='numpy')
                response_data -= stability[data['output'][:-5]].values

        response_data += np.array(data['values'], dtype=np.float)
        response_data = response_data.flatten()
        if not bar_chart:
            extra_kwargs = {}
            if len(response_data) < 10:
                extra_kwargs['markersize'] = 20
                extra_kwargs['marker'] = '.'
                extra_kwargs['linestyle'] = 'none'
                extra_kwargs['clip_on'] = False

            fig.gca().plot(indep_var_data, response_data, label=data.get('reference', None),
                           **extra_kwargs)
        else:
            bar_labels.append(data.get('reference', None))
            bar_data.append(response_data[0])
    if bar_chart:
        fig.gca().barh(0.02 * np.arange(len(bar_data)), bar_data,
                       color='k', height=0.01)
        endmember_title = ' to '.join([':'.join(i) for i in endpoints])
        fig.suptitle('{} (T = {} K)'.format(endmember_title, temperatures), fontsize=20)
        fig.gca().set_yticks(0.02 * np.arange(len(bar_data)))
        fig.gca().set_yticklabels(bar_labels, fontsize=20)
        # This bar chart is rotated 90 degrees, so "y" is now x
        fig.gca().set_xlabel(plot_mapping.get(y, y))
    else:
        fig.gca().set_frame_on(False)
        leg = fig.gca().legend(loc='best')
        leg.get_frame().set_edgecolor('black')
    fig.canvas.draw()


def plot_parameters(dbf, comps, phase_name, configuration, symmetry, datasets=None):
    em_plots = [('T', 'CPM'), ('T', 'CPM_FORM'), ('T', 'SM'), ('T', 'SM_FORM'),
                ('T', 'HM'), ('T', 'HM_FORM')]
    mix_plots = [('Z', 'HM_FORM'), ('Z', 'HM_MIX'), ('Z', 'SM_MIX')]
    comps = sorted(comps)
    mod = Model(dbf, comps, phase_name)
    # This is for computing properties of formation
    mod_norefstate = Model(dbf, comps, phase_name, parameters={'GHSER'+c.upper(): 0 for c in comps})
    # Is this an interaction parameter or endmember?
    if any([isinstance(conf, list) or isinstance(conf, tuple) for conf in configuration]):
        plots = mix_plots
    else:
        plots = em_plots
    for x_val, y_val in plots:
        if datasets is not None:
            if y_val.endswith('_MIX'):
                desired_props = [y_val.split('_')[0]+'_FORM', y_val]
            else:
                desired_props = [y_val]
            desired_data = _get_data(comps, phase_name, configuration, symmetry, datasets, desired_props)
        else:
            desired_data = []
        if len(desired_data) == 0:
            continue
        if y_val.endswith('_FORM'):
            _compare_data_to_parameters(dbf, comps, phase_name, desired_data, mod_norefstate, configuration, x_val, y_val)
        else:
            _compare_data_to_parameters(dbf, comps, phase_name, desired_data, mod, configuration, x_val, y_val)


def _list_to_tuple(x):
    def _tuplify(y):
        if isinstance(y, list) or isinstance(y, tuple):
            return tuple(_tuplify(i) if isinstance(i, (list, tuple)) else i for i in y)
        else:
            return y
    return tuple(map(_tuplify, x))


def _tuple_to_list(x):
    def _listify(y):
        if isinstance(y, list) or isinstance(y, tuple):
            return list(y)
        else:
            return y
    return list(map(_listify, x))


def _generate_symmetric_group(configuration, symmetry):
    configurations = [_list_to_tuple(configuration)]
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
    """
    Generate an initial CALPHAD model for a given phase and
    sublattice model.

    Parameters
    ==========
    dbf : Database
        Database to add parameters to.
    phase_name : str
        Name of the phase.
    symmetry : set of set of int or None
        Sublattice model symmetry.
    subl_model : list of tuple
        Sublattice model for the phase of interest.
    site_ratios : list of float
        Number of sites in each sublattice, normalized to one atom.
    datasets : tinydb of datasets
        All datasets to consider for the calculation.
    refdata : dict
        Maps tuple(element, phase_name) -> SymPy object defining energy relative to SER
    aliases : list or None
        Alternative phase names. Useful for matching against reference data or other datasets.
    """
    # First fit endmembers
    all_em_count = len(list(itertools.product(*subl_model)))
    endmembers = sorted(set(canonicalize(i, symmetry) for i in itertools.product(*subl_model)))
    # Number of significant figures in parameters
    numdigits = 6
    em_dict = {}
    aliases = [] if aliases is None else aliases
    aliases = sorted(set(aliases + [phase_name]))
    print('FITTING: ', phase_name)
    print('{0} endmembers ({1} distinct by symmetry)'.format(all_em_count, len(endmembers)))

    def _to_tuple(x):
        if isinstance(x, list) or isinstance(x, tuple):
            return tuple(x)
        else:
            return tuple([x])

    all_endmembers = []
    for endmember in endmembers:
        print('ENDMEMBER: '+str(endmember))
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
                    dbf.symbols[sym_name] = stability
                    break
            if dbf.symbols.get(sym_name, None) is not None:
                num_moles = sum([sites for elem, sites in zip(endmember, site_ratios) if elem != 'VA'])
                fit_eq = num_moles * sympy.Symbol(sym_name)
        if fit_eq is None:
            # No reference lattice stability data -- we have to fit it
            parameters = fit_formation_energy(dbf, sorted(dbf.elements), phase_name, endmember, symmetry, datasets)
            fit_eq = sympy.Add(*[sigfigs(value, numdigits) * key for key, value in parameters.items()])
            ref = 0
            for subl, ratio in zip(endmember, site_ratios):
                if subl == 'VA':
                    continue
                ref = ref + ratio * sympy.Symbol('GHSER'+subl)
            fit_eq += ref
        symmetric_endmembers = _generate_symmetric_group(endmember, symmetry)
        print('SYMMETRIC_ENDMEMBERS: ', symmetric_endmembers)
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

    bin_interactions = sorted(set(canonicalize(i, symmetry) for i in transformed_bin_interactions),
                              key=bin_int_sort_key)
    print('{0} distinct binary interactions'.format(len(bin_interactions)))
    for interaction in bin_interactions:
        ixx = []
        for i in interaction:
            if isinstance(i, (tuple, list)):
                ixx.append(tuple(i))
            else:
                ixx.append(i)
        ixx = tuple(ixx)
        print('INTERACTION: '+str(ixx))
        parameters = fit_formation_energy(dbf, sorted(dbf.elements), phase_name, ixx, symmetry, datasets)
        # Organize parameters by polynomial degree
        degree_polys = np.zeros(10, dtype=np.object)
        for degree in reversed(range(10)):
            check_symbol = sympy.Symbol('YS') * sympy.Symbol('Z')**degree
            keys_to_remove = []
            for key, value in parameters.items():
                if key.has(check_symbol):
                    coef = sigfigs(parameters[key], numdigits) * (key / check_symbol)
                    # Try converting to float to work around a print precision problem with sympy
                    try:
                        coef = float(coef)
                    except TypeError:
                        pass
                    degree_polys[degree] += coef
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                parameters.pop(key)
        print(degree_polys)
        # Insert into database
        symmetric_interactions = _generate_symmetric_group(interaction, symmetry)
        for degree in np.arange(degree_polys.shape[0]):
            if degree_polys[degree] != 0:
                for syminter in symmetric_interactions:
                    dbf.add_parameter('L', phase_name, tuple(map(_to_tuple, syminter)), degree, degree_polys[degree])
    # Now fit ternary interactions


def multi_plot(dbf, comps, phases, datasets, ax=None):
    import matplotlib.pyplot as plt
    plots = [('ZPF', 'T')]
    real_components = sorted(set(comps) - {'VA'})
    legend_handles, phase_color_map = phase_legend(phases)
    for output, indep_var in plots:
        desired_data = datasets.search((tinydb.where('output') == output) &
                                       (tinydb.where('components').test(lambda x: set(x).issubset(comps))) &
                                       (tinydb.where('phases').test(lambda x: len(set(phases).intersection(x)) > 0)))
        ax = ax if ax is not None else plt.gca()
        # TODO: There are lot of ways this could break in multi-component situations
        chosen_comp = real_components[-1]
        ax.set_xlabel('X({})'.format(chosen_comp))
        ax.set_ylabel(indep_var)
        ax.set_xlim((0, 1))
        symbol_map = {1: "o", 2: "s", 3: "^"}
        for data in desired_data:
            payload = data['values']
            # TODO: Add broadcast_conditions support
            # Repeat the temperature (or whatever variable) vector to align with the unraveled data
            temp_repeats = np.zeros(len(np.atleast_1d(data['conditions'][indep_var])), dtype=np.int)
            for idx, x in enumerate(payload):
                temp_repeats[idx] = len(x)
            temps_ravelled = np.repeat(data['conditions'][indep_var], temp_repeats)
            payload_ravelled = []
            phases_ravelled = []
            comps_ravelled = []
            symbols_ravelled = []
            # TODO: Fix to only include equilibria listed in 'phases'
            for p in payload:
                symbols_ravelled.extend([symbol_map[len(p)]] * len(p))
                payload_ravelled.extend(p)
            for rp in payload_ravelled:
                phases_ravelled.append(rp[0])
                comp_dict = dict(zip([x.upper() for x in rp[1]], rp[2]))
                dependent_comp = list(set(real_components) - set(comp_dict.keys()))
                if len(dependent_comp) > 1:
                    raise ValueError('Dependent components greater than one')
                elif len(dependent_comp) == 1:
                    dependent_comp = dependent_comp[0]
                    # TODO: Assuming N=1
                    comp_dict[dependent_comp] = 1 - sum(np.array(list(comp_dict.values()), dtype=np.float))
                chosen_comp_value = comp_dict[chosen_comp]
                comps_ravelled.append(chosen_comp_value)
            symbols_ravelled = np.array(symbols_ravelled)
            comps_ravelled = np.array(comps_ravelled)
            temps_ravelled = np.array(temps_ravelled)
            phases_ravelled = np.array(phases_ravelled)
            # We can't pass an array of markers to scatter, sadly
            for sym in symbols_ravelled:
                selected = symbols_ravelled == sym
                ax.scatter(comps_ravelled[selected], temps_ravelled[selected], marker=sym, s=100,
                           c='none', edgecolors=[phase_color_map[x] for x in phases_ravelled[selected]])
        ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1, 0.5))


def _sublattice_model_to_variables(phase_name, subl_model):
    """
    Convert a sublattice model to a list of variables.
    """
    result = []
    for idx, subl in enumerate(subl_model):
        result.extend([v.SiteFraction(phase_name, idx, s) for s in sorted(subl, key=str)])
    return result


def choose_interaction(phase_name, subl_model, site_fractions):
    """
    Get the interaction or end-member with the largest contribution to the given site fractions.
    """
    subl_model = copy.deepcopy(subl_model)
    for idx in range(len(subl_model)):
        if isinstance(subl_model[idx], set):
            subl_model[idx] = sorted(subl_model[idx])
    #print('SUBL_MODEL', subl_model)
    #print('SITE_FRACTIONS', site_fractions)
    sitefrac_dict = dict(zip(_sublattice_model_to_variables(phase_name, subl_model), site_fractions))
    #print('SITEFRAC_DICT', sitefrac_dict)
    all_endmembers = sorted(set(i for i in itertools.product(*subl_model)))
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

    bin_interactions = sorted(set(tuple(i) for i in transformed_bin_interactions),
                              key=bin_int_sort_key)
    # Return interaction with largest numerical value for the given site fractions
    important_interaction = None
    important_k = 0
    interaction_score = 0
    for endmember in all_endmembers:
        em_prod = sympy.Mul(*[v.SiteFraction(phase_name, idx, s) for idx, s in enumerate(endmember)])
        #print('EM_PROD', em_prod)
        em_prod = em_prod.subs(sitefrac_dict).evalf()
        #print('ENDMEMBER', endmember, '- SCORE', em_prod)
        if em_prod > interaction_score:
            interaction_score = em_prod
            important_interaction = endmember
            important_k = 0
    max_value = {0: 0.5, 1: 1./6*(3**0.5), 2: 1./16, 3: 0.0464758}
    for interaction in bin_interactions:
        interacting_subls = sum([isinstance(subl, (list, tuple, set)) for subl in interaction])
        for k in range(2):
            in_prod = sympy.S.One
            for idx, subl in enumerate(interaction):
                if isinstance(subl, (list, tuple)):
                    in_prod *= sympy.Mul(*[v.SiteFraction(phase_name, idx, s) for s in subl])
                    # Only works for binary interactions
                    in_prod *= (v.SiteFraction(phase_name, idx, subl[0]) - v.SiteFraction(phase_name, idx, subl[1])) ** k
                else:
                    in_prod *= v.SiteFraction(phase_name, idx, subl)
            in_prod = np.abs(in_prod.subs(sitefrac_dict).evalf())
            # Rescale to be percent of maximum value
            # TODO: Not sure if this scaling is correct for reciprocal interactions
            in_prod /= (interacting_subls * (max_value[k] ** interacting_subls))
            #print('INTERACTION', interaction, ' (k={})'.format(k), '- SCORE', in_prod)
            if in_prod > interaction_score:
                interaction_score = in_prod
                important_interaction = interaction
                important_k = k
    return important_interaction, important_k


def multi_phase_fit(dbf, comps, phases, inpd, datasets, x, param_symbols=None):
    desired_data = datasets.search((tinydb.where('output') == 'ZPF') &
                                   (tinydb.where('components').test(lambda x: set(x).issubset(comps))) &
                                   (tinydb.where('phases').test(lambda x: len(set(phases).intersection(x)) > 0)))
    phase_errors = tinydb.TinyDB(storage=tinydb.storages.MemoryStorage)
    phase_interactions = Counter()
    if param_symbols is not None:
        param_dict = dict(zip(param_symbols, x))
        print('PARAM DICT', param_dict)
    else:
        param_dict = None
    phase_models = {phase: Model(dbf, comps, phase, parameters=param_dict) for phase in phases}
    phase_obj_callables = {}
    phase_grad_callables = {}
    phase_hess_callables = {}
    for phase, phase_mod in phase_models.items():
        phase_obj_callables[phase], phase_grad_callables[phase], phase_hess_callables[phase] = \
            build_functions(phase_mod.GM, [v.P, v.T] + phase_mod.site_fractions)
    for data in desired_data:
        payload = data['values']
        conditions = data['conditions']
        broadcast = data.get('broadcast_conditions', False)
        #print(conditions)
        phase_regions = defaultdict(lambda: list())
        # TODO: Fix to only include equilibria listed in 'phases'
        for idx, p in enumerate(payload):
            phase_key = tuple(sorted(rp[0] for rp in p))
            if len(phase_key) < 2:
                # Skip single-phase regions for fitting purposes
                continue
            # Need to sort 'p' here so we have the sorted ordering used in 'phase_key'
            #print('SORTED P', sorted(p, key=operator.itemgetter(0)))
            comp_dicts = [dict(zip([v.X(x.upper()) for x in rp[1]], rp[2]))
                          for rp in sorted(p, key=operator.itemgetter(0))]
            cur_conds = {}
            for key, value in conditions.items():
                value = np.atleast_1d(np.asarray(value))
                if len(value) > 1:
                    value = value[idx]
                cur_conds[getattr(v, key)] = float(value)
            phase_regions[phase_key].append((cur_conds, comp_dicts))
        print('PHASE_REGIONS', phase_regions)
        for region, region_eq in phase_regions.items():
            print('REGION', region)
            for req in region_eq:
                # We are now considering a particular tie region
                current_statevars, comp_dicts = req
                region_chemical_potentials = []
                for cond_dict in comp_dicts:
                    # We are now considering a particular tie vertex
                    for key, val in cond_dict.items():
                        if val is None:
                            cond_dict[key] = np.nan
                    cond_dict.update(current_statevars)
                    print('COND_DICT (MULTI)', cond_dict)
                    if np.any(np.isnan(list(cond_dict.values()))):
                        # This composition is unknown -- it doesn't contribute to hyperplane estimation
                        pass
                    else:
                        # Extract chemical potential hyperplane from multi-phase calculation
                        # Note that we consider all phases in the system, not just ones in this tie region
                        multi_eqdata = equilibrium(dbf, comps, phases, cond_dict, pbar=False, verbose=False,
                                                   callables=phase_obj_callables, grad_callables=phase_grad_callables,
                                                   hess_callables=phase_hess_callables)
                        print('MULTI_EQDATA', multi_eqdata)
                        # Does there exist only a single phase in the result with zero internal degrees of freedom?
                        # We should exclude those chemical potentials from the average because they are meaningless.
                        num_phases = len(np.squeeze(multi_eqdata['Phase'].values != ''))
                        zero_dof = np.all((multi_eqdata['Y'].values == 1.) | np.isnan(multi_eqdata['Y'].values))
                        if (num_phases == 1) and zero_dof:
                            region_chemical_potentials.append(np.full_like(np.squeeze(multi_eqdata['MU'].values), np.nan))
                        else:
                            region_chemical_potentials.append(np.squeeze(multi_eqdata['MU'].values))
                print('REGION_CHEMICAL_POTENTIALS', region_chemical_potentials)
                region_chemical_potentials = np.nanmean(region_chemical_potentials, axis=0, dtype=np.float)
                print('REGION_CHEMICAL_POTENTIALS', region_chemical_potentials)
                # Now perform the equilibrium calculation for the isolated phases and add the result to the error record
                for current_phase, cond_dict in zip(region, comp_dicts):
                    # We are now considering a particular tie vertex
                    for key, val in cond_dict.items():
                        if val is None:
                            cond_dict[key] = np.nan
                    cond_dict.update(current_statevars)
                    print('COND_DICT ({})'.format(current_phase), cond_dict)
                    if np.any(np.isnan(list(cond_dict.values()))):
                        # We don't actually know the phase composition here, so we estimate it
                        single_eqdata = calculate(dbf, comps, [current_phase], mode='numpy', output='GM', pdens=1000,
                                                  T=cond_dict[v.T], P=cond_dict[v.P])
                        print('SINGLE_EQDATA (UNKNOWN COMP)', single_eqdata)
                        driving_force = np.multiply(region_chemical_potentials,
                                                    single_eqdata['X'].values).sum(axis=-1) - single_eqdata['GM'].values
                        desired_sitefracs = single_eqdata['Y'].values[..., np.argmax(driving_force), :]
                        error = float(driving_force.max())
                    else:
                        # Extract energies from single-phase calculations
                        single_eqdata = equilibrium(dbf, comps, [current_phase], cond_dict, pbar=False, verbose=False,
                                                    callables=phase_obj_callables, grad_callables=phase_grad_callables,
                                                    hess_callables=phase_hess_callables)
                        print('SINGLE_EQDATA', single_eqdata)
                        # Sometimes we can get a miscibility gap in our "single-phase" calculation
                        # Choose the weighted mixture of site fractions
                        print('Y FRACTIONS', single_eqdata['Y'].values)
                        phases_idx = np.nonzero(~np.isnan(np.squeeze(single_eqdata['NP'].values)))
                        cur_vertex = np.nanargmax(np.squeeze(single_eqdata['NP'].values))
                        desired_sitefracs = np.multiply(single_eqdata['NP'].values[..., phases_idx, np.newaxis],
                                                        single_eqdata['Y'].values[..., phases_idx, :]).sum(axis=-2)
                        #desired_sitefracs = single_eqdata['Y'].values[..., cur_vertex, :]
                        select_energy = float(single_eqdata['GM'].values)
                        region_comps = []
                        for comp in [c for c in sorted(comps) if c != 'VA']:
                            region_comps.append(cond_dict.get(v.X(comp), np.nan))
                        region_comps[region_comps.index(np.nan)] = 1 - np.nansum(region_comps)
                        print('REGION_COMPS', region_comps)
                        error = np.multiply(region_chemical_potentials, region_comps).sum() - select_energy
                        error = float(error)
                    print('DESIRED_SITEFRACS', desired_sitefracs)

                    best_interaction = choose_interaction(current_phase, phase_models[current_phase].constituents,
                                                          np.squeeze(desired_sitefracs))
                    # best_interaction returns a tuple of interaction, interaction_k
                    num_sites = _site_ratio_normalization(dbf.phases[current_phase])
                    symmetry = inpd['phases'][current_phase].get('equivalent_sublattices', None)
                    constituent_array = canonicalize(best_interaction[0], symmetry)
                    # Need to get indices of swapped sublattices, then swap them around in the sitefrac array
                    desired_sitefracs = np.squeeze(desired_sitefracs)
                    if symmetry is not None:
                        sidx = 0
                        symslices = []
                        for subl in phase_models[current_phase].constituents:
                            symslices.append(slice(sidx, sidx+len(subl)))
                            sidx += len(subl)
                        print('OLD_SITEFRACS', desired_sitefracs)
                        for subls in symmetry:
                            subls = sorted(subls)
                            print('SUBLS', subls)
                            items = [best_interaction[0][i] for i in subls]
                            print('ITEMS', items)
                            sorted_subls = sorted(subls, key=lambda x: canonical_sort_key(items[x]))
                            print('SORTED SUBLS', sorted_subls)
                            old_sitefracs = [copy.deepcopy(desired_sitefracs[sl])
                                             for sl in [symslices[i] for i in subls]]
                            old_sitefracs = [old_sitefracs[i] for i in sorted_subls]
                            print('OLD_SF', old_sitefracs)
                            for old_sf, new_subl in zip(old_sitefracs, subls):
                                print('desired_sitefracs[{}] = {}'.format(str(symslices[new_subl]), old_sf))
                                desired_sitefracs[symslices[new_subl]] = old_sf
                            print('NEW_SITEFRACS', desired_sitefracs)
                    symmetric_interactions = _generate_symmetric_group(constituent_array, symmetry)
                    interarray = sympy.S.Zero
                    for syminter in symmetric_interactions:
                        print('SYMINTER', syminter)
                        symarray = sympy.S.One
                        for idx, subl in enumerate(syminter):
                            if isinstance(subl, (list, tuple)):
                                symarray *= sympy.Mul(*[v.SiteFraction(current_phase, idx, s) for s in subl])
                                # Only works for binary interactions and assumes canonicalized
                                # If not we could end up with sign errors in binary parameters
                                symarray *= (v.SiteFraction(current_phase, idx, subl[0]) -
                                             v.SiteFraction(current_phase, idx, subl[1])) ** best_interaction[1]
                            else:
                                symarray *= v.SiteFraction(current_phase, idx, subl)
                        interarray += symarray
                    sorted_constituents = phase_models[current_phase].constituents
                    for idx in range(len(sorted_constituents)):
                        if isinstance(sorted_constituents[idx], set):
                            sorted_constituents[idx] = sorted(sorted_constituents[idx])
                    sitefrac_dict = dict(zip(_sublattice_model_to_variables(current_phase,
                                                                            sorted_constituents),
                                             desired_sitefracs))
                    error_record = {'phase_name': current_phase, 'parameter_order': best_interaction[1],
                                    'constituent_array': constituent_array, 'components': tuple(comps),
                                    'error': error, 'num_sites': num_sites,
                                    'intercoef': interarray, 'site_fractions': sitefrac_dict}
                    error_record.update(current_statevars)
                    phase_errors.insert(error_record)
                    phase_interactions[(current_phase, constituent_array, best_interaction[1])] += 1
    return phase_errors, phase_interactions


def _generate_phase_features(phase_name, symmetry, subl_model, num_sites):
    """
    Generate a list of all possible features for a phase in our solution matrix.
    """
    max_parameter_order = 3
    endmembers = sorted(set(canonicalize(i, symmetry) for i in itertools.product(*subl_model)))
    all_endmembers = []
    #all_desired_coefficients = [v.T**n for n in range(-1, 3)] + [v.T*sympy.log(v.T)]
    all_desired_coefficients = [v.T**n for n in range(0, 2)]
    # Key is tuple of form (phase_name, interaction, parameter_order, coefficient)
    interaction_to_sympy_object = OrderedDict()
    for endmember in endmembers:
        symmetric_endmembers = _generate_symmetric_group(endmember, symmetry)
        all_endmembers.extend(symmetric_endmembers)
        interarray = sympy.S.Zero
        for em in symmetric_endmembers:
            symarray = sympy.S.One
            for idx, subl in enumerate(em):
                    symarray *= v.SiteFraction(phase_name, idx, subl)
            interarray += symarray
        for coef in all_desired_coefficients:
            interaction_to_sympy_object[(phase_name, endmember, 0, coef)] = coef*interarray/num_sites

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

    bin_interactions = sorted(set(tuple(i) for i in transformed_bin_interactions),
                              key=bin_int_sort_key)
    # placeholder for parameter order
    k = sympy.Symbol('k')
    for constituent_array in bin_interactions:
        symmetric_interactions = _generate_symmetric_group(constituent_array, symmetry)
        interarray = sympy.S.Zero
        for syminter in symmetric_interactions:
            symarray = sympy.S.One
            for idx, subl in enumerate(syminter):
                if isinstance(subl, (list, tuple)):
                    symarray *= sympy.Mul(*[v.SiteFraction(phase_name, idx, s) for s in subl])
                    # Only works for binary interactions and assumes canonicalized
                    # If not we could end up with sign errors in binary parameters
                    symarray *= (v.SiteFraction(phase_name, idx, subl[0]) -
                                 v.SiteFraction(phase_name, idx, subl[1])) ** k
                else:
                    symarray *= v.SiteFraction(phase_name, idx, subl)
            interarray += symarray
        for parameter_order in range(max_parameter_order+1):
            for coef in all_desired_coefficients:
                interaction_to_sympy_object[(phase_name, constituent_array, parameter_order, coef)] = \
                    coef * interarray.xreplace({k: parameter_order}) / num_sites
    return interaction_to_sympy_object


def _site_ratio_normalization(phase):
    """
    Calculates the normalization factor based on the number of sites
    in each sublattice.
    """
    site_ratio_normalization = sympy.S.Zero
    # Normalize by the sum of site ratios times a factor
    # related to the site fraction of vacancies
    comps = set()
    for consts in phase.constituents:
        comps |= consts
    for idx, sublattice in enumerate(phase.constituents):
        if ('VA' in set(sublattice)) and ('VA' in comps):
            site_ratio_normalization += phase.sublattices[idx] * \
                (1 - v.SiteFraction(phase.name, idx, 'VA'))
        else:
            site_ratio_normalization += phase.sublattices[idx]
    return site_ratio_normalization


def fit(input_fname, datasets, saveall=True, resume=None):
    """
    Fit thermodynamic and phase equilibria data to a model.

    Parameters
    ==========
    input_fname : str
        Filename for input JSON configuration file.
    datasets : tinydb
    saveall : bool, optional
        If True, write TDB out at each iteration.
    resume : Database, optional
        If specified, start multi-phase fitting using this Database.
        Useful for resuming calculations from Databases generated by 'saveall'.

    Returns
    =======
    dbf : Database
    """
    start_time = datetime.utcnow()
    # TODO: Validate input JSON
    data = json.load(open(input_fname))
    if resume is None:
        dbf = Database()
        dbf.elements = sorted(data['components'])
        # Write reference state to Database
        refdata = getattr(pycalphad.refdata, data['refdata'])
        stabledata = getattr(pycalphad.refdata, data['refdata']+'Stable')
        comp_refs = {c.upper(): stabledata[c.upper()] for c in dbf.elements if c.upper() != 'VA'}
        comp_refs['VA'] = 0
        dbf.symbols.update({'GHSER'+c.upper(): data for c, data in comp_refs.items()})
        for phase_name, phase_obj in data['phases'].items():
            # Perform parameter selection and single-phase fitting based on input
            # TODO: Need to pass particular models to include: magnetic, order-disorder, etc.
            symmetry = phase_obj.get('equivalent_sublattices', None)
            aliases = phase_obj.get('aliases', None)
            # TODO: More advanced phase data searching
            site_ratios = phase_obj['sublattice_site_ratios']
            subl_model = phase_obj['sublattice_model']
            dbf.add_phase(phase_name, {}, site_ratios)
            dbf.add_phase_constituents(phase_name, subl_model)
            # phase_fit() adds parameters to dbf
            phase_fit(dbf, phase_name, symmetry, subl_model, site_ratios, datasets, refdata, aliases=aliases)
    else:
        print('STARTING FROM USER-SPECIFIED DATABASE')
        dbf = resume

    if saveall:
        dbf.to_file('{}-{}-iter{}.tdb'.format(start_time,
                                              ''.join([c for c in sorted(data['components']) if c != 'VA']), 0),
                    fmt='tdb')

    max_iterations = 10

    def abstol(x, tol):
        return np.abs(x) > tol

    def update_param(coef, pnum, value, database):
        print('UPDATE_PARAM', (coef, pnum, value))
        def transform(element):
            if isinstance(element['parameter'], sympy.Piecewise):
                newvals = [(arg.expr + coef * value, arg.cond) for arg in element['parameter'].args]
                newval = sympy.Piecewise(*newvals)
            else:
                newval = element['parameter'] + coef * value
            element['parameter'] = newval
            return element
        return transform

    min_error = 1e-3
    pcount = 0
    stepsize = 1
    converged = False

    for iteration in range(max_iterations):
        print('MAIN ITERATION ', iteration+1, flush=True)
        errors, interactions = multi_phase_fit(dbf, dbf.elements, sorted(data['phases'].keys()), data, datasets, None)
        print(errors.all())
        print(interactions)
        features = OrderedDict()
        data_rows = 0
        for phase_name in sorted(data['phases'].keys()):
            symmetry = data['phases'][phase_name].get('equivalent_sublattices', None)
            subl_model = data['phases'][phase_name]['sublattice_model']
            num_sites = _site_ratio_normalization(dbf.phases[phase_name])
            new_features = _generate_phase_features(phase_name, symmetry, subl_model, num_sites)
            print(phase_name, len(new_features), 'features')
            features.update(new_features)
        #print('FEATURES', list(features.keys()))
        dof_columns = len(features)
        for interaction in interactions.keys():
            phase_name, constituent_array, parameter_order = interaction
            intercount = errors.count((tinydb.where('constituent_array') == constituent_array) &
                                      (tinydb.where('parameter_order') == parameter_order) &
                                      (tinydb.where('phase_name') == phase_name) &
                                      (tinydb.where('error').test(abstol, min_error)))
            data_rows += intercount
        # Now add all the single-phase data points
        # We add these to help the optimizer stay in a reasonable range of solutions
        # Remember to apply the feature transforms for the relevant thermodynamic property
        comps = sorted(dbf.elements)
        phases = sorted(dbf.phases.keys())
        dq = OrderedDict()
        # Use weights to get residuals to the same order of magnitude
        data_multipliers = {'CPM': 100, 'SM': 100, 'HM': 1}
        for phase_name in phases:
            desired_props = ["CPM_FORM", "CPM_MIX", "SM_FORM", "SM_MIX", "HM_FORM", "HM_MIX"]
            # Subtract out all of these contributions (zero out reference state because these are formation properties)
            fixed_model = Model(dbf, comps, phase_name, parameters={'GHSER'+c.upper(): 0 for c in comps})
            fixed_model.models['idmix'] = 0
            # TODO: What about phase name aliases?
            desired_data = datasets.search((tinydb.where('output').test(lambda k: k in desired_props)) &
                                           (tinydb.where('components').test(lambda k: set(k).issubset(comps))) &
                                           (tinydb.where('solver').test(lambda k: k.get('mode', None) == 'manual')) &
                                           (tinydb.where('phases') == [phase_name]))
            #print('DESIRED_DATA', desired_data)
            if len(desired_data) == 0:
                continue
            total_data = 0
            for idx, dd in enumerate(desired_data):
                temp_filter = np.nonzero(np.atleast_1d(dd['conditions']['T']) >= 298.15)
                desired_data[idx]['conditions']['T'] = np.atleast_1d(dd['conditions']['T'])[temp_filter]
                # Don't use data['values'] because we rewrote it above; not sure what 'data' references now
                desired_data[idx]['values'] = np.asarray(desired_data[idx]['values'])[..., temp_filter, :]
                total_data += len(desired_data[idx]['values'].flat)
            #print('TOTAL DATA', total_data)
            all_samples = _get_samples(desired_data)
            #print('LEN ALL SAMPLES', len(all_samples))
            #print('ALL SAMPLES', all_samples)
            assert len(all_samples) == total_data
            data_quantities = [np.concatenate(_shift_reference_state([ds],
                                                                     feature_transforms[ds['output']],
                                                                     fixed_model), axis=-1)
                               for ds in desired_data]
            # Flatten list
            data_quantities = np.asarray(list(itertools.chain(*data_quantities)), dtype=np.object)
            site_fractions = [_build_sitefractions(phase_name, ds['solver']['sublattice_configurations'],
                ds['solver'].get('sublattice_occupancies',
                                 np.ones((len(ds['solver']['sublattice_configurations']),
                                          len(ds['solver']['sublattice_configurations'][0])), dtype=np.float)))
                              for ds in desired_data for _ in np.atleast_1d(ds['conditions']['T'])]
            # Flatten list
            site_fractions = list(itertools.chain(*site_fractions))
            #print('SITE_FRACTIONS', site_fractions)
            # Add dependent site fractions to dictionary
            for idx in range(len(site_fractions)):
                sf = site_fractions[idx]
                for subl_idx, subl_species in enumerate(dbf.phases[phase_name].constituents):
                    for spec in subl_species:
                        if v.SiteFraction(phase_name, subl_idx, spec) not in sf.keys():
                            sfsum = sum([val for key, val in sf.items() if key.sublattice_index == subl_idx])
                            sfsum = max(sfsum, 1e-16)
                            site_fractions[idx][v.Y(phase_name, subl_idx, spec)] = 1 - sfsum
            # Remove existing partial model contributions from the data
            #print('DATA_QUANTITIES 1', data_quantities)
            data_quantities = data_quantities - np.repeat([feature_transforms[ds['output']](fixed_model.GM)
                                                           for ds in desired_data],
                                                          [len(ds['values'].flat) for ds in desired_data])
            print('LEN DATA_QUANTITIES', len(data_quantities))
            #print('DATA_QUANTITIES 2', data_quantities)
            assert len(data_quantities) == total_data
            data_quantities = [sympy.S(i).xreplace(sf).xreplace({v.T: ixx[0]}).evalf()
                               for i, sf, ixx in zip(data_quantities, site_fractions, all_samples)]
            #print('LEN DATA_QUANTITIES', len(data_quantities))
            #print('DATA_QUANTITIES 3', data_quantities)
            data_quantities = np.array(data_quantities, dtype=np.float)
            # Reweight data based on the output type
            multiply_array = [np.repeat(data_multipliers[ds['output'].split('_')[0]], len(ds['values'].flat))
                              for ds in desired_data]
            multiply_array = list(itertools.chain(*multiply_array))
            #print('LEN MULTIPLY_ARRAY', len(multiply_array))
            #print('LEN DATA_QUANTITIES', len(data_quantities))
            #data_quantities *= np.asarray(multiply_array, dtype=np.float)
            output_types = [np.repeat(ds['output'], len(ds['values'].flat)) for ds in desired_data]
            output_types = list(itertools.chain(*output_types))
            dq[phase_name] = (data_quantities, site_fractions, all_samples, output_types)
        total_singlephase_dq = sum([len(x[0]) for x in dq.values()])
        solution_matrix = np.zeros((data_rows + total_singlephase_dq, dof_columns), dtype=np.object)
        error_vector = np.zeros((data_rows + total_singlephase_dq,), dtype=np.float)
        for idx, feature in enumerate(features.values()):
            solution_matrix[:, idx] = feature
        iter_idx = 0
        for x in dq.values():
            error_vector[data_rows+iter_idx:data_rows+iter_idx+len(x[0])] = x[0]
            iter_idx += len(x[0])
        data_idx = 0
        all_records = []
        for interaction in interactions.keys():
            phase_name, constituent_array, parameter_order = interaction
            records = errors.search((tinydb.where('constituent_array') == constituent_array) &
                                    (tinydb.where('parameter_order') == parameter_order) &
                                    (tinydb.where('phase_name') == phase_name) &
                                    (tinydb.where('error').test(abstol, min_error)))
            all_records.extend(records)
            for record in records:
                error_vector[data_idx] = stepsize * record['error']
                data_idx += 1
        for rec_idx, record in enumerate(all_records):
            # XXX: Assumes all features are only single-phase
            phase_name = record['phase_name']
            same_phase = np.array([x[0] == phase_name for x in features.keys()])
            nonzero_features = np.nonzero(same_phase)[0]
            zero_features = np.nonzero(~same_phase)[0]
            solution_matrix[rec_idx, zero_features] = 0
            for col_idx in nonzero_features:
                solution_matrix[rec_idx, col_idx] = \
                    sympy.S(sympy.S(solution_matrix[rec_idx, col_idx]).xreplace(record)).xreplace(record['site_fractions'])
        rec_idx = len(all_records)
        for x in dq.values():
            data_quantities, site_fractions, all_samples, output_types = x
            #print('DQ, SF, AS, OUT', x)
            for i, sf, ixx, output_type in zip(data_quantities, site_fractions, all_samples, output_types):
                # XXX: Assumes all features are only single-phase
                phase_name = list(sf)[0].phase_name
                same_phase = np.array([x[0] == phase_name for x in features.keys()])
                nonzero_features = np.nonzero(same_phase)[0]
                zero_features = np.nonzero(~same_phase)[0]
                solution_matrix[rec_idx, zero_features] = 0
                for col_idx in nonzero_features:
                    # TODO: This will break if feature_transforms modifies site fraction dof!
                    solution_matrix[rec_idx, col_idx] = \
                        sympy.S(feature_transforms[output_type](sympy.S(solution_matrix[rec_idx, col_idx]).xreplace(sf))).xreplace({v.T: ixx[0]})
                rec_idx += 1
        #print('SOLUTION MATRIX (SYMBOLIC)', solution_matrix, flush=True)
        solution_matrix = solution_matrix.astype(np.float)
        #print('FEATURES', list(features.keys()))
        #print('SOLUTION MATRIX', solution_matrix, flush=True)
        #print('ERROR VECTOR', error_vector, flush=True)
        import functools
        if solution_matrix.shape[0] <= 2:
            reg = Lasso(fit_intercept=False, normalize=False, max_iter=1000)
        else:
            reg = LassoCV(n_alphas=100, fit_intercept=False, normalize=False, max_iter=1000)
        multi_solver = Pipeline([('scaler', StandardScaler(with_mean=False, with_std=True)),
                                 ('downselect', SelectKBest(score_func=functools.partial(f_regression,
                                                                                         center=False),
                                                            k='all')),
                                 ('lasso', reg)
                                 ])
        multi_solver.fit(solution_matrix[data_rows:, :], error_vector[data_rows:])
        if isinstance(reg, LassoCV):
            print('CHOSEN ALPHA', multi_solver.named_steps['lasso'].alpha_)

        multi_solver.fit(solution_matrix, error_vector)
        selected_feature_indices = np.nonzero(np.atleast_1d(multi_solver.steps[-1][1].coef_))[0]
        if len(selected_feature_indices) > 0:
            print('SELECTED FEATURE INDICES', selected_feature_indices)
            #print('SOLUTION MATRIX (SCALED)',
            #      multi_solver.named_steps['scaler'].transform(solution_matrix)[:, selected_feature_indices], flush=True)
            #print('SOLUTION MATRIX', solution_matrix, flush=True)
            #print('SOLUTION MATRIX', solution_matrix[:, selected_feature_indices], flush=True)
            new_features = operator.itemgetter(*selected_feature_indices)(list(features.keys()))
            if len(selected_feature_indices) == 1:
                new_features = [new_features]
            # Because we rescaled our features, we need to descale the result
            selected_param_values = np.atleast_1d(multi_solver.steps[-1][1].coef_)[selected_feature_indices]
            scales = multi_solver.named_steps['scaler'].scale_[selected_feature_indices]
            print('SCALER', scales)
            new_params = list(zip(new_features, selected_param_values / scales))
            #print('DIFF', solution_matrix[:, selected_feature_indices].dot(np.atleast_1d(multi_solver.steps[-1][1].coef_)) - error_vector)
        else:
            new_params = []
            converged = True

        print('NEW_PARAMS', new_params)
        score = multi_solver.score(solution_matrix, error_vector)
        if score < 0.1:
            # Not making good progress
            new_params = []
            converged = True
        print('SCORE', multi_solver.score(solution_matrix, error_vector))

        # Commit new parameter values to the database
        for param, val in new_params:
            current_phase = param[0]
            parameter_order = param[2]
            # single components in sublattices need to be wrapped in a tuple so they'll match in the database
            symmetry = data['phases'][current_phase].get('equivalent_sublattices', None)
            symmetric_interactions = _generate_symmetric_group(param[1], symmetry)
            for syminter in symmetric_interactions:
                const_array = []
                for subl in syminter:
                    if isinstance(subl, (tuple, list, set)):
                        const_array.append(subl)
                    else:
                        const_array.append((subl,))
                const_array = tuple(const_array)
                param_query = ((tinydb.where('phase_name') == current_phase) &
                               (tinydb.where('parameter_order') == parameter_order) &
                               ((tinydb.where('parameter_type') == 'G') |
                                (tinydb.where('parameter_type') == 'L')) &
                               (tinydb.where('constituent_array') == const_array)
                               )
                result = dbf.search(param_query)
                if len(result) == 0:
                    dbf.add_parameter('G', current_phase, const_array, parameter_order, sympy.S.Zero)
                dbf._parameters.update(update_param(param[3], pcount, val, dbf), param_query)
            pcount += 1
        if saveall:
            dbf.to_file('{}-{}-iter{}.tdb'.format(start_time,
                                                  ''.join([c for c in sorted(comps) if c != 'VA']), iteration+1),
                        fmt='tdb')
        if converged:
            print('CONVERGED')
            break
    return dbf
