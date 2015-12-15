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
from pycalphad import calculate, Database, Model
from pycalphad.refstates import SGTE91
from sklearn.linear_model import LinearRegression
import tinydb
import sympy
import numpy as np
import json
from collections import OrderedDict
import itertools
import operator
import copy
from functools import reduce


# Mapping of energy polynomial coefficients to corresponding property coefficients
feature_transforms = {"CPM_FORM": lambda x: -v.T*sympy.diff(x, v.T, 2),
                      "CPM": lambda x: -v.T*sympy.diff(x, v.T, 2),
                      "SM_FORM": lambda x: -sympy.diff(x, v.T),
                      "SM": lambda x: -sympy.diff(x, v.T),
                      "HM_FORM": lambda x: x - v.T*sympy.diff(x, v.T),
                      "HM": lambda x: x - v.T*sympy.diff(x, v.T)}

plot_mapping = {
    'T': 'Temperature (K)',
    'CPM': 'Heat Capacity (J/K-mol-atom)',
    'HM': 'Enthalpy (J/mol-atom)',
    'SM': 'Entropy (J/K-mol-atom)',
    'CPM_FORM': 'Heat Capacity of Formation (J/K-mol-atom)',
    'HM_FORM': 'Enthalpy of Formation (J/mol-atom)',
    'SM_FORM': 'Entropy of Formation (J/K-mol-atom)'
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


def _symmetry_filter(x, config):
    if x['mode'] == 'manual':
        if len(config) != len(x['sublattice_configurations'][0]):
            return False
        # If even one matches, it's a match
        # We do more filtering downstream
        for data_config in x['sublattice_configurations']:
            if config == data_config:
                return True
    return False


def _get_data(comps, phase_name, configuration, datasets, prop):
    configuration = list(configuration)
    desired_data = datasets.search((tinydb.where('output') == prop) &
                                   (tinydb.where('components') == comps) &
                                   (tinydb.where('solver').test(_symmetry_filter, configuration)) &
                                   (tinydb.where('phases') == [phase_name]))
    # This seems to be necessary because the 'values' member does not modify 'datasets'
    # But everything else does!
    desired_data = copy.deepcopy(desired_data)
    #if len(desired_data) == 0:
    #    raise ValueError('No datasets for the system of interest containing {} were in \'datasets\''.format(prop))

    for idx, data in enumerate(desired_data):
        # Filter output values to only contain data for matching sublattice configurations
        matching_configs = np.array([(sblconf == configuration) for sblconf in data['solver']['sublattice_configurations']])
        matching_configs = np.arange(len(data['solver']['sublattice_configurations']))[matching_configs]
        # Rewrite output values with filtered data
        desired_data[idx]['values'] = np.array(data['values'], dtype=np.float)[..., matching_configs]
        desired_data[idx]['solver']['sublattice_configurations'] = np.array(data['solver']['sublattice_configurations'],
                                                                            dtype=np.object)[matching_configs].tolist()
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
        rss = np.square(np.dot(current_matrix, clf.coef_) - data_quantities).sum()
        # Compute Aikaike Information Criterion
        # Form valid under assumption all sample variances are equal and unknown
        score = 2*num_params + current_matrix.shape[-2] * np.log(rss)
        model_scores.append(score)
        results[num_params - 1, :num_params] = clf.coef_
        print(feature_tuple[:num_params], 'rss:', rss, 'AIC:', score)
    return OrderedDict(zip(feature_tuple, results[np.argmin(model_scores), :]))


def _get_samples(desired_data):
    all_samples = []
    for data in desired_data:
        temperatures = np.atleast_1d(data['conditions']['T'])
        site_fractions = data['solver'].get('sublattice_occupancies', [[1]])
        site_fraction_product = [reduce(operator.mul, list(itertools.chain(*[np.atleast_1d(f) for f in fracs])), 1)
                                 for fracs in site_fractions]
        # TODO: Subtle sorting bug here, if the interactions aren't already in sorted order...
        # TODO: This also looks like it won't work if we add more than one interaction here
        interaction_product = [f[0] - f[1] for fracs in site_fractions for f in fracs
                               if isinstance(f, list) and len(f) == 2]
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


def fit_formation_energy(comps, phase_name, configuration,
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
        Database containing the relevant phase.
    comps : list of str
        Names of the relevant components.
    phase_name : str
        Name of the desired phase for which the parameters will be found.
    configuration : ndarray
        Configuration of the sublattices for the fitting procedure.
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
                    ("HM_FORM", (1,))
                    ]
        features = OrderedDict(features)
    if any([isinstance(conf, list) for conf in configuration]):
        # Product of all nonzero site fractions in all sublattices
        YS = sympy.Symbol('YS')
        # Product of all binary interaction terms
        Z = sympy.Symbol('Z')
        redlich_kister_features = (YS, YS*Z, YS*(Z**2), YS*(Z**3))
        for feature in features.keys():
            all_features = list(itertools.product(redlich_kister_features, features[feature]))
            features[feature] = [i[0]*i[1] for i in all_features]
    parameters = {}
    for feature in features.values():
        for coef in feature:
            parameters[coef] = 0
    # ENDMEMBERS
    #
    # HEAT CAPACITY OF FORMATION
    desired_data = _get_data(comps, phase_name, configuration, datasets, "CPM_FORM")
    if len(desired_data) > 0:
        cp_matrix = _build_feature_matrix("CPM_FORM", features["CPM_FORM"], desired_data)
        data_quantities = np.concatenate([np.asarray(i['values']).flatten() for i in desired_data], axis=-1)
        parameters.update(_fit_parameters(cp_matrix, data_quantities, features["CPM_FORM"]))
    # ENTROPY OF FORMATION
    desired_data = _get_data(comps, phase_name, configuration, datasets, "SM_FORM")
    if len(desired_data) > 0:
        sm_matrix = _build_feature_matrix("SM_FORM", features["SM_FORM"], desired_data)
        data_quantities = np.concatenate([np.asarray(i['values']).flatten() for i in desired_data], axis=-1)
        # Subtract out the fixed contribution (from CPM_FORM) from our SM_FORM response vector
        all_samples = _get_samples(desired_data)
        fixed_portion = [feature_transforms["SM_FORM"](i).subs({v.T: temp, 'YS': compf[0],
                                                                'Z': compf[1]}).evalf()
                         for temp, compf in all_samples for i in features["CPM_FORM"]]
        fixed_portion = np.array(fixed_portion, dtype=np.float).reshape(len(all_samples), len(features["CPM_FORM"]))
        fixed_portion = np.dot(fixed_portion, [parameters[feature] for feature in features["CPM_FORM"]])
        parameters.update(_fit_parameters(sm_matrix, data_quantities - fixed_portion, features["SM_FORM"]))
    # ENTHALPY OF FORMATION
    desired_data = _get_data(comps, phase_name, configuration, datasets, "HM_FORM")
    if len(desired_data) > 0:
        hm_matrix = _build_feature_matrix("HM_FORM", features["HM_FORM"], desired_data)
        data_quantities = np.concatenate([np.asarray(i['values']).flatten() for i in desired_data], axis=-1)
        # Subtract out the fixed contribution (from CPM_FORM+SM_FORM) from our HM_FORM response vector
        all_samples = _get_samples(desired_data)
        fixed_portion = [feature_transforms["HM_FORM"](i).subs({v.T: temp, 'YS': compf[0],
                                                                'Z': compf[1]}).evalf()
                         for temp, compf in all_samples for i in features["CPM_FORM"]+features["SM_FORM"]]
        fixed_portion = np.array(fixed_portion, dtype=np.float).reshape(len(all_samples),
                                                                        len(features["CPM_FORM"]+features["SM_FORM"]))
        fixed_portion = np.dot(fixed_portion, [parameters[feature] for feature in features["CPM_FORM"]+features["SM_FORM"]])
        parameters.update(_fit_parameters(hm_matrix, data_quantities - fixed_portion, features["HM_FORM"]))
    return parameters


def _compare_data_to_parameters(desired_data, parameters, x, y, refdata):
    import matplotlib.pyplot as plt
    fit_eq = sympy.Add(*[feature_transforms[y](key)*value for key, value in parameters.items()])
    fit_eq += refdata
    all_samples = np.array(_get_samples(desired_data), dtype=np.object)
    temperatures = np.array([i[0] for i in all_samples], dtype=np.float)
    interactions = np.array([i[1][1] for i in all_samples], dtype=np.float)
    temp_min = temperatures.min()
    temp_max = temperatures.max()
    if (temp_min == temp_max) and (x == 'Z'):
        x_vals = np.linspace(0, 1, 100)
        # TODO: Not the correct 'YS' value if we have more than one interaction
        predicted_quantities = [fit_eq.subs({v.T: temp_min, 'YS': x_val * (1 - x_val),
                                             'Z': x_val - (1 - x_val)}).evalf() for x_val in x_vals]
    elif x == 'T':
        x_vals = np.linspace(temp_min, temp_max, 100)
        predicted_quantities = [fit_eq.subs({v.__dict__[x]: x_val}).evalf() for x_val in x_vals]
    else:
        raise ValueError('No support for plotting multiple temperatures on a composition plot yet: {}-{}'.format(x, y))

    fig = plt.figure(figsize=(9, 9))
    fig.gca().plot(x_vals, predicted_quantities, label='This work', color='k')
    fig.gca().set_xlim((x_vals.min()), (x_vals.max()))
    fig.gca().set_xlabel(plot_mapping.get(x, x))
    fig.gca().set_ylabel(plot_mapping.get(y, y))
    for data in desired_data:
        indep_var_data = None
        if x == 'T' or x == 'P':
            indep_var_data = np.array(data['conditions'][x], dtype=np.float).flatten()
        elif x == 'Z':
            indep_var_data = (interactions+1)/2
        response_data = np.array(data['values'], dtype=np.float).flatten()

        if (data['output'] == 'HM') or (data['output'] == 'GM'):
            # Shift data by reference state
            # This assumes that HM data has been shifted to zero at 298.15 K
            response_data += np.array(fit_eq.subs({v.__dict__[x]: 298.15}).evalf(), dtype=np.float)
        if len(response_data) < 10:
            plot_func = 'scatter'
        else:
            plot_func = 'plot'
        getattr(fig.gca(), plot_func)(indep_var_data,
                                      response_data,
                                      label=data.get('reference', None))
    fig.gca().legend(loc='best')
    fig.canvas.draw()


def plot_parameters(comps, phase_name, configuration, subl_ratios,
                    datasets, parameters, plots=None, refstate=SGTE91):
    if plots is None:
        plots = [('T', 'CPM'), ('T', 'CPM_FORM'), ('T', 'SM'), ('T', 'SM_FORM'),
                 ('T', 'HM'), ('T', 'HM_FORM'), ('Z', 'HM'), ('Z', 'HM_FORM')]
    comps = sorted(comps)
    for x_val, y_val in plots:
        refdata = 0
        if '_FORM' not in y_val:
            # Add reference state contribution to properties not "of formation"
            comp_refs = {c.upper(): feature_transforms[y_val](refstate[c.upper()]) for c in comps if c.upper() != 'VA'}
            comp_refs['VA'] = 0
            for subl, ratio in zip(configuration, subl_ratios):
                refdata = refdata + ratio * comp_refs[subl]
        desired_data = _get_data(comps, phase_name, configuration, datasets, y_val)
        _compare_data_to_parameters(desired_data, parameters, x_val, y_val, refdata)

def _symmetrize(configurations, symmetry):
    symmetrized = None
    if symmetry == 'B2':
        symmetrized = sorted({tuple(sorted(config[0:2])+list(config[2:])) for config in configurations})
    else:
        symmetrized = sorted(configurations)
    return symmetrized


def generate_parameter_file(phase_name, symmetry, subl_model, site_ratios, datasets, refstate=SGTE91):
    """
    Generate an initial CALPHAD model for a given phase and
    sublattice model.

    Parameters
    ==========
    phase_name : str
        Name of the phase.
    symmetry : str or None
        Sublattice model symmetry.
    subl_model : list of tuple
        Sublattice model for the phase of interest.
    site_ratios : list of float
        Number of sites in each sublattice, normalized to one atom.
    datasets : tinydb of datasets
        All datasets to consider for the calculation.
    """
    dbf = Database()
    dbf.elements = {c.upper() for subl in subl_model for c in subl}
    dbf.add_phase(phase_name, [], site_ratios)
    dbf.add_phase_constituents(phase_name, subl_model)
    # Write reference state to Database
    comp_refs = {c.upper(): refstate[c.upper()] for c in dbf.elements if c.upper() != 'VA'}
    comp_refs['VA'] = 0
    dbf.symbols.update({'GHSER'+c.upper(): data for c, data in comp_refs.items()})
    # First fit endmembers
    all_em_count = len(list(itertools.product(*subl_model)))
    endmembers = _symmetrize(itertools.product(*subl_model), symmetry)
    print('{0} endmembers ({1} distinct by symmetry)'.format(all_em_count, len(endmembers)))
    for endmember in endmembers:
        print('ENDMEMBER: '+str(endmember))
        parameters = fit_formation_energy(sorted(dbf.elements), phase_name, endmember, datasets)
        refdata = 0
        for subl, ratio in zip(endmember, site_ratios):
            if subl == 'VA':
                continue
            refdata = refdata + ratio * sympy.Symbol('GHSER'+subl)
        fit_eq = sympy.Add(*[key*value for key, value in parameters.items()])
        fit_eq += refdata
        print(fit_eq)
    # Now fit all binary interactions
    bin_interactions = list(itertools.combinations(endmembers, 2))
    transformed_bin_interactions = []
    for first_endmember, second_endmember in bin_interactions:
        interaction = []
        for first_occupant, second_occupant in zip(first_endmember, second_endmember):
            if first_occupant == second_occupant:
                interaction.append(first_occupant)
            else:
                interaction.append(tuple(sorted([first_occupant, second_occupant])))
        transformed_bin_interactions.append(interaction)
    bin_interactions = _symmetrize(transformed_bin_interactions, symmetry)
    print('{0} distinct binary interactions'.format(len(bin_interactions)))
    print(bin_interactions)
    # Now fit ternary interactions
    # Generate parameter file
    pass
