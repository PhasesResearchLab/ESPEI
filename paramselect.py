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
from pycalphad import calculate, Model
from sklearn.linear_model import Ridge
import tinydb
import sympy
import numpy as np
import json
from collections import OrderedDict
import matplotlib.pyplot as plt


def load_datasets(dataset_filenames):
    ds_database = tinydb.TinyDB(storage=tinydb.storages.MemoryStorage)
    for fname in dataset_filenames:
        with open(fname) as file_:
            ds_database.insert(json.load(file_))
    return ds_database


def choose_parameters(dbf, comps, phase_name, sublattice_configuration,
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
        Database containing the relevant phase. Only Standard Reference
        terms should be in the phase description!
    comps : list of str
        Names of the relevant components.
    phase_name : str
        Name of the desired phase for which the parameters will be found.
    sublattice_configuration : ndarray
        Configuration of the sublattices for the fitting procedure.
    datasets : tinydb of Dataset
        All the datasets desired to fit to.
    features : dict (optional)
        Maps "property" to a list of features for the linear model.
        e.g., {"CPM": (v.T*sympy.log(v.T), v.T**2, v.T**3)}

    Returns
    =======
    dict of feature: estimated value
    """
    if features is None:
        features = [("CPM", (v.T * sympy.log(v.T), v.T**2, v.T**-1, v.T**3))
                    ]
        features = OrderedDict(features)
    # Mapping of energy polynomial coefficients to corresponding property coefficients
    feature_transforms = {"CPM": lambda x: -v.T*sympy.diff(x, v.T, 2)}
    mod = Model(dbf, comps, phase_name)

    # ENDMEMBERS
    for prop in features.keys():
        # Construct Model and query datasets containing this property
        mod_prop = getattr(mod, prop)
        desired_data = datasets.search((tinydb.where('output') == prop) &
                                       (tinydb.where('components') == comps) &
                                       (tinydb.where('phases') == [phase_name]))
        if len(desired_data) == 0:
            raise ValueError('No datasets for the system of interest were in \'datasets\'')
        transformed_features = sympy.Matrix([feature_transforms[prop](i) for i in features[prop]])
        # These are the variables we are linearizing over; for now should just be T
        # Note: This section will need to be rewritten when we start fitting over P or other potentials
        target_variables = v.T  #sorted(transformed_features.atoms(v.StateVariable), key=str)
        all_variables = np.concatenate([i['conditions'][str(target_variables)] for i in desired_data], axis=-1)
        temp_filter = all_variables >= 298.15
        all_variables = all_variables[temp_filter]
        # full feature_matrix is built, and we add columns one at a time for our model
        feature_matrix = np.empty((len(all_variables), len(transformed_features)), dtype=np.float)
        feature_matrix[:, :] = [transformed_features.subs({v.T: i}).evalf() for i in all_variables]
        data_quantities = np.concatenate([np.asarray(i['values']).flatten() for i in desired_data], axis=-1)
        data_quantities = data_quantities[temp_filter]
        # Subtract out weighted sum of property since we only fit properties of formation
        data_quantities -= calculate(dbf, comps, phase_name, points=sublattice_configuration,
                                     mode='numpy', output=prop, model=mod,
                                     T=all_variables, P=101325)[prop].values.flatten()
        # Now generate candidate models; add parameters one at a time
        model_scores = []
        param_sets = []
        clf = Ridge(fit_intercept=False, alpha=0)
        plt.figure(figsize=(15, 12))
        variance_array = np.array([1., 1., 1e6, 1.])
        for num_params in range(1, len(transformed_features) + 1):
            current_matrix = np.multiply(feature_matrix, variance_array)[:, :num_params]
            clf.fit(current_matrix, data_quantities)
            # This may not exactly be the correct form for the likelihood
            error = np.square(np.dot(current_matrix, clf.coef_) - data_quantities).sum() + \
                clf.alpha * np.square(clf.coef_).sum()
            # Compute Aikaike Information Criterion
            score = 2*num_params + 2*np.log(error)
            model_scores.append(score)
            param_sets.append(transformed_features[:num_params])
            x_plot = np.linspace(all_variables.min(), all_variables.max(), 100)
            x_pred = [sympy.Matrix(transformed_features[:num_params]).subs({v.T: i}).evalf() for i in x_plot]
            y_plot = clf.predict(np.multiply(x_pred, variance_array[:num_params]))
            plt.plot(x_plot, y_plot, label=str(num_params)+' parameters')
            print(transformed_features[:num_params], 'error:', error, 'AIC:', score)
            print('Parameters:', np.multiply(clf.coef_, variance_array[:num_params]))
        plt.scatter(all_variables, data_quantities)
        plt.ylabel('Heat Capacity of Formation (J/mol-K)', fontsize=20)
        plt.xlabel('Temperature (K)', fontsize=20)
        plt.legend()


    # First, fit the heat capacities of formation
    # Next, entropies of formation
    # Finally, enthalpies of formation
    pass