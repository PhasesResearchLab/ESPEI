"""
Fit, score and select models
"""

import logging
import numpy as np
from sklearn.linear_model import Ridge, LinearRegression

_log = logging.getLogger(__name__)


def fit_model(feature_matrix, data_quantities, ridge_alpha, weights=None):
    """
    Return model coefficients fit by scikit-learn's LinearRegression

    Parameters
    ----------
    feature_matrix : ndarray
        (M*N) regressor matrix. The transformed model inputs (y_i, T, P, etc.)
    data_quantities : ndarray
        (M,) response vector. Target values of the output (e.g. HM_MIX) to reproduce.
    ridge_alpha : float
        Value of the $alpha$ hyperparameter used in ridge regression. Defaults to 1.0e-100, which should be degenerate
        with ordinary least squares regression. For now, the parameter is applied to all features.

    Returns
    -------
    list
        List of model coefficients of shape (N,)

    Notes
    -----
    Solve Ax = b. `x` are the desired model coefficients. `A` is the
    'feature_matrix'. `b` corrresponds to 'data_quantities'.
    """
    if ridge_alpha is not None:
        clf = Ridge(fit_intercept=False, normalize=True, alpha=ridge_alpha)
    else:
        clf = LinearRegression(fit_intercept=False, normalize=True)
    clf.fit(feature_matrix, data_quantities, sample_weight=weights)
    return clf.coef_


def score_model(feature_matrix, data_quantities, model_coefficients, feature_list, weights, aicc_factor=None, rss_numerical_limit=1.0e-16):
    """
    Use the AICc to score a model that has been fit.

    Parameters
    ----------
    feature_matrix : ndarray
        (M*N) regressor matrix. The transformed model inputs (y_i, T, P, etc.)
    data_quantities : ndarray
        (M,) response vector. Target values of the output (e.g. HM_MIX) to reproduce.
    model_coefficients : list
        List of fitted model coefficients to be scored. Has shape (N,).
    feature_list : list
        Polynomial coefficients corresponding to each column of 'feature_matrix'.
        Has shape (N,). Purely a logging aid.
    aicc_factor : float
        Multiplication factor for the AICc's parameter penalty.
    rss_numerical_limit : float
        Anything with an absolute value smaller than this is set to zero.

    Returns
    -------
    float
        A model score

    Notes
    -----
    Solve Ax = b, where 'feature_matrix' is A and 'data_quantities' is b.

    The likelihood function is a simple least squares with no regularization. The form of the AIC is valid under
    assumption all sample variances are random and Gaussian, model is univariate. It is assumed the model here is
    univariate with T.
    """
    factor = aicc_factor if aicc_factor is not None else 1.0
    num_params = len(feature_list)
    rss = np.square(np.dot(feature_matrix, model_coefficients) - data_quantities.astype(np.float_)*np.array(weights)).sum()
    if np.abs(rss) < rss_numerical_limit:
        rss = rss_numerical_limit
    # Compute the corrected Akaike Information Criterion
    # The correction is (2k^2 + 2k)/(n - k - 1)
    # Our denominator for the correction must always be an integer by this equation.
    # Our correction can blow up if (n-k-1) = 0 and if n - 1 < k (we will actually be *lowering* the AICc)
    # So we will prevent blowing up by taking the denominator as 1/(p-n+1) for p > n - 1
    num_samples = data_quantities.size
    if (num_samples - 1.0) > num_params*factor:
        correction_denom = num_samples - factor*num_params - 1.0
    elif (num_samples - 1.0) == num_params*factor:
        correction_denom = 0.99
    else:
        correction_denom = 1.0 / (factor*num_params - num_samples + 1.0)
    correction = factor*(2.0*num_params**2 + 2.0*num_params)/correction_denom
    aic = 2.0*factor*num_params + num_samples * np.log(rss/num_samples)
    aicc = aic + correction  # model score
    _log.trace('%s rss: %s, AIC: %s, AICc: %s', feature_list, rss, aic, aicc)
    return aicc


def select_model(candidate_models, ridge_alpha, weights, aicc_factor=None):
    """
    Select a model from a series of candidates by fitting and scoring them

    Parameters
    ----------
    candidate_models : list
        List of tuples of (features, feature_matrix, data_quantities)
    ridge_alpha : float
        Value of the $alpha$ hyperparameter used in ridge regression. Defaults to 1.0e-100, which should be degenerate
        with ordinary least squares regression. For now, the parameter is applied to all features.
    aicc_factor : float
        Multiplication factor for the AICc's parameter penalty.
    Returns
    -------
    tuple
        Tuple of (feature_list, model_coefficients) for the highest scoring model
    """
    opt_model_score = np.inf
    opt_model = None  # will hold a (feature_list, model_coefficients)
    for feature_list, feature_matrix, data_quantities in candidate_models:
        model_coefficients = fit_model(feature_matrix, data_quantities, ridge_alpha, weights=weights)
        model_score = score_model(feature_matrix, data_quantities, model_coefficients, feature_list, weights, aicc_factor=aicc_factor)
        if np.isneginf(model_score):  # exact fit, stop here
            return (feature_list, model_coefficients)
        if model_score < opt_model_score:
            opt_model_score = model_score
            opt_model = (feature_list, model_coefficients)
    return opt_model

