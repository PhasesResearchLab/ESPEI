"""
Fit, score and select models
"""

import logging
import numpy as np
from numpy.typing import ArrayLike
from sklearn.linear_model import Ridge, LinearRegression

_log = logging.getLogger(__name__)


def fit_model(feature_matrix, data_quantities, ridge_alpha, weights=None):
    """
    Return model coefficients fit by scikit-learn's LinearRegression

    Parameters
    ----------
    feature_matrix : ArrayLike
        (:math:`M \\times N`) regressor matrix. The transformed model inputs (y_i, T, P, etc.)
    data_quantities : ArrayLike
        Size (:math:`M`) response vector. Target values of the output (e.g. HM_MIX) to reproduce.
    ridge_alpha : float
        Value of the :math:`\\alpha` hyperparameter used in ridge regression. Defaults to 1.0e-100, which should be degenerate
        with ordinary least squares regression. For now, the parameter is applied to all features.

    Returns
    -------
    list
        List of model coefficients of size (:math:`N`)

    Notes
    -----
    Solve :math:`Ax = b` where :math:`x` are the desired model coefficients,
    :math:`A` is the ``feature_matrix`` and
    :math:`b` corrresponds to ``data_quantities``.

    """
    if ridge_alpha is not None:
        clf = Ridge(fit_intercept=False, normalize=True, alpha=ridge_alpha)
    else:
        clf = LinearRegression(fit_intercept=False, normalize=True)
    clf.fit(feature_matrix, data_quantities, sample_weight=weights)
    return clf.coef_


def score_model(feature_matrix, data_quantities, model_coefficients, feature_list, weights, aicc_factor=None, rss_numerical_limit=1.0e-16):
    """
    Use the modified AICc to score a model that has been fit.

    The modified AICc is given by

    .. math::

       \\mathrm{mAICc} = n \\ln \\frac{\\mathrm{RSS}}{n} + 2pk + \\frac {2p^2k^2 + 2pk} {n - pk - 1}

    Parameters
    ----------
    feature_matrix : ArrayLike
        (:math:`M \\times N`) regressor matrix. The transformed model inputs (y_i, T, P, etc.)
    data_quantities : ArrayLike
        Size (:math:`M`) response vector. Target values of the output (e.g. HM_MIX) to reproduce.
    model_coefficients : list
        Size (:math:`N`) list of fitted model coefficients to be scored.
    feature_list : list
        Polynomial coefficients corresponding to each column of ``feature_matrix``.
        Has shape (N,). Purely a logging aid.
    aicc_factor : float
        Multiplication factor for the AICc's parameter penalty.
    rss_numerical_limit : float
        Anything with an absolute value smaller than this is set to zero.

    Returns
    -------
    float
        A model score

    """
    p = aicc_factor if aicc_factor is not None else 1.0
    k = len(feature_list)
    rss = np.square((np.dot(feature_matrix, model_coefficients) - data_quantities.astype(np.float_))*np.array(weights)).sum()
    if np.abs(rss) < rss_numerical_limit:
        rss = rss_numerical_limit
    # Compute the corrected Akaike Information Criterion
    n = data_quantities.size
    pk = k*p
    aic = n * np.log(rss / n) + 2.0 * pk
    if pk >= (n - 1.0):
        # Prevent the denominator of the proper mAICc from blowing up (pk = n - 1) or negative (pk > n - 1)
        correction = (2.0 * p**2 * k**2 + 2.0 * pk) * (-n + pk + 3.0)
    else:
        correction = (2.0 * p**2 * k**2 + 2.0 * pk) / (n - pk - 1.0)
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
        Value of the :math:`\\alpha` hyperparameter used in ridge regression. Defaults to 1.0e-100, which should be degenerate
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
