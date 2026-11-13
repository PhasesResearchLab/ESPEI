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
        clf = Ridge(fit_intercept=False, alpha=ridge_alpha)
    else:
        # Features of different magnitudes (e.g. 1/T and T**3) make the feature matrix poorly conditioned.
        # This tolerance is the standard for LAPACK least squares and is needed
        # because scikit-learn sets a high tol that makes fits less accurate.
        tol = max(np.shape(feature_matrix)) * np.finfo(float).eps
        clf = LinearRegression(fit_intercept=False, tol=tol)
    clf.fit(feature_matrix, data_quantities, sample_weight=weights)
    return clf.coef_


def select_model(feature_matrix, data_quantities, std_deviation, candidates, penalty_factors=None, ridge_alpha=None, estimate_scale=False):
    """
    Return the candidate model with the lowest modified AICc and its coefficients.

    The score of a candidate with :math:`k` features is the AIC,
    :math:`\\chi^2 + 2pk`, where :math:`\\chi^2` is the sum of the squared
    residuals divided by the standard deviations of the :math:`n` data points
    and :math:`p` is the penalty factor. If ``estimate_scale``, the standard
    deviations are only relative and the scale is estimated from the residuals
    of each candidate, so the score is the modified AICc,

    .. math::

       \\mathrm{mAICc} = n \\ln \\frac{\\chi^2}{n} + 2pk + \\frac {2p^2k^2 + 2pk} {n - pk - 1}

    Parameters
    ----------
    feature_matrix : ArrayLike
        (:math:`n \\times N`) values of all the features of the candidates at the data.
    data_quantities : ArrayLike
        Size (:math:`n`) response vector to reproduce.
    std_deviation : ArrayLike
        Size (:math:`n`) standard deviation of each data point.
    candidates : List[Sequence[int]]
        Each candidate is the columns of ``feature_matrix`` in the model.
    penalty_factors : Optional[ArrayLike]
        Size (:math:`N`) AICc penalty factor of each feature. Defaults to 1.
    ridge_alpha : Optional[float]
        Value of the :math:`\\alpha` hyperparameter used in ridge regression.
        If None, ordinary least squares is used.
    estimate_scale : bool
        If True, estimate the scale of the standard deviations.

    Returns
    -------
    Tuple[Sequence[int], ArrayLike]
        Columns and coefficients of the selected candidate.

    """
    n = data_quantities.size
    # relative to the most precise data, so ridge_alpha has the units of the data without weights
    weights = (std_deviation.min() / std_deviation)**2
    penalty_factors = np.ones(feature_matrix.shape[1]) if penalty_factors is None else np.asarray(penalty_factors, dtype=float)
    opt_score = np.inf
    opt_model = None
    for columns in candidates:
        columns = list(columns)
        coefficients = fit_model(feature_matrix[:, columns], data_quantities, ridge_alpha, weights=weights)
        chi2 = np.sum(((feature_matrix[:, columns] @ coefficients - data_quantities) / std_deviation)**2)
        pk = penalty_factors[columns].sum()
        if not estimate_scale:
            score = chi2 + 2.0 * pk
        elif pk >= (n - 1.0):
            # Prevent the denominator of the proper mAICc from blowing up (pk = n - 1) or negative (pk > n - 1)
            score = n * np.log(max(chi2, 1.0e-16) / n) + 2.0 * pk + (2.0 * pk**2 + 2.0 * pk) * (-n + pk + 3.0)
        else:
            score = n * np.log(max(chi2, 1.0e-16) / n) + 2.0 * pk + (2.0 * pk**2 + 2.0 * pk) / (n - pk - 1.0)
        _log.debug('columns %s chi2: %s, score: %s', columns, chi2, score)
        if score < opt_score:
            opt_score = score
            opt_model = (columns, coefficients)
    _log.trace('selected model columns: %s, coefficients: %s, model score: %s', opt_model[0], list(opt_model[1]), opt_score)
    return opt_model
