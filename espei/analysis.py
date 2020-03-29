"""
Tools for analyzing ESPEI runs
"""

import numpy as np


def truncate_arrays(trace_array, prob_array=None):
    """
    Return slides of ESPEI output arrays with any empty remaining iterations (zeros) removed.

    Parameters
    ----------
    trace_array : np.ndarray
        Array of the trace from an ESPEI run. Should have shape (chains, iterations, parameters)
    prob_array : np.ndarray
        Array of the lnprob output from an ESPEI run. Should have shape (chains, iterations)

    Returns
    -------
    np.ndarry or (np.ndarray, np.ndarray)
        A slide of the zeros-removed trace array is returned if only the trace
        is passed. Otherwise a tuple of both the trace and lnprob are returned.

    Examples
    --------
    >>> from espei.analysis import truncate_arrays
    >>> trace = np.array([[[1, 0], [2, 0], [3, 0], [0, 0]], [[0, 2], [0, 4], [0, 6], [0, 0]]])  # 3 iterations of 4 allocated
    >>> truncate_arrays(trace).shape
    (2, 3, 2)

    """
    nz = np.nonzero(np.any(trace_array != 0, axis=-1))
    s = trace_array.shape
    # number of iterations that are non-zero
    iterations = trace_array[nz].reshape(s[0], -1, s[2]).shape[1]

    if prob_array is None:
        return trace_array[:, :iterations, :]
    else:
        return trace_array[:, :iterations, :], prob_array[:, :iterations]
