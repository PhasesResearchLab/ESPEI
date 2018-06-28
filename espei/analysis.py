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

    """
    nz = np.nonzero(np.all(trace_array != 0, axis=-1))
    s = trace_array.shape
    # number of iterations that are non-zero
    iterations = trace_array[nz].reshape(s[0], -1, s[2]).shape[1]

    if prob_array is None:
        return trace_array[:,:iterations,:]
    else:
        return trace_array[:,:iterations,:], prob_array[:, :iterations]
