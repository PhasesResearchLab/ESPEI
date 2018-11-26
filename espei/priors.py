"""
Classes and functions for retrieving statistical priors for given parameters.
"""

import copy

import numpy as np
from scipy.stats import norm, uniform, triang


class rv_zero(object):
    """
    A simple class that mimics the scipy.stats.rv_continuous object's logpdf method, always returning zero.

    This class mainly exists for backwards compatibility where no prior is specified.

    Examples
    --------
    >>> import numpy as np
    >>> rv = rv_zero()
    >>> np.isclose(rv.logpdf(-np.inf), 0.0)
    True
    >>> np.isclose(rv.logpdf(1.0), 0.0)
    True
    >>> np.isclose(rv.logpdf(0.0), 0.0)
    True

    """

    def __init__(self, *args, **kwargs):
        pass

    def logpdf(self, *args, **kwargs):
        return 0.0


class DistributionParameter(object):
    """
    Handle generating absolute, scaling, shifting parameters.

    Examples
    --------
    >>> dp = DistributionParameter(5.0, 'absolute')  # always get back 5
    >>> dp.value(1.0) == 5.0
    True
    >>> dp = DistributionParameter(-2.0, 'relative')  # multiply by -2
    >>> dp.value(2.0) == -4.0
    True
    >>> dp = DistributionParameter(-1.0, 'shift_absolute')  # subtract 1
    >>> dp.value(2.0) == 1.0
    True
    >>> dp = DistributionParameter(-0.5, 'shift_relative')  # subtract 1/2 value
    >>> dp.value(2.0) == 1.0
    True

    """
    SUPPORTED_TYPES = ('absolute', 'relative', 'shift_absolute', 'shift_relative', 'identity')

    def __init__(self, parameter, param_type='absolute'):
        """

        Parameters
        ----------
        parameter : float
            Value of the distribution parameter.
        param_type : str
            Type of parameter, can be absolute, relative, or shift.

        """
        self.parameter = parameter

        if param_type not in self.SUPPORTED_TYPES:
            raise ValueError("Parameter type {} not in supported parameter types: {}".format(param_type, self.SUPPORTED_TYPES))

        self.modfunc = getattr(self, '_'+param_type)
        self.param_type = param_type


    def value(self, p):
        """
        Return the distribution parameter value modified by the parameter and type.

        Parameters
        ----------
        p : float
            Input parameter to modify.

        Returns
        -------
        float

        """
        return self.modfunc(p)

    def _absolute(self, _):
        return self.parameter

    def _relative(self, p):
        return p * self.parameter

    def _shift_relative(self, p):
        return p + np.abs(p)*self.parameter

    def _shift_absolute(self, p):
        return p + self.parameter

    def _identity(self, p):
        # tecnically equivalent to _relative with self.parameter = 1.0 or _shift with self.parmaeter = 0
        return p

    def __repr__(self):
        return "DistributionParameter<{} of {}>".format(self.param_type, self.parameter)


class PriorSpec(object):
    """
    Specification template for instantiating priors.
    """
    SUPPORTED_PRIORS = ('normal', 'uniform', 'triangular', 'zero')

    def __init__(self, name, **parameters):
        """

        Parameters
        ----------
        name : parameter name
        parameters : dict
            Parameters to be passed

        """
        if name not in self.SUPPORTED_PRIORS:
            raise ValueError("Selected prior is not ")
        self.name = name
        self.prior_generator = getattr(self, '_prior_'+name)

        # process parameters
        distribution_params = {}
        for k, v in parameters.items():
            # the key is the parameter type, e.g 'scale', 'scale_relative', 'loc_absolute'
            split_type = k.split('_')
            param = split_type[0]
            if len(split_type) > 1:
                param_type = '_'.join(split_type[1:])
            else:
                param_type = 'absolute'
            distribution_params[param] = DistributionParameter(v, param_type=param_type)

        self.parameters = distribution_params

    def get_prior(self, value):
        """Instantiate a prior as described in the spec

        Examples
        --------
        >>> import numpy as np
        >>> from espei.priors import PriorSpec
        >>> tri_spec = {'name': 'triangular', 'loc_shift_relative': -0.5, 'scale_shift_relative': 0.5, 'c': 0.5}
        >>> np.isneginf(PriorSpec(**tri_spec).get_prior(10).logpdf(5.1))
        False
        >>> np.isneginf(PriorSpec(**tri_spec).get_prior(10).logpdf(4.9))
        True

        """
        params = {k: dp.value(value) for k, dp in self.parameters.items()}
        return self.prior_generator(params)

    def _prior_normal(self, params):
        """Instantiate a normal prior"""
        # make the standard deviation is positive
        params['scale'] = np.abs(params.get('scale', 1.0))
        return norm(**params)

    def _prior_uniform(self, params):
        """Instantiate a uniform prior"""
        # make sure loc is min and scale is max
        params['scale'] = np.abs(params.get('scale', 1.0))
        return uniform(**params)

    def _prior_triangular(self, params):
        """Instantiate a triangular prior"""
        # make sure loc is min and scale is max
        params['scale'] = np.abs(params.get('scale', 1.0))
        # make sure c is set
        params['c'] = params.get('c', 0.5)
        return triang(**params)

    def _prior_zero(self, _):
        """Instantiate a zero prior"""
        return rv_zero()


def build_prior_specs(prior_spec, parameters):
    """
    Get priors from given parameters

    Parameters
    ----------
    prior_spec : PriorSpec or dict
        Either a prior spec dict (to instantiate), a PriorSpec, or a list of either.
        If a list is passed, it must correspond to the parameters.
    parameters : list
        List of parameters that the priors will be instantiated by

    Returns
    -------
    [PriorSpec]

    Examples
    --------
    >>> s_norm = {'name': 'normal', 'scale_relative': 0.1, 'loc_identity': 1.0}
    >>> len(build_prior_specs(s_norm, [10, 100])) == 2
    True
    >>> s_tri = {'name': 'triangular', 'loc_shift_relative': -0.5, 'scale_shift_relative': 0.5, 'c': 0.5}
    >>> from espei.priors import PriorSpec
    >>> len(build_prior_specs([s_norm, PriorSpec(**s_tri)], [10, 100])) == 2
    True

    """
    # Check for single instance, otherwise assume some list/iterable matching the parameters
    if isinstance(prior_spec, (dict, PriorSpec)):
        prior_spec = [copy.deepcopy(prior_spec) for _ in parameters]

    prior_specs = []
    for spec, param in zip(prior_spec, parameters):
        if isinstance(spec, dict):
            prior_specs.append(PriorSpec(**spec))
        elif isinstance(spec, PriorSpec):
            prior_specs.append(spec)
        elif hasattr(spec, "logpdf"):
            # this is an instance of a prior we'll add it to the spec list and it should be handled later
            prior_spec.append(spec)
        else:
            raise ValueError("Unknown prior spec {}. Should be either a PriorSpec instance or a dict".format(spec))
    return prior_specs
