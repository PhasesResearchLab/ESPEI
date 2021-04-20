"""
Utilities for ESPEI

Classes and functions defined here should have some reuse potential.
"""

import itertools
import re
from collections import namedtuple

import numpy as np
import sympy
from distributed import Client
from pycalphad import variables as v
from sympy import Symbol
from tinydb import TinyDB, where
from tinydb.storages import MemoryStorage


def unpack_piecewise(x):
    if isinstance(x, sympy.Piecewise):
        return float(x.args[0].expr)
    else:
        return float(x)


class PickleableTinyDB(TinyDB):
    """A pickleable version of TinyDB that uses MemoryStorage as a default."""
    def __getstate__(self):
        # first remove the query cache. The cache speed is not important to us.
        for table_name in self.tables():
            self.table(table_name)._query_cache = {}
        pickle_dict = {}
        for key, value in self.__dict__.items():
            if key == '_table':
                pickle_dict[key] = value.all()
            else:
                pickle_dict[key] = value
        return pickle_dict

    def __setstate__(self, state):
        self.__init__(storage=MemoryStorage)
        self.insert_multiple(state['_tables']['_default'])


class ImmediateClient(Client):
    """
    A subclass of distributed.Client that automatically unwraps the Futures
    returned by map.
    """
    def map(self, f, *iterators, **kwargs):
        """Map a function on a sequence of arguments.

        Any keyword arguments are passed to distributed.Client.map
        """
        _client = super(ImmediateClient, self)
        result = _client.gather(_client.map(f, *[list(it) for it in iterators], **kwargs))
        return result


def sigfigs(x, n):
    """Round x to n significant digits"""
    if x != 0:
        return np.around(x, -(np.floor(np.log10(np.abs(x)))).astype(np.int_) + (n - 1))
    else:
        return x


def optimal_parameters(trace_array, lnprob_array, kth=0):
    """
    Return the optimal parameters in the trace based on the highest likelihood.
    If kth is specified, return the kth set of *unique* optimal parameters.

    Parameters
    ----------
    trace_array : ndarray
        Array of shape (number of chains, iterations, number of parameters)
    lnprob_array : ndarray
        Array of shape (number of chains, iterations)
    kth : int
        Zero-indexed optimum. 0 (the default) is the most optimal solution. 1 is
        the second most optimal, etc.. Only *unique* solutions will be returned.

    Returns
    -------
    Array of optimal parameters

    Notes
    -----
    It is ok if the calculation did not finish and the arrays are padded with
    zeros. The number of chains and iterations in the trace and lnprob arrays
    must match.

    Examples
    --------
    >>> from espei.utils import optimal_parameters
    >>> trace = np.array([[[1, 0], [2, 0], [3, 0], [0, 0]], [[0, 2], [0, 4], [0, 6], [0, 0]]])  # 3 iterations of 4 allocated
    >>> lnprob = np.array([[-6, -4, -2, 0], [-3, -1, -2, 0]])
    >>> np.all(np.isclose(optimal_parameters(trace, lnprob), np.array([0, 4])))
    True

    """
    # indicies of chains + iterations that have non-zero parameters (that step has run)
    nz = np.nonzero(np.any(trace_array != 0, axis=-1))
    # chain + iteration index with the highest likelihood
    unique_params = np.zeros(trace_array.shape[-1])
    unique_params_found = -1
    # loop through all possible nonzero iterations
    for i in range(nz[-1][-1]):
        # find the next set of parameters parameters
        candidate_index = np.argpartition(-lnprob_array[nz], i)[i]
        candidate_params = trace_array[nz][candidate_index]
        # if the parameters are unique, make them the new unique parameters
        if np.any(candidate_params != unique_params):
            unique_params = candidate_params
            unique_params_found += 1
        # if we have found the kth set of unique parameters, stop
        if unique_params_found == kth:
            return unique_params
    return np.zeros(trace_array.shape[-1])


def database_symbols_to_fit(dbf, symbol_regex="^V[V]?([0-9]+)$"):
    """
    Return names of the symbols to fit that match the regular expression

    Parameters
    ----------
    dbf : Database
        pycalphad Database
    symbol_regex : str
        Regular expression of the fitting symbols. Defaults to V or VV followed by one or more numbers.

    Returns
    -------
    dict
        Context dictionary for different methods of calculation the error.
    """
    pattern = re.compile(symbol_regex)
    return sorted([x for x in sorted(dbf.symbols.keys()) if pattern.match(x)])


def bib_marker_map(bib_keys, markers=None):
    """
    Return a dict with reference keys and marker dicts

    Parameters
    ----------
    bib_keys :
    markers : list
        List of 2-tuples of ('fillstyle', 'marker') e.g. [('top', 'o'), ('full', 's')].
        Defaults to cycling through the filled markers, the different fill styles.

    Returns
    -------
    dict
        Dictionary with bib_keys as keys, dict values of formatted strings and marker dicts

    Examples
    --------
    >>> mm = bib_marker_map(['otis2016', 'bocklund2018'])
    >>> mm == {'bocklund2018': {'formatted': 'bocklund2018', 'markers': {'fillstyle': 'none', 'marker': 'o'}}, 'otis2016': {'formatted': 'otis2016', 'markers': {'fillstyle': 'none', 'marker': 'v'}}}
    True

    """
    # TODO: support custom formatting from looking up keys in a bib_db
    if not markers:
        filled_markers = ['o', 'v', 's', 'd', 'P', 'X', '^', '<', '>']
        fill_styles = ['none', 'full', 'top', 'right', 'bottom', 'left']
        markers = itertools.product(fill_styles, filled_markers)
    b_m_map = dict()
    for ref, marker_tuple in zip(sorted(bib_keys), markers):
        fill, mark = marker_tuple
        b_m_map[ref] = {
            'formatted': ref, # just use the key for formatting
            'markers': {
                'fillstyle': fill,
                'marker': mark
            }
        }
    return b_m_map


def parameter_term(expression, symbol):
    """
    Determine the term, e.g. T*log(T) that belongs to the symbol in expression

    Parameters
    ----------
    expression :
    symbol :

    Returns
    -------

    """
    if expression == symbol:
        # the parameter is the symbol, so the multiplicative term is 1.
        term = 1
    else:
        if isinstance(expression, sympy.Piecewise):
            expression = expression.args[0][0]
        if isinstance(expression, sympy.Symbol):
            # this is not mathematically correct, but we just need to be able to split it into args
            expression = sympy.Add(expression, 1)
        if not isinstance(expression, sympy.Add):
            raise ValueError('Parameter {} is a {} not a sympy.Add or a Piecewise Add'.format(expression, type(expression)))

        expression_terms = expression.args
        term = None
        for term_coeff in expression_terms:
            coeff, root = term_coeff.as_coeff_mul(symbol)
            if root == (symbol,):
                term = coeff
                break
    if term is None:
        raise ValueError('No multiplicative terms found for Symbol {} in parameter {}'.format(symbol, expression))
    return term


def formatted_constituent_array(constituent_array):
    """
    Given a constituent array of Species, return the classic CALPHAD-style interaction.

    Parameters
    ----------
    constituent_array : list
        List of sublattices, which are lists of Species in that sublattice

    Returns
    -------
    str
        String of the constituent array formatted in the classic CALPHAD style

    Examples
    --------
    >>> from pycalphad import variables as v
    >>> const_array = [[v.Species('CU'), v.Species('MG')], [v.Species('MG')]]
    >>> formatted_constituent_array(const_array)
    'CU,MG:MG'

    """
    return ':'.join([','.join([sp.name for sp in subl]) for subl in constituent_array])


def formatted_parameter(dbf, symbol, unique=True):
    """
    Get the deconstructed pretty parts of the parameter/term a symbol belongs to in a Database.

    Parameters
    ----------
    dbf : pycalphad.Database
    symbol : string or sympy.Symbol
        Symbol in the Database to get the parameter for.
    unique : bool
        If True, will raise if more than one parameter containing the symbol is found.


    Returns
    -------
    FormattedParameter
        A named tuple with the following attributes:
        ``phase_name``, ``interaction``, ``symbol``, ``term``, ``parameter_type``
        or ``term_symbol`` (which is just the Symbol * temperature term)
    """
    FormattedParameter = namedtuple('FormattedParameter', ['phase_name', 'interaction', 'symbol', 'term', 'parameter_type', 'term_symbol'])

    if not isinstance(symbol, Symbol):
        symbol = Symbol(symbol)
    search_res = dbf._parameters.search(
        where('parameter').test(lambda x: symbol in x.free_symbols))

    if len(search_res) == 0:
        raise ValueError('Symbol {} not found in any parameters.'.format(symbol))
    elif (len(search_res) > 1) and unique:
        raise ValueError('Parameters found containing Symbol {} are not unique. Found {}.'.format(symbol, search_res))

    formatted_parameters = []
    for result in search_res:
        const_array = formatted_constituent_array(result['constituent_array'])
        # format the paramter type to G or L0, L1, ...
        parameter_type = '{}{}'.format(result['parameter_type'], result['parameter_order'])
        # override non-interacting to G if there's no interaction
        has_interaction = ',' in const_array
        if not has_interaction:
            if (result['parameter_type'] == 'G') or (result['parameter_type'] == 'L'):
                parameter_type = 'G'

        term = parameter_term(result['parameter'], symbol)
        formatted_param = FormattedParameter(result['phase_name'],
                                             const_array,
                                             symbol,
                                             term,
                                             parameter_type,
                                             term*symbol
                                             )
        formatted_parameters.append(formatted_param)

    if unique:
        return formatted_parameters[0]
    else:
        return formatted_parameters


def build_sitefractions(phase_name, sublattice_configurations, sublattice_occupancies):
    """Convert nested lists of sublattice configurations and occupancies to a list
    of dictionaries. The dictionaries map SiteFraction symbols to occupancy
    values. Note that zero occupancy site fractions will need to be added
    separately since the total degrees of freedom aren't known in this function.

    Parameters
    ----------
    phase_name : str
        Name of the phase
    sublattice_configurations : [[str]]
        sublattice configuration
    sublattice_occupancies : [[float]]
        occupancy of each sublattice

    Returns
    -------
    [[float]]
        a list of site fractions over sublattices

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


def extract_aliases(phase_models):
    """Map possible phase name aliases to the desired phase model phase name.

    This function enforces that each alias is only claimed by one phase.
    Each phase name in the phase models is added as an alias for itself to
    support an "identity" operation.

    Parameters
    ----------
    phase_models
        Phase models ESPEI uses to initialize databases. Must contain a mapping
        of phase names to phase data (sublattices, site ratios, aliases)

    Returns
    -------
    Dict[str, str]

    """
    # Intialize aliases with identity for each phase first
    aliases = {phase_name: phase_name for phase_name in phase_models["phases"].keys()}
    for phase_name, phase_dict in phase_models["phases"].items():
        for alias in phase_dict.get("aliases", []):
            if alias not in aliases:
                aliases[alias] = phase_name
            else:
                raise ValueError(f"Cannot add {alias} as an alias for {phase_name} because it is already used by {aliases[alias]}")
    return aliases
