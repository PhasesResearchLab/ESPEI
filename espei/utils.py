"""
Utilities for ESPEI

Classes and functions defined here should have some reuse potential.
"""

import re, itertools
import numpy as np
from sympy import Symbol
from distributed import Client
from tinydb import TinyDB
from tinydb.storages import MemoryStorage
from six import string_types
import bibtexparser
from bibtexparser.bparser import BibTexParser
from bibtexparser.customization import convert_to_unicode

from pycalphad import Model, variables as v
from pycalphad.core.sympydiff_utils import build_functions
from pycalphad.core.utils import unpack_kwarg, unpack_components
from pycalphad.core.phase_rec import PhaseRecord_from_cython


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
        self.insert_multiple(state['_table'])


class ImmediateClient(Client):
    """
    A subclass of distributed.Client that automatically unwraps the Futures
    returned by map.
    """
    def map (self, *args, **kwargs):
        result = super(ImmediateClient, self).map(*args, **kwargs)
        result = [x.result() for x in result]
        return result


def sigfigs(x, n):
    """Round x to n significant digits"""
    if x != 0:
        return np.around(x, -(np.floor(np.log10(np.abs(x)))).astype(np.int) + (n - 1))
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
    """
    # indicies of chains + iterations that have non-zero parameters (that step has run)
    nz = np.nonzero(np.all(trace_array != 0, axis=-1))
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
    list
    """
    pattern = re.compile(symbol_regex)
    return sorted([x for x in sorted(dbf.symbols.keys()) if pattern.match(x)])


def flexible_open_string(obj):
    """
    Return the string of a an object that is either file-like, a file path, or the raw string.

    Parameters
    ----------
    obj : string-like or file-like
        Either a multiline string, a path, or a file-like object

    Returns
    -------
    str
    """
    if isinstance(obj, string_types):
        # the obj is a string
        if '\n' in obj:
            # if the string has linebreaks, then we assume it's a raw string. Return it.
            return obj
        else:
            # assume it is a path
            with open(obj) as fp:
                read_string = fp.read()
            return read_string
    elif hasattr(obj, 'read'):
        # assume it is file-like
        read_string = obj.read()
        return read_string
    else:
        raise ValueError('Unable to determine how to extract the string of the passed object ({}) of type {}. Expected a raw string, file-like, or path-like.'.format(obj, type(obj)))


bibliography_database = PickleableTinyDB(storage=MemoryStorage)

def add_bibtex_to_bib_database(bibtex, bib_db=None):
    """
    Add entries from a BibTeX file to the bibliography database

    Parameters
    ----------
    bibtex : str
        Either a multiline string, a path, or a file-like object of a BibTeX file
    bib_db: PickleableTinyDB
        Database to put the BibTeX entries. Defaults to a module-level default database

    Returns
    -------
    The modified bibliographic database
    """
    if not bib_db:
        bib_db = bibliography_database
    bibtex_string = flexible_open_string(bibtex)
    parser = BibTexParser()
    parser.customization = convert_to_unicode
    parsed_bibtex = bibtexparser.loads(bibtex_string, parser=parser)
    bib_db.insert_multiple(parsed_bibtex.entries)
    return bib_db


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
    >>> bib_marker_map(['otis2016', 'bocklund2018'])
    {
    'bocklund2018': {
                    'formatted': 'bocklund2018',
                    'markers': {'fillstyle': 'full', 'marker': 'o'}
                },
    'otis2016': {
                    'formatted': 'otis2016',
                    'markers': {'fillstyle': 'full', 'marker': 'v'}
                }
    }
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


def get_pure_elements(dbf, comps):
    """
    Return a list of pure elements in the system

    Parameters
    ----------
    dbf : pycalphad.Database
        A Database object
    comps : list
        A list of component names (species and pure elements)

    Returns
    -------
    list
        A list of pure elements in the Database
    """
    comps = sorted(unpack_components(dbf, comps))
    components = [x for x in comps]
    desired_active_pure_elements = [list(x.constituents.keys()) for x in components]
    desired_active_pure_elements = [el.upper() for constituents in desired_active_pure_elements for el in constituents]
    pure_elements = sorted(set([x for x in desired_active_pure_elements if x != 'VA']))
    return pure_elements


def eq_callables_dict(dbf, comps, phases, model=None, param_symbols=None, output='GM', build_gradients=True):
    """
    Create a dictionary of callable dictionaries for phases in equilibrium

    Parameters
    ----------
    dbf : pycalphad.Database
        A pycalphad Database object
    comps : list
        List of component names
    phases : list
        List of phase names
    model : dict or type
        Dictionary of {phase_name: Model subclass} or a type corresponding to a
        Model subclass. Defaults to ``Model``.
    param_symbols : list
        SymPy Symbol objects that will be preserved in the callable functions.
    output : str
        Output property of the particular Model to sample
    build_gradients : bool
        Whether or not to build gradient functions. Defaults to True.

    Returns
    -------
    dict
        Dictionary of keyword argument callables to pass to equilibrium.

    Notes
    -----
    Based on the pycalphad equilibrium method for building phases as of commit 37ff75ce.

    Examples
    --------
    >>> dbf = Database('AL-NI.tdb')
    >>> comps = ['AL', 'NI', 'VA']
    >>> phases = ['FCC_L12', 'BCC_B2', 'LIQUID', 'AL3NI5', 'AL3NI2', 'AL3NI']
    >>> eq_callables = eq_callables_dict(dbf, comps, phases)
    >>> equilibrium(dbf, comps, phases, conditions, **eq_callables)
    """
    comps = sorted(unpack_components(dbf, comps))
    pure_elements = get_pure_elements(dbf, comps)

    eq_callables = {
        'massfuncs': {},
        'massgradfuncs': {},
        'callables': {},
        'grad_callables': {},
        'hess_callables': {},
    }

    models = unpack_kwarg(model, default_arg=Model)
    param_symbols = param_symbols if param_symbols is not None else []
    # wrap param symbols in Symbols if they are strings
    if all([isinstance(sym, string_types) for sym in param_symbols]):
        param_symbols = [Symbol(sym) for sym in param_symbols]
    param_values = np.zeros_like(param_symbols, dtype=np.float64)

    phase_records = {}
    # create models
    # starting from pycalphad
    for name in phases:
        mod = models[name]
        if isinstance(mod, type):
            models[name] = mod = mod(dbf, comps, name)
        site_fracs = mod.site_fractions
        variables = sorted(site_fracs, key=str)
        try:
            out = getattr(mod, output)
        except AttributeError:
            raise AttributeError('Missing Model attribute {0} specified for {1}'
                                 .format(output, mod.__class__))

        # Build the callables of the output
        # Only force undefineds to zero if we're not overriding them
        undefs = list(out.atoms(Symbol) - out.atoms(v.StateVariable) - set(param_symbols))
        for undef in undefs:
            out = out.xreplace({undef: float(0)})
        build_output = build_functions(out, tuple([v.P, v.T] + site_fracs), parameters=param_symbols, include_grad=build_gradients)
        if build_gradients:
            cf, gf = build_output
        else:
            cf = build_output
            gf = None
        hf = None
        eq_callables['callables'][name] = cf
        eq_callables['grad_callables'][name] = gf
        eq_callables['hess_callables'][name] = hf

        # Build the callables for mass
        # TODO: In principle, we should also check for undefs in mod.moles()

        if build_gradients:
            mcf, mgf = zip(*[build_functions(mod.moles(el), [v.P, v.T] + variables,
                                           include_obj=True,
                                           include_grad=build_gradients,
                                           parameters=param_symbols)
                           for el in pure_elements])
        else:
            mcf = tuple([build_functions(mod.moles(el), [v.P, v.T] + variables,
                                           include_obj=True,
                                           include_grad=build_gradients,
                                           parameters=param_symbols)
                           for el in pure_elements])
            mgf = None
        eq_callables['massfuncs'][name] = mcf
        eq_callables['massgradfuncs'][name] = mgf

        # creating the phase records triggers the compile
        phase_records[name.upper()] = PhaseRecord_from_cython(comps, variables,
                                                           np.array(dbf.phases[name].sublattices, dtype=np.float),
                                                           param_values, eq_callables['callables'][name],
                                                           eq_callables['grad_callables'][name], eq_callables['hess_callables'][name],
                                                           eq_callables['massfuncs'][name], eq_callables['massgradfuncs'][name])

    # finally, add the models to the eq_callables
    eq_callables['model'] = dict(models)
    return eq_callables
