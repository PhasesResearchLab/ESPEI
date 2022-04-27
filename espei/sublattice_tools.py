"""
Utilities for manipulating sublattice models.
"""

from typing import Any, Sequence, Union
import itertools


def tuplify(x):
    """Convert a list to a tuple, or wrap an object in a tuple if it's not a list or tuple."""
    if isinstance(x, list) or isinstance(x, tuple):
        return tuple(x)
    else:
        return tuple([x])


def recursive_tuplify(x):
    """Recursively convert a nested list to a tuple"""
    def _tuplify(y):
        if isinstance(y, list) or isinstance(y, tuple):
            return tuple(_tuplify(i) if isinstance(i, (list, tuple)) else i for i in y)
        else:
            return y
    return tuple(map(_tuplify, x))


def canonical_sort_key(x):
    """
    Wrap strings in tuples so they'll sort.

    Parameters
    ----------
    x : list
        List of strings to sort

    Returns
    -------
    tuple
        tuple of strings that can be sorted
    """
    return [tuple(i) if isinstance(i, (tuple, list)) else (i,) for i in x]


def canonicalize(configuration, equivalent_sublattices):
    """
    Sort a sequence with symmetry. This routine gives the sequence
    a deterministic ordering while respecting symmetry.

    Parameters
    ----------
    configuration : [str]
        Sublattice configuration to sort.
    equivalent_sublattices : {{int}}
        Indices of 'configuration' which should be equivalent by symmetry, i.e.,
        [[0, 4], [1, 2, 3]] means permuting elements 0 and 4, or 1, 2 and 3, respectively,
        has no effect on the equivalence of the sequence.

    Returns
    -------
    str
        sorted tuple that has been canonicalized.

    """
    canonicalized = list(configuration)
    if equivalent_sublattices is not None:
        for subl in equivalent_sublattices:
            subgroup = sorted([configuration[idx] for idx in sorted(subl)], key=canonical_sort_key)
            for subl_idx, conf_idx in enumerate(sorted(subl)):
                if isinstance(subgroup[subl_idx], list):
                    canonicalized[conf_idx] = tuple(subgroup[subl_idx])
                else:
                    canonicalized[conf_idx] = subgroup[subl_idx]

    return recursive_tuplify(canonicalized)


def generate_symmetric_group(configuration: Sequence[Any], symmetry: Union[None, Sequence[Sequence[int]]]):
    """
    For a particular configuration and list of sublattices that are symmetric,
    generate all the symmetrically equivalent configurations.

    Parameters
    ----------
    configuration : Sequence[Any]
        Typically a constituent array. The length should correspond to the number of
        sublattices in the phase.
    symmetry : Union[None, Sequence[Sequence[int]]]
        A list of lists giving the indices of symmetrically equivalent sublattices.
        For example: a symmetry of `[[0, 1, 2, 3]]` means that the first four
        sublattices are symmetric to each other. If multiple sublattices are given, the
        sublattices are internally equivalent and the sublattices themselves are assumed
        interchangeble. That is, for a symmetry of `[[0, 1], [2, 3]]`, sublattices
        0 and 1 are equivalent to each other (i.e. `[0, 1] == [1, 0]`) and similarly for
        sublattices 2 and 3. It also implies that the sublattices are interchangeable,
        (i.e. `[[0, 1], [2, 3]] == [[2, 3], [0, 1]]`), but note that constituents cannot
        change sublattices (i.e. `[[0, 1], [2, 3]] != [[0, 3], [2, 1]]`).
        If `symmetry=None` is given, no new configurations are generated.

    Returns
    -------
    tuple
        Tuple of configuration tuples that are all symmetrically equivalent.

    Notes
    -----
    In the general case, equivalency between sublattices, for example
    (`[[0, 1], [2, 3]] == [[2, 3], [0, 1]]`), is not necessarily required. It
    could be that sublattices 0 and 1 represent equivalent substitutional
    sublattices, while 2 and 3 represent equivalent interstitial sites.
    Interchanging sublattices between substitutional sublattices is allowed, but
    the substitutional sites would not be interchangeable with the interstitial
    sites. To achieve this kind of effect with this function, you would need to
    call it once with the equivalent substitutional sublattices, then for each
    generated configuration, call this function again, giving the unique
    configurations for symmetric interstitial sublattices.
    """
    # recursively casting sequences to tuples ensures that the generated configurations are hashable
    configuration = recursive_tuplify(configuration)
    sublattice_indices = list(range(len(configuration)))
    if symmetry is None:
        return [configuration]
    seen_subl_indices = sorted([i for equiv_subl in symmetry for i in equiv_subl])
    # fixed_subl_indices were not given, they are assumed to be inequivalent and constant
    fixed_subl_indices = sorted(set(sublattice_indices) - set(seen_subl_indices))

    # permute within each sublattice, i.e. [0, 1] -> [[0, 1], [1, 0]]
    intra_sublattice_permutations = (itertools.permutations(equiv_subl) for equiv_subl in symmetry)
    # product, combining all internal sublattice permutations, i.e.
    # [[0, 1], [1, 0]] and [[2, 3], [3, 2]] become [ ([0, 1], [2, 3]), ... ]
    sublattice_products = itertools.product(*intra_sublattice_permutations)
    # finally, swap sets of equivalent sublattices, i.e.
    # [ ([0, 1], [2, 3]), ... ] -> [[ ([0, 1], [2, 3]),  ([2, 3], [0, 1]) ], ... ]
    inter_sublattice_permutations = (itertools.permutations(x) for x in sublattice_products)

    symmetrically_distinct_configurations = set()
    # chain.from_iterable calls flatten out nested permutation lists, i.e.
    # ([0, 1], [2, 3]) -> [0, 1, 2, 3]
    for proposed_distinct_indices in itertools.chain.from_iterable(inter_sublattice_permutations):
        new_config = list(configuration[i] for i in itertools.chain.from_iterable(proposed_distinct_indices))
        # The configuration only contains indices for symmetric sublattices. For the
        # inequivalent sublattices, we need to insert them at their proper indices.
        # Indices _must_ be in sorted order because we are changing the array size on insertion.
        for fixed_idx in fixed_subl_indices:
            new_config.insert(fixed_idx, configuration[fixed_idx])
        symmetrically_distinct_configurations.add(tuple(new_config))
    return sorted(symmetrically_distinct_configurations, key=canonical_sort_key)


def sorted_interactions(interactions, max_interaction_order, symmetry):
    """
    Return interactions sorted by interaction order

    Parameters
    ----------
    interactions : list
        List of tuples/strings of potential interactions
    max_interaction_order : int
        Highest expected interaction order, e.g. ternary interactions should be 3
    symmetry : list of lists
        List of lists containing symmetrically equivalent sublattice indices,
        e.g. [[0, 1], [2, 3]] means that sublattices 0 and 1 are equivalent and
        sublattices 2 and 3 are also equivalent.

    Returns
    -------
    list
        Sorted list of interactions

    Notes
    -----
    Sort by number of full interactions, e.g. (A:A,B) is before (A,B:A,B)
    The goal is to return a sort key that can sort through multiple interaction
    orders, e.g. (A:A,B,C), which should be before (A,B:A,B,C), which should be
    before (A,B,C:A,B,C).

    """
    def int_sort_key(x):
        # Each interaction is given a sort score for the number of interactions
        # it has at each level. For example, a 4 sublattice phase with
        # interaction of (A:A:A,B:A,B,C) has 2 order-1 interactions, 1 order-2
        # interaction and 1 order-3 interaction. It's sort score would be (1, 1, 2).
        # thus it should sort below a (2, 1, 2), for example.
        sort_score = []
        for interaction_order in reversed(range(1, max_interaction_order+1)):
            sort_score.append(sum((isinstance(n, (list, tuple)) and len(n) == interaction_order) for n in x))
        return canonical_sort_key(recursive_tuplify(sort_score) + x)

    interactions = sorted(set(canonicalize(i, symmetry) for i in interactions), key=int_sort_key)
    # filter out interactions that have ternary and binary parameters (cross interactions)
    # for now, I'm not really sure how the mathematics work out
    # I think they are treated as a binary interaction
    # in reality, most people would probably not want to fit these cross interactions
    filtered_interactions = []
    for inter in interactions:
        if not (interaction_test(inter, 2) and interaction_test(inter, 3)):
            # check for multiple 3 order interactions
            order_3_interactions_count = 0
            for subl in inter:
                if interaction_test((subl,), 3):
                    order_3_interactions_count += 1
            if order_3_interactions_count <= 1:
                filtered_interactions.append(inter)
    return filtered_interactions


def generate_interactions(endmembers, order, symmetry, forbid_cross_interactions=None):
    """
    Returns a list of sorted interactions of a certain order

    Parameters
    ----------
    endmembers : list
        List of tuples/strings of all endmembers (including symmetrically equivalent)
    order : int
        Highest expected interaction order, e.g. ternary interactions should be 3
    symmetry : list of lists
        List of lists containing symmetrically equivalent sublattice indices,
        e.g. [[0, 1], [2, 3]] means that sublattices 0 and 1 are equivalent and
        sublattices 2 and 3 are also equivalent.
    forbid_cross_interactions : Optiona[bool]
        If True, will prevent cross interations from being generated. If None,
        automatically determine based on the symmetry. If there is no symmetry,
        cross interactions are forbidden. Symmetry usually implies that there
        is a disordered state that is interesting, so cross interactions are
        allowed.

    Returns
    -------
    list
        List of interaction tuples, e.g. [('A', ('A', 'B'))]

    """
    if forbid_cross_interactions is None:
        # Profiling indicates that it is worth skipping cross interactions in
        # multi-component systems due to the cost of the sorted_interactions
        if symmetry is None:
            forbid_cross_interactions = True  # symmetry may imply ordering, so keep these
        else:
            forbid_cross_interactions = False
    interactions = []
    for endmembers in itertools.combinations(endmembers, order):
        config = []
        has_desired_order = False  # Some interactions make for degenerate endmember
        for subl in zip(*endmembers):
            subl = set(subl)
            num_constit = len(subl)
            if num_constit == 1:
                config.append(tuple(subl)[0])  # Canonical form: a pure sublattice is not a sequence
            else:
                if has_desired_order and forbid_cross_interactions:
                    has_desired_order = False  # setting this to false ensures the interaction won't be added
                    break
                if num_constit == order:
                    has_desired_order = True
                config.append(tuple(sorted(subl)))
        if has_desired_order:
            interactions.append(config)
    return sorted_interactions(interactions, order, symmetry)


def interaction_test(configuration, order=None):
    """
    Returns True if the configuration has an interaction

    Parameters
    ----------
    order : int, optional
        Specific order to check for. E.g. a value of 3 checks for ternary interactions

    Returns
    -------
    bool
        True if there is an interaction.

    Examples
    --------
    >>> configuration = [['A'], ['A','B']]
    >>> interaction_test(configuration)
    True
    >>> interaction_test(configuration, order=2)
    True
    >>> interaction_test(configuration, order=3)
    False

    """
    interacting_species = [len(subl) for subl in configuration if isinstance(subl, (tuple,list))]
    if order is None:  # checking for any interaction
        return any([subl_occupation > 1 for subl_occupation in interacting_species])
    else:
        return any([subl_occupation == order for subl_occupation in interacting_species])


def endmembers_from_interaction(configuration):
    """For a given configuration with possible interactions, return all the endmembers"""
    config = []
    for c in configuration:
        if isinstance(c, (list, tuple)):
            config.append(c)
        else:
            config.append([c])
    return list(itertools.product(*[tuple(c) for c in config]))


def generate_endmembers(sublattice_model, symmetry=None):
    """Return all the unique endmembers by symmetry for a given sublattice model.

    Parameters
    ----------
    sublattice_model : list of lists
        General sublattice model, with each sublattice as a sublist.
    symmetry : list of lists, optional
        List of lists containing symmetrically equivalent sublattice indices.
        If None (default), all endmembers will be returned.

    Returns
    -------
    list
        List of endmember tuples

    Examples
    --------
    >>> subl_model = [['A', 'B'], ['A','B']]
    >>> generate_endmembers(subl_model)  # four endmembers
    [('A', 'A'), ('A', 'B'), ('B', 'A'), ('B', 'B')]
    >>> # three endmembers, ('A', 'B') is equivalent to ('B', 'A') by symmetry.
    >>> generate_endmembers(subl_model, [[0, 1]])  # the first and second sublattices are symmetrically equivalent.
    [('A', 'A'), ('A', 'B'), ('B', 'B')]

    """
    return sorted(set(canonicalize(i, symmetry) for i in list(itertools.product(*sublattice_model))))
