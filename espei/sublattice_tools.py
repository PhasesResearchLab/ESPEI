"""
Utilities for manipulating sublattice models.
"""

import itertools

import numpy as np


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


def generate_symmetric_group(configuration, symmetry):
    """
    For a particular configuration and list of sublattices with symmetry,
    generate all the symmetrically equivalent configurations.

    Parameters
    ----------
    configuration : tuple
        Tuple of a sublattice configuration.
    symmetry : list of lists
        List of lists containing symmetrically equivalent sublattice indices,
        e.g. [[0, 1], [2, 3]] means that sublattices 0 and 1 are equivalent and
        sublattices 2 and 3 are also equivalent.

    Returns
    -------
    tuple
        Tuple of configuration tuples that are all symmetrically equivalent.

    """
    configurations = [recursive_tuplify(configuration)]
    permutation = np.array(symmetry, dtype=np.object_)

    def permute(x):
        if len(x) == 0:
            return x
        x[0] = np.roll(x[0], 1)
        x[:] = np.roll(x, 1, axis=0)
        return x

    if symmetry is not None:
        while np.any(np.array(symmetry, dtype=np.object_) != permute(permutation)):
            new_conf = np.array(configurations[0], dtype=np.object_)
            subgroups = []
            # There is probably a more efficient way to do this
            for subl in permutation:
                subgroups.append([configuration[idx] for idx in subl])
            # subgroup is ordered according to current permutation
            # but we'll index it based on the original symmetry
            # This should permute the configurations
            for subl, subgroup in zip(symmetry, subgroups):
                for subl_idx, conf_idx in enumerate(subl):
                    new_conf[conf_idx] = subgroup[subl_idx]
            configurations.append(recursive_tuplify(new_conf.tolist()))

    return sorted(set(configurations), key=canonical_sort_key)


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


def generate_interactions(endmembers, order, symmetry):
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

    Returns
    -------
    list
        List of interaction tuples, e.g. [('A', ('A', 'B'))]

    """
    interactions = list(itertools.combinations(endmembers, order))
    transformed_interactions = []
    for endmembers in interactions:
        interaction = []
        has_correct_interaction_order = False  # flag to check that we have interactions of at least interaction_order
        # occupants is a tuple of each endmember's ith constituent, looping through i
        for occupants in zip(*endmembers):
            # if all occupants are the same, the ith element of the interaction is not an interacting element
            if all([occupants[0] == x for x in occupants[1:]]):
                interaction.append(occupants[0])
            else:  # there is an interaction
                interacting_species = tuple(sorted(set(occupants)))
                if len(interacting_species) == order:
                    has_correct_interaction_order = True
                interaction.append(interacting_species)
        # only add this interaction if it has an interaction of the desired order.
        # that is, throw away interactions that degenerate to a lower order
        if has_correct_interaction_order:
            transformed_interactions.append(interaction)
    return sorted_interactions(transformed_interactions, order, symmetry)


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
