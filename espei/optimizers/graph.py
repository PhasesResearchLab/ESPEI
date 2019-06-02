"""
Defines a OptNode and OptGraph to be used by OptimizerBase subclasses.
Together they define the path of one or more optimizations and can be used to
store and replay optimization history.
"""
import copy


class OptNode:
    """
    Node as the result of an optimization.

    Attributes
    ----------
    parameters : dict
    datasets : PickleableTinyDB
    id : int
    parent : OptNode
    children : list of OptNode

    Notes
    -----
    OptNodes are individual nodes in the graph that correspond to the result of
    a call to fit - they represent optimized parameters given the parent state
    and some data (also part of the OptNode).

    Each OptNode can only be derived from one set of parameters, however one
    parameter state may be a branching point to many new parameter states, so an
    OptNode can have only one parent, but many children.

    """
    def __init__(self, parameters, datasets, node_id=None):
        self.parameters = copy.deepcopy(parameters)
        self.datasets = copy.deepcopy(datasets)
        self.id = node_id
        self.parent = None
        self.children = set()

    def __repr__(self):
        str_params = str(self.parameters)
        return "<OptNode({}, node_id={})>".format(str_params, self.id)

    def __str__(self):
        str_params = str(self.parameters)
        return "<OptNode({}, node_id={})>".format(str_params, self.id)

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        return self.id == other.id


class OptGraph:
    """
    Directed acyclic graph of optimal parameters.

    Attributes
    ----------

    Notes
    -----
    The OptGraph defines a directed acyclic graph of commits. Each commit
    corresponds to a single OptNode. The root node is intended to be the fresh
    parameters from the database before any optimization. Therefore, any path
    from the root node to any other node represents a set of optimizations to
    the parameters in the database.
    """
    def __init__(self, root):
        root.id = 0
        root.parent = None
        self._id_counter = 0
        self.root = root
        self.nodes = {self.root.id: root}

    def add_node(self, node, parent):
        node.id = self._get_next_id()
        node.parent = parent
        parent.children.add(node)
        self.nodes[node.id] = node

    def _get_next_id(self):
        self._id_counter += 1
        return self._id_counter

    def __str__(self):
        all_nodes = ''
        for nid, node in self.nodes.items():
            children = ', '.join([str(c.id) for c in node.children])
            all_nodes += '{}: [{}], '.format(nid, children)
        return "<OptGraph({})>".format(all_nodes)

    @staticmethod
    def get_path_to_node(node):
        """
        Return the path from the root to the node.

        Parameters
        ----------
        node : OptNode

        Returns
        -------
        list of OptNode

        """
        rev_path = [node]  # leaf -> root
        while node.parent is not None:
            rev_path.append(node.parent)
            node = node.parent
        return list(reversed(rev_path))  # root -> leaf

    def get_transformation_dict(self, node):
        """
        Return a dictionary of parameters from the path walked from the root to
        the passed node.

        Parameters
        ----------
        node : OptNode

        Returns
        -------
        dict

        """
        path = self.get_path_to_node(node)
        transform_dict = {}
        for node in path:
            transform_dict.update(node.parameters)
        return transform_dict
