import copy
from espei.datasets import load_datasets
from espei.utils import unpack_piecewise, database_symbols_to_fit
from .graph import OptNode, OptGraph
from .utils import OptimizerError

class OptimizerBase(object):
    """Enable fitting and replaying fitting steps"""
    def __init__(self, dbf):
        self.orig_dbf = copy.deepcopy(dbf)
        self.dbf = copy.deepcopy(dbf)
        parameters = {sym: unpack_piecewise(dbf.symbols[sym]) for sym in database_symbols_to_fit(dbf)}
        ds = load_datasets([])  # empty TinyDB
        root = OptNode(parameters, ds)
        self.current_node = root
        self.staged_nodes = []
        self.graph = OptGraph(root)

    def _fit(self, symbols, datasets, *args, **kwargs):
        """
        Optimize a set of symbols to the passed datasets

        Parameters
        ----------
        symbols : list of str
        datasets : PickleableTinyDB

        Returns
        -------
        OptNode

        """
        raise NotImplementedError("The `_fit` method not implemented. Create a subclass of OptimizerBase with `_fit` overridden to use it")

    def fit(self, symbols, datasets, *args, **kwargs):
        node = self._fit(symbols, datasets, *args, **kwargs)
        self.staged_nodes.append(node)
        self.dbf.symbols.update(node.parameters)
        return self.dbf

    @staticmethod
    def predict(params, context):
        """Given a set of parameters and a context, return the resulting sum of square error.

        Parameters
        ----------
        params : list
            1 dimensional array of parameters
        context : dict
            Dictionary of arguments/keyword arguments to pass to functions

        Returns
        -------
        float
        """
        raise NotImplementedError("The `predict` method not implemented. Create a subclass of OptimizerBase with `predict` overridden to use it")

    def commit(self,):
        if len(self.staged_nodes) > 0:
            for staged in self.staged_nodes:
                self.graph.add_node(staged, self.current_node)
                self.current_node = staged
            self.staged_nodes = []
            self.reset_database()
        else:
            raise OptimizerError("Nothing to commit. Stage a commit by running the `fit` method.")

    def discard(self):
        """Discard all staged nodes"""
        self.staged_nodes = []
        self.reset_database()

    def reset_database(self):
        """Set the Database to the state of the current node"""
        trans_dict = self.graph.get_transformation_dict(self.current_node)
        self.dbf = copy.deepcopy(self.orig_dbf)
        self.dbf.symbols.update(trans_dict)
