import copy
from espei.datasets import load_datasets
from espei.utils import unpack_piecewise, database_symbols_to_fit
from .graph import OptNode, OptGraph
from .utils import OptimizerError

class OptimizerBase():
    """Enable fitting and replaying fitting steps"""
    def __init__(self, dbf):
        self.orig_dbf = copy.deepcopy(dbf)
        self.dbf = copy.deepcopy(dbf)
        parameters = {sym: unpack_piecewise(dbf.symbols[sym]) for sym in database_symbols_to_fit(dbf)}
        ds = load_datasets([])  # empty TinyDB
        root = OptNode(parameters, ds)
        self.current_node = root
        self.graph = OptGraph(root)

    def _fit(self, symbols, datasets, **kwargs):
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

    def fit(self, symbols, datasets, **kwargs):
        node = self._fit(symbols, datasets, **kwargs)
        self.staged_node = node
        return node

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
        if self.staged_node is not None:
            self.graph.add_node(self.staged_node, self.current_node)
            self.current_node = self.staged_node
            self.staged_node = None
            self.dbf.symbols.update(self.current_node.parameters)
        else:
            raise OptimizerError("Nothing to commit. Stage a commit by running the `fit` method.")
