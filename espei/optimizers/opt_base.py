import copy
from espei.datasets import load_datasets
from espei.utils import unpack_piecewise, database_symbols_to_fit
from .utils import OptimizerError

def _copy_database(dbf):
    "Copy of a Database whose symbols can be updated without changing the original."
    # pycalphad's Database.__deepcopy__ shares symbols (and phases) with the original
    dbf_copy = copy.deepcopy(dbf)
    dbf_copy.symbols = dict(dbf.symbols)
    return dbf_copy

class OptimizerBase(object):
    """Enable fitting and replaying fitting steps"""
    def __init__(self, dbf):
        self.orig_dbf = _copy_database(dbf)
        self.dbf = _copy_database(dbf)

    def _fit(self, symbols, datasets, *args, **kwargs):
        """
        Optimize a set of symbols to the passed datasets

        Parameters
        ----------
        symbols : list of str
        datasets : PickleableTinyDB

        Returns
        -------
        Dict[str, float]

        """
        raise NotImplementedError("The `_fit` method not implemented. Create a subclass of OptimizerBase with `_fit` overridden to use it")

    def fit(self, symbols, datasets, *args, **kwargs):
        parameters = self._fit(symbols, datasets, *args, **kwargs)
        self.dbf.symbols.update(parameters)
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
