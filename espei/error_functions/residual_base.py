from typing import Dict, List, Optional, Protocol, Tuple, Type

from numpy.typing import ArrayLike

from pycalphad import Database

from espei.phase_models import PhaseModelSpecification
from espei.typing import SymbolName
from espei.utils import PickleableTinyDB


class ResidualFunction(Protocol):
    """
    Protocol class for computing the error (residual) between data and a model
    prediction given a set of parameters.

    Classes that implement this protocol typically will concerned with
    implementing a residual/likelihood function for a certain type of data. The
    protocol is intentitionally left to be simple to implement while enabling
    flexibility to implement additional methods and attributes for performance
    and debugging purposes.

    Parameters
    ----------
    database : Database
    datasets : PickleableTinyDB
        The candidate datasets for the a contribution. Usually the datasets
        are a superset (in components, phases, data types, etc.) of the actual
        data used to compute the residual for any particular contribution.
    phase_models : PhaseModels
        Defines the active set of components that should be fit and any
        user-provided overrides to the PyCalphad Model class for each phase.
    symbols_to_fit : Optional[List[SymbolName]]
        User-provided symbols to fit. By default, the symbols to fit should be
        set by ``espei.utils.database_symbols_to_fit``.
    weight : Optional[float]
        When computing the likelihood, this should be used to modify the
        probability distribution. Higher weights should correspond to narrower
        probability distributions, but it's exact use will depend on the
        particular probability distribution.

    Attributes
    ----------
    weight : Optional[Union[float, Dict[str, float]]]

    """

    def __init__(
        self,
        database: Database,
        datasets: PickleableTinyDB,
        phase_models: PhaseModelSpecification,
        symbols_to_fit: Optional[List[SymbolName]] = None,
        weight: Optional[Dict[str, float]] = None,
        ):
        ...

    def get_residuals(self, parameters: ArrayLike) -> Tuple[List[float], List[float]]:
        """
        Return the residual comparing the selected data to the set of parameters.

        The residual is zero if the database predictions under the given
        parameters agrees with the data exactly.

        Parameters
        ----------
        parameters : ArrayLike
            1D parameter vector. The size of the parameters array should match
            the number of fitting symbols used to build the models. This is
            _not_ checked.

        Returns
        -------
        Tuple[List[float], List[float]]
            Tuple of (residuals, weights), which must obey len(residuals) == len(weights)

        """
        ...

    def get_likelihood(self, parameters) -> float:
        """
        Return log-likelihood for the set of parameters.

        Parameters
        ----------
        parameters : ArrayLike
            1D parameter vector. The size of the parameters array should match
            the number of fitting symbols used to build the models. This is
            _not_ checked.

        Returns
        -------
        float
            Value of log-likelihood for the given set of parameters
        """
        ...


class ResidualRegistry():
    def __init__(self) -> None:
        self._registered_residual_functions: Type[ResidualFunction] = []

    def get_registered_residual_functions(self) -> List[Type[ResidualFunction]]:
        return self._registered_residual_functions

    def register(self, residual_function: Type[ResidualFunction]):
        # Don't allow duplicates
        if residual_function not in self._registered_residual_functions:
            self._registered_residual_functions.append(residual_function)

residual_function_registry = ResidualRegistry()
