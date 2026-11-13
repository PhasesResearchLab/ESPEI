from typing import Any, Dict, List, Optional, Tuple, Type
import itertools
import logging
from numpy.typing import ArrayLike
import numpy as np
import symengine
from tinydb import where
from pycalphad import Model, variables as v
from espei.core_utils import get_prop_data, filter_configurations, filter_temperatures, symmetry_filter
from espei.error_functions.non_equilibrium_thermochemical_error import PROPERTY_STD_DEVIATION, get_prop_samples, get_sample_condition_dicts
from espei.parameter_selection.model_building import make_successive, build_feature_matrix
from espei.sublattice_tools import recursive_tuplify, tuplify
from espei.utils import build_sitefractions

_log = logging.getLogger(__name__)

Dataset = Dict[str, Any]  # Typing stub

__all__ = [
    "FittingStep",
    "AbstractLinearPropertyStep",
    "StepHM",
    "StepSM",
    "StepCPM",
    "StepGM",
    "StepGibbsEnergy",
    "StepV0",
    "StepLogVA",
]


def _data_site_fractions(phase_name: str, data: [Dataset]) -> List[Dict[v.SiteFraction, float]]:
    """Site fractions of each data point of the datasets, in the order of their ravelled values."""
    site_fractions = []
    for ds in data:
        configurations = ds['solver']['sublattice_configurations']
        occupancies = ds['solver'].get('sublattice_occupancies', np.ones((len(configurations), len(configurations[0])), dtype=np.float64))
        for _ in ds['conditions']['T']:
            site_fractions.extend(build_sitefractions(phase_name, configurations, occupancies))
    return site_fractions


class FittingStep():
    parameter_name: str
    data_types_read: str
    features: [symengine.Expr]
    supported_reference_states: [str]
    data_std_deviation: float = 1.0
    data_steps: Tuple[Type["FittingStep"], ...] = ()

    @classmethod
    def transform_feature(cls, expr: symengine.Expr, model: Optional[Model] = None) -> symengine.Expr:
        return expr

    @classmethod
    def get_feature_sets(cls) -> [[symengine.Expr]]:
        return make_successive(cls.features)

    @classmethod
    def get_response_vector(cls, fixed_model: Model, fixed_portions: [symengine.Basic], data: [Dataset], sample_condition_dicts: [Dict[str, Any]]) -> ArrayLike:  # np.float_
        """
        Get the response vector, b, for the a linear models in Ax=b with
        features, A, and coefficients, x, in units of [qty]/mole-formula.

        Parameters
        ----------
        fixed_model : Model
            Model with all lower order (in composition) terms already fit.
        fixed_portions : [symengine.Basic]
            SymEngine expressions for the parts of the parameter being fit
            that are already known, i.e. the reference part of the parameter
            (e.g. GHSER functions for endmember Gibbs energies) and the terms
            fit by previous steps, e.g. [GHSERAL, 3.0*YS*v.T]. In
            [qty]/mole-formula.
        data : [Dict[str, Any]]
            ESPEI single phase datasets for this property.

        Returns
        -------
        np.ndarray[:]
            Ravelled data quantities in [qty]/mole-formula

        Notes
        -----
        pycalphad Model parameters (and therefore fixed_portions) are stored as per
        mole-formula quantites, but the calculated properties and our data are all
        in [qty]/mole-atoms. We multiply by mole-atoms/mole-formula to convert the
        units to [qty]/mole-formula.

        """
        raise NotImplementedError()

    @classmethod
    def get_data(cls, comps: List[str], phase_name: str, configuration, symmetry, datasets) -> List[Dataset]:
        """
        Return copies of the datasets used by this step to fit a configuration.

        Parameters
        ----------
        comps : List[str]
            Names of the relevant components.
        phase_name : str
            Name of the phase being fit.
        configuration : Tuple[Union[str, Tuple[str]]]
            Sublattice configuration being fit.
        symmetry : Optional[List[List[int]]]
            Symmetry of the sublattice configuration.
        datasets : PickleableTinyDB
            All the datasets that could be fit.

        Returns
        -------
        List[Dataset]

        """
        solver_qry = (where('solver').test(symmetry_filter, configuration, recursive_tuplify(symmetry) if symmetry else symmetry))
        desired_props = [step.data_types_read + refstate for step in (cls.data_steps or (cls,)) for refstate in step.supported_reference_states]
        desired_data = get_prop_data(comps, phase_name, desired_props, datasets, additional_query=solver_qry)
        desired_data = filter_configurations(desired_data, configuration, symmetry)
        desired_data = filter_temperatures(desired_data)
        _log.trace('%s: datasets found: %s', desired_props, len(desired_data))
        return desired_data

    @classmethod
    def get_linear_system(cls, fixed_model: Model, fixed_portions: [symengine.Basic], data: [Dataset], configuration, features: [symengine.Expr]) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
        """
        Return the features at the data, the response vector, and the standard deviation and weight of each data point.

        Parameters
        ----------
        fixed_model : Model
            Model with all lower order (in composition) terms already fit.
        fixed_portions : [symengine.Basic]
            SymEngine expressions for the known parts of the parameter being
            fit, in [qty]/mole-formula.
        data : [Dataset]
            Datasets returned by ``get_data``.
        configuration : Tuple[Union[str, Tuple[str]]]
            Sublattice configuration being fit.
        features : [symengine.Expr]
            Features of the parameter, e.g. ``YS*v.T``.

        Returns
        -------
        Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]
            Feature matrix (data points by features), response vector, standard deviations and weights.

        """
        config_tup = tuple(map(tuplify, configuration))
        feature_matrices, responses, std_deviations, weights = [], [], [], []
        for step in (cls.data_steps or (cls,)):
            outputs = [step.data_types_read + refstate for refstate in step.supported_reference_states]
            step_data = [ds for ds in data if ds['output'] in outputs]
            if len(step_data) == 0:
                continue
            calculate_dict = get_prop_samples(step_data, config_tup)
            sample_condition_dicts = get_sample_condition_dicts(calculate_dict, config_tup, fixed_model.phase_name)
            responses.append(step.get_response_vector(fixed_model, fixed_portions, step_data, sample_condition_dicts))
            feature_matrices.append(build_feature_matrix(sample_condition_dicts, [step.transform_feature(f) for f in features]))
            std_deviations.append(np.full(len(sample_condition_dicts), step.data_std_deviation))
            weights.append(np.asarray(calculate_dict['weights'], dtype=np.float64))
        return np.vstack(feature_matrices), np.concatenate(responses), np.concatenate(std_deviations), np.concatenate(weights)


class AbstractLinearPropertyStep(FittingStep):
    """
    This class is a base class for generic linear properties.

    "Generic" meaning, essentially, that
    `property_name == parameter_name == data_type` and PyCalphad models
    set this property something like
    ```python
    self.<parameter_name> = self.redlich_kister_sum(phase, param_search, <param_name>_query)
    ```
    Note that `redlich_kister_sum` specifically is an implementation detail and
    isn't relevant for this class in particular. Any mixing model could work,
    although ESPEI currently only generates features by
    `build_redlich_kister_candidate_models`.

    Fitting steps that want to use this base class should need to subclass this
    class, override the `parameter_name` and `data_type_read` (typically these
    match), and optionally override the `features` if a desired.

    For models that are 'nearly linear', and the `transform_data` method can be
    overriden to try to linearize the data with respect to the model parameters.

    Most parameters are fit per mole of formula units, but there are some
    exceptions (e.g. VA parameters). The `normalize_parameter_per_mole_formula`
    attribute handles normalization.
    """
    features: [symengine.Expr] = [symengine.S.One, v.T, v.T**2, v.T**3, v.T**(-1)]
    supported_reference_states: [str] = ["", "_MIX"]
    normalize_parameter_per_mole_formula: bool = True

    @staticmethod
    def transform_data(d: ArrayLike, model: Optional[Model] = None) -> ArrayLike:  # np.object_
        """Helper function to linearize data in terms of the model parameters.

        If data is already linear w.r.t. the parameters, the default
        implementation of identity (`return d` can be preserved).
        """
        return d

    @classmethod
    def shift_reference_state(cls, desired_data: [Dataset], fixed_model: Model) -> ArrayLike:  # np.object_
        # Shift all data into absolute values.
        # Old versions of this code used to shift to units of per-mole-formula,
        # but we no longer do that here.
        total_response = []
        for dataset in desired_data:
            values = np.asarray(dataset['values'], dtype=np.object_)
            for config_idx in range(len(dataset['solver']['sublattice_configurations'])):
                occupancy = dataset['solver'].get('sublattice_occupancies', None)
                if dataset['output'].endswith('_FORM'):
                    raise NotImplementedError(f"The formation reference state for data type {dataset['output']} is not yet supported for {cls.parameter_name} parameters.")
                elif dataset['output'].endswith('_MIX'):
                    # we treat this using the special "endmember" reference
                    # state (i.e. where the property of interest is zero for all
                    # the endmembers). To shift our VM_MIX data into VM, we need
                    # to add the VM computed from the endmember reference model.
                    values[..., config_idx] += getattr(fixed_model.endmember_reference_model, cls.data_types_read)
                else:
                    pass
            total_response.append(values.flatten())
        return total_response

    @classmethod
    def get_response_vector(cls, fixed_model: Model, fixed_portions: [symengine.Basic], data: [Dataset], sample_condition_dicts: [Dict[str, Any]]) -> ArrayLike:  # np.float_
        mole_atoms_per_mole_formula_unit = fixed_model._site_ratio_normalization
        # This function takes Dataset objects (`data`) -> values array (of np.object_)
        rhs = np.concatenate(cls.shift_reference_state(data, fixed_model), axis=-1)
        rhs = cls.transform_data(rhs, fixed_model)

        # RKM models are already liner in the parameters, so we don't need to
        # do anything too special here.
        # subtract off lower order contributions.
        for i in range(rhs.shape[0]):
            rhs[i] -= getattr(fixed_model, cls.parameter_name)
            if cls.normalize_parameter_per_mole_formula:
                # Convert the quantity per-mole-atoms to per-mole-formula
                rhs[i] *= mole_atoms_per_mole_formula_unit

        # Previous steps may have introduced some symbolic terms.
        # Now we remove all the symbols:
        site_fractions = _data_site_fractions(fixed_model.phase_name, data)
        # If any site fractions show up in our rhs that aren't in these
        # datasets' site fractions, set them to zero. This can happen if we're
        # fitting a multi-component model that has site fractions from
        # components that aren't in a particular dataset
        for sf, value_with_symbols, cond_dict in zip(site_fractions, rhs, sample_condition_dicts):
            missing_variables = symengine.S(value_with_symbols).atoms(v.SiteFraction) - set(sf.keys())
            sf.update({x: 0. for x in missing_variables})
            sf.update(cond_dict)
        # also replace with database symbols in case we did higher order fitting
        rhs = [fixed_model.symbol_replace(symengine.S(value_with_symbols).xreplace(sf), fixed_model._symbols).evalf() for value_with_symbols, sf in zip(rhs, site_fractions)]
        # cast to float, confirming that these are concrete values with no sybolics
        rhs = np.asarray(rhs, dtype=np.float64)
        return rhs


class StepGM(FittingStep):
    """Fit Gibbs energy parameters to Gibbs energy (GM) data."""
    parameter_name: str = "G"
    data_types_read: str = "GM"
    supported_reference_states: [str] = ["", "_MIX", "_FORM"]
    features: [symengine.Expr] = [symengine.S.One, v.T]
    data_std_deviation: float = PROPERTY_STD_DEVIATION["GM"]

    @classmethod
    def transform_feature(cls, expr: symengine.Expr, model: Optional[Model] = None) -> symengine.Expr:
        # expr is the AST for GM, so GM features need no transformation
        return expr

    @classmethod
    def get_formation_reference(cls, fixed_model: Model) -> symengine.Expr:
        """
        Return the Gibbs energy of the pure element reference state of
        ``_FORM`` data in J/mole-formula as a function of site fractions.

        The reference state is the stable element reference (SER) defined by
        the GHSER functions (e.g. ``GHSERAL``) in the Database. Subclasses can
        override this to use a different definition of the reference state.

        Parameters
        ----------
        fixed_model : pycalphad.Model
            Model of the phase being fit.

        Returns
        -------
        symengine.Expr

        """
        reference = symengine.S.Zero
        for subl_idx, (site_ratio, constituents) in enumerate(zip(fixed_model.site_ratios, fixed_model.constituents)):
            for species in constituents:
                for element, amount in species.constituents.items():
                    if element == "VA":
                        continue
                    ghser = symengine.Symbol("GHSER" + (element.upper()*2)[:2])
                    reference += site_ratio * amount * v.Y(fixed_model.phase_name, subl_idx, species) * ghser
        return fixed_model.symbol_replace(reference, fixed_model._symbols)

    @classmethod
    def shift_reference_state(cls, desired_data: [Dataset], fixed_model: Model, mole_atoms_per_mole_formula_unit: symengine.Expr) -> ArrayLike:  # np.object_
        """
        Shift absolute, _MIX or _FORM data to absolute values in per mole-formula units.

        Parameters
        ----------
        desired_data : List[Dict[str, Any]]
            ESPEI single phase dataset
        fixed_model : pycalphad.Model
            Model with all lower order (in composition) terms already fit.
        mole_atoms_per_mole_formula_unit : float
            Number of moles of atoms in every mole atom unit.

        Returns
        -------
        np.ndarray
            Absolute data for this feature in [qty]/mole-formula.

        Raises
        ------
        ValueError

        Notes
        -----
        pycalphad Model parameters are stored as per mole-formula quantites, but
        the calculated properties and our data are all in [qty]/mole-atoms. We
        multiply by mole-atoms/mole-formula to convert the units to
        [qty]/mole-formula.

        """
        total_response = []
        formation_reference = None  # delay instantiation until needed
        for dataset in desired_data:
            values = np.asarray(dataset['values'], dtype=np.object_)*mole_atoms_per_mole_formula_unit
            unique_excluded_contributions = set(dataset.get('excluded_model_contributions', []))
            for config_idx in range(len(dataset['solver']['sublattice_configurations'])):
                occupancy = dataset['solver'].get('sublattice_occupancies', None)
                if dataset['output'] == cls.data_types_read:
                    # absolute data need no shift
                    pass
                elif dataset['output'].endswith('_FORM'):
                    # add the pure element reference (already per mole-formula)
                    if formation_reference is None:
                        formation_reference = cls.transform_feature(cls.get_formation_reference(fixed_model))
                    values[..., config_idx] += formation_reference
                elif dataset['output'].endswith('_MIX'):
                    if occupancy is None:
                        raise ValueError('Cannot have a _MIX property without sublattice occupancies.')
                    else:
                        values[..., config_idx] += cls.transform_feature(fixed_model.models['ref'])
                else:
                    raise ValueError(f'Unknown property to shift: {dataset["output"]}')
                for excluded_contrib in unique_excluded_contributions:
                    values[..., config_idx] += cls.transform_feature(fixed_model.models[excluded_contrib])
            total_response.append(values.flatten())
        return total_response


    @classmethod
    def get_response_vector(cls, fixed_model: Model, fixed_portions: [symengine.Basic], data: [Dataset], sample_condition_dicts: [Dict[str, Any]]) -> ArrayLike:  # np.float_
        mole_atoms_per_mole_formula_unit = fixed_model._site_ratio_normalization
        # Define site fraction symbols that will be reused
        phase_name = fixed_model.phase_name

        site_fractions = _data_site_fractions(phase_name, data)

        data_qtys = np.concatenate(cls.shift_reference_state(data, fixed_model, mole_atoms_per_mole_formula_unit), axis=-1)
        # Remove existing partial model contributions from the data
        data_qtys = data_qtys - cls.transform_feature(fixed_model.ast)
        # Subtract out high-order (in T) parameters we've already fit, already in per mole-formula units
        data_qtys = data_qtys - cls.transform_feature(sum(fixed_portions))
        # If any site fractions show up in our rhs that aren't in these
        # datasets' site fractions, set them to zero. This can happen if we're
        # fitting a multi-component model that has site fractions from
        # components that aren't in a particular dataset
        for sf, i, cond_dict in zip(site_fractions, data_qtys, sample_condition_dicts):
            missing_variables = symengine.S(i).atoms(v.SiteFraction) - set(sf.keys())
            sf.update({x: 0. for x in missing_variables})
            sf.update(cond_dict)
        # also replace with database symbols in case we did higher order fitting
        data_qtys = [fixed_model.symbol_replace(symengine.S(i).xreplace(sf), fixed_model._symbols).evalf() for i, sf in zip(data_qtys, site_fractions)]
        data_qtys = np.asarray(data_qtys, dtype=np.float64)
        return data_qtys


class StepHM(StepGM):
    data_types_read: str = "HM"
    features: [symengine.Expr] = [symengine.S.One]
    data_std_deviation: float = PROPERTY_STD_DEVIATION["HM"]

    @classmethod
    def transform_feature(cls, expr: symengine.Expr, model: Optional[Model] = None) -> symengine.Expr:
        # expr is the AST for GM, so we need to transform GM features into HM
        # H = G + T*S, with S = -dG/dT
        return expr - v.T*symengine.diff(expr, v.T)


class StepSM(StepGM):
    data_types_read: str = "SM"
    features: [symengine.Expr] = [v.T]
    data_std_deviation: float = PROPERTY_STD_DEVIATION["SM"]

    @classmethod
    def transform_feature(cls, expr: symengine.Expr, model: Optional[Model] = None) -> symengine.Expr:
        # expr is the AST for GM, so we need to transform GM features into SM
        # S = -dG/dT
        return -symengine.diff(expr, v.T)


class StepCPM(StepGM):
    data_types_read: str = "CPM"
    features: [symengine.Expr] = [v.T * symengine.log(v.T), v.T**2, v.T**-1, v.T**3]
    data_std_deviation: float = PROPERTY_STD_DEVIATION["CPM"]

    @classmethod
    def transform_feature(cls, expr: symengine.Expr, model: Optional[Model] = None) -> symengine.Expr:
        # expr is the AST for GM, so we need to transform GM features into CPM
        # CP = T * dS/dT = - T * d^2G / dT^2
        return -v.T*symengine.diff(expr, v.T, 2)


class StepGibbsEnergy(StepGM):
    """Fit Gibbs energy parameters to GM, HM, SM and CPM data together."""
    data_steps: Tuple[Type[FittingStep], ...] = (StepCPM, StepSM, StepHM, StepGM)
    features: [symengine.Expr] = StepHM.features + StepSM.features + StepCPM.features


class StepV0(AbstractLinearPropertyStep):
    parameter_name: str = "V0"
    data_types_read: str = "V0"
    features: [symengine.Expr] = [symengine.S.One]

class StepLogVA(AbstractLinearPropertyStep):
    parameter_name: str = "VA"
    data_types_read: str = "VM"
    features: [symengine.Expr] = [v.T, v.T**2, v.T**3, v.T**(-1)]
    supported_reference_states: [str] = ["", "_MIX"]
    normalize_parameter_per_mole_formula: bool = False

    @staticmethod
    def transform_data(d: ArrayLike, model: Model) -> ArrayLike:  # np.object_
        # We are given samples of volume (VM) as our data (d) with the model:
        # \[ V_0 * exp( V_A ) = VM \]
        # We linearize in terms of the parameter that we want to fit (VA) by:
        # \[ V_A = \log(VM / V_0) \]
        # cast to object_ because the real type may become a symengine.Expr
        d = np.asarray(d, dtype=np.object_)
        for i in range(d.shape[0]):
            d[i] = symengine.log(d[i] / model.V0)
        return d

    @classmethod
    def get_feature_sets(cls) -> [[symengine.Expr]]:
        # All combinations of features
        return list(itertools.chain(*(itertools.combinations(cls.features, n) for n in range(1, len(cls.features)+1))))