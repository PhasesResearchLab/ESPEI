from typing import Any, Dict, Optional
import itertools
from numpy.typing import ArrayLike
import numpy as np
import symengine
from pycalphad import Model, variables as v
from espei.parameter_selection.model_building import make_successive
from espei.utils import build_sitefractions

Dataset = Dict[str, Any]  # Typing stub

__all__ = [
    "FittingStep",
    "AbstractLinearPropertyStep",
    "StepHM",
    "StepSM",
    "StepCPM",
    "StepV0",
    "StepLogVA",
]


class FittingStep():
    parameter_name: str
    data_types_read: str
    features: [symengine.Expr]
    supported_reference_states: [str]

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
            Model with all lower order (in composition) terms already fit. Pure
            element reference state (GHSER functions) should be set to zero.
        fixed_portions : [symengine.Basic]
            SymEngine expressions for model parameters and interaction productions for
            higher order (in T) terms for this property, e.g. [0, 3.0*YS*v.T]. In
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
        # Construct flattened list of site fractions corresponding to the ravelled data
        site_fractions = []
        for ds in data:
            for _ in ds['conditions']['T']:
                sf = build_sitefractions(fixed_model.phase_name, ds['solver']['sublattice_configurations'], ds['solver'].get('sublattice_occupancies', np.ones((len(ds['solver']['sublattice_configurations']), len(ds['solver']['sublattice_configurations'][0])), dtype=np.float_)))
                site_fractions.append(sf)
        site_fractions = list(itertools.chain(*site_fractions))
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
        rhs = np.asarray(rhs, dtype=np.float_)
        return rhs


class StepHM(FittingStep):
    parameter_name: str = "G"
    data_types_read: str = "HM"
    supported_reference_states: [str] = ["_MIX", "_FORM"]
    features: [symengine.Expr] = [symengine.S.One]

    @classmethod
    def transform_feature(cls, expr: symengine.Expr, model: Optional[Model] = None) -> symengine.Expr:
        # expr is the AST for GM, so we need to transform GM features into HM
        # H = G + T*S, with S = -dG/dT
        return expr - v.T*symengine.diff(expr, v.T)

    @classmethod
    def shift_reference_state(cls, desired_data: [Dataset], fixed_model: Model, mole_atoms_per_mole_formula_unit: symengine.Expr) -> ArrayLike:  # np.object_
        """
        Shift _MIX or _FORM data to a common reference state in per mole-atom units.

        Parameters
        ----------
        desired_data : List[Dict[str, Any]]
            ESPEI single phase dataset
        fixed_model : pycalphad.Model
            Model with all lower order (in composition) terms already fit. Pure
            element reference state (GHSER functions) should be set to zero.
        mole_atoms_per_mole_formula_unit : float
            Number of moles of atoms in every mole atom unit.

        Returns
        -------
        np.ndarray
            Data for this feature in [qty]/mole-formula in a common reference state.

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
        for dataset in desired_data:
            values = np.asarray(dataset['values'], dtype=np.object_)*mole_atoms_per_mole_formula_unit
            unique_excluded_contributions = set(dataset.get('excluded_model_contributions', []))
            for config_idx in range(len(dataset['solver']['sublattice_configurations'])):
                occupancy = dataset['solver'].get('sublattice_occupancies', None)
                if dataset['output'].endswith('_FORM'):
                    # we don't shift the reference state because we assume our
                    # models are already in the formation reference state (by us
                    # setting GHSERXX functions to zero explictly)
                    pass
                elif dataset['output'].endswith('_MIX'):
                    if occupancy is None:
                        raise ValueError('Cannot have a _MIX property without sublattice occupancies.')
                    else:
                        values[..., config_idx] += cls.transform_feature(fixed_model.models['ref'])*mole_atoms_per_mole_formula_unit
                else:
                    raise ValueError(f'Unknown property to shift: {dataset["output"]}')
                for excluded_contrib in unique_excluded_contributions:
                    values[..., config_idx] += cls.transform_feature(fixed_model.models[excluded_contrib])*mole_atoms_per_mole_formula_unit
            total_response.append(values.flatten())
        return total_response


    @classmethod
    def get_response_vector(cls, fixed_model: Model, fixed_portions: [symengine.Basic], data: [Dataset], sample_condition_dicts: [Dict[str, Any]]) -> ArrayLike:  # np.float_
        mole_atoms_per_mole_formula_unit = fixed_model._site_ratio_normalization
        # Define site fraction symbols that will be reused
        phase_name = fixed_model.phase_name

        # Construct flattened list of site fractions corresponding to the ravelled data (from shift_reference_state)
        site_fractions = []
        for ds in data:
            for _ in ds['conditions']['T']:
                sf = build_sitefractions(phase_name, ds['solver']['sublattice_configurations'], ds['solver'].get('sublattice_occupancies', np.ones((len(ds['solver']['sublattice_configurations']), len(ds['solver']['sublattice_configurations'][0])), dtype=np.float_)))
                site_fractions.append(sf)
        site_fractions = list(itertools.chain(*site_fractions))

        data_qtys = np.concatenate(cls.shift_reference_state(data, fixed_model, mole_atoms_per_mole_formula_unit), axis=-1)
        # Remove existing partial model contributions from the data, convert to per mole-formula units
        data_qtys = data_qtys - cls.transform_feature(fixed_model.ast)*mole_atoms_per_mole_formula_unit
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
        data_qtys = np.asarray(data_qtys, dtype=np.float_)
        return data_qtys


class StepSM(StepHM):
    data_types_read: str = "SM"
    features: [symengine.Expr] = [v.T]

    @classmethod
    def transform_feature(cls, expr: symengine.Expr, model: Optional[Model] = None) -> symengine.Expr:
        # expr is the AST for GM, so we need to transform GM features into SM
        # S = -dG/dT
        return -symengine.diff(expr, v.T)


class StepCPM(StepHM):
    data_types_read: str = "CPM"
    features: [symengine.Expr] = [v.T * symengine.log(v.T), v.T**2, v.T**-1, v.T**3]

    @classmethod
    def transform_feature(cls, expr: symengine.Expr, model: Optional[Model] = None) -> symengine.Expr:
        # expr is the AST for GM, so we need to transform GM features into CPM
        # CP = T * dS/dT = - T * d^2G / dT^2
        return -v.T*symengine.diff(expr, v.T, 2)


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