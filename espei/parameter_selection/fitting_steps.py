from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Type
import itertools
import logging
from numpy.typing import ArrayLike
import numpy as np
import symengine
from tinydb import where
from pycalphad import Model, variables as v
from espei.core_utils import get_prop_data, filter_configurations, filter_temperatures, symmetry_filter
from espei.error_functions.non_equilibrium_thermochemical_error import get_prop_samples, get_sample_condition_dicts
from espei.parameter_selection.model_building import make_successive, build_feature_matrix, build_redlich_kister_candidate_models
from espei.parameter_selection.selection import select_model, fit_model, aicc_correction
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
    "StepJointGibbsEnergy",
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
        desired_props = [cls.data_types_read + refstate for refstate in cls.supported_reference_states]
        desired_data = get_prop_data(comps, phase_name, desired_props, datasets, additional_query=solver_qry)
        desired_data = filter_configurations(desired_data, configuration, symmetry)
        desired_data = filter_temperatures(desired_data)
        _log.trace('%s: datasets found: %s', desired_props, len(desired_data))
        return desired_data

    @classmethod
    def select_parameters(cls, fixed_model: Model, fixed_portions: [symengine.Basic], desired_data: [Dataset], configuration, ridge_alpha: Optional[float] = None, aicc_feature_factors: Optional[Dict[str, float]] = None) -> Optional[Tuple[List[symengine.Expr], ArrayLike]]:
        """
        Build candidate models for this step, then fit and select the best one.

        Parameters
        ----------
        fixed_model : Model
            Model with all lower order (in composition) terms already fit.
        fixed_portions : [symengine.Basic]
            SymEngine expressions for the known parts of the parameter being
            fit (reference part and terms fit by previous steps), in
            [qty]/mole-formula.
        desired_data : [Dataset]
            Datasets returned by ``get_data``.
        configuration : Tuple[Union[str, Tuple[str]]]
            Sublattice configuration being fit.
        ridge_alpha : Optional[float]
            Value of the :math:`\\alpha` hyperparameter used in ridge regression.
            If None, ordinary least squares is used.
        aicc_feature_factors : Optional[Dict[str, float]]
            Map of data type (e.g. ``"HM"``) to a multiplication factor for the AICc's parameter penalty.

        Returns
        -------
        Optional[Tuple[List[symengine.Expr], ArrayLike]]
            Tuple of (selected features, fitted coefficients), or None if no model was selected.

        """
        aicc_feature_factors = aicc_feature_factors if aicc_feature_factors is not None else {}
        config_tup = tuple(map(tuplify, configuration))
        calculate_dict = get_prop_samples(desired_data, config_tup)
        sample_condition_dicts = get_sample_condition_dicts(calculate_dict, config_tup, fixed_model.phase_name)
        response_vector = cls.get_response_vector(fixed_model, fixed_portions, desired_data, sample_condition_dicts)
        candidate_models = []
        for features in build_redlich_kister_candidate_models(configuration, cls.get_feature_sets()):
            feature_matrix = build_feature_matrix(sample_condition_dicts, list(map(cls.transform_feature, features)))
            candidate_models.append((features, feature_matrix, response_vector))
        aicc_factor = aicc_feature_factors.get(cls.data_types_read, 1.0)
        weights = calculate_dict['weights']
        assert len(sample_condition_dicts) == len(weights)
        return select_model(candidate_models, ridge_alpha, weights=weights, aicc_factor=aicc_factor)


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
                sf = build_sitefractions(fixed_model.phase_name, ds['solver']['sublattice_configurations'], ds['solver'].get('sublattice_occupancies', np.ones((len(ds['solver']['sublattice_configurations']), len(ds['solver']['sublattice_configurations'][0])), dtype=np.float64)))
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
        rhs = np.asarray(rhs, dtype=np.float64)
        return rhs


class StepGM(FittingStep):
    """Fit Gibbs energy parameters to Gibbs energy (GM) data."""
    parameter_name: str = "G"
    data_types_read: str = "GM"
    supported_reference_states: [str] = ["", "_MIX", "_FORM"]
    features: [symengine.Expr] = [symengine.S.One, v.T]

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

        # Construct flattened list of site fractions corresponding to the ravelled data (from shift_reference_state)
        site_fractions = []
        for ds in data:
            for _ in ds['conditions']['T']:
                sf = build_sitefractions(phase_name, ds['solver']['sublattice_configurations'], ds['solver'].get('sublattice_occupancies', np.ones((len(ds['solver']['sublattice_configurations']), len(ds['solver']['sublattice_configurations'][0])), dtype=np.float64)))
                site_fractions.append(sf)
        site_fractions = list(itertools.chain(*site_fractions))

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

    @classmethod
    def transform_feature(cls, expr: symengine.Expr, model: Optional[Model] = None) -> symengine.Expr:
        # expr is the AST for GM, so we need to transform GM features into HM
        # H = G + T*S, with S = -dG/dT
        return expr - v.T*symengine.diff(expr, v.T)


class StepSM(StepGM):
    data_types_read: str = "SM"
    features: [symengine.Expr] = [v.T]

    @classmethod
    def transform_feature(cls, expr: symengine.Expr, model: Optional[Model] = None) -> symengine.Expr:
        # expr is the AST for GM, so we need to transform GM features into SM
        # S = -dG/dT
        return -symengine.diff(expr, v.T)


class StepCPM(StepGM):
    data_types_read: str = "CPM"
    features: [symengine.Expr] = [v.T * symengine.log(v.T), v.T**2, v.T**-1, v.T**3]

    @classmethod
    def transform_feature(cls, expr: symengine.Expr, model: Optional[Model] = None) -> symengine.Expr:
        # expr is the AST for GM, so we need to transform GM features into CPM
        # CP = T * dS/dT = - T * d^2G / dT^2
        return -v.T*symengine.diff(expr, v.T, 2)


class _JointDataBlock(NamedTuple):
    step: Type[FittingStep]
    rows: slice
    sample_condition_dicts: List[Dict[Any, float]]
    response_vector: ArrayLike
    weights: ArrayLike


class _JointFeatureGroup(NamedTuple):
    step: Type[FittingStep]
    candidates: List[Tuple[symengine.Expr, ...]]
    universe: List[symengine.Expr]
    owned: bool  # True if the step's own data type is present
    resolving_blocks: List[int]  # indices of data blocks where these features are non-zero


def _column_rank(matrix: ArrayLike, rtol: float) -> int:
    """Numerical rank of a matrix after normalizing each non-zero column to unit length."""
    if matrix.size == 0:
        return 0
    norms = np.linalg.norm(matrix, axis=0)
    matrix = matrix[:, norms > 0] / norms[norms > 0]
    if matrix.shape[1] == 0:
        return 0
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    return int(np.sum(singular_values > rtol * singular_values[0]))


class StepJointGibbsEnergy(FittingStep):
    """
    Fit Gibbs energy parameters to GM, HM, SM and CPM data simultaneously.

    The sequential Gibbs energy fitting description fits CPM, then SM, then HM
    data, fixing the parameters selected in each step. Each sequential step
    only fits the features of its own data type, so any temperature dependence
    in HM or SM data is not used to fit heat capacity features. Since GM, HM,
    SM and CPM are all linear transformations of the same Gibbs energy
    coefficients, this step stacks all the data into one linear least squares
    problem instead.

    Candidate models are the Cartesian product of the candidate models of each
    step in ``parameter_steps``, where each step's features form a group:

    * If a group's own data type is present, the group always contributes
      features, exactly as in the sequential fit.
    * If a group's own data type is absent, the group may contribute no
      features. It may also contribute features that the other data types
      can resolve, i.e. features whose columns are linearly independent of
      all the columns of the groups with their own data. For example, HM data
      at several temperatures can resolve heat capacity features, but HM data
      at a single temperature cannot. Temperature dependent features also
      require the data that resolve them to span at least
      ``min_temperature_range``.

    Each data type's rows are scaled by ``1/property_std_deviation`` so that
    residuals with different units are commensurate in the fit. Dataset
    weights are applied as in the sequential steps.

    Candidates are scored by summing a modified AICc log-likelihood term for
    each data type, each with its own profiled variance, and the AICc penalty
    for each group of features. A group with its own data is penalized
    against the number of points of that data type. A group without its own
    data is penalized against the number of points of the data types that
    resolve it. The AICc penalty factor for each group is looked up by the
    group's data type (e.g. ``"CPM"`` for the heat capacity features).

    When the data decouple, the parameters are the same as the sequential
    fit. One example is CPM data at many temperatures with HM and SM data at
    a single temperature. Another is HM and SM data at a single temperature
    with no CPM data. The results differ when HM, SM or GM data have
    temperature dependence that the sequential fit discards.
    """
    parameter_name: str = "G"
    # Same order as the sequential fitting description, so the fitted parameters are inserted in the same order
    parameter_steps: Tuple[Type[FittingStep], ...] = (StepCPM, StepSM, StepHM)
    data_steps: Tuple[Type[FittingStep], ...] = (StepGM, StepHM, StepSM, StepCPM)
    # Same as the defaults for non-equilibrium thermochemical data in MCMC
    property_std_deviation: Dict[str, float] = {"GM": 500.0, "HM": 500.0, "SM": 0.2, "CPM": 0.2}
    # Temperature dependent features of groups without their own data are only
    # candidates if the data that resolve them span at least this temperature
    # range (K). Otherwise, e.g., HM data from different sources at 298.15 K
    # and 299 K could be "resolved" by unphysical heat capacity features.
    min_temperature_range: float = 100.0
    rank_rtol: float = 1.0e-9
    rss_numerical_limit: float = 1.0e-16

    @classmethod
    def get_data(cls, comps: List[str], phase_name: str, configuration, symmetry, datasets) -> List[Dataset]:
        desired_data = []
        for data_step in cls.data_steps:
            desired_data.extend(data_step.get_data(comps, phase_name, configuration, symmetry, datasets))
        return desired_data

    @classmethod
    def _build_data_blocks(cls, fixed_model: Model, fixed_portions: [symengine.Basic], desired_data: [Dataset], configuration) -> List[_JointDataBlock]:
        config_tup = tuple(map(tuplify, configuration))
        blocks = []
        num_rows = 0
        for data_step in cls.data_steps:
            outputs = [data_step.data_types_read + refstate for refstate in data_step.supported_reference_states]
            data = [ds for ds in desired_data if ds['output'] in outputs]
            if len(data) == 0:
                continue
            calculate_dict = get_prop_samples(data, config_tup)
            if calculate_dict['values'].size == 0:
                continue
            sample_condition_dicts = get_sample_condition_dicts(calculate_dict, config_tup, fixed_model.phase_name)
            response_vector = data_step.get_response_vector(fixed_model, fixed_portions, data, sample_condition_dicts)
            weights = np.asarray(calculate_dict['weights'], dtype=np.float64)
            assert len(sample_condition_dicts) == len(weights) == len(response_vector)
            rows = slice(num_rows, num_rows + len(response_vector))
            num_rows += len(response_vector)
            blocks.append(_JointDataBlock(data_step, rows, sample_condition_dicts, response_vector, weights))
        return blocks

    @classmethod
    def select_parameters(cls, fixed_model: Model, fixed_portions: [symengine.Basic], desired_data: [Dataset], configuration, ridge_alpha: Optional[float] = None, aicc_feature_factors: Optional[Dict[str, float]] = None) -> Optional[Tuple[List[symengine.Expr], ArrayLike]]:
        aicc_feature_factors = aicc_feature_factors if aicc_feature_factors is not None else {}
        blocks = cls._build_data_blocks(fixed_model, fixed_portions, desired_data, configuration)
        if len(blocks) == 0:
            return None
        present_data_types = [block.step.data_types_read for block in blocks]

        groups = []
        for parameter_step in cls.parameter_steps:
            candidates = [tuple(c) for c in build_redlich_kister_candidate_models(configuration, parameter_step.get_feature_sets())]
            universe = list(dict.fromkeys(itertools.chain.from_iterable(candidates)))
            resolving_blocks = [i for i, block in enumerate(blocks) if any(block.step.transform_feature(f) != 0 for f in parameter_step.features)]
            owned = parameter_step.data_types_read in present_data_types
            groups.append(_JointFeatureGroup(parameter_step, candidates, universe, owned, resolving_blocks))
        groups = [g for g in groups if g.owned or len(g.resolving_blocks) > 0]

        # Feature matrix (unscaled) with all possible features as columns
        all_features = list(itertools.chain.from_iterable(g.universe for g in groups))
        feature_column = {f: j for j, f in enumerate(all_features)}
        feature_matrix = np.vstack([build_feature_matrix(block.sample_condition_dicts, [block.step.transform_feature(f) for f in all_features]) for block in blocks])
        response_vector = np.concatenate([block.response_vector for block in blocks])
        weights = np.concatenate([block.weights for block in blocks])
        row_scale = np.concatenate([np.full(block.response_vector.size, 1.0/cls.property_std_deviation[block.step.data_types_read]) for block in blocks])
        scaled_feature_matrix = feature_matrix * row_scale[:, np.newaxis]
        scaled_response_vector = response_vector * row_scale

        # Features of groups without their own data are only candidates if the data resolve them
        weighted_scaled_feature_matrix = scaled_feature_matrix * weights[:, np.newaxis]
        basis = [feature_column[f] for g in groups if g.owned for f in g.universe]
        basis_rank = _column_rank(weighted_scaled_feature_matrix[:, basis], cls.rank_rtol)
        group_candidates = [g.candidates if g.owned else None for g in groups]
        # Resolve features of groups without their own data from the last
        # group (lowest order in temperature, e.g. HM) to the first, so that
        # e.g. GM data at two temperatures resolve the constant and T features
        # rather than heat capacity features.
        for group_idx, g in reversed(list(enumerate(groups))):
            if g.owned:
                continue
            is_temperature_dependent = any(symengine.diff(f, v.T) != 0 for f in g.step.features)
            resolving_temperatures = [cond[v.T] for i in g.resolving_blocks for cond in blocks[i].sample_condition_dicts]
            if is_temperature_dependent and (max(resolving_temperatures) - min(resolving_temperatures)) < cls.min_temperature_range:
                _log.trace('%s features not resolved: data span less than %s K', g.step.data_types_read, cls.min_temperature_range)
                group_candidates[group_idx] = [()]
                continue
            unresolved = set()
            for f in g.universe:
                trial_rank = _column_rank(weighted_scaled_feature_matrix[:, basis + [feature_column[f]]], cls.rank_rtol)
                if trial_rank > basis_rank:
                    basis.append(feature_column[f])
                    basis_rank = trial_rank
                else:
                    unresolved.add(f)
            _log.trace('%s features unresolved by data: %s', g.step.data_types_read, unresolved)
            group_candidates[group_idx] = [()] + [c for c in g.candidates if unresolved.isdisjoint(c)]

        # Groups penalized against the same data points share one AICc penalty
        penalty_units = {}  # Dict[Tuple[int, ...], List[int]]: block indices -> group indices
        for group_idx, g in enumerate(groups):
            if g.owned:
                key = (present_data_types.index(g.step.data_types_read),)
            else:
                key = tuple(g.resolving_blocks)
            penalty_units.setdefault(key, []).append(group_idx)
        penalty_unit_num_points = {key: sum(blocks[i].response_vector.size for i in key) for key in penalty_units}
        group_aicc_factors = [aicc_feature_factors.get(g.step.data_types_read, 1.0) for g in groups]

        # A block that is exactly fit by features that are non-zero only in
        # that block carries no information about other features, regardless
        # of the numerical residual from the solver.
        nonzero = feature_matrix != 0
        private_to_block = []
        for block in blocks:
            outside = np.ones(feature_matrix.shape[0], dtype=bool)
            outside[block.rows] = False
            private_to_block.append(nonzero[block.rows].any(axis=0) & ~nonzero[outside].any(axis=0))
        saturation_cache = {}
        def is_saturated(block_idx, columns):
            block = blocks[block_idx]
            private_columns = tuple(j for j in columns if private_to_block[block_idx][j])
            if len(private_columns) < block.response_vector.size:
                return False
            key = (block_idx, private_columns)
            if key not in saturation_cache:
                saturation_cache[key] = _column_rank(feature_matrix[block.rows][:, list(private_columns)], cls.rank_rtol) == block.response_vector.size
            return saturation_cache[key]

        opt_score = np.inf
        opt_model = None
        for candidate in itertools.product(*group_candidates):
            features = list(itertools.chain.from_iterable(candidate))
            if len(features) == 0:
                continue
            columns = [feature_column[f] for f in features]
            if ridge_alpha is None:
                # Features span many orders of magnitude (e.g. T**3 and 1/T),
                # so normalize the columns to improve the conditioning. This
                # doesn't change the least squares solution, but it would
                # change the meaning of a ridge penalty.
                column_norms = np.linalg.norm(scaled_feature_matrix[:, columns], axis=0)
                column_norms[column_norms == 0] = 1.0
                coefficients = fit_model(scaled_feature_matrix[:, columns] / column_norms, scaled_response_vector, ridge_alpha, weights=weights) / column_norms
            else:
                coefficients = fit_model(scaled_feature_matrix[:, columns], scaled_response_vector, ridge_alpha, weights=weights)
            residuals = (np.dot(feature_matrix[:, columns], coefficients) - response_vector) * weights
            score = 0.0
            for block_idx, block in enumerate(blocks):
                num_points = block.response_vector.size
                if is_saturated(block_idx, columns):
                    rss = cls.rss_numerical_limit
                else:
                    rss = max(np.square(residuals[block.rows]).sum(), cls.rss_numerical_limit)
                score += num_points * np.log(rss / num_points)
            for key, group_idxs in penalty_units.items():
                pk = sum(group_aicc_factors[i] * len(candidate[i]) for i in group_idxs)
                score += 2.0 * pk + aicc_correction(pk, penalty_unit_num_points[key])
            _log.debug('%s score: %s', features, score)
            if score < opt_score:
                opt_score = score
                opt_model = (features, coefficients)
        if opt_model is not None:
            _log.trace('selected model features: %s, coefficients: %s, model score: %s', opt_model[0], list(opt_model[1]), opt_score)
        return opt_model


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