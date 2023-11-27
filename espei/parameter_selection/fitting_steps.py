# TODO: do all typing before merging

from typing import Any, Dict, Optional
import itertools
from numpy.typing import ArrayLike
import numpy as np
import symengine
from pycalphad import Model, variables as v
from espei.parameter_selection.model_building import make_successive
from espei.parameter_selection.utils import build_sitefractions
# from espei.paramselect import *


__all__ = [
    "FittingStep",
    "AbstractRKMPropertyStep",
    "StepHM",
    "StepSM",
    "StepCPM",
    "StepElasticC11",
    "StepElasticC12",
    "StepElasticC44",
    "StepV0",
    "StepLogVA",
]

Dataset = Dict[str, Any]  # Typing stub

class FittingStep():
    parameter_name: str
    # TODO: can we think of a situtation where it makes sense to go to multilpe data types read in a single step?
    data_types_read: str
    features: [symengine.Expr]
    # TODO: does a reference state support list here make sense?
    # If we instead make shift_reference_state a part of the API, it would be
    # more Pythonic to raise errors, but what would be better for end users?
    # Presumably front-facing APIs could use something like this to check
    # reference state support at dataset creation time, but maybe that couples
    # too many things.
    supported_reference_states: [str]

    @staticmethod
    def transform_data(d: ArrayLike, model: Optional[Model] = None) -> ArrayLike:  # data may be muddied with symbols from Model
        return d

    @classmethod
    def transform_feature(cls, f: symengine.Expr, model: Optional[Model] = None) -> symengine.Expr:
        return f

    @classmethod
    def get_feature_sets(cls):
        return make_successive(cls.features)

    # TODO: rename, maybe get_regression_rhs (and build_feature_matrix => get_regression_matrix)
    @classmethod
    def get_data_quantities(cls, desired_property, fixed_model, fixed_portions, data, sample_condition_dicts):
        """
        Parameters
        ----------
        desired_property : str
            String property corresponding to the features that could be fit, e.g. HM, SM_FORM, CPM_MIX
        fixed_model : pycalphad.Model
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


class AbstractRKMPropertyStep(FittingStep):
    """
    This class is a base class for generic Redlich-Kister-Muggianu (RKM) properties.

    "Generic" meaning, essentially, that
    `property_name == parameter_name == data_type` and PyCalphad models
    set this property something like
    ```python
    self.<parameter_name> = self.redlich_kister_sum(phase, param_search, <param_name>_query)
    ```

    Fitting steps that want to use this base class should need to subclass this
    class, override the `parameter_name` and `data_type_read` (typically these
    match), and optionally override the `features` if a desired.
    """
    features = [symengine.S.One, v.T, v.T**2, v.T**3, v.T**(-1)]
    supported_reference_states = ["", "_MIX"]  # TODO: add _FORM support

    @classmethod
    def shift_reference_state(cls, desired_data, fixed_model, _):
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
                    values[..., config_idx] += getattr(fixed_model.endmember_reference_model, cls.parameter_name)
                else:
                    pass
            total_response.append(values.flatten())
        return total_response

    @classmethod
    def get_data_quantities(cls, desired_property: str, fixed_model: Model, fixed_portions: [symengine.Basic], data: [Dataset], sample_condition_dicts):
        """
        Parameters
        ----------
        desired_property : str
            String property corresponding to the features that could be fit, e.g. HM, SM_FORM, CPM_MIX
        fixed_model : pycalphad.Model
            Model with all lower order (in composition) terms already fit.
        fixed_portions : [symengine.Basic]
            API compatibility. Not used for this method.
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
        # TODO: make sure to normalize per mole of formula correctly!
        #   For this fitting step, everything we are working with from model
        #   should be per mole of atoms (except the volume_data, that
        #   shift_reference_state shifts).If we modify our internal
        #   shift_reference_state, we might be able to just normalize per mole
        #   of formula units at the end.
        mole_atoms_per_mole_formula_unit = fixed_model._site_ratio_normalization
        # This function takes Dataset objects (`data`) -> values array (of np.object_)
        rhs = np.concatenate(cls.shift_reference_state(data, fixed_model, None), axis=-1)

        # RKM models are already liner in the parameters, so we don't need to
        # do anything too special here.
        # subtract off lower order contributions.
        for i in range(rhs.shape[0]):
            rhs[i] -= getattr(fixed_model, cls.parameter_name)
            # convert to moles_per_formula_unit
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
        # if any site fractions show up in our data_qtys that aren't in these datasets' site fractions, set them to zero.
        # TODO: when does this (^) actually happen?
        for sf, value_with_symbols, cond_dict in zip(site_fractions, rhs, sample_condition_dicts):
            missing_variables = symengine.S(value_with_symbols).atoms(v.SiteFraction) - set(sf.keys())
            sf.update({x: 0. for x in missing_variables})
            sf.update(cond_dict)
        rhs = [symengine.S(value_with_symbols).xreplace(sf).evalf() for value_with_symbols, sf in zip(rhs, site_fractions)]
        # cast to float, confirming that these are concrete values with no sybolics
        rhs = np.asarray(rhs, dtype=np.float_)
        return rhs


# Legacy imports while we're transitioning, fix these
from espei.parameter_selection.utils import feature_transforms
# TODO: support fitting GM_MIX and GM_FORM directly from DFT?
# TODO: for HM, SM, and CPM, refactor to stop using the transforms and build the transforms into the subclasses
# Maybe this is where we introduce the data and feature transforms class methods?
class StepHM(FittingStep):
    parameter_name = "GM"
    data_types_read = "HM"
    supported_reference_states = ["_MIX", "_FORM"]
    features = [symengine.S.One]

    @classmethod
    def transform_feature(cls, f: symengine.Expr, model: Optional[Model] = None) -> symengine.Expr:
        transform = feature_transforms[cls.data_types_read]
        return transform(f)

    # TODO: this function actually does 2 things that should be split up into separate functions:
    # 1. Extract data from Dataset objects into an array of raw values
    # 2. Shifts the reference state of the values
    #    For Gibbs energy (and derivatives), we always shift to _FORM reference state
    # This is the original s_r_s method from ESPEI
    @classmethod
    def shift_reference_state(cls, desired_data, feature_transform, fixed_model, mole_atoms_per_mole_formula_unit):
        """
        Shift _MIX or _FORM data to a common reference state in per mole-atom units.

        Parameters
        ----------
        desired_data : List[Dict[str, Any]]
            ESPEI single phase dataset
        feature_transform : Callable
            Function to transform an AST for the GM property to the property of
            interest, i.e. entropy would be ``lambda GM: -symengine.diff(GM, v.T)``
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
                    pass
                elif dataset['output'].endswith('_MIX'):
                    if occupancy is None:
                        raise ValueError('Cannot have a _MIX property without sublattice occupancies.')
                    else:
                        values[..., config_idx] += feature_transform(fixed_model.models['ref'])*mole_atoms_per_mole_formula_unit
                else:
                    raise ValueError(f'Unknown property to shift: {dataset["output"]}')
                for excluded_contrib in unique_excluded_contributions:
                    values[..., config_idx] += feature_transform(fixed_model.models[excluded_contrib])*mole_atoms_per_mole_formula_unit
            total_response.append(values.flatten())
        return total_response


    @classmethod
    def get_data_quantities(cls, desired_property, fixed_model, fixed_portions, data, sample_condition_dicts):
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

        feat_transform = feature_transforms[desired_property]
        data_qtys = np.concatenate(cls.shift_reference_state(data, feat_transform, fixed_model, mole_atoms_per_mole_formula_unit), axis=-1)
        # Remove existing partial model contributions from the data, convert to per mole-formula units
        data_qtys = data_qtys - feat_transform(fixed_model.ast)*mole_atoms_per_mole_formula_unit
        # Subtract out high-order (in T) parameters we've already fit, already in per mole-formula units
        data_qtys = data_qtys - feat_transform(sum(fixed_portions))
        # if any site fractions show up in our data_qtys that aren't in this datasets site fractions, set them to zero.
        for sf, i, cond_dict in zip(site_fractions, data_qtys, sample_condition_dicts):
            missing_variables = symengine.S(i).atoms(v.SiteFraction) - set(sf.keys())
            sf.update({x: 0. for x in missing_variables})
            # The equations we have just have the site fractions as YS
            # and interaction products as Z, so take the product of all
            # the site fractions that we see in our data qtys
            sf.update(cond_dict)
        data_qtys = [symengine.S(i).xreplace(sf).evalf() for i, sf in zip(data_qtys, site_fractions)]
        data_qtys = np.asarray(data_qtys, dtype=np.float_)
        return data_qtys


# TODO: does it make sense to inherit from HM? Do we need an abstract class? Or does fixing the transforms issue and having each implementation be separate be correct?
# TODO: support "" (absolute) entropy reference state?
class StepSM(StepHM):
    data_types_read = "SM"
    features = [v.T]


# TODO: support "" (absolute) heat capacity reference state?
class StepCPM(StepHM):
    data_types_read = "CPM"
    features = [v.T * symengine.log(v.T), v.T**2, v.T**-1, v.T**3]



class StepElasticC11(AbstractRKMPropertyStep):
    parameter_name = "C11"
    data_types_read = "C11"

class StepElasticC12(AbstractRKMPropertyStep):
    parameter_name = "C12"
    data_types_read = "C12"

class StepElasticC44(AbstractRKMPropertyStep):
    parameter_name = "C44"
    data_types_read = "C44"

class StepV0(AbstractRKMPropertyStep):
    parameter_name = "V0"
    data_types_read = "V0"
    features = [symengine.S.One]

class StepLogVA(FittingStep):
    # V = V0*exp(VA), to linearize in terms of VA features, we want to fit
    # VA = ln(V/V0)
    parameter_name = "VA"
    data_types_read = "VM"
    features = [v.T, v.T**2, v.T**3, v.T**(-1)]
    supported_reference_states = ["", "_MIX"]  # TODO: add formation support

    # TODO: This is probably deprecated now that we have a get_data_quantities implemented in this class
    @staticmethod
    def transform_data(d, model: Model) -> ArrayLike:
        # We are given samples of volume (V) as our data (d).
        # ln(V/V0) = VA
        # cast to object_ because the real type may become a symengine.Expr
        d = np.asarray(d, dtype=np.object_)
        for i in range(d.shape[0]):
            d[i] = symengine.log(d[i] / model.V0)
        return d

    @classmethod
    def get_feature_sets(cls):
        # All combinations of features
        # TODO: this might be what is expensive when we're generating interaction parameters
        return list(itertools.chain(*(itertools.combinations(cls.features, n) for n in range(1, len(cls.features)+1))))

    @classmethod
    def shift_reference_state(cls, desired_data, fixed_model, mole_atoms_per_mole_formula_unit):
        """
        Shift _MIX or _FORM data to a common reference state in per mole-atom units.

        Parameters
        ----------
        desired_data : [Dict[str, Any]]
            ESPEI single phase dataset
        feature_transform : Callable
            Function to transform an AST for the GM property to the property of
            interest, i.e. entropy would be ``lambda GM: -symengine.diff(GM, v.T)``
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
        # Because of the log transform used to linearize the data it must be
        # satisfied that \( V/V0 > 0 \). It's therefore not easy to work with
        # excess volumes (which may be positive or negative).
        # The solution is to shift other reference states into absolute volume.
        total_response = []
        for dataset in desired_data:
            values = np.asarray(dataset['values'], dtype=np.object_)*mole_atoms_per_mole_formula_unit
            for config_idx in range(len(dataset['solver']['sublattice_configurations'])):
                occupancy = dataset['solver'].get('sublattice_occupancies', None)
                if dataset['output'].endswith('_FORM'):
                    raise NotImplementedError(f"The formation reference state for data type {dataset['output']} is not yet supported for {cls.parameter_name} parameters.")
                elif dataset['output'].endswith('_MIX'):
                    # for backwards compatibility, we treat this using the
                    # special "endmember" reference state (i.e. where the
                    # property of interest is zero for all the endmembers).
                    # To shift our VM_MIX data into VM, we need to add the VM
                    # computed from the endmember reference model.
                    values[..., config_idx] += fixed_model.endmember_reference_model.VM
                else:
                    pass
            total_response.append(values.flatten())
        return total_response

    # TODO: maybe we could refactor the existing AbstractRKMPropertyStep to use
    # the cls.transform_data method, as that's really the only difference
    # # between this function and the V0 function.
    @classmethod
    def get_data_quantities(cls, desired_property, fixed_model, fixed_portions, data, sample_condition_dicts):
        """
        Parameters
        ----------
        desired_property : str
            String property corresponding to the features that could be fit, e.g. HM, SM_FORM, CPM_MIX
        fixed_model : pycalphad.Model
            Model with all lower order (in composition) terms already fit.
        fixed_portions : [symengine.Basic]
            API compatibility. Not used for this method.
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
        # VM (VA parameters) can't easily fit excess volumes with _MIX or _FORM
        #   reference states because the excess volume might be negative and our
        #   linearization transformation would take the log of a negative
        #   value for `log(V_xs / V0)`. We use shift_reference_state to work in
        #   terms of total volume, which should always be positive.

        # TODO: make sure to normalize per mole of formula correctly!
        #   For this fitting step, everything we are working with from model
        #   should be per mole of atoms (except the volume_data, that
        #   shift_reference_state shifts).If we modify our internal
        #   shift_reference_state, we might be able to just normalize per mole
        #   of formula units at the end.
        mole_atoms_per_mole_formula_unit = fixed_model._site_ratio_normalization
        # This function takes Dataset objects (`data`) -> values array
        volume_data = np.concatenate(cls.shift_reference_state(data, fixed_model, mole_atoms_per_mole_formula_unit), axis=-1)

        # Now we start to build the rhs:
        # cast to np.object_ so we can do symengine ops inplace
        # the rhs may contain some symbolic terms during the intermediate steps,
        # which we will remove later
        rhs = np.empty_like(volume_data, dtype=np.object_)
        for i in range(volume_data.shape[0]):
            # Our model is of the form:
            # \[ V_0 * exp( V_A ) = V \]
            # In this fitting step, all \(V_0\) are fixed, so that's always just a constant.
            # We start to linearize by
            # \[ V_A = \log(V / V_0) \]
            rhs[i] = symengine.log( volume_data[i] / fixed_model.V0 )
            # We are fitting V_A parameters here, but remember that there may be
            # lower order V_A parameters already in the model.
            # \[ V_A^{\mathrm{prev. fit}} + V_A^{\mathrm{curr. fit}} = \log(V / V_0) \]
            # We subtract those off to get the final linearized form of the rhs that
            # we can express as an Ax = b form:
            # \[ V_A^{\mathrm{curr. fit}} = \log(V / V_0) - V_A^{\mathrm{prev. fit}} \]
            rhs[i] -= fixed_model.VA

        # TODO: there shouldn't really YS and Z here since all the symbols come
        # from the existing model. The comment below inside the for loop is from
        # copy and pasted from get_data_quantities in the ESPEI source, and the
        # comments may be incorrect (in this context) and maybe can be removed?

        # Now we remove all the symbols:
        # Construct flattened list of site fractions corresponding to the ravelled data
        site_fractions = []
        for ds in data:
            for _ in ds['conditions']['T']:
                sf = build_sitefractions(fixed_model.phase_name, ds['solver']['sublattice_configurations'], ds['solver'].get('sublattice_occupancies', np.ones((len(ds['solver']['sublattice_configurations']), len(ds['solver']['sublattice_configurations'][0])), dtype=np.float_)))
                site_fractions.append(sf)
        site_fractions = list(itertools.chain(*site_fractions))

        # if any site fractions show up in our data_qtys that aren't in these datasets' site fractions, set them to zero.
        for sf, value_with_symbols, cond_dict in zip(site_fractions, rhs, sample_condition_dicts):
            missing_variables = symengine.S(value_with_symbols).atoms(v.SiteFraction) - set(sf.keys())
            sf.update({x: 0. for x in missing_variables})
            # The equations we have just have the site fractions as YS
            # and interaction products as Z, so take the product of all
            # the site fractions that we see in our data qtys
            sf.update(cond_dict)
        rhs = [symengine.S(value_with_symbols).xreplace(sf).evalf() for value_with_symbols, sf in zip(rhs, site_fractions)]
        # cast to float, confirming that these are concrete values with no sybolics
        rhs = np.asarray(rhs, dtype=np.float_)
        return rhs
