"""
Calculate error due to thermochemical quantities: heat capacity, entropy, enthalpy.
"""

import logging
from collections import OrderedDict
from typing import Any, Dict, List, NewType, Optional, Tuple, Union

import symengine
from scipy.stats import norm
import numpy as np
import numpy.typing as npt
from symengine import Symbol
from tinydb import where
from pycalphad.codegen.phase_record_factory import PhaseRecordFactory
from pycalphad import Database, Model, ReferenceState, variables as v
from pycalphad.core.utils import unpack_species, get_pure_elements, filter_phases
from pycalphad import Workspace
from pycalphad.property_framework import IsolatedPhase
from pycalphad.property_framework.metaproperties import find_first_compset
from pycalphad.core.solver import Solver, SolverResult


from espei.datasets import Dataset
from espei.core_utils import ravel_conditions, get_prop_data, filter_temperatures
from espei.parameter_selection.redlich_kister import calc_interaction_product
from espei.phase_models import PhaseModelSpecification
from espei.sublattice_tools import canonicalize, recursive_tuplify, tuplify
from espei.typing import SymbolName
from espei.utils import database_symbols_to_fit, PickleableTinyDB
from .residual_base import ResidualFunction, residual_function_registry

_log = logging.getLogger(__name__)

class NoSolveSolver(Solver):
    def solve(self, composition_sets, conditions):
        """
        Initialize a solver, but don't actually do anything - i.e. return the SolverResult as if the calculation converged.

        """
        spec = self.get_system_spec(composition_sets, conditions)
        self._fix_state_variables_in_compsets(composition_sets, conditions)
        state = spec.get_new_state(composition_sets)
        # DO NOT: converged = spec.run_loop(state, 1000)

        if self.remove_metastable:
            phase_idx = 0
            compsets_to_remove = []
            for compset in composition_sets:
                # Mark unstable phases for removal
                if compset.NP <= 0.0 and not compset.fixed:
                    compsets_to_remove.append(int(phase_idx))
                phase_idx += 1
            # Watch removal order here, as the indices of composition_sets are changing!
            for idx in reversed(compsets_to_remove):
                del composition_sets[idx]

        phase_amt = [compset.NP for compset in composition_sets]

        x = composition_sets[0].dof
        state_variables = composition_sets[0].phase_record.state_variables
        num_statevars = len(state_variables)
        for compset in composition_sets[1:]:
            x = np.r_[x, compset.dof[num_statevars:]]
        x = np.r_[x, phase_amt]
        chemical_potentials = np.array(state.chemical_potentials)

        if self.verbose:
            print('Chemical Potentials', chemical_potentials)
            print(np.asarray(x))
        return SolverResult(converged=True, x=x, chemical_potentials=chemical_potentials)


# TODO: make into an object similar to how ZPF data works?
FixedConfigurationCalculationData = NewType("FixedConfigurationCalculationData", Dict[str, Any])

def filter_sublattice_configurations(desired_data: List[Dataset], subl_model) -> List[Dataset]:  # TODO: symmetry support
    """Modify the desired_data to remove any configurations that cannot be represented by the sublattice model."""
    subl_model_sets = [set(subl) for subl in subl_model]
    for data in desired_data:
        matching_configs = []  # binary mask of whether a configuration is represented by the sublattice model
        for config in data['solver']['sublattice_configurations']:
            config = recursive_tuplify(canonicalize(config, None))
            if (
                len(config) == len(subl_model) and
                all(subl.issuperset(tuplify(config_subl)) for subl, config_subl in zip(subl_model_sets, config))
            ):
                matching_configs.append(True)
            else:
                matching_configs.append(False)
        matching_configs = np.asarray(matching_configs, dtype=np.bool_)

        # Rewrite output values with filtered data
        data['values'] = np.array(data['values'], dtype=np.float64)[..., matching_configs]
        data['solver']['sublattice_configurations'] = np.array(data['solver']['sublattice_configurations'], dtype=np.object_)[matching_configs].tolist()
        if 'sublattice_occupancies' in data['solver']:
            data['solver']['sublattice_occupancies'] = np.array(data['solver']['sublattice_occupancies'], dtype=np.object_)[matching_configs].tolist()
    return desired_data


def calculate_points_array(phase_constituents, configuration, occupancies=None):
    """
    Calculate the points array to use in pycalphad calculate calls.

    Converts the configuration data (and occupancies for mixing data) into the
    points array by looking up the indices in the active phase constituents.

    Parameters
    ----------
    phase_constituents : list
        List of active constituents in a phase
    configuration : list
        List of the sublattice configuration
    occupancies : list
        List of sublattice occupancies. Required for mixing sublattices, otherwise takes no effect.

    Returns
    -------
    numpy.ndarray

    Notes
    -----
    Errors will be raised if components in the configuration are not in the
    corresponding phase constituents sublattice.
    """
    # pad the occupancies for zipping if none were passed (the case for non-mixing)
    if occupancies is None:
        occupancies = [0] * len(configuration)

    # construct the points array from zeros
    points = np.zeros(sum(len(subl) for subl in phase_constituents))
    current_subl_idx = 0  # index that marks the beginning of the sublattice
    for phase_subl, config_subl, subl_occupancies in zip(phase_constituents, configuration, occupancies):
        phase_subl = list(phase_subl)
        if isinstance(config_subl, (tuple, list)):
            # we have mixing on the sublattice
            for comp, comp_occupancy in zip(config_subl, subl_occupancies):
                points[current_subl_idx + phase_subl.index(comp)] = comp_occupancy
        else:
            points[current_subl_idx + phase_subl.index(config_subl)] = 1
        current_subl_idx += len(phase_subl)
    return points


def get_prop_samples(desired_data, constituents):
    """
    Return data values and the conditions to calculate them using pycalphad.calculate

    Parameters
    ----------
    desired_data : List[Dict[str, Any]]
        List of dataset dictionaries that contain the values to sample
    constituents : List[List[str]]
        Names of constituents in each sublattice.

    Returns
    -------
    Dict[str, Union[float, ArrayLike, List[float]]]
        Dictionary of condition kwargs for pycalphad's calculate and the expected values

    """
    # TODO: assumes T, P, N as conditions
    # calculate needs points, state variable lists, and values to compare to
    num_dof = sum(map(len, constituents))
    calculate_dict = {
        'N': np.array([]),
        'P': np.array([]),
        'T': np.array([]),
        'points': np.atleast_2d([[]]).reshape(-1, num_dof),
        'values': np.array([]),
        'weights': [],
        'references': [],
    }

    for datum in desired_data:
        # extract the data we care about
        datum_T = datum['conditions']['T']
        datum_P = datum['conditions']['P']
        # TODO: fix this when N different from 1 allowed in pycalphad
        datum_N = np.full_like(datum['values'], 1.0)
        configurations = datum['solver']['sublattice_configurations']
        occupancies = datum['solver'].get('sublattice_occupancies')
        values = np.array(datum['values'])
        if values.size == 0:
            # Skip any data that don't have any values left (e.g. after filtering)
            continue
        # Broadcast the weights to the shape of the values. This ensures that
        # the sizes of the weights and values are the same, which is important
        # because they are flattened later (so the shape information is lost).
        weights = np.broadcast_to(np.asarray(datum.get('weight', 1.0)), values.shape)

        # broadcast and flatten the conditions arrays
        P, T, N = ravel_conditions(values, datum_P, datum_T, datum_N)
        if occupancies is None:
            occupancies = [None] * len(configurations)

        # calculate the points arrays, should be 2d array of points arrays
        points = np.array([calculate_points_array(constituents, config, occup) for config, occup in zip(configurations, occupancies)])
        assert values.shape == weights.shape, f"Values data shape {values.shape} does not match weights shape {weights.shape}"

        # add everything to the calculate_dict
        calculate_dict['P'] = np.concatenate([calculate_dict['P'], P])
        calculate_dict['T'] = np.concatenate([calculate_dict['T'], T])
        calculate_dict['N'] = np.concatenate([calculate_dict['N'], N])
        calculate_dict['points'] = np.concatenate([calculate_dict['points'], np.tile(points, (values.shape[0]*values.shape[1], 1))], axis=0)
        calculate_dict['values'] = np.concatenate([calculate_dict['values'], values.flatten()])
        calculate_dict['weights'].extend(weights.flatten())
        calculate_dict['references'].extend([datum.get('reference', "") for _ in range(values.flatten().size)])
    return calculate_dict


def get_sample_condition_dicts(calculate_dict: Dict[Any, Any], configuration_tuple: Tuple[Union[str, Tuple[str]]], phase_name: str) -> List[Dict[Symbol, float]]:
    sublattice_dof = list(map(len, configuration_tuple))
    sample_condition_dicts = []
    for sample_idx in range(calculate_dict["values"].size):
        cond_dict = {}
        points = calculate_dict["points"][sample_idx, :]

        # T and P
        cond_dict[v.T] = calculate_dict["T"][sample_idx]
        cond_dict[v.P] = calculate_dict["P"][sample_idx]

        # YS site fraction product
        site_fraction_product = np.prod(points)
        cond_dict[Symbol("YS")] = site_fraction_product

        # Reconstruct site fractions in sublattice form from points
        # Required so we can identify which sublattices have interactions
        points_idxs = [0] + np.cumsum(sublattice_dof).tolist()
        site_fractions = []
        for subl_idx in range(len(points_idxs)-1):
            subl_site_fractions = points[points_idxs[subl_idx]:points_idxs[subl_idx+1]]
            for species_name, site_frac in zip(configuration_tuple[subl_idx], subl_site_fractions):
                cond_dict[v.Y(phase_name, subl_idx, species_name)] = site_frac
            site_fractions.append(subl_site_fractions.tolist())

        # Z (binary) or V_I, V_J, V_K (ternary) interaction products
        interaction_product = calc_interaction_product(site_fractions)
        if hasattr(interaction_product, "__len__"):
            # Ternary interaction
            assert len(interaction_product) == 3
            cond_dict[Symbol("V_I")] = interaction_product[0]
            cond_dict[Symbol("V_J")] = interaction_product[1]
            cond_dict[Symbol("V_K")] = interaction_product[2]
        else:
            cond_dict[Symbol("Z")] = interaction_product

        sample_condition_dicts.append(cond_dict)
    return sample_condition_dicts


def get_thermochemical_data(dbf, comps, phases, datasets, model=None, weight_dict=None, symbols_to_fit=None):
    """

    Parameters
    ----------
    dbf : pycalphad.Database
        Database to consider
    comps : list
        List of active component names
    phases : list
        List of phases to consider
    datasets : espei.utils.PickleableTinyDB
        Datasets that contain single phase data
    model : Optional[Dict[str, Type[Model]]]
        Dictionary phase names to pycalphad Model classes.
    weight_dict : dict
        Dictionary of weights for each data type, e.g. {'HM': 200, 'SM': 2}
    symbols_to_fit : list
        Parameters to fit. Used to build the models and PhaseRecords.

    Returns
    -------
    list
        List of data dictionaries to iterate over
    """
    # phase by phase, then property by property, then by model exclusions
    if weight_dict is None:
        weight_dict = {}

    if model is None:
        model = {}

    if symbols_to_fit is not None:
        symbols_to_fit = sorted(symbols_to_fit)
    else:
        symbols_to_fit = database_symbols_to_fit(dbf)

    species_comps = set(unpack_species(dbf, comps))

    # estimated from NIST TRC uncertainties
    property_std_deviation = {
        'HM': 500.0/weight_dict.get('HM', 1.0),  # J/mol
        'SM':   0.2/weight_dict.get('SM', 1.0),  # J/K-mol
        'CPM':  0.2/weight_dict.get('CPM', 1.0),  # J/K-mol
    }
    properties = ['HM_FORM', 'SM_FORM', 'CPM_FORM', 'HM_MIX', 'SM_MIX', 'CPM_MIX']

    ref_states = []
    for el in get_pure_elements(dbf, comps):
        ref_state = ReferenceState(el, dbf.refstates[el]['phase'])
        ref_states.append(ref_state)
    all_data_dicts = []
    for phase_name in phases:
        if phase_name not in dbf.phases:
            continue
        # phase constituents are Species objects, so we need to be doing intersections with those
        phase_constituents = dbf.phases[phase_name].constituents
        # phase constituents must be filtered to only active:
        constituents = [[sp.name for sp in sorted(subl_constituents.intersection(species_comps))] for subl_constituents in phase_constituents]
        for prop in properties:
            desired_data = get_prop_data(comps, phase_name, prop, datasets, additional_query=(where('solver').exists()))
            if len(desired_data) == 0:
                continue
            unique_exclusions = set([tuple(sorted(set(d.get('excluded_model_contributions', [])))) for d in desired_data])
            for exclusion in unique_exclusions:
                data_dict = {
                    'phase_name': phase_name,
                    'prop': prop,
                    # needs the following keys to be added:
                    # species, calculate_dict, phase_record_factory, model, output, weights
                }
                # get all the data with these model exclusions
                if exclusion == tuple([]):
                    exc_search = (~where('excluded_model_contributions').exists()) & (where('solver').exists())
                else:
                    exc_search = (where('excluded_model_contributions').test(lambda x: tuple(sorted(set(x))) == exclusion)) & (where('solver').exists())
                curr_data = get_prop_data(comps, phase_name, prop, datasets, additional_query=exc_search)
                curr_data = filter_sublattice_configurations(curr_data, constituents)
                curr_data = filter_temperatures(curr_data)
                calculate_dict = get_prop_samples(curr_data, constituents)
                model_cls = model.get(phase_name, Model)
                mod = model_cls(dbf, comps, phase_name, parameters=symbols_to_fit)
                if prop.endswith('_FORM'):
                    output = ''.join(prop.split('_')[:-1])+"R"
                    mod.shift_reference_state(ref_states, dbf, contrib_mods={e: symengine.S.Zero for e in exclusion})
                else:
                    output = prop
                for contrib in exclusion:
                    mod.models[contrib] = symengine.S.Zero
                    try:
                        # TODO: we can remove this try/except block when pycalphad 0.8.5
                        # is released with these internal API changes
                        mod.endmember_reference_model.models[contrib] = symengine.S.Zero
                    except AttributeError:
                        mod.reference_model.models[contrib] = symengine.S.Zero
                model_dict = {phase_name: mod}
                species = sorted(unpack_species(dbf, comps), key=str)
                data_dict['species'] = species
                statevar_dict = {getattr(v, c, None): vals for c, vals in calculate_dict.items() if isinstance(getattr(v, c, None), v.StateVariable)}
                statevar_dict = OrderedDict(sorted(statevar_dict.items(), key=lambda x: str(x[0])))
                phase_record_factory = PhaseRecordFactory(dbf, species, statevar_dict, model_dict,
                                                   parameters={s: 0 for s in symbols_to_fit})
                str_statevar_dict = OrderedDict((str(k), vals) for k, vals in statevar_dict.items())
                data_dict['str_statevar_dict'] = str_statevar_dict
                data_dict['phase_record_factory'] = phase_record_factory
                data_dict['calculate_dict'] = calculate_dict
                data_dict['model'] = model_dict
                data_dict['output'] = output
                data_dict['weights'] = np.array(property_std_deviation[prop.split('_')[0]])/np.array(calculate_dict.pop('weights'))
                data_dict['constituents'] = constituents
                all_data_dicts.append(data_dict)
    return all_data_dicts


def compute_fixed_configuration_property_differences(dbf, calc_data: FixedConfigurationCalculationData, parameters):
    species = calc_data['species']
    phase_name = calc_data['phase_name']
    models = calc_data['model']  # Dict[PhaseName: Model]
    output = calc_data['output']
    phase_record_factory = calc_data['phase_record_factory']
    sample_values = calc_data['calculate_dict']['values']
    str_statevar_dict = calc_data['str_statevar_dict']

    constituent_list = []
    sublattice_list = []
    counter = 0
    for sublattice in calc_data['constituents']:
        for const in sublattice:
            sublattice_list.append(counter)
            constituent_list.append(const)
        counter = counter + 1

    differences = []
    for index in range(len(sample_values)):
        cond_dict = {}
        for sv_key, sv_val in str_statevar_dict.items():
            cond_dict.update({sv_key: sv_val[index]})

        # Build internal DOF as if they were used in conditions
        dof = {}
        for site_frac in range(len(constituent_list)):
            comp = constituent_list[site_frac]
            occupancy = calc_data['calculate_dict']['points'][index,site_frac]
            sublattice = sublattice_list[site_frac]
            dof.update({v.Y(phase_name,sublattice,comp): occupancy})

        # TODO: active_pure_elements should be replaced with wks.components when wks.components no longer includes phase constituent Species
        # Build composition conditions, probably not necessary given that we don't actually solve anything, but still useful in terms of derivatives probably.
        active_pure_elements = [list(x.constituents.keys()) for x in species]
        active_pure_elements = sorted(set(el.upper() for constituents in active_pure_elements for el in constituents) - {"VA"})
        ind_comps = len(active_pure_elements) - 1
        for comp in active_pure_elements:
            if v.Species(comp) != v.Species('VA') and ind_comps > 0:
                cond_dict[v.X(comp)] = float(models[phase_name].moles(comp).xreplace(dof))
                ind_comps = ind_comps - 1
        # Need to be careful here. Making a workspace erases the custom models that have some contributions excluded (which are passed in). Not sure exactly why.
        # The models themselves are preserved, but the ones inside the workspace's phase_record_factory get clobbered.
        # We workaround this by replacing the phase_record_factory models with ours, but this is definitely a hack we'd like to avoid.
        wks = Workspace(database=dbf, components=species, phases=[phase_name], conditions={**cond_dict}, models=models, phase_record_factory=phase_record_factory, parameters=parameters, solver=NoSolveSolver())
        # We then get a composition set and we use a special "NoSolveSolver" to
        # ensure that we don't change from the data-specified DOF.
        compset = find_first_compset(phase_name, wks)
        new_sitefracs = np.array([sf for _, sf in sorted(dof.items(), key=lambda y: (y[0].phase_name, y[0].sublattice_index, y[0].species.name))])
        new_statevars = np.array(compset.dof[:len(compset.phase_record.state_variables)])  # no updates expected
        compset.update(new_sitefracs, 1.0, new_statevars)
        iso_phase = IsolatedPhase(compset, wks=wks)
        iso_phase.solver = NoSolveSolver()
        results = wks.get(iso_phase(output))
        sample_differences = results - sample_values[index]
        differences.append(sample_differences)
    return differences


def calculate_non_equilibrium_thermochemical_probability(thermochemical_data: List[FixedConfigurationCalculationData], dbf, parameters=None):
    """
    Calculate the weighted single phase error in the Database

    Parameters
    ----------
    thermochemical_data : list
        List of thermochemical data dicts
    parameters : np.ndarray
        Array of parameters to calculate the error with.

    Returns
    -------
    float
        A single float of the residual sum of square errors

    """
    if parameters is None:
        parameters = {}

    prob_error = 0.0
    for data in thermochemical_data:
        phase_name = data['phase_name']
        sample_values = data['calculate_dict']['values']
        differences = compute_fixed_configuration_property_differences(dbf, data, parameters)
        differences = np.array(differences)
        probabilities = norm.logpdf(differences, loc=0, scale=data['weights'])
        prob_sum = np.sum(probabilities)
        _log.debug("%s(%s) - probability sum: %0.2f, data: %s, differences: %s, probabilities: %s, references: %s", data['prop'], phase_name, prob_sum, sample_values, differences, probabilities, data['calculate_dict']['references'])
        prob_error += prob_sum
    return prob_error


class FixedConfigurationPropertyResidual(ResidualFunction):
    def __init__(
        self,
        database: Database,
        datasets: PickleableTinyDB,
        phase_models: Union[PhaseModelSpecification, None],
        symbols_to_fit: Optional[List[SymbolName]] = None,
        weight: Optional[Dict[str, float]] = None,
        ):
        super().__init__(database, datasets, phase_models, symbols_to_fit, weight)

        if weight is not None:
            self.weight = weight
        else:
            self.weight = {}

        if phase_models is not None:
            comps = sorted(phase_models.components)
            model_dict = phase_models.get_model_dict()
        else:
            comps = sorted(database.elements)
            model_dict = dict()
        phases = sorted(filter_phases(database, unpack_species(database, comps), database.phases.keys()))
        if symbols_to_fit is None:
            symbols_to_fit = database_symbols_to_fit(database)
        self.thermochemical_data = get_thermochemical_data(database, comps, phases, datasets, model_dict, weight_dict=self.weight, symbols_to_fit=symbols_to_fit)
        self._symbols_to_fit = symbols_to_fit
        self.dbf = database

    def get_residuals(self, parameters: npt.ArrayLike) -> Tuple[List[float], List[float]]:
        residuals = []
        weights = []
        for data in self.thermochemical_data:
            dataset_residuals = compute_fixed_configuration_property_differences(self.dbf, data, dict(zip(self._symbols_to_fit, parameters)))
            residuals.extend(dataset_residuals)
            dataset_weights = np.asarray(data["weights"], dtype=float).flatten().tolist()
            if len(dataset_weights) != len(dataset_residuals):
                # we need to broadcast the residuals. For now, assume the weights are a scalar, since that's all that's supported
                assert len(dataset_weights) == 1
                dataset_weights = [float(dataset_weights[0]) for _ in range(len(dataset_residuals))]
            weights.extend(dataset_weights)
        return residuals, weights

    def get_likelihood(self, parameters) -> float:
        parameters = {param_name: param for param_name, param in zip(self._symbols_to_fit, parameters.tolist())}
        likelihood = calculate_non_equilibrium_thermochemical_probability(self.thermochemical_data, self.dbf, parameters)
        return likelihood


residual_function_registry.register(FixedConfigurationPropertyResidual)