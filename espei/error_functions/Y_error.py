
import yaml
YAML_LOADER = yaml.FullLoader
from espei.datasets import DatasetError, load_datasets
from espei.core_utils import ravel_conditions

import copy
import logging
from collections import OrderedDict
from typing import NamedTuple, Sequence, Dict, Optional, Tuple, Type
import numpy as np
import tinydb
from tinydb import TinyDB, Query, where
from scipy.stats import norm
from pycalphad.plot.eqplot import _map_coord_to_variable
from pycalphad import Database, Model, ReferenceState, variables as v
from pycalphad.core.equilibrium import _eqcalculate
from pycalphad.codegen.callables import build_phase_records
from pycalphad.core.utils import instantiate_models, filter_phases, extract_parameters, unpack_components, unpack_condition
from pycalphad.core.phase_rec import PhaseRecord

from espei.utils import PickleableTinyDB, MemoryStorage
from espei.shadow_functions import equilibrium_, calculate_, no_op_equilibrium_, update_phase_record_parameters



_log = logging.getLogger(__name__)

YPropData = NamedTuple('YPropData', (('dbf', Database),
                                       ('species', Sequence[v.Species]),
                                       ('phases', Sequence[str]),
                                       ('potential_conds', Dict[v.StateVariable, float]),
                                       ('composition_conds', Sequence[Dict[v.X, float]]),
                                       ('models', Dict[str, Model]),
                                       ('params_keys', Dict[str, float]),
                                       ('phase_records', Sequence[Dict[str, PhaseRecord]]),
                                       ('output', str),
                                       ('samples', np.ndarray),
                                       ('weight', np.ndarray),
                                       ('reference', str),
                                       ))


def build_Ypropdata(data: tinydb.database.Document,
                     dbf: Database,
                     model: Optional[Dict[str, Type[Model]]] = None,
                     parameters: Optional[Dict[str, float]] = None,
                     data_weight_dict: Optional[Dict[str, float]] = None
                     ) -> YPropData:
    """
    Build YPropData for the calculations corresponding to a single dataset.

    Parameters
    ----------
    data : tinydb.database.Document
        Document corresponding to a single ESPEI dataset.
    dbf : Database
        Database that should be used to construct the `Model` and `PhaseRecord` objects.
    model : Optional[Dict[str, Type[Model]]]
        Dictionary phase names to pycalphad Model classes.
    parameters : Optional[Dict[str, float]]
        Mapping of parameter symbols to values.
    data_weight_dict : Optional[Dict[str, float]]
        Mapping of a data type (e.g. `HM` or `SM`) to a weight.

    Returns
    -------
    YPropData
    """
    parameters = parameters if parameters is not None else {}
    data_weight_dict = data_weight_dict if data_weight_dict is not None else {}
    property_std_deviation = {
        'Y': 0.01,
    }

    params_keys, _ = extract_parameters(parameters)

    data_comps = list(set(data['components']).union({'VA'}))
    species = sorted(unpack_components(dbf, data_comps), key=str)
    data_phases = filter_phases(dbf, species, candidate_phases=data['phases'])
    models = instantiate_models(dbf, species, data_phases, model=model, parameters=parameters)
    output = data['output']
    property_output = output.split('_')[0]  # property without _FORM, _MIX, etc.
    samples = np.array(data['values']).flatten()
    reference = data.get('reference', '')

    # Models are now modified in response to the data from this data


    data['conditions'].setdefault('N', 1.0)  # Add default for N. Nothing else is supported in pycalphad anyway.
    pot_conds = OrderedDict([(getattr(v, key), unpack_condition(data['conditions'][key])) for key in sorted(data['conditions'].keys()) if not key.startswith('X_')])
    comp_conds = OrderedDict([(v.X(key[2:]), unpack_condition(data['conditions'][key])) for key in sorted(data['conditions'].keys()) if key.startswith('X_')])

    phase_records = build_phase_records(dbf, species, data_phases, {**pot_conds, **comp_conds}, models, parameters=parameters, build_gradients=True, build_hessians=True)

    # Now we need to unravel the composition conditions
    # (from Dict[v.X, Sequence[float]] to Sequence[Dict[v.X, float]]), since the
    # composition conditions are only broadcast against the potentials, not
    # each other. Each individual composition needs to be computed
    # independently, since broadcasting over composition cannot be turned off
    # in pycalphad.
    rav_comp_conds = [OrderedDict(zip(comp_conds.keys(), pt_comps)) for pt_comps in zip(*comp_conds.values())]

    # Build weights, should be the same size as the values
    total_num_calculations = len(rav_comp_conds)*np.prod([len(vals) for vals in pot_conds.values()])
    dataset_weights = np.array(data.get('weight', 10.0)) * np.ones(total_num_calculations)
    weights = (property_std_deviation.get(property_output, 1.0)/data_weight_dict.get(property_output, 1.0)/dataset_weights).flatten()

    return YPropData(dbf, species, data_phases, pot_conds, rav_comp_conds, models, params_keys, phase_records, output, samples, weights, reference)


def get_Y_thermochemical_data(dbf: Database, comps: Sequence[str],
                                        phases: Sequence[str],
                                        datasets: PickleableTinyDB,
                                        model: Optional[Dict[str, Model]] = None,
                                        parameters: Optional[Dict[str, float]] = None,
                                        data_weight_dict: Optional[Dict[str, float]] = None,
                                        ) -> Sequence[YPropData]:
    """
    Get all the YPropData for each matching equilibrium thermochemical dataset in the datasets

    Parameters
    ----------
    dbf : Database
        Database with parameters to fit
    comps : Sequence[str]
        List of pure element components used to find matching datasets.
    phases : Sequence[str]
        List of phases used to search for matching datasets.
    datasets : PickleableTinyDB
        Datasets that contain single phase data
    model : Optional[Dict[str, Type[Model]]]
        Dictionary phase names to pycalphad Model classes.
    parameters : Optional[Dict[str, float]]
        Mapping of parameter symbols to values.
    data_weight_dict : Optional[Dict[str, float]]
        Mapping of a data type (e.g. `HM` or `SM`) to a weight.

    Notes
    -----
    Found datasets will be subsets of the components and phases. Equilibrium
    thermochemical data is assumed to be any data that does not have the
    `solver` key, and does not have an output of `ZPF` or `ACR` (which
    correspond to different data types than can be calculated here.)

    Returns
    -------
    Sequence[YPropData]
    """

    desired_data = datasets.search(
        (where('output').test(lambda x: 'Y' in x)) &
        (where('components').test(lambda x: set(x).issubset(comps))))

    Y_thermochemical_data = []  # 1:1 correspondence with each dataset
    for data in desired_data:
        Y_thermochemical_data.append(build_Ypropdata(data, dbf, model=model, parameters=parameters, data_weight_dict=data_weight_dict))
    return Y_thermochemical_data







def calculate_Y_probability_differences(Ypropdata: YPropData,
                          parameters: np.ndarray,
                          approximate_equilibrium: Optional[bool] = False,
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return the sum of square error from site fraction data

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
    parameters : dict
        Dictionary of symbols that will be overridden in pycalphad.equilibrium
    phase_models : dict
        Phase models to pass to pycalphad calculations
    callables : dict
        Callables to pass to pycalphad
    data_weight : float
        Weight for standard deviation of activity measurements, dimensionless.
        Corresponds to the standard deviation of differences in chemical
        potential in typical measurements of activity, in J/mol.

    Returns
    -------
    float
        A single float of the sum of square errors

    Notes
    -----
    General procedure:
    1. Get the datasets
    2. For each dataset
        a. Calculate current site fraction 
        b. Find the target site fraction
        c. Calculate error due to site fraction

    """
    if approximate_equilibrium:
        _equilibrium = no_op_equilibrium_
    else:
        _equilibrium = equilibrium_

    dbf = Ypropdata.dbf
    species = Ypropdata.species
    phases = Ypropdata.phases

    pot_conds = Ypropdata.potential_conds
    models = Ypropdata.models
    phase_records = Ypropdata.phase_records
    update_phase_record_parameters(phase_records, parameters)
    params_dict = OrderedDict(zip(map(str, Ypropdata.params_keys), parameters))
    output = 'GM'
    weights = np.array(Ypropdata.weight, dtype=np.float_)
    samples = Ypropdata.samples
    sublattice=models[phases[0]].site_fractions

    calculated_data = []
    for comp_conds in Ypropdata.composition_conds:
        cond_dict = OrderedDict(**pot_conds, **comp_conds)
        # str_statevar_dict must be sorted, assumes that pot_conds are.
        str_statevar_dict = OrderedDict([(str(key), vals) for key, vals in pot_conds.items()])
        grid = calculate_(species, phases, str_statevar_dict, models, phase_records, pdens=50, fake_points=True)
        multi_eqdata = _equilibrium(phase_records, cond_dict, grid)
        result_st=multi_eqdata.Y.flatten()

        result = result_st[np.logical_not(np.isnan(result_st))]
        if len(result)==0 or len(result)<len(sublattice):
            return -np.inf 
        elif len(result) > len(sublattice):
            result_st=result_st[0]
            result = result_st[np.logical_not(np.isnan(result_st))]
        # TODO: could be kind of slow. Callables (which are cachable) must be built.
        #propdata = _eqcalculate(dbf, species, phases, cond_dict, 'Y', data=multi_eqdata, per_phase=False, callables=None, parameters=params_dict, model=models)
        #print('_pro',propdata.get_dataset())
        #if 'vertex' in propdata.data_vars[output][0]:
        #    raise ValueError(f"Property {output} cannot be used to calculate equilibrium thermochemical error because each phase has a unique value for this property.")

        calculated_data.extend(result)

    calculated_data = np.array(calculated_data, dtype=np.float_)
    ind=[i for i,v in enumerate(samples) if v == None]
    calculated_data=np.delete(calculated_data,ind)
    samples=np.delete(samples,ind)
    final_weights=[]
    for i in weights:
        final_weights.append([i]*len(sublattice))
    final_weights=np.array(final_weights)
    final_weights=np.delete(final_weights,ind)
    assert calculated_data.shape == samples.shape, f"Calculated data shape {calculated_data.shape} does not match samples shape {samples.shape}"
    assert calculated_data.shape == final_weights.shape, f"Calculated data shape {calculated_data.shape} does not match weights shape {weights.shape}"
    samples = np.array(samples, dtype=np.float_)
    differences = np.array(calculated_data - samples, dtype=np.float64)
    _log.debug('Output: %s differences: %s, weights: %s, reference: %s', output, differences, final_weights, Ypropdata.reference)
    return differences, final_weights


def calculate_Y_probability(Y_thermochemical_data: Sequence[YPropData],
                                                     parameters: np.ndarray,
                                                     approximate_equilibrium: Optional[bool] = False,
                                                     ) -> float:
    """
    Calculate the total equilibrium thermochemical probability for all EqPropData

    Parameters
    ----------
    Y_thermochemical_data : Sequence[EqPropData]
        List of site occupancy data corresponding to the datasets.
    parameters : np.ndarray
        Values of parameters for this iteration to be updated in PhaseRecords.
    approximate_equilibrium : Optional[bool], optional

    eq_thermochemical_data : Sequence[EqPropData]

    Returns
    -------
    float
        Sum of log-probability for all thermochemical data.

    """
    if len(Y_thermochemical_data) == 0:
        return 0.0

    differences = []
    weights = []
    for Ypropdata in Y_thermochemical_data:
        diffs, wts = calculate_Y_probability_differences(Ypropdata, parameters, approximate_equilibrium)

        if np.any(np.isinf(diffs) | np.isnan(diffs)):
            # NaN or infinity are assumed calculation failures. If we are
            # calculating log-probability, just bail out and return -infinity.
            return -np.inf
        differences.append(diffs)
        weights.append(wts)

    differences = np.concatenate(differences, axis=0)
    weights = np.concatenate(weights, axis=0)
    probs = norm(loc=0.0, scale=weights).logpdf(differences)
    return np.sum(probs)






