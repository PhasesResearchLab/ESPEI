
import yaml
YAML_LOADER = yaml.FullLoader
from tinydb import TinyDB, Query, where
from espei.utils import PickleableTinyDB, MemoryStorage
from espei.datasets import DatasetError, load_datasets
from pycalphad import Database, calculate, equilibrium, variables as v
from pycalphad.core.utils import filter_phases
from espei.core_utils import ravel_conditions
from pycalphad.plot.eqplot import _map_coord_to_variable
from pycalphad.core.calculate import instantiate_models
import numpy as np
import copy
from scipy.stats import norm
import logging


def calculate_Y_probability(dbf, comps, phases, datasets, parameters=None, phase_models=None, callables=None, data_weight=1.0):
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
    std_dev = 500
    if parameters is None:
        parameters = {}
    error = 0
    Y_data=datasets.search(
        (where('output').test(lambda x: 'Y' in x)) &
        (where('components').test(lambda x: set(x).issubset(comps))))
    error = 0
    if len(Y_data) == 0:
        return error
    for ds in Y_data:
        data_comps=ds['components']
        species = list(map(v.Species, data_comps))
        data_phases = phases
        conditions = {}
        conds_list = [(cond, value) for cond, value in ds['conditions'].items() if cond not in ('P', 'T')]
        print(conds_list)
        for comp_name, comp_x in conds_list:
            print(comp_name, comp_x)
            P, T, X = ravel_conditions(ds['values'], ds['conditions']['P'], ds['conditions']['T'], comp_x,Y=True)
            conditions[v.P] = P
            conditions[v.T] = T
            conditions[_map_coord_to_variable(comp_name)] = X
        print(P,T,X)
        conditions_list = [{c: conditions[c][i] for c in conditions.keys()} for i in range(len(conditions[v.T]))]
        print('cond',len(conditions[v.T]),X,conditions_list)
        current_Y = []
        model=instantiate_models(dbf, data_comps, ds['phases'])
        sublattice=model[ds['phases'][0]].site_fractions
        for conds in conditions_list:
            sample_eq_res = equilibrium(dbf, data_comps, data_phases, conds,
                                    model=phase_models, parameters=parameters,
                                        callables=callables)
        
            result_st=sample_eq_res.Y.where(sample_eq_res.Phase==ds['phases']).squeeze().values
            result = result_st[np.logical_not(np.isnan(result_st))]
            if len(result)==0 or len(result)<len(sublattice):
                return -np.inf 
            elif len(result) > len(sublattice):
                result_st=result_st[0]
                result = result_st[np.logical_not(np.isnan(result_st))]
            current_Y=np.hstack((current_Y,result))
        target_Y=np.array(ds['values']).flatten()
        weight = ds.get('weight', 10.00)
        ind=[i for i,v in enumerate(target_Y) if v == None]
        target_Y=np.delete(target_Y,ind)
        current_Y=np.delete(current_Y,ind)
        print(target_Y,current_Y)
        pe =norm(loc=0, scale=0.01/(weight*data_weight)).logpdf(np.array(target_Y - current_Y, dtype=np.float64))
        error += np.sum(pe)
        logging.debug('Site_fraction error - data: {}, site_fraction difference: {}, probability: {}, reference: {}'.format(target_Y, current_Y-target_Y, pe, ds["reference"]))

        # TODO: write a test for this
    if np.any(np.isnan(np.array([error], dtype=np.float64))):  # must coerce sympy.core.numbers.Float to float64
        return -np.inf
    return error






