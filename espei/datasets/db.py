import fnmatch, json, os
from typing import Any, Dict, List, TypeAlias
import warnings
import numpy as np
from tinydb.storages import MemoryStorage
from tinydb import where

from espei.utils import PickleableTinyDB

from .dataset_models import Dataset, ActivityPropertyDataset, BroadcastSinglePhaseFixedConfigurationDataset, EquilibriumPropertyDataset, ZPFDataset, DatasetError



def recursive_map(f, x):
    """
    map, but over nested lists

    Parameters
    ----------
    f : callable
        Function to apply to x
    x : list or value
        Value passed to v

    Returns
    -------
    list or value
    """
    if isinstance(x, list):
        if [isinstance(xx, list) for xx in x]:
            # we got a nested list
            return [recursive_map(f, xx) for xx in x]
        else:
            # it's a list with some values inside
            return list(map(f, x))
    else:
        # not a list, probably just a singular value
        return f(x)


def check_dataset(dataset: dict[str, Any]) -> Dataset:
    """Ensure that the dataset is valid and consistent.

    Currently supports the following validation checks:
    * data shape is valid
    * phases and components used match phases and components entered
    * individual shapes of keys, such as ZPF, sublattice configs and site ratios

    Planned validation checks:
    * all required keys are present

    Note that this follows some of the implicit assumptions in ESPEI at the time
    of writing, such that conditions are only P, T, configs for single phase and
    essentially only T for ZPF data.

    Parameters
    ----------
    dataset : Dataset
        Dictionary of the standard ESPEI dataset.

    Returns
    -------
    Dataset

    Raises
    ------
    DatasetError
        If an error is found in the dataset
    """
    is_equilibrium = 'solver' not in dataset.keys() and dataset['output'] != 'ZPF'
    is_activity = dataset['output'].startswith('ACR')
    components = dataset['components']
    conditions = dataset['conditions']
    values = dataset['values']
    phases = dataset['phases']
    if is_equilibrium:
        conditions = dataset['conditions']
        comp_conditions = {k: v for k, v in conditions.items() if k.startswith('X_')}
    if is_activity:
        ref_state = dataset['reference_state']
    elif is_equilibrium:
        for el, vals in dataset.get('reference_states', {}).items():
            if 'phase' not in vals:
                raise DatasetError(f'Reference state for element {el} must define the `phase` key with the reference phase name.')

    # check that the shape of conditions match the values
    num_pressure = np.atleast_1d(conditions['P']).size
    num_temperature = np.atleast_1d(conditions['T']).size
    if is_equilibrium:
        values_shape = np.array(values).shape
        # check each composition condition is the same shape
        num_x_conds = [len(v) for _, v in comp_conditions.items()]
        if num_x_conds.count(num_x_conds[0]) != len(num_x_conds):
            raise DatasetError('All compositions in conditions are not the same shape. Note that conditions cannot be broadcast. Composition conditions are {}'.format(comp_conditions))
        conditions_shape = (num_pressure, num_temperature, num_x_conds[0])
        if conditions_shape != values_shape:
            raise DatasetError('Shape of conditions (P, T, compositions): {} does not match the shape of the values {}.'.format(conditions_shape, values_shape))

    # check that all of the components used match the components entered
    if is_equilibrium:  # and is_activity
        components_entered = set(components)
        components_used = set()
        components_used.update({c.split('_')[1] for c in comp_conditions.keys()})
        # mass balance of components
        comp_dof = len(comp_conditions.keys())
        if len(components_entered - components_used - {'VA'}) > comp_dof or len(components_used - components_entered) > 0:
            raise DatasetError('Components entered {} do not match components used {}.'.format(components_entered, components_used))

    if dataset["output"] == "ZPF":
        dataset_obj = ZPFDataset(**dataset)
    elif is_activity:
        dataset_obj = ActivityPropertyDataset(**dataset)
    elif is_equilibrium:
        dataset_obj = EquilibriumPropertyDataset(**dataset)
    elif 'solver' in dataset.keys():
        dataset_obj = BroadcastSinglePhaseFixedConfigurationDataset(**dataset)
    else:
        raise ValueError(f"Unknown dataset type for dataset {dataset}")
    return dataset_obj


def clean_dataset(dataset: dict[str, Any]) -> dict[str, Any]:
    """
    No-op
    """
    warnings.warn(f"clean_dataset deprecated will be removed in ESPEI 0.11. Behavior has been migrated to the pydantic dataset implementations in espei.datasets.dataset_models.", DeprecationWarning)
    return dataset


def apply_tags(datasets: PickleableTinyDB, tags):
    """
    Modify datasets using the tags system

    Parameters
    ----------
    datasets : PickleableTinyDB
        Datasets to modify
    tags : dict
        Dictionary of {tag: update_dict}

    Returns
    -------
    None

    Notes
    -----
    In general, everything replaces or is additive. We use the following update rules:
    1. If the update value is a list, extend the existing list (empty list if key does not exist)
    2. If the update value is scalar, override the previous (deleting any old value, if present)
    3. If the update value is a dict, update the exist dict (empty dict if dict does not exist)
    4. Otherwise, the value is updated, overriding the previous

    Examples
    --------
    >>> from espei.utils import PickleableTinyDB
    >>> from tinydb.storages import MemoryStorage
    >>> ds = PickleableTinyDB(storage=MemoryStorage)
    >>> doc_id = ds.insert({'tags': ['dft'], 'excluded_model_contributions': ['contrib']})
    >>> my_tags = {'dft': {'excluded_model_contributions': ['idmix', 'mag'], 'weight': 5.0}}
    >>> from espei.datasets import apply_tags
    >>> apply_tags(ds, my_tags)
    >>> all_data = ds.all()
    >>> all(d['excluded_model_contributions'] == ['contrib', 'idmix', 'mag'] for d in all_data)
    True
    >>> all(d['weight'] == 5.0 for d in all_data)
    True

    """
    for tag, update_dict in tags.items():
        matching_datasets = datasets.search(where("tags").test(lambda x: tag in x))
        for newkey, newval in update_dict.items():
            for match in matching_datasets:
                if isinstance(newval, list):
                    match[newkey] = match.get(newkey, []) + newval
                elif np.isscalar(newval):
                    match[newkey] = newval
                elif isinstance(newval, dict):
                    d = match.get(newkey, dict())
                    d.update(newval)
                    match[newkey] = d
                else:
                    match[newkey] = newval
                datasets.update(match, doc_ids=[match.doc_id])


def load_datasets(dataset_filenames, include_disabled=False) -> PickleableTinyDB:
    """
    Create a PickelableTinyDB with the data from a list of filenames.

    Parameters
    ----------
    dataset_filenames : [str]
        List of filenames to load as datasets

    Returns
    -------
    PickleableTinyDB
    """
    ds_database = PickleableTinyDB(storage=MemoryStorage)
    for fname in dataset_filenames:
        with open(fname) as file_:
            try:
                d = json.load(file_)
                if not include_disabled and d.get('disabled', False):
                    # The dataset is disabled and not included
                    continue
                dataset_obj = check_dataset(d)
                ds_database.insert(dataset_obj.model_dump())
            except ValueError as e:
                raise ValueError('JSON Error in {}: {}'.format(fname, e))
            except DatasetError as e:
                raise DatasetError('Dataset Error in {}: {}'.format(fname, e))
    return ds_database


def recursive_glob(start, pattern='*.json'):
    """
    Recursively glob for the given pattern from the start directory.

    Parameters
    ----------
    start : str
        Path of the directory to walk while for file globbing
    pattern : str
        Filename pattern to match in the glob.

    Returns
    -------
    [str]
        List of matched filenames

    """
    matches = []
    for root, dirnames, filenames in os.walk(start, followlinks=True):
        for filename in fnmatch.filter(filenames, pattern):
            matches.append(os.path.join(root, filename))
    return sorted(matches)
