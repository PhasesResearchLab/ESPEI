import fnmatch, json, os
from typing import Any, Dict, List, TypeAlias
import warnings
import numpy as np
from tinydb.storages import MemoryStorage
from tinydb import where

from espei.utils import PickleableTinyDB
from .dataset_models import to_Dataset, Dataset, ActivityPropertyDataset, BroadcastSinglePhaseFixedConfigurationDataset, EquilibriumPropertyDataset, ZPFDataset, DatasetError

__all__ = [
    "load_datasets",
    "recursive_glob",
    "apply_tags",
    "check_dataset",
    "clean_dataset"
]

def check_dataset(dataset: dict[str, Any]) -> dict[str, Any]:
    """Ensure that the dataset is valid and consistent by round-tripping through pydantic."""
    warnings.warn(f"check_dataset is deprecated will be removed in ESPEI 0.11. Behavior has been migrated to the pydantic dataset implementations in espei.datasets.dataset_models. To get a Dataset object, use espei.datasets.to_Dataset.", DeprecationWarning)
    return to_Dataset(dataset).model_dump()


def clean_dataset(dataset: dict[str, Any]) -> dict[str, Any]:
    """Ensure that the dataset is valid and consistent by round-tripping through pydantic."""
    warnings.warn(f"clean_dataset is deprecated will be removed in ESPEI 0.11. Behavior has been migrated to the pydantic dataset implementations in espei.datasets.dataset_models. To get a Dataset object, use espei.datasets.to_Dataset.", DeprecationWarning)
    return to_Dataset(dataset).model_dump()


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
                ds_database.insert(to_Dataset(d).model_dump())
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
