"""Fixtures for use in tests"""

import os, time

import pytest
from espei.datasets import validate_dataset
from espei.utils import PickleableTinyDB, MemoryStorage


class ValidatedDatasets(PickleableTinyDB):
    """Datasets that are validated when inserted, as by espei.datasets.load_datasets"""
    def insert(self, dataset):
        return self.table(self.default_table_name).insert(validate_dataset(dataset).model_dump(exclude_unset=True))


@pytest.fixture
def datasets_db():
    """Returns a clean instance of ValidatedDatasets"""
    db = ValidatedDatasets(storage=MemoryStorage)
    yield db
    db.close()

@pytest.fixture
def tmp_file():
    """Create a temporary file with content and return the file name"""
    fname = 'tmp_file-' + str(time.time()).split('.')[0]
    def _tmp_file(content):
        with open(fname, 'w') as fp:
            fp.write(content)
        return fname
    yield _tmp_file
    os.remove(fname)
