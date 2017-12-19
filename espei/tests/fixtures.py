"""Fixtures for use in tests"""

import os, time

import pytest
from espei.utils import PickleableTinyDB, MemoryStorage

@pytest.fixture
def datasets_db():
    """Returns a clean instance of a PickleableTinyDB for datasets"""
    db = PickleableTinyDB(storage=MemoryStorage)
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
