"""Fixtures for use in tests"""

import pytest
from espei.utils import PickleableTinyDB, MemoryStorage

@pytest.fixture
def datasets_db():
    """Returns a clean instance of a PickleableTinyDB for datasets"""
    db = PickleableTinyDB(storage=MemoryStorage)
    yield db
    db.close()
