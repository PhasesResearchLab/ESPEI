"""
Test different error functions as isolated units.
"""

import numpy as np

from pycalphad import Database

from espei.error_functions import calculate_activity_error

from .fixtures import datasets_db
from .testing_data import CU_MG_TDB, CU_MG_EXP_ACTIVITY


def test_activity_error(datasets_db):
    """Test that activity error returns a correct result"""

    datasets_db.insert(CU_MG_EXP_ACTIVITY)

    dbf = Database(CU_MG_TDB)
    error = calculate_activity_error(dbf, ['CU','MG','VA'], list(dbf.phases.keys()), datasets_db, {}, {}, {}, {}, {}, {}, {})
    assert np.isclose(float(error), -93037371.27, atol=0.01)
