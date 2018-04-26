"""
Test different error functions as isolated units.
"""

import numpy as np

from pycalphad import Database

from espei.error_functions import calculate_activity_error, calculate_thermochemical_error

from .fixtures import datasets_db
from .testing_data import CU_MG_TDB, CU_MG_EXP_ACTIVITY, CU_MG_HM_FORM_T_CUMG2, CU_MG_SM_FORM_T_X_FCC_A1


def test_activity_error(datasets_db):
    """Test that activity error returns a correct result"""

    datasets_db.insert(CU_MG_EXP_ACTIVITY)

    dbf = Database(CU_MG_TDB)
    error = calculate_activity_error(dbf, ['CU','MG','VA'], list(dbf.phases.keys()), datasets_db, {}, {}, {}, {}, {}, {}, {})
    assert np.isclose(float(error), -93037371.27, atol=0.01)


def test_thermochemical_error_with_multiple_T_points(datasets_db):
    """Multiple temperature datapoints in a dataset for a stoichiometric comnpound should be successful."""
    datasets_db.insert(CU_MG_HM_FORM_T_CUMG2)

    dbf = Database(CU_MG_TDB)
    error = calculate_thermochemical_error(dbf, ['CU','MG','VA'], list(dbf.phases.keys()), datasets_db, {}, {}, {}, {})
    assert np.isclose(float(error), 0, atol=0.01)


def test_thermochemical_error_with_multiple_T_X_points(datasets_db):
    """Multiple temperature and composition datapoints in a dataset for a mixing phase should be successful."""
    datasets_db.insert(CU_MG_SM_FORM_T_X_FCC_A1)

    dbf = Database(CU_MG_TDB)
    error = calculate_thermochemical_error(dbf, ['CU', 'MG', 'VA'], list(dbf.phases.keys()), datasets_db, {}, {}, {}, {})
    assert np.isclose(float(error), 0, atol=0.01)

