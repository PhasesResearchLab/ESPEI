# pylint: disable=redefined-outer-name
"""
Test that setting up and fitting never modifies the caller's Database or pycalphad.variables.
"""

import numpy as np
import pytest
from pycalphad import Database, variables as v

from espei.error_functions.context import setup_context
from espei.optimizers.opt_base import OptimizerBase
from espei.optimizers.opt_scipy import SciPyOptimizer
from espei.utils import unpack_piecewise

from .fixtures import datasets_db
from .testing_data import *


@pytest.fixture
def cu_mg_datasets(datasets_db):
    """Cu-Mg datasets with data for every residual type"""
    datasets_db.insert(CU_MG_DATASET_ZPF_WORKING)
    datasets_db.insert(CU_MG_EXP_ACTIVITY)
    datasets_db.insert(CU_MG_EQ_HMR_LIQUID)
    datasets_db.insert(CU_MG_HM_MIX_CUMG2_ANTISITE)
    return datasets_db


def _symbols_snapshot(dbf):
    return {name: (type(value), value) for name, value in dbf.symbols.items()}


def _initial_guess(dbf, ctx):
    return np.array([unpack_piecewise(dbf.symbols[s]) for s in ctx['symbols_to_fit']])


def test_setup_context_does_not_modify_database(cu_mg_datasets):
    """setup_context and get_likelihood leave the Database symbols and pycalphad.variables unchanged"""
    dbf = Database(CU_MG_TDB)
    symbols_before = _symbols_snapshot(dbf)
    variables_before = set(dir(v))
    ctx = setup_context(dbf, cu_mg_datasets)
    assert _symbols_snapshot(dbf) == symbols_before
    assert set(dir(v)) == variables_before

    params = _initial_guess(dbf, ctx)
    for residual_obj in ctx['residual_objs']:
        residuals, _ = residual_obj.get_residuals(params)
        assert len(residuals) > 0
        assert np.isfinite(residual_obj.get_likelihood(params))
    assert _symbols_snapshot(dbf) == symbols_before
    assert set(dir(v)) == variables_before


def test_setup_context_twice_gives_identical_likelihoods(cu_mg_datasets):
    """Two contexts set up from the same Database give identical likelihoods"""
    dbf = Database(CU_MG_TDB)
    ctx_1 = setup_context(dbf, cu_mg_datasets)
    ctx_2 = setup_context(dbf, cu_mg_datasets)
    params = _initial_guess(dbf, ctx_1) * 1.01
    for residual_1, residual_2 in zip(ctx_1['residual_objs'], ctx_2['residual_objs']):
        assert residual_1.get_likelihood(params) == residual_2.get_likelihood(params)


class _ConstantOptimizer(OptimizerBase):
    def _fit(self, symbols, datasets):
        return {'VV0001': 123.0}


def test_optimizer_fit_does_not_modify_database():
    """OptimizerBase.fit updates only the returned Database"""
    dbf = Database(CU_MG_TDB)
    original_value = unpack_piecewise(dbf.symbols['VV0001'])
    opt = _ConstantOptimizer(dbf)
    fit_dbf = opt.fit(['VV0001'], None)
    assert fit_dbf is opt.dbf
    assert fit_dbf.symbols['VV0001'] == 123.0
    assert unpack_piecewise(dbf.symbols['VV0001']) == original_value
    assert unpack_piecewise(opt.orig_dbf.symbols['VV0001']) == original_value


def test_scipy_optimizer_predict_is_negative_log_likelihood(cu_mg_datasets):
    """SciPyOptimizer.predict returns the negative total log-likelihood"""
    dbf = Database(CU_MG_TDB)
    ctx = setup_context(dbf, cu_mg_datasets)
    params = _initial_guess(dbf, ctx)
    lnlike = sum(residual_obj.get_likelihood(params) for residual_obj in ctx['residual_objs'])
    assert np.isclose(SciPyOptimizer.predict(params, ctx), -lnlike, rtol=1e-12)
