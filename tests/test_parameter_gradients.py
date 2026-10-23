# pylint: disable=redefined-outer-name
"""
Test opt-in likelihood gradients with respect to the fitted parameters.
"""
from unittest import mock

import numpy as np
import pytest
from pycalphad import Database

from espei import _parameter_sensitivities as sensitivities
from espei._parameter_sensitivities import BACKEND
from espei.error_functions.context import setup_context
from espei.error_functions.residual_base import ResidualFunction
from espei.optimizers.opt_mcmc import EmceeOptimizer
from espei.optimizers.opt_scipy import SciPyOptimizer
from espei.utils import unpack_piecewise

from .fixtures import datasets_db
from .testing_data import *

BACKEND_FUNCTIONS = ['get_tangent', 'get_gradient', 'isolated_phase_gm_gradient', 'fixed_configuration_gradient']


@pytest.fixture
def cu_mg_context(datasets_db):
    """Cu-Mg context with data for every residual type, and its Database's parameter values"""
    datasets_db.insert(CU_MG_DATASET_ZPF_WORKING)
    datasets_db.insert(CU_MG_EXP_ACTIVITY)
    datasets_db.insert(CU_MG_EQ_HMR_LIQUID)
    datasets_db.insert(CU_MG_HM_MIX_CUMG2_ANTISITE)
    dbf = Database(CU_MG_TDB)
    ctx = setup_context(dbf, datasets_db)
    params = np.array([unpack_piecewise(dbf.symbols[s]) for s in ctx['symbols_to_fit']])
    return ctx, params


def _no_backend(*args, **kwargs):
    raise NotImplementedError("no parameter-sensitivity backend")


def test_stub_backend_raises(cu_mg_context):
    """Every residual's gradient calls the backend, so without one it raises"""
    if BACKEND is None:
        for name in BACKEND_FUNCTIONS:
            func = getattr(sensitivities, name)
            with pytest.raises(NotImplementedError, match="no parameter-sensitivity backend"):
                func(*[None] * func.__code__.co_argcount)
    ctx, params = cu_mg_context
    with mock.patch.multiple(sensitivities, **{name: _no_backend for name in BACKEND_FUNCTIONS}):
        for residual_obj in ctx['residual_objs']:
            with pytest.raises(NotImplementedError, match="no parameter-sensitivity backend"):
                residual_obj.get_likelihood_and_gradient(params)


def test_likelihoods_never_call_the_backend(cu_mg_context):
    """get_residuals, get_likelihood and the optimizers' predict compute no gradients"""
    ctx, params = cu_mg_context
    with mock.patch.multiple(sensitivities, **{name: mock.Mock(side_effect=AssertionError(name)) for name in BACKEND_FUNCTIONS}):
        for residual_obj in ctx['residual_objs']:
            residuals, _ = residual_obj.get_residuals(params)
            assert len(residuals) > 0
            assert np.isfinite(residual_obj.get_likelihood(params))
        assert np.isfinite(EmceeOptimizer.predict(params, **ctx))
        assert np.isfinite(SciPyOptimizer.predict(params, ctx))


def test_residual_without_gradient_raises():
    """The protocol's get_likelihood_and_gradient names the residual that lacks it"""
    class LikelihoodOnlyResidual(ResidualFunction):
        def __init__(self):
            pass

        def get_likelihood(self, parameters):
            return 0.0

    with pytest.raises(NotImplementedError, match="LikelihoodOnlyResidual"):
        LikelihoodOnlyResidual().get_likelihood_and_gradient(np.zeros(1))
