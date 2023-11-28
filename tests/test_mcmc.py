from unittest import mock
import numpy as np
import pytest

from numpy.linalg import LinAlgError
from scipy.stats import norm
from pycalphad import Database, variables as v

from espei.error_functions.context import setup_context
from espei.optimizers.opt_mcmc import EmceeOptimizer
from espei.error_functions import get_zpf_data, get_thermochemical_data
from espei.error_functions.zpf_error import ZPFResidual
from espei.priors import rv_zero
from .fixtures import datasets_db
from .testing_data import *


def test_lnprob_calculates_multi_phase_probability_for_success(datasets_db):
    """lnprob() successfully calculates the probability for equilibrium """
    dbf = Database.from_string(CU_MG_TDB, fmt='tdb')
    datasets_db.insert(CU_MG_DATASET_ZPF_WORKING)
    comps = ['CU', 'MG', 'VA']
    phases = ['LIQUID', 'FCC_A1', 'HCP_A3', 'LAVES_C15', 'CUMG2']
    param = 'VV0001'
    orig_val = dbf.symbols[param].args[0]
    initial_params = {param: orig_val}

    residual_objs = [
        ZPFResidual(dbf, datasets_db, None, [param])
    ]
    opt = EmceeOptimizer(dbf)
    res = opt.predict([10], prior_rvs=[rv_zero()], symbols_to_fit=[param], residual_objs=residual_objs)

    assert np.isreal(res)
    assert not np.isinf(res)
    assert np.isclose(res, -31.309645520830344, rtol=1e-6)

    # The purpose of this part is to test that the driving forces (and probability)
    # are different than the case of VV0001 = 10.
    res_2 = opt.predict([-10000000], prior_rvs=[rv_zero()], symbols_to_fit=[param], residual_objs=residual_objs)

    assert np.isreal(res_2)
    assert not np.isinf(res_2)
    # Accept a large rtol becuase the results should be _very_ different
    assert not np.isclose(res_2, -31.309645520830344, rtol=1e-2)

def test_lnprob_calculates_single_phase_probability_for_success(datasets_db):
    """lnprob() succesfully calculates the probability from single phase data"""
    dbf = Database.from_string(CU_MG_TDB_FCC_ONLY, fmt='tdb')
    datasets_db.insert(CU_MG_HM_MIX_SINGLE_FCC_A1)
    comps = ['CU', 'MG', 'VA']
    phases = ['FCC_A1']
    param = 'VV0003'
    orig_val = -14.0865
    opt = EmceeOptimizer(dbf)

    ctx = setup_context(dbf, datasets_db, symbols_to_fit=[param])
    res_orig = opt.predict([orig_val], prior_rvs=[rv_zero()], **ctx)
    assert np.isreal(res_orig)
    assert np.isclose(res_orig, -9.119484935312146, rtol=1e-6)

    res_10 = opt.predict([10.0], prior_rvs=[rv_zero()], **ctx)
    assert np.isreal(res_10)
    assert np.isclose(res_10, -9.143559131626864, rtol=1e-6)

    res_1e5 = opt.predict([1e5], prior_rvs=[rv_zero()], **ctx)
    assert np.isreal(res_1e5)
    assert np.isclose(res_1e5, -1359.1335466316268, rtol=1e-6)


def test_optimizer_computes_probability_with_activity_data(datasets_db):
    """EmceeOptimizer correctly computed probability with activity data

    This test is mathematically redundant with test_error_functions.test_activity_error, but aims to test the functionality of using the Optimizer / ResidualFunction API
    """
    datasets_db.insert(CU_MG_EXP_ACTIVITY)
    dbf = Database(CU_MG_TDB)
    opt = EmceeOptimizer(dbf)
    # Having no degrees of freedom isn't currently allowed by setup_context
    # we use VV0000 and the current value in the database
    ctx = setup_context(dbf, datasets_db, symbols_to_fit=["VV0000"])
    error = opt.predict(np.array([-32429.6]), **ctx)
    assert np.isclose(error, -257.41020886970756, rtol=1e-6)


def _eq_LinAlgError(*args, **kwargs):
    raise LinAlgError()


def _eq_ValueError(*args, **kwargs):
    raise ValueError()


@mock.patch('espei.error_functions.zpf_error.equilibrium', _eq_LinAlgError)
@pytest.mark.xfail
def test_lnprob_does_not_raise_on_LinAlgError(datasets_db):
    """lnprob() should catch LinAlgError raised by equilibrium and return -np.inf"""
    dbf = Database.from_string(CU_MG_TDB, fmt='tdb')
    comps = ['CU', 'MG', 'VA']
    phases = ['LIQUID', 'FCC_A1', 'HCP_A3', 'LAVES_C15', 'CUMG2']
    datasets_db.insert(CU_MG_DATASET_ZPF_WORKING)
    opt = EmceeOptimizer(dbf)
    residual_objs = [
        ZPFResidual(dbf, datasets_db, None, ["VV0001"])
    ]
    res = opt.predict([10], prior_rvs=[rv_zero()], symbols_to_fit=['VV0001'], residual_objs=residual_objs)
    assert np.isneginf(res)


@mock.patch('espei.error_functions.zpf_error.equilibrium', _eq_ValueError)
@pytest.mark.xfail
def test_lnprob_does_not_raise_on_ValueError(datasets_db):
    """lnprob() should catch ValueError raised by equilibrium and return -np.inf"""
    dbf = Database.from_string(CU_MG_TDB, fmt='tdb')
    opt = EmceeOptimizer(dbf)
    comps = ['CU', 'MG', 'VA']
    phases = ['LIQUID', 'FCC_A1', 'HCP_A3', 'LAVES_C15', 'CUMG2']
    datasets_db.insert(CU_MG_DATASET_ZPF_WORKING)
    opt = EmceeOptimizer(dbf)
    residual_objs = [
        ZPFResidual(dbf, datasets_db, None, ["VV0001"])
    ]
    res = opt.predict([10], prior_rvs=[rv_zero()], symbols_to_fit=['VV0001'], residual_objs=residual_objs)
    assert np.isneginf(res)


def test_parameter_initialization():
    """Determinisitically generated parameters should match."""
    initial_parameters = np.array([1, 10, 100, 1000])
    opt = EmceeOptimizer(Database())
    deterministic_params = opt.initialize_new_chains(initial_parameters, 1, 0.10, deterministic=True)
    print(repr(deterministic_params))
    expected_parameters = np.array([
        list(initial_parameters), # The first element is always a start at the initial parameters
        #These values are known due to deterministic=True above
        [1.03116874e+00, 9.01412995e+00, 1.12795943e+02, 9.16447258e+02],
        [1.00664662e+00, 1.07178898e+01, 9.63696718e+01, 1.36872292e+03],
        [1.07642366e+00, 1.16413520e+01, 8.71742457e+01, 9.61836382e+02],        
    ])
    assert np.all(np.isclose(deterministic_params, expected_parameters))


def test_emcee_opitmizer_can_restart(datasets_db):
    """A restart trace can be passed to the Emcee optimizer """
    dbf = Database.from_string(CU_MG_TDB, fmt='tdb')
    datasets_db.insert(CU_MG_DATASET_ZPF_WORKING)
    param = 'VV0001'
    opt = EmceeOptimizer(dbf)
    restart_tr = -4*np.ones((2, 10, 1))  # 2 chains, 10 iterations, 1 parameter
    opt.fit([param], datasets_db, iterations=1, chains_per_parameter=2, restart_trace=restart_tr)
    assert opt.sampler.chain.shape == (2, 1, 1)


def test_equilibrium_thermochemical_correct_probability(datasets_db):
    """Integration test for equilibrium thermochemical error."""
    dbf = Database(CU_MG_TDB)
    opt = EmceeOptimizer(dbf)
    datasets_db.insert(CU_MG_EQ_HMR_LIQUID)
    ctx = setup_context(dbf, datasets_db, ['VV0017'])
    ctx.update(opt.get_priors(None, ['VV0017'], [0]))

    prob = opt.predict(np.array([-31626.6]), **ctx)
    expected_prob = norm(loc=0, scale=500).logpdf([-31626.6*0.5*0.5]).sum()
    assert np.isclose(prob, expected_prob)

    # change to -40000
    prob = opt.predict(np.array([-40000], dtype=np.float_), **ctx)
    expected_prob = norm(loc=0, scale=500).logpdf([-40000*0.5*0.5]).sum()
    assert np.isclose(prob, expected_prob)

def test_lnprob_calculates_associate_tdb(datasets_db):
    """lnprob() successfully calculates the probability for equilibrium """
    dbf = Database.from_string(CU_MG_TDB_ASSOC, fmt='tdb')
    datasets_db.insert(CU_MG_DATASET_ZPF_WORKING)
    comps = ['CU', 'MG', 'VA']
    phases = ['LIQUID', 'FCC_A1', 'HCP_A3', 'LAVES_C15', 'CUMG2']
    param = 'VV0001'
    orig_val = dbf.symbols[param].args[0]
    initial_params = {param: orig_val}

    residual_objs = [
        ZPFResidual(dbf, datasets_db, None, [param])
    ]
    opt = EmceeOptimizer(dbf)
    res = opt.predict([10], prior_rvs=[rv_zero()], symbols_to_fit=[param], residual_objs=residual_objs)

    assert np.isreal(res)
    assert not np.isinf(res)
    assert np.isclose(res, -31.309645520830344, rtol=1e-6)

    # The purpose of this part is to test that the driving forces (and probability)
    # are different than the case of VV0001 = 10.
    res_2 = opt.predict([-10000000], prior_rvs=[rv_zero()], symbols_to_fit=[param], residual_objs=residual_objs)

    assert np.isreal(res_2)
    assert not np.isinf(res_2)
    # Accept a large rtol becuase the results should be _very_ different
    assert not np.isclose(res_2, -31.309645520830344, rtol=1e-2)
