import mock
import numpy as np
import pytest

from numpy.linalg import LinAlgError
from pycalphad import Database, variables as v
from pycalphad.codegen.callables import build_callables
from pycalphad.core.utils import instantiate_models

from espei.optimizers.opt_mcmc import EmceeOptimizer
from espei.error_functions import get_zpf_data, get_thermochemical_data
from espei.priors import rv_zero
from .fixtures import datasets_db
from .testing_data import CU_MG_TDB, CU_MG_DATASET_ZPF_WORKING, CU_MG_TDB_FCC_ONLY, CU_MG_HM_MIX_SINGLE_FCC_A1


def test_lnprob_calculates_multi_phase_probability_for_success(datasets_db):
    """lnprob() successfully calculates the probability for equilibrium """
    dbf = Database.from_string(CU_MG_TDB, fmt='tdb')
    datasets_db.insert(CU_MG_DATASET_ZPF_WORKING)
    comps = ['CU', 'MG', 'VA']
    phases = ['LIQUID', 'FCC_A1', 'HCP_A3', 'LAVES_C15', 'CUMG2']
    param = 'VV0001'
    orig_val = dbf.symbols[param].args[0].expr
    models = instantiate_models(dbf, comps, phases, parameters={param: orig_val})
    eq_callables = build_callables(dbf, comps, phases, models, parameter_symbols=[param],
                        output='GM', build_gradients=True, build_hessians=False,
                        additional_statevars={v.N, v.P, v.T})

    zpf_kwargs = {
        'dbf': dbf, 'phases': phases, 'zpf_data': get_zpf_data(comps, phases, datasets_db),
        'phase_models': models, 'callables': eq_callables,
        'data_weight': 1.0,
    }
    opt = EmceeOptimizer(dbf)
    res = opt.predict([10], prior_rvs=[rv_zero()], symbols_to_fit=[param], zpf_kwargs=zpf_kwargs)

    assert np.isreal(res)
    assert np.isclose(res, -31.309645520830344, rtol=1e-6)

    res_2 = opt.predict([10000000], prior_rvs=[rv_zero()], symbols_to_fit=[param], zpf_kwargs=zpf_kwargs)

    assert not np.isclose(res_2, -31.309645520830344, rtol=1e-6)


def test_lnprob_calculates_single_phase_probability_for_success(datasets_db):
    """lnprob() succesfully calculates the probability from single phase data"""
    dbf = Database.from_string(CU_MG_TDB_FCC_ONLY, fmt='tdb')
    datasets_db.insert(CU_MG_HM_MIX_SINGLE_FCC_A1)
    comps = ['CU', 'MG', 'VA']
    phases = ['FCC_A1']
    param = 'VV0003'
    orig_val = -14.0865
    opt = EmceeOptimizer(dbf)

    thermochemical_data = get_thermochemical_data(dbf, comps, phases, datasets_db, parameters={param: orig_val})
    thermochemical_kwargs = {'dbf': dbf, 'comps': comps, 'thermochemical_data': thermochemical_data}
    res_orig = opt.predict([orig_val], prior_rvs=[rv_zero()], symbols_to_fit=[param], thermochemical_kwargs=thermochemical_kwargs)
    assert np.isreal(res_orig)
    assert np.isclose(res_orig, -9.119484935312146, rtol=1e-6)

    res_10 = opt.predict([10], prior_rvs=[rv_zero()], symbols_to_fit=[param], thermochemical_kwargs=thermochemical_kwargs)
    assert np.isreal(res_10)
    assert np.isclose(res_10, -9.143559131626864, rtol=1e-6)

    res_1e5 = opt.predict([1e5], prior_rvs=[rv_zero()], symbols_to_fit=[param], thermochemical_kwargs=thermochemical_kwargs)
    assert np.isreal(res_1e5)
    assert np.isclose(res_1e5, -1359.1335466316268, rtol=1e-6)


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
    zpf_kwargs = {'dbf': dbf, 'phases': phases, 'zpf_data': get_zpf_data(comps, phases, datasets_db), 'data_weight': 1.0}
    res = opt.predict([10], prior_rvs=[rv_zero()], symbols_to_fit=['VV0001'], zpf_kwargs=zpf_kwargs)
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
    zpf_kwargs = {'dbf': dbf, 'phases': phases, 'zpf_data': get_zpf_data(comps, phases, datasets_db), 'data_weight': 1.0}
    res = opt.predict([10], prior_rvs=[rv_zero()], symbols_to_fit=['VV0001'], zpf_kwargs=zpf_kwargs)
    assert np.isneginf(res)


def test_parameter_initialization():
    """Determinisitically generated parameters should match."""
    initial_parameters = np.array([1, 10, 100, 1000])
    opt = EmceeOptimizer(Database())
    deterministic_params = opt.initialize_new_chains(initial_parameters, 1, 0.10, deterministic=True)
    expected_parameters = np.array([
        [9.81708401e-01, 9.39027722e+00, 1.08016748e+02, 9.13512881e+02],
        [1.03116874, 9.01412995, 112.79594345, 916.44725799],
        [1.00664662e+00, 1.07178898e+01, 9.63696718e+01, 1.36872292e+03],
        [1.07642366e+00, 1.16413520e+01, 8.71742457e+01, 9.61836382e+02]])
    assert np.all(np.isclose(deterministic_params, expected_parameters))
