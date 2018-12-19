import mock
import numpy as np
import sympy

from numpy.linalg import LinAlgError
from pycalphad import Database
from pycalphad.codegen.callables import build_callables
from pycalphad import Model

from espei.mcmc import lnprob, generate_parameter_distribution
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
    eq_callables = build_callables(dbf, comps, phases, model=Model, parameters={param: orig_val})
    eq_callables['phase_models'] = eq_callables.pop('model')
    eq_callables.pop('phase_records')

    res = lnprob([10], prior_rvs=[rv_zero()], comps=comps, dbf=dbf, phases=phases,
                 datasets=datasets_db, symbols_to_fit=[param], callables=eq_callables)

    assert np.isreal(res)
    assert np.isclose(res, -31.309645520830344, rtol=1e-6)

    res_2 = lnprob([10000000], prior_rvs=[rv_zero()], comps=comps, dbf=dbf, phases=phases,
                 datasets=datasets_db, symbols_to_fit=[param], callables=eq_callables)

    assert not np.isclose(res_2, -31.309645520830344, rtol=1e-6)


def test_lnprob_calculates_single_phase_probability_for_success(datasets_db):
    """lnprob() succesfully calculates the probability from single phase data"""
    dbf = Database.from_string(CU_MG_TDB_FCC_ONLY, fmt='tdb')
    datasets_db.insert(CU_MG_HM_MIX_SINGLE_FCC_A1)

    comps = ['CU', 'MG', 'VA']
    phases = ['FCC_A1']
    param = 'VV0003'
    orig_val = -14.0865
    eq_callables = build_callables(dbf, comps, phases, model=Model, parameters={param: orig_val})
    pm = eq_callables.pop('model')
    eq_callables.pop('phase_records')

    mods_no_idmix = {}
    for phase_name in phases:
        mods_no_idmix[phase_name] = Model(dbf, comps, phase_name, parameters=[sympy.Symbol(param)])
        mods_no_idmix[phase_name].models['idmix'] = 0

    prop = 'HM_MIX'  # from the dataset
    thermochemical_callables = {}
    from sympy import Symbol
    thermochemical_callables[prop] = build_callables(dbf, comps, phases, model=mods_no_idmix, output=prop, parameters={param: orig_val}, build_gradients=False)
    # pop off the callables not used in properties because we don't want them around (they should be None, anyways)
    thermochemical_callables[prop].pop('phase_records')
    # thermochemical_callables[prop].pop('model')

    res_orig = lnprob([orig_val], prior_rvs=[rv_zero()], comps=comps, dbf=dbf, phases=phases, phase_models=pm,
                 datasets=datasets_db, symbols_to_fit=[Symbol(param)], callables=eq_callables,
                 thermochemical_callables=thermochemical_callables)

    assert np.isreal(res_orig)
    assert np.isclose(res_orig, -9.119484935312146, rtol=1e-6)


    res_10 = lnprob([10], prior_rvs=[rv_zero()], comps=comps, dbf=dbf, phases=phases, phase_models=pm,
                 datasets=datasets_db, symbols_to_fit=[Symbol(param)], callables=eq_callables,
                 thermochemical_callables=thermochemical_callables)

    assert np.isreal(res_10)
    assert np.isclose(res_10, -9.143559131626864, rtol=1e-6)

    res_1e5 = lnprob([1e5], prior_rvs=[rv_zero()], comps=comps, dbf=dbf, phases=phases, phase_models=pm,
                 datasets=datasets_db, symbols_to_fit=[param], callables=eq_callables,
                 thermochemical_callables=thermochemical_callables)
    assert np.isreal(res_1e5)
    assert np.isclose(res_1e5, -1359.1335466316268, rtol=1e-6)


def _eq_LinAlgError(*args, **kwargs):
    raise LinAlgError()


def _eq_ValueError(*args, **kwargs):
    raise ValueError()


@mock.patch('espei.error_functions.zpf_error.equilibrium', _eq_LinAlgError)
def test_lnprob_does_not_raise_on_LinAlgError(datasets_db):
    """lnprob() should catch LinAlgError raised by equilibrium and return -np.inf"""
    dbf = Database.from_string(CU_MG_TDB, fmt='tdb')
    datasets_db.insert(CU_MG_DATASET_ZPF_WORKING)
    res = lnprob([10], prior_rvs=[rv_zero()], comps=['CU', 'MG', 'VA'], dbf=dbf,
                 phases=['LIQUID', 'FCC_A1', 'HCP_A3', 'LAVES_C15', 'CUMG2'],
                 datasets=datasets_db, symbols_to_fit=['VV0001'], phase_models=None)
    assert np.isneginf(res)


@mock.patch('espei.error_functions.zpf_error.equilibrium', _eq_ValueError)
def test_lnprob_does_not_raise_on_ValueError(datasets_db):
    """lnprob() should catch ValueError raised by equilibrium and return -np.inf"""
    dbf = Database.from_string(CU_MG_TDB, fmt='tdb')
    datasets_db.insert(CU_MG_DATASET_ZPF_WORKING)
    res = lnprob([10], prior_rvs=[rv_zero()], comps=['CU', 'MG', 'VA'], dbf=dbf,
                 phases=['LIQUID', 'FCC_A1', 'HCP_A3', 'LAVES_C15', 'CUMG2'],
                 datasets=datasets_db, symbols_to_fit=['VV0001'], phase_models=None)
    assert np.isneginf(res)


def test_parameter_initialization():
    """Determinisitically generated parameters should match."""
    initial_parameters = np.array([1, 10, 100, 1000])
    deterministic_params = generate_parameter_distribution(initial_parameters, 4, 0.10, deterministic=True)
    expected_parameters = np.array([
        [9.81708401e-01, 9.39027722e+00, 1.08016748e+02, 9.13512881e+02],
        [1.03116874, 9.01412995, 112.79594345, 916.44725799],
        [1.00664662e+00, 1.07178898e+01, 9.63696718e+01, 1.36872292e+03],
        [1.07642366e+00, 1.16413520e+01, 8.71742457e+01, 9.61836382e+02]])
    assert np.all(np.isclose(deterministic_params, expected_parameters))
