import mock
import yaml
import numpy as np
from numpy.linalg import LinAlgError
from pycalphad import Database

from espei.mcmc import lnprob, generate_parameter_distribution
from espei.tests.fixtures import datasets_db
from espei.tests.testing_data import CU_MG_TDB, CU_MG_DATASET_ZPF_WORKING

dbf = Database.from_string(CU_MG_TDB, fmt='tdb')


single_phase_data = """
{
    "components": ["CU", "MG"],
    "phases": ["FCC_A1"],
    "solver": {
        "sublattice_site_ratios": [1],
        "sublattice_occupancies": [[[0.5, 0.5], 1]],
        "sublattice_configurations": [[["CU", "MG"], "VA"]],
        "mode": "manual"
    },
    "conditions": {
        "P": 101325,
        "T": 298.15
    },
    "output": "HM_MIX",
    "values": [[[-1000]]]
}
"""
single_phase_json = yaml.load(single_phase_data)


def test_lnprob_calculates_multi_phase_probability_for_success(datasets_db):
    """lnprob() successfully calculates the probability for equilibrium """
    datasets_db.insert(CU_MG_DATASET_ZPF_WORKING)
    from espei.utils import eq_callables_dict
    from pycalphad import Model
    import sympy
    comps = ['CU', 'MG', 'VA']
    phases = ['LIQUID', 'FCC_A1', 'HCP_A3', 'LAVES_C15', 'CUMG2']
    param = 'VV0001'
    eq_callables = eq_callables_dict(dbf, comps, phases, model=Model, param_symbols=sorted([sympy.Symbol(sym) for sym in [param]], key=str))
    eq_callables['phase_models'] = eq_callables.pop('model')
    orig_val = dbf.symbols.pop(param)

    res = lnprob([10], comps=comps, dbf=dbf,
                 phases=phases,
                 datasets=datasets_db, symbols_to_fit=[param], scheduler=None, **eq_callables)

    # replace the value in the database
    # TODO: make a fixture for this
    dbf.symbols[param] = orig_val

    assert np.isreal(res)
    assert np.isclose(res, -5741.61962949)


def test_lnprob_calculates_single_phase_probability_for_success(datasets_db):
    """lnprob() succesfully calculates the probability from single phase data"""
    datasets_db.insert(single_phase_json)
    res = lnprob([10], comps=['CU', 'MG', 'VA'], dbf=dbf,
                 phases=['LIQUID', 'FCC_A1', 'HCP_A3', 'LAVES_C15', 'CUMG2'],
                 datasets=datasets_db, symbols_to_fit=['VV0001'],
                 phase_models=None,
                 scheduler=None, )
    assert np.isreal(res)
    assert np.isclose(res, -19859.38)


def _eq_LinAlgError(*args, **kwargs):
    raise LinAlgError()


def _eq_ValueError(*args, **kwargs):
    raise ValueError()


@mock.patch('espei.error_functions.zpf_error.equilibrium', _eq_LinAlgError)
def test_lnprob_does_not_raise_on_LinAlgError(datasets_db):
    """lnprob() should catch LinAlgError raised by equilibrium and return -np.inf"""
    datasets_db.insert(CU_MG_DATASET_ZPF_WORKING)
    res = lnprob([10], comps=['CU', 'MG', 'VA'], dbf=dbf,
                 phases=['LIQUID', 'FCC_A1', 'HCP_A3', 'LAVES_C15', 'CUMG2'],
                 datasets=datasets_db, symbols_to_fit=['VV0001'], phase_models=None, scheduler=None)
    assert np.isneginf(res)


@mock.patch('espei.error_functions.zpf_error.equilibrium', _eq_ValueError)
def test_lnprob_does_not_raise_on_ValueError(datasets_db):
    """lnprob() should catch ValueError raised by equilibrium and return -np.inf"""
    datasets_db.insert(CU_MG_DATASET_ZPF_WORKING)
    res = lnprob([10], comps=['CU', 'MG', 'VA'], dbf=dbf,
                 phases=['LIQUID', 'FCC_A1', 'HCP_A3', 'LAVES_C15', 'CUMG2'],
                 datasets=datasets_db, symbols_to_fit=['VV0001'], phase_models=None, scheduler=None)
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
