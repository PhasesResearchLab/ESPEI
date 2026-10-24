# pylint: disable=redefined-outer-name
"""
Test opt-in likelihood gradients with respect to the fitted parameters.
"""
from unittest import mock

import numpy as np
import pytest
from pycalphad import Database, Workspace, variables as v

from espei import _parameter_sensitivities as sensitivities
from espei.error_functions.activity_error import ActivityResidual
from espei.error_functions.equilibrium_thermochemical_error import EquilibriumPropertyResidual
from espei.error_functions.non_equilibrium_thermochemical_error import FixedConfigurationPropertyResidual
from espei.error_functions.residual_base import ResidualFunction
from espei.error_functions.zpf_error import ZPFResidual
from espei.utils import MemoryStorage, PickleableTinyDB, unpack_piecewise

from .fixtures import datasets_db
from .testing_data import *

RESIDUALS = [ZPFResidual, ActivityResidual, EquilibriumPropertyResidual, FixedConfigurationPropertyResidual]


def _parameters(dbf, symbols):
    return np.array([unpack_piecewise(dbf.symbols[s]) for s in symbols])


def test_residual_without_gradient_raises():
    """The protocol's get_likelihood_and_gradient names the residual that lacks it"""
    class LikelihoodOnlyResidual(ResidualFunction):
        def __init__(self):
            pass

        def get_likelihood(self, parameters):
            return 0.0

    with pytest.raises(NotImplementedError, match="LikelihoodOnlyResidual"):
        LikelihoodOnlyResidual().get_likelihood_and_gradient(np.zeros(1))


def test_tangent_and_gradient_match_the_analytic_binary():
    """Tangent and HM gradient at a two-phase equilibrium match the exact derivatives"""
    dbf = Database(A_B_ANALYTIC_TDB)
    symbols = ['VV0000', 'VV0001', 'VV0002', 'VV0003', 'VV0004']
    wks = Workspace(dbf, ['A', 'B', 'VA'], ['ALPHA', 'BETA'], {v.N: 1, v.P: 101325, v.T: 900, v.X('B'): 0.3},
                    parameters=dict(zip(symbols, _parameters(dbf, symbols))))
    tangent = sensitivities.get_tangent(wks, symbols)[0]
    # exact derivatives, solved symbolically
    dmu = [[0.2914650335580264, -0.5392055237449641], [262.3185302022238, -485.28497137046776],
           [0.18877352812872164, -0.34922792576958506], [-0.0026294876738948226, 0.012296843572457023],
           [-0.3401063929145322, 1.5905132977740268]]
    dhm = [0.16344869864498984, 109.06634905007459, 0.11029376378672937, -0.011806413300450692, 0.15365484369263027]
    np.testing.assert_allclose(tangent['chemical_potentials'], dmu, rtol=1e-9)
    np.testing.assert_allclose(np.ravel(sensitivities.get_gradient(wks, 'HM', symbols)), dhm, rtol=1e-9)
    np.testing.assert_array_equal(np.ravel(sensitivities.get_gradient(wks, v.MU('B'), symbols)), tangent['chemical_potentials'][:, 1])


def test_tangent_at_a_pure_element():
    """At pure A, rounding-level right-hand sides are not mistaken for infeasible ones"""
    dbf = Database(A_B_C_TWO_PHASE_TDB)
    symbols = ['VV0000', 'VV0001', 'VV0002']
    wks = Workspace(dbf, ['A', 'B', 'C'], ['ALPHA'], {v.N: 1, v.P: 101325, v.T: [1000, 1150], v.X('B'): 0, v.X('C'): 0},
                    parameters=dict(zip(symbols, _parameters(dbf, symbols))))
    for tangent in sensitivities.get_tangent(wks, symbols):
        # dMU_B/dL_AB = dMU_C/dL_AC = 1 at infinite dilution
        np.testing.assert_allclose(tangent['chemical_potentials'], [[0, 1, 0], [0, 0, 1], [0, 0, 0]], atol=1e-5)
        np.testing.assert_allclose(tangent['phase_amounts'], 0, atol=1e-12)


def test_residual_gradients_match_closed_forms():
    """On a regular solution, d(MU_A - MU_A^ref)/dL = x_B^2, dHMR/dL = dHM_FORM/dL = x_A x_B, and the endmembers' derivatives vanish"""
    T, L, x_B = 600.0, 2000.0, np.array([0.1, 0.3, 0.6])
    dbf = Database(A_B_REGULAR_SOLUTION_TDB)
    symbols = ['L_ALPHA', 'G_ALPHA_A', 'G_ALPHA_B']
    params = np.array([L, 0.0, 0.0])
    # the closed forms hold at the equilibria's converged compositions
    wks = Workspace(dbf, ['A', 'B'], ['ALPHA'], {v.N: 1, v.P: 101325, v.T: T, v.X('B'): x_B}, parameters={'L_ALPHA': L})
    x_eq = np.ravel(wks.get(v.X('B')))
    activity = {
        "components": ["A", "B"], "phases": ["ALPHA"], "output": "ACR_A",
        "reference_state": {"phases": ["ALPHA"], "conditions": {"P": 101325, "T": T, "X_B": 0.0}},
        "conditions": {"P": 101325, "T": T, "X_B": x_B.tolist()}, "values": [[[0.8, 0.6, 0.3]]], "reference": "closed form",
    }
    equilibrium = {
        "components": ["A", "B"], "phases": ["ALPHA"], "output": "HMR",
        "reference_states": {"A": {"phase": "ALPHA"}, "B": {"phase": "ALPHA"}},
        "conditions": {"P": 101325, "T": [T], "X_B": x_B.tolist()}, "values": [[[0.0, 0.0, 0.0]]], "reference": "closed form",
    }
    fixed = {
        "components": ["A", "B"], "phases": ["ALPHA"], "output": "HM_FORM",
        "solver": {"mode": "manual", "sublattice_site_ratios": [1], "sublattice_configurations": [[["A", "B"]]] * 3,
                   "sublattice_occupancies": [[[1 - x, x]] for x in x_B]},
        "conditions": {"P": 101325, "T": T}, "values": [[[0.0, 0.0, 0.0]]], "reference": "closed form",
    }
    for residual_class, dataset, closed_form in ((ActivityResidual, activity, x_eq**2),
                                                 (EquilibriumPropertyResidual, equilibrium, (1 - x_eq) * x_eq),
                                                 (FixedConfigurationPropertyResidual, fixed, (1 - x_B) * x_B)):
        datasets = PickleableTinyDB(storage=MemoryStorage)
        datasets.insert(dataset)
        residual_obj = residual_class(dbf, datasets, None, symbols)
        residuals, weights = map(np.asarray, residual_obj.get_residuals(params))
        _, gradient = residual_obj.get_likelihood_and_gradient(params)
        expected = [-np.sum(residuals / weights**2 * closed_form), 0.0, 0.0]
        np.testing.assert_allclose(gradient, expected, rtol=1e-10, atol=1e-15, err_msg=residual_class.__name__)


def test_zpf_gradient_of_each_vertex_type(datasets_db):
    """ZPF gradient with tie-line, null, disordered and stoichiometric vertices, and a NaN hyperplane"""
    disordered = {**CU_MG_DATASET_ZPF_ZERO_ERROR,
                  "values": [[["FCC_A1", ["CU"], [0.9998], "disordered"], ["LAVES_C15", ["CU"], [0.6671], "disordered"]]]}
    stoichiometric = {
        "components": ["CU", "MG"], "phases": ["CUMG2", "HCP_A3"], "conditions": {"P": 101325, "T": 733.15}, "output": "ZPF",
        "values": [[["CUMG2", ["CU"], [0.3333333333]], ["HCP_A3", ["CU"], [0.005]]]], "reference": "testing",
    }
    for dataset in (CU_MG_DATASET_ZPF_WORKING, CU_MG_DATASET_ZPF_STOICH_COMPOUND, disordered, stoichiometric):
        datasets_db.insert(dataset)
    dbf = Database(CU_MG_TDB)
    # LIQUID, CUMG2, LAVES_C15, FCC_A1 and HCP_A3, not sorted
    symbols = ['VV0017', 'VV0000', 'VV0008', 'VV0003', 'VV0007']
    params = _parameters(dbf, symbols)
    _, gradient = ZPFResidual(dbf, datasets_db, None, symbols).get_likelihood_and_gradient(params)
    # each vertex's driving force gradient checked against central differences
    np.testing.assert_allclose(gradient, [0.00023788619, -0.00026016945, 0.0082605113, 3.8510858e-08, -5.2589904e-06], rtol=1e-6)


@pytest.mark.parametrize('residual_class, dataset', [
    (ZPFResidual, A_B_DATASET_ZPF_ALPHA_BETA),
    (ActivityResidual, A_B_DATASET_ACR_A_ALPHA),
    (EquilibriumPropertyResidual, A_B_DATASET_HM_ALPHA),
    (FixedConfigurationPropertyResidual, A_B_DATASET_HM_MIX_ALPHA),
], ids=['ZPF', 'Activity', 'EquilibriumProperty', 'FixedConfigurationProperty'])
def test_residuals_follow_the_order_of_symbols_to_fit(datasets_db, residual_class, dataset):
    """Likelihoods and gradients follow symbols_to_fit, and VV0001, outside the A-B subsystem, has zero gradient"""
    datasets_db.insert(dataset)
    dbf = Database(A_B_C_TWO_PHASE_TDB)
    expected = residual_class(dbf, datasets_db, None, ['VV0000']).get_likelihood(np.array([-12000.0]))
    residual_obj = residual_class(dbf, datasets_db, None, ['VV0001', 'VV0000'])
    params = np.array([-2000.0, -12000.0])
    likelihood, gradient = residual_obj.get_likelihood_and_gradient(params)
    assert np.isfinite(expected)
    assert residual_obj.get_likelihood(params) == pytest.approx(expected, rel=1e-12)
    assert likelihood == pytest.approx(expected, rel=1e-12)
    assert gradient[0] == 0.0 and gradient[1] != 0.0


def test_empty_data_gives_zero_gradient(datasets_db):
    """Without data, every residual's likelihood is zero and its gradient a zero vector"""
    dbf = Database(A_B_REGULAR_SOLUTION_TDB)
    for residual_class in RESIDUALS:
        likelihood, gradient = residual_class(dbf, datasets_db, None, ['L_ALPHA']).get_likelihood_and_gradient(np.array([0.0]))
        assert likelihood == 0.0
        np.testing.assert_array_equal(gradient, [0.0])


def test_failed_driving_force_gives_nan_gradient(datasets_db):
    """A failed driving force short circuits, and the -inf likelihood has a NaN gradient"""
    datasets_db.insert(A_B_DATASET_ZPF_ALPHA_BETA)
    residual_obj = ZPFResidual(Database(A_B_C_TWO_PHASE_TDB), datasets_db, None, ['VV0000'])
    with mock.patch('espei.error_functions.zpf_error.driving_force_to_hyperplane', return_value=(np.inf, np.zeros(1))):
        likelihood, gradient = residual_obj.get_likelihood_and_gradient(np.array([-10000.0]))
    assert likelihood == -np.inf
    assert gradient.shape == (1,) and np.all(np.isnan(gradient))
