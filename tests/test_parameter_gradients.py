# pylint: disable=redefined-outer-name
"""
Test opt-in likelihood gradients with respect to the fitted parameters.
"""
import copy
import dataclasses
from unittest import mock

import numpy as np
import pytest
from pycalphad import Database, Model, Workspace, variables as v
from pycalphad.property_framework import IsolatedPhase, JanssonDerivative

from espei import _parameter_sensitivities as sensitivities
from espei._parameter_sensitivities import BACKEND
from espei.datasets import check_dataset
from espei.error_functions.activity_error import ActivityResidual
from espei.error_functions.context import setup_context
from espei.error_functions.equilibrium_thermochemical_error import EquilibriumPropertyResidual
from espei.error_functions.non_equilibrium_thermochemical_error import FixedConfigurationPropertyResidual
from espei.error_functions.residual_base import ResidualFunction
from espei.error_functions.zpf_error import ZPFResidual, calculate_zpf_driving_forces, calculate_zpf_error, estimate_hyperplane
from espei.optimizers.opt_mcmc import EmceeOptimizer
from espei.optimizers.opt_scipy import SciPyOptimizer
from espei.utils import MemoryStorage, PickleableTinyDB, database_symbols_to_fit, unpack_piecewise

from .fixtures import datasets_db
from .gradient_data import ABCD_GRADIENT_TDB, FD_SYMBOLS, load_gradient_datasets
from .gradient_data import analytic
from .testing_data import *

requires_backend = pytest.mark.skipif(BACKEND is None, reason="no parameter-sensitivity backend")
BACKEND_FUNCTIONS = ['get_tangent', 'get_gradient', 'isolated_phase_gm_gradient', 'fixed_configuration_gradient']
RESIDUALS = {'ZPF': ZPFResidual, 'Activity': ActivityResidual, 'EquilibriumProperty': EquilibriumPropertyResidual,
             'FixedConfigurationProperty': FixedConfigurationPropertyResidual}
SYSTEMS = ['cu_mg', 'ternary', 'quaternary']

# Batalin's liquid mixing enthalpies as an equilibrium property
CU_MG_EQ_HMR_LIQUID_BATALIN = {
    "components": ["CU", "MG"], "phases": ["LIQUID"],
    "conditions": {"P": 101325, "T": 1100, "X_MG": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]},
    "reference_states": {"CU": {"phase": "LIQUID"}, "MG": {"phase": "LIQUID"}},
    "output": "HMR",
    "values": [[[-3700, -5610, -6800, -7600, -7950, -7700, -6700, -5100, -2600]]],
    "reference": "Batalin (1987)",
}
CU_MG_GRADIENT_DATASETS = [
    CU_MG_DATASET_ZPF_ZERO_ERROR, CU_MG_DATASET_ZPF_WORKING, CU_MG_DATASET_ZPF_HYPERPLANE_TWOPHASE,
    CU_MG_DATASET_ZPF_STOICH_COMPOUND, CU_MG_EXP_ACTIVITY, CU_MG_EQ_HMR_LIQUID_BATALIN, CU_MG_HM_MIX_CUMG2_ANTISITE,
    CU_MG_HM_FORM_LAVES_C15_ANTISITE, CU_MG_SM_MIX_T_X_FCC_A1, CU_MG_CPM_MIX_X_HCP_A3,
]


def _load_system(system, datasets_db, names=None):
    """Database of a gradient test system, with its datasets (or the named ones) inserted"""
    if system == 'cu_mg':
        dbf, datasets = Database(CU_MG_TDB), CU_MG_GRADIENT_DATASETS
    else:
        generated = load_gradient_datasets(system)
        dbf, datasets = Database(ABCD_GRADIENT_TDB), [generated[name] for name in (names or generated)]
    for dataset in datasets:
        datasets_db.insert(copy.deepcopy(dataset))
    return dbf


def _true_parameters(dbf, symbols):
    return np.array([unpack_piecewise(dbf.symbols[s]) for s in symbols], dtype=np.float64)


def _perturbed(dbf, symbols, seed=0):
    """theta* (1 + 0.05 s) + s for a fixed random sign vector s, so residuals are nonzero"""
    signs = np.random.default_rng(seed).choice([-1.0, 1.0], size=len(symbols))
    return _true_parameters(dbf, symbols) * (1 + 0.05 * signs) + signs


def central_difference_check(func, x, names, gradient, steps=(1e-4, 1e-3, 1e-2), rtol=1e-5, atol=1e-8):
    """Compare a likelihood gradient with central differences, taking the best step per parameter."""
    def fd(j, step):
        h = step * max(abs(x[j]), 1.0)
        up, down = x.copy(), x.copy()
        up[j] += h
        down[j] -= h
        return (func(up) - func(down)) / (2 * h)

    best = np.array([fd(j, steps[0]) for j in range(x.size)])
    for step in steps[1:]:
        bad = np.abs(gradient - best) > rtol * np.max(np.abs(best)) + atol
        if not np.any(bad):
            break
        for j in np.nonzero(bad)[0]:
            candidate = fd(j, step)
            if abs(gradient[j] - candidate) < abs(gradient[j] - best[j]):
                best[j] = candidate
    tolerance = rtol * np.max(np.abs(best)) + atol
    bad = np.abs(gradient - best) > tolerance
    assert not np.any(bad), f"{np.asarray(names)[bad]}: gradient {gradient[bad]}, central differences {best[bad]} (tolerance {tolerance})"


def _relative_error(actual, expected):
    actual, expected = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    return np.max(np.abs(actual - expected)) / np.max(np.abs(expected))


@pytest.fixture
def cu_mg_context(datasets_db):
    """Cu-Mg context with data for every residual type, and its Database's parameter values"""
    datasets_db.insert(CU_MG_DATASET_ZPF_WORKING)
    datasets_db.insert(CU_MG_EXP_ACTIVITY)
    datasets_db.insert(CU_MG_EQ_HMR_LIQUID)
    datasets_db.insert(CU_MG_HM_MIX_CUMG2_ANTISITE)
    dbf = Database(CU_MG_TDB)
    ctx = setup_context(dbf, datasets_db)
    return ctx, _true_parameters(dbf, ctx['symbols_to_fit'])


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


def test_generated_datasets_valid():
    """Every generated dataset passes check_dataset"""
    for system in ('ternary', 'quaternary'):
        for dataset in load_gradient_datasets(system).values():
            check_dataset(dataset)


@pytest.mark.parametrize('system', ['ternary', 'quaternary'])
def test_values_at_true_parameters(system):
    """The generated data's residuals vanish at the true parameters, so the value path works in multicomponent systems"""
    dbf = Database(ABCD_GRADIENT_TDB)
    symbols = database_symbols_to_fit(dbf)
    true = _true_parameters(dbf, symbols)
    for name, dataset in load_gradient_datasets(system).items():
        datasets = PickleableTinyDB(storage=MemoryStorage)
        datasets.insert(dataset)
        if dataset['output'] == 'ZPF':
            residual_obj = ZPFResidual(dbf, datasets, None, symbols)
            driving_forces = np.concatenate(calculate_zpf_driving_forces(residual_obj.zpf_data, true)[0])
            vertices = [vertex for region in dataset['values'] for vertex in region if vertex[0] != '__HYPERPLANE__']
            for driving_force, vertex in zip(driving_forces, vertices, strict=True):
                if all(x is None for x in vertex[2]) and vertex[0] != 'AB2':
                    # the sampled maximum is on or below the hyperplane
                    assert driving_force <= 1e-3, (name, vertex, driving_force)
                else:
                    assert abs(driving_force) <= 1e-3, (name, vertex, driving_force)
            continue
        residual_class = ActivityResidual if dataset['output'].startswith('ACR') else \
            FixedConfigurationPropertyResidual if 'solver' in dataset else EquilibriumPropertyResidual
        residuals, _ = residual_class(dbf, datasets, None, symbols).get_residuals(true)
        values = np.ravel(dataset['values'])
        assert np.all(np.abs(residuals) <= 1e-6 * np.abs(values) + 1e-4), (name, residuals)


@requires_backend
@pytest.mark.parametrize('residual', RESIDUALS)
@pytest.mark.parametrize('system', SYSTEMS)
def test_residual_gradient_matches_central_differences(datasets_db, system, residual):
    """Each residual's likelihood gradient matches central differences, and its likelihood is get_likelihood's"""
    dbf = _load_system(system, datasets_db)
    symbols = FD_SYMBOLS[system]
    residual_obj = RESIDUALS[residual](dbf, datasets_db, None, symbols)
    x = _perturbed(dbf, symbols)
    likelihood, gradient = residual_obj.get_likelihood_and_gradient(x)
    assert likelihood == residual_obj.get_likelihood(x)
    assert gradient.shape == (len(symbols),)
    assert np.all(np.isfinite(gradient)) and np.any(gradient != 0)
    central_difference_check(residual_obj.get_likelihood, x, symbols, gradient)


ZPF_VERTEX_DATASETS = ['zpf_two_phase', 'zpf_null', 'zpf_disordered', 'zpf_hyperplane', 'zpf_stoichiometric',
                       'zpf_three_phase', 'zpf_edge_stoichiometric']


@requires_backend
@pytest.mark.parametrize('name', ZPF_VERTEX_DATASETS)
@pytest.mark.parametrize('system', ['ternary', 'quaternary'])
def test_zpf_vertex_gradient_matches_central_differences(datasets_db, system, name):
    """The ZPF likelihood gradient matches central differences for each vertex type alone"""
    dbf = _load_system(system, datasets_db, [name])
    symbols = FD_SYMBOLS[system]
    residual_obj = ZPFResidual(dbf, datasets_db, None, symbols)
    x = _perturbed(dbf, symbols)
    likelihood, gradient = residual_obj.get_likelihood_and_gradient(x)
    assert likelihood == residual_obj.get_likelihood(x)
    assert np.all(np.isfinite(gradient)) and np.any(gradient != 0)
    central_difference_check(residual_obj.get_likelihood, x, symbols, gradient)


@requires_backend
def test_masked_hyperplane_vertex_is_masked_from_the_gradient(datasets_db):
    """A vertex whose equilibrium is one phase without internal degrees of freedom is left out of the hyperplane's gradient too"""
    dbf = _load_system('ternary', datasets_db, ['zpf_edge_stoichiometric'])
    symbols = ['VV0015', 'VV0030']
    x = _perturbed(dbf, symbols)
    region = ZPFResidual(dbf, datasets_db, None, symbols).zpf_data[0]['phase_regions'][0]
    hyperplane, gradient = estimate_hyperplane(region, dbf, x, compute_gradient=True)
    stoichiometric = [vtx for vtx in region.hyperplane_vertices if vtx.phase_name == 'AB2']
    others = [vtx for vtx in region.hyperplane_vertices if vtx.phase_name != 'AB2']
    masked = estimate_hyperplane(dataclasses.replace(region, hyperplane_vertices=stoichiometric), dbf, x, compute_gradient=True)
    assert np.all(np.isnan(masked[0])) and np.all(np.isnan(masked[1]))
    unmasked = estimate_hyperplane(dataclasses.replace(region, hyperplane_vertices=others), dbf, x, compute_gradient=True)
    assert np.all(np.isfinite(gradient)) and np.any(gradient != 0)
    np.testing.assert_array_equal(hyperplane, unmasked[0])
    np.testing.assert_array_equal(gradient, unmasked[1])


@requires_backend
def test_tangent_and_gradient_match_the_analytic_binary():
    """Tangent and property gradients at the analytic binary's two-phase equilibrium match its exact derivatives"""
    dbf = Database(analytic.AB_ANALYTIC_TDB)
    names = analytic.AB_ANALYTIC_SYMBOLS
    conds = {getattr(v, k) if k != 'X_B' else v.X('B'): val for k, val in analytic.AB_ANALYTIC_CONDITIONS.items()}
    wks = Workspace(dbf, ['A', 'B', 'VA'], ['ALPHA', 'BETA'], conditions=conds,
                    parameters=dict(zip(names, _true_parameters(dbf, names))))
    np.testing.assert_allclose(np.ravel(wks.eq.MU), analytic.MU, rtol=1e-9)
    phases = [compset.phase_record.phase_name for compset in wks.get_composition_sets()]
    tangent = sensitivities.get_tangent(wks, names)[0]
    assert _relative_error(tangent['chemical_potentials'], analytic.DMU_DTHETA) < 1e-9
    for phase in ('ALPHA', 'BETA'):
        idx = phases.index(phase)
        assert _relative_error(tangent['site_fractions'][idx], analytic.DY_DTHETA[phase]) < 1e-9
        assert _relative_error(tangent['phase_amounts'][:, idx], analytic.DNP_DTHETA[phase]) < 1e-9
    assert np.all(tangent['state_variables'] == 0)
    assert _relative_error(np.ravel(sensitivities.get_gradient(wks, 'HM', names)), analytic.DHM_DTHETA) < 1e-9
    assert _relative_error(np.ravel(sensitivities.get_gradient(wks, 'GM', names)), analytic.DGM_DTHETA) < 1e-9
    assert _relative_error(np.ravel(sensitivities.get_gradient(wks, v.MU('B'), names)), np.asarray(analytic.DMU_DTHETA)[:, 1]) < 1e-9


@requires_backend
def test_regular_solution_closed_forms(datasets_db):
    """Each residual's gradient on a regular solution matches its closed form: d(MU_A - MU_A^ref)/dL = x_B^2, dHM/dL = dHM_MIX/dL = x_A x_B"""
    T, L, x_B = 600.0, 2000.0, np.array([0.1, 0.3, 0.6])
    dbf = Database(A_B_REGULAR_SOLUTION_TDB)
    # the closed forms hold at the equilibria's converged compositions
    wks = Workspace(dbf, ['A', 'B'], ['ALPHA'], {v.N: 1, v.P: 101325, v.T: T, v.X('B'): x_B}, parameters={'L_ALPHA': L})
    x_eq = np.ravel(wks.get(v.X('B')))
    activity = {
        "components": ["A", "B"], "phases": ["ALPHA"], "output": "ACR_A",
        "reference_state": {"phases": ["ALPHA"], "conditions": {"P": 101325, "T": T, "X_B": 0.0}},
        "conditions": {"P": 101325, "T": T, "X_B": x_B.tolist()}, "values": [[[0.8, 0.6, 0.3]]], "reference": "closed form",
    }
    equilibrium = {
        "components": ["A", "B"], "phases": ["ALPHA"], "output": "HM",
        "conditions": {"P": 101325, "T": [T], "X_B": x_B.tolist()}, "values": [[[0.0, 0.0, 0.0]]], "reference": "closed form",
    }
    fixed = {
        "components": ["A", "B"], "phases": ["ALPHA"], "output": "HM_MIX",
        "solver": {"mode": "manual", "sublattice_site_ratios": [1], "sublattice_configurations": [[["A", "B"]]] * 3,
                   "sublattice_occupancies": [[[1 - x, x]] for x in x_B]},
        "conditions": {"P": 101325, "T": T}, "values": [[[0.0, 0.0, 0.0]]], "reference": "closed form",
    }
    for residual_class, dataset, closed_form in ((ActivityResidual, activity, x_eq**2),
                                                 (EquilibriumPropertyResidual, equilibrium, (1 - x_eq) * x_eq),
                                                 (FixedConfigurationPropertyResidual, fixed, (1 - x_B) * x_B)):
        datasets = PickleableTinyDB(storage=MemoryStorage)
        datasets.insert(dataset)
        residual_obj = residual_class(dbf, datasets, None, ['L_ALPHA'])
        residuals, weights = map(np.asarray, residual_obj.get_residuals(np.array([L])))
        _, gradient = residual_obj.get_likelihood_and_gradient(np.array([L]))
        expected = -np.sum(residuals / weights**2 * closed_form)
        assert gradient[0] == pytest.approx(expected, rel=1e-10), residual_class.__name__


@pytest.fixture
def parameters_as_conditions(monkeypatch):
    """PR #268's route: each fitted parameter an IndependentPotential condition, on a Database without their symbols"""
    variables_before = set(dir(v))

    def make(tdb, names):
        for name in names:
            monkeypatch.setattr(v, name, v.IndependentPotential(name), raising=False)
        dbf = Database(tdb)
        dbf.symbols = {name: value for name, value in dbf.symbols.items() if name not in names}
        return dbf, [getattr(v, name) for name in names]

    yield make
    monkeypatch.undo()
    assert set(dir(v)) == variables_before


def _model_symbols(dbf, comps, phases):
    """Fitted symbols that appear in the models of a subsystem"""
    symbols = database_symbols_to_fit(dbf)
    found = set()
    for phase in phases:
        found |= {str(s) for s in Model(dbf, comps, phase, parameters=symbols).GM.free_symbols}
    return [s for s in symbols if s in found]


ORACLE_POINTS = [
    ('cu_mg', ['CU', 'MG', 'VA'], {v.T: 600, v.X('MG'): 0.1}, 'FCC_A1'),
    ('ternary', ['A', 'B', 'C', 'VA'], {v.T: 900, v.X('B'): 0.05, v.X('C'): 0.15}, 'TAU'),
    ('ternary', ['A', 'B', 'C', 'VA'], {v.T: 1150, v.X('B'): 0.25, v.X('C'): 0.15}, 'LIQUID'),
    ('quaternary', ['A', 'B', 'C', 'D', 'VA'], {v.T: 900, v.X('B'): 0.35, v.X('C'): 0.15, v.X('D'): 0.1}, 'ORD'),
]


@requires_backend
@pytest.mark.parametrize('system, comps, conds, isolated_phase', ORACLE_POINTS)
def test_gradients_match_parameters_as_conditions(parameters_as_conditions, system, comps, conds, isolated_phase):
    """dMU/dtheta, dHM/dtheta and the isolated phase's dGM/dtheta match JanssonDerivative with parameters as conditions"""
    tdb = CU_MG_TDB if system == 'cu_mg' else ABCD_GRADIENT_TDB
    dbf = Database(tdb)
    phases = list(dbf.phases)
    names = _model_symbols(dbf, comps, phases)
    values = _true_parameters(dbf, names)
    conds = {v.N: 1, v.P: 101325, **conds}
    elements = [c for c in comps if c != 'VA']
    wks = Workspace(dbf, comps, phases, conditions=conds, parameters=dict(zip(names, values)))
    dmu = sensitivities.get_tangent(wks, names)[0]['chemical_potentials']
    dhm = np.ravel(sensitivities.get_gradient(wks, 'HM', names))
    isolated = IsolatedPhase(isolated_phase, wks=wks)
    wks.get(isolated('GM'))
    dgm_isolated = sensitivities.isolated_phase_gm_gradient(isolated._compset, names)

    dbf_conditions, keys = parameters_as_conditions(tdb, names)
    wks_conditions = Workspace(dbf_conditions, comps, phases, conditions={**conds, **dict(zip(keys, values))})
    oracle_mu = wks_conditions.get(*[JanssonDerivative(v.MU(el), key) for key in keys for el in elements])
    oracle_hm = wks_conditions.get(*[JanssonDerivative('HM', key) for key in keys])
    isolated_conditions = IsolatedPhase(isolated_phase, wks=wks_conditions)
    oracle_isolated = wks_conditions.get(*[isolated_conditions(JanssonDerivative('GM', key)) for key in keys])
    assert _relative_error(dmu, np.reshape(oracle_mu, (len(keys), len(elements)))) <= 1e-8
    assert _relative_error(dhm, np.ravel(oracle_hm)) <= 1e-8
    assert _relative_error(dgm_isolated, np.ravel(oracle_isolated)) <= 1e-8


@requires_backend
def test_shuffled_symbols_permute_the_gradient(datasets_db):
    """The gradient follows each residual's parameter order: activity and fixed-configuration residuals read the
    parameters in the given order, ZPF and equilibrium-property residuals in sorted order"""
    dbf = _load_system('cu_mg', datasets_db)
    symbols = FD_SYMBOLS['cu_mg']
    order = np.random.default_rng(1).permutation(len(symbols))
    shuffled = [symbols[i] for i in order]
    x = _perturbed(dbf, symbols)
    for residual_class in RESIDUALS.values():
        likelihood, gradient = residual_class(dbf, datasets_db, None, symbols).get_likelihood_and_gradient(x)
        shuffled_obj = residual_class(dbf, datasets_db, None, shuffled)
        if residual_class in (ActivityResidual, FixedConfigurationPropertyResidual):
            shuffled_likelihood, shuffled_gradient = shuffled_obj.get_likelihood_and_gradient(x[order])
            np.testing.assert_allclose(shuffled_gradient, gradient[order], rtol=1e-12)
        else:
            shuffled_likelihood, shuffled_gradient = shuffled_obj.get_likelihood_and_gradient(x)
            np.testing.assert_allclose(shuffled_gradient, gradient, rtol=1e-12)
        assert shuffled_likelihood == pytest.approx(likelihood, rel=1e-12)


@requires_backend
def test_parameter_without_data_has_zero_gradient(datasets_db):
    """A parameter of the quaternary only (TAU's B:D endmember) has exactly zero gradient for ternary data"""
    dbf = _load_system('ternary', datasets_db)
    symbols = sorted(FD_SYMBOLS['ternary'] + ['VV0024'])
    x = _perturbed(dbf, symbols)
    for residual_class in RESIDUALS.values():
        _, gradient = residual_class(dbf, datasets_db, None, symbols).get_likelihood_and_gradient(x)
        assert gradient[symbols.index('VV0024')] == 0.0, residual_class.__name__
        assert np.any(gradient != 0), residual_class.__name__


def test_empty_data_gives_zero_gradient(datasets_db):
    """Without data, every residual's gradient is a zero vector of the right length"""
    dbf = Database(CU_MG_TDB)
    symbols = FD_SYMBOLS['cu_mg']
    for residual_class in RESIDUALS.values():
        likelihood, gradient = residual_class(dbf, datasets_db, None, symbols).get_likelihood_and_gradient(_true_parameters(dbf, symbols))
        assert likelihood == 0.0
        np.testing.assert_array_equal(gradient, np.zeros(len(symbols)))


def test_nan_hyperplane_gives_zero_gradient_rows(datasets_db):
    """A failed hyperplane gives zero driving forces and zero gradient rows"""
    dbf = Database(CU_MG_TDB)
    datasets_db.insert(CU_MG_DATASET_ZPF_NAN_EQUILIBRIUM)
    symbols = FD_SYMBOLS['cu_mg']
    residual_obj = ZPFResidual(dbf, datasets_db, None, symbols)

    def nan_hyperplane(*args, compute_gradient=False, **kwargs):
        return (np.full(2, np.nan), np.full((2, len(symbols)), np.nan)) if compute_gradient else np.full(2, np.nan)

    with mock.patch('espei.error_functions.zpf_error.estimate_hyperplane', side_effect=nan_hyperplane):
        driving_forces, _, gradients = calculate_zpf_driving_forces(residual_obj.zpf_data, _true_parameters(dbf, symbols), compute_gradient=True)
    assert driving_forces == [[0, 0]]
    np.testing.assert_array_equal(np.asarray(gradients[0]), np.zeros((2, len(symbols))))


def test_failed_driving_force_short_circuits_with_nan_gradient(datasets_db):
    """A failed driving force short circuits to 3-tuples, and its -inf likelihood has a NaN gradient"""
    dbf = Database(CU_MG_TDB)
    datasets_db.insert(CU_MG_DATASET_ZPF_WORKING)
    symbols = FD_SYMBOLS['cu_mg']
    params = _true_parameters(dbf, symbols)
    residual_obj = ZPFResidual(dbf, datasets_db, None, symbols)

    def hyperplane(*args, compute_gradient=False, **kwargs):
        return (np.zeros(2), np.zeros((2, len(symbols)))) if compute_gradient else np.zeros(2)

    def failed_driving_force(*args, compute_gradient=False, **kwargs):
        return (np.inf, np.full(len(symbols), np.nan)) if compute_gradient else np.inf

    with mock.patch('espei.error_functions.zpf_error.estimate_hyperplane', side_effect=hyperplane), \
         mock.patch('espei.error_functions.zpf_error.driving_force_to_hyperplane', side_effect=failed_driving_force):
        result = calculate_zpf_driving_forces(residual_obj.zpf_data, params, short_circuit=True, compute_gradient=True)
        likelihood, gradient = calculate_zpf_error(residual_obj.zpf_data, params, compute_gradient=True)
    assert len(result) == 3
    assert result[0] == [[np.inf]] and result[1] == [[np.inf]] and np.all(np.isnan(result[2][0][0]))
    assert likelihood == -np.inf and gradient.shape == (len(symbols),) and np.all(np.isnan(gradient))


@requires_backend
def test_scipy_optimizer_fits_with_the_gradient(datasets_db):
    """L-BFGS-B with the likelihood gradient raises the likelihood, and fit leaves the caller's Database unchanged"""
    dbf = _load_system('cu_mg', datasets_db)
    symbols = FD_SYMBOLS['cu_mg']
    start = _perturbed(dbf, symbols)
    for name, value in zip(symbols, start):
        dbf.symbols[name] = value
    symbols_before = dict(dbf.symbols)
    ctx = setup_context(dbf, datasets_db, symbols)
    start_likelihood = sum(residual_obj.get_likelihood(start) for residual_obj in ctx['residual_objs'])
    opt = SciPyOptimizer(dbf)
    fit_dbf = opt.fit(symbols, datasets_db, method='L-BFGS-B', jac=True, maxiter=20)
    fitted = np.array([fit_dbf.symbols[name] for name in ctx['symbols_to_fit']])
    np.testing.assert_array_equal(fitted, opt.minimize_result.x)
    assert sum(residual_obj.get_likelihood(fitted) for residual_obj in ctx['residual_objs']) > start_likelihood
    assert dbf.symbols == symbols_before
