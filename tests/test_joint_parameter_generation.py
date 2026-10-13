"""Tests for fitting GM, HM, SM and CPM data jointly with StepJointGibbsEnergy"""

import copy
import numpy as np
import pytest
import symengine
from pycalphad import Database, variables as v
from tinydb import where

from espei.paramselect import generate_parameters
from espei.refdata import SGTE91, SGTE91Stable
from espei.parameter_selection.fitting_descriptions import gibbs_energy_fitting_description, joint_gibbs_energy_fitting_description
from .testing_data import *
from .fixtures import datasets_db

# Coefficients of the Gibbs energy features below, in J/mol-formula
GIBBS_ENERGY_FEATURES = (symengine.S.One, v.T, v.T*symengine.log(v.T), v.T**2, v.T**(-1), v.T**3)

# Unary and compound Gibbs energies from the Cu-Mg assessment in pycalphad's
# test databases (cumg.tdb, M. Palumbo, T. Abe and K. Hashimoto, NIMS 2008).
# Only the low temperature pieces are used: 298.15 K < T < 923 K.
GHSERCU = (-7770.458, 130.485403, -24.112392, -0.00265684, 52478, 1.29223E-07)
GHSERMG = (-8367.34, 143.677875, -26.1849782, 4.858E-04, 78950, -1.393669E-06)
G_CU2MG_CU_MG = (-54690.99, 364.73085, -69.276417, -5.1925E-4, 143502, -5.65953E-6)
G_CUMG2_CU_MG_FORMATION = (-28620.00, 1.859733, 0, 0, 0, 0)  # relative to GHSERCU + 2*GHSERMG
# Unary and corundum Gibbs energies from zrlayalo.tdb in pycalphad's test databases.
# Only the low temperature pieces are used: 298.15 K < T < 600 K.
GHSERAL = (-7976.15, 137.093038, -24.3671976, -0.001884662, 74092, -8.77664E-07)
GHSEROO = (-3480.87, -25.503038, -11.136, -0.005098888, -38365, 6.61846E-07)
GCORUND = (-1707351.3, 448.021092, -67.4804, -.06747, 938780, 1.4205433E-05)

CU2MG_FORMATION = tuple(g - 2*cu - mg for g, cu, mg in zip(G_CU2MG_CU_MG, GHSERCU, GHSERMG))
CORUNDUM_FORMATION = tuple(g - 2*al - 3*o for g, al, o in zip(GCORUND, GHSERAL, GHSEROO))

CU2MG_TEMPERATURES = [298.15] + list(range(300, 901, 50))
CORUNDUM_TEMPERATURES = [298.15] + list(range(300, 601, 25))

# (phase models, configuration, Gibbs energy of formation coefficients, temperatures)
CU2MG = (
    {"components": ["CU", "MG"], "phases": {"CU2MG": {"sublattice_model": [["CU"], ["MG"]], "sublattice_site_ratios": [2, 1]}}},
    ["CU", "MG"], CU2MG_FORMATION, CU2MG_TEMPERATURES,
)
CUMG2 = (
    {"components": ["CU", "MG"], "phases": {"CUMG2": {"sublattice_model": [["CU"], ["MG"]], "sublattice_site_ratios": [1, 2]}}},
    ["CU", "MG"], G_CUMG2_CU_MG_FORMATION, CU2MG_TEMPERATURES,
)
CORUNDUM = (
    {"components": ["AL", "O"], "phases": {"CORUNDUM": {"sublattice_model": [["AL"], ["O"]], "sublattice_site_ratios": [2, 3]}}},
    ["AL", "O"], CORUNDUM_FORMATION, CORUNDUM_TEMPERATURES,
)


def _gibbs_energy(coefficients):
    return sum(coef*feature for coef, feature in zip(coefficients, GIBBS_ENERGY_FEATURES))


def _property(gibbs_energy, prop):
    if prop == "GM":
        return gibbs_energy
    elif prop == "HM":
        return gibbs_energy - v.T*symengine.diff(gibbs_energy, v.T)
    elif prop == "SM":
        return -symengine.diff(gibbs_energy, v.T)
    elif prop == "CPM":
        return -v.T*symengine.diff(gibbs_energy, v.T, 2)
    raise ValueError(prop)


def _evaluate(expr, temperatures):
    return np.array([float(symengine.S(expr).xreplace({v.T: T})) for T in temperatures])


def _formation_datasets(compound, outputs, temperatures=None, noise=None):
    """ESPEI datasets (per mole-atom) generated from the Gibbs energy of formation of a compound"""
    phase_models, configuration, coefficients, default_temperatures = compound
    phase_name = list(phase_models["phases"].keys())[0]
    site_ratios = phase_models["phases"][phase_name]["sublattice_site_ratios"]
    gibbs_energy = _gibbs_energy(coefficients) / sum(site_ratios)
    noise = noise if noise is not None else {}
    datasets = []
    for output in outputs:
        temps = temperatures.get(output, default_temperatures) if temperatures is not None else default_temperatures
        values = _evaluate(_property(gibbs_energy, output), temps)
        values = values + noise.get(output, np.zeros_like(values))
        datasets.append({
            "components": phase_models["components"],
            "phases": [phase_name],
            "solver": {"mode": "manual", "sublattice_site_ratios": site_ratios, "sublattice_configurations": [configuration]},
            "conditions": {"P": 101325, "T": list(temps)},
            "output": output + "_FORM",
            "values": [[[float(val)] for val in values]],
            "reference": "Generated from a CALPHAD database",
        })
    return datasets


def _absolute_datasets(compound, outputs, temperatures=None, noise=None):
    """Like _formation_datasets, but absolute (SER referenced) using ESPEI's SGTE91 unary data"""
    phase_models, configuration, _, _ = compound
    phase_name = list(phase_models["phases"].keys())[0]
    site_ratios = phase_models["phases"][phase_name]["sublattice_site_ratios"]
    ser_gibbs_energy = sum(ratio*SGTE91Stable[el] for ratio, el in zip(site_ratios, configuration)) / sum(site_ratios)
    datasets = _formation_datasets(compound, outputs, temperatures=temperatures, noise=noise)
    for ds in datasets:
        prop = ds["output"].split("_")[0]
        ser_values = _evaluate(_property(ser_gibbs_energy, prop), ds["conditions"]["T"])
        ds["values"] = [[[val[0] + ser] for val, ser in zip(ds["values"][0], ser_values)]]
        ds["output"] = prop
    return datasets


def _fitted_formation_gibbs_energy(dbf, phase_name):
    """Fitted parameter of a single endmember phase with the pure element reference (GHSER) symbols removed"""
    params = dbf._parameters.search((where('phase_name') == phase_name) & (where('parameter_type') == 'G'))
    assert len(params) == 1
    expr = symengine.S(params[0]['parameter']).xreplace({symengine.Symbol(name): val for name, val in dbf.symbols.items() if name.startswith('VV')})
    return expr.xreplace({sym: 0 for sym in expr.free_symbols if sym.name.startswith('GHSER')})


def _fitted_symbols(dbf):
    return {name: float(val) for name, val in dbf.symbols.items() if name.startswith('VV')}


def _fit(phase_models, datasets, fitting_description, datasets_db, **kwargs):
    datasets_db.truncate()
    datasets_db.insert_multiple(copy.deepcopy(datasets))
    return generate_parameters(phase_models, datasets_db, 'SGTE91', 'linear', fitting_description=fitting_description, **kwargs)


def _assert_equivalent_databases(dbf_a, dbf_b):
    """Assert that two databases have the same parameters and fitted symbol values"""
    def param_keys(dbf):
        return sorted((p['phase_name'], str(p['constituent_array']), p['parameter_type'], p['parameter_order'], str(p['parameter'])) for p in dbf._parameters.all())
    assert param_keys(dbf_a) == param_keys(dbf_b)
    symbols_a, symbols_b = _fitted_symbols(dbf_a), _fitted_symbols(dbf_b)
    assert symbols_a.keys() == symbols_b.keys()
    for name in symbols_a:
        # Fitted symbols are rounded to 6 significant figures
        assert np.isclose(symbols_a[name], symbols_b[name], rtol=2e-6, atol=1e-8), f"{name}: {symbols_a[name]} != {symbols_b[name]}"


EXISTING_TEST_CASES = {
    "cpm_mix_and_sm_mix": (CU_ZN_LIQUID_PHASE_MODEL, [CU_ZN_CPM_MIX_EXPR_TO_FLOAT, CU_ZN_SM_MIX_EXPR_TO_FLOAT], None),
    "sm_mix_higher_order": (CU_ZN_LIQUID_PHASE_MODEL, [CU_ZN_SM_MIX_L1], None),
    "ternary_symmetric_with_binary": (AL_CO_CR_A2_PHASE_MODELS, [AL_CO_A2_BINARY_SYMMETRIC_DATASET, AL_CO_CR_A2_TERNARY_SYMMETRIC_DATASET], None),
    "ternary_asymmetric": (AL_CO_CR_A2_PHASE_MODELS, [AL_CO_CR_BCC_A2_TERNARY_NON_SYMMETRIC_DATASET], None),
    "ternary_asymmetric_2_sublattice": (AL_CO_CR_B2_PHASE_MODELS, [AL_CO_CR_BCC_B2_TERNARY_NON_SYMMETRIC_DATASET], None),
    "excluded_contributions": (CR_FE_PHASE_MODELS, [CR_FE_HM_MIX_EXCLUDED_MAG, CR_FE_HM_MIX_WITH_MAG], CR_FE_INITIAL_TDB_CONTRIBUTIONS),
}


@pytest.mark.parametrize("case", EXISTING_TEST_CASES.keys())
def test_joint_fit_matches_sequential_fit_for_existing_test_data(datasets_db, case):
    """Joint fitting should reproduce the sequential fit when HM and SM data have no temperature dependence"""
    phase_models, datasets, initial_tdb = EXISTING_TEST_CASES[case]
    dbf_sequential = _fit(phase_models, datasets, gibbs_energy_fitting_description, datasets_db, dbf=Database(initial_tdb) if initial_tdb else None)
    dbf_joint = _fit(phase_models, datasets, joint_gibbs_energy_fitting_description, datasets_db, dbf=Database(initial_tdb) if initial_tdb else None)
    _assert_equivalent_databases(dbf_sequential, dbf_joint)


@pytest.mark.parametrize("compound", [CU2MG, CORUNDUM], ids=["CU2MG", "CORUNDUM"])
@pytest.mark.parametrize("cpm_noise", [0.0, 0.05])
def test_joint_fit_matches_sequential_fit_for_typical_formation_data(datasets_db, compound, cpm_noise):
    """CPM data at many temperatures with HM and SM data at 298.15 K should fit the same parameters jointly or sequentially"""
    phase_models, _, coefficients, temperatures = compound
    rng = np.random.default_rng(1234)
    datasets = _formation_datasets(compound, ["HM", "SM", "CPM"], temperatures={"HM": [298.15], "SM": [298.15]}, noise={"CPM": rng.normal(0.0, cpm_noise, len(temperatures))})
    dbf_sequential = _fit(phase_models, datasets, gibbs_energy_fitting_description, datasets_db)
    dbf_joint = _fit(phase_models, datasets, joint_gibbs_energy_fitting_description, datasets_db)
    _assert_equivalent_databases(dbf_sequential, dbf_joint)
    if cpm_noise == 0.0:
        # all heat capacity features are needed to reproduce the database
        assert len(_fitted_symbols(dbf_joint)) == 6
        phase_name = list(phase_models["phases"].keys())[0]
        expected = _evaluate(_gibbs_energy(coefficients), temperatures)
        assert np.allclose(_evaluate(_fitted_formation_gibbs_energy(dbf_joint, phase_name), temperatures), expected, atol=1.0)


@pytest.mark.parametrize("compound", [CU2MG, CORUNDUM], ids=["CU2MG", "CORUNDUM"])
def test_joint_fit_reproduces_database_from_all_properties(datasets_db, compound):
    """Consistent GM, HM, SM and CPM data should be fit exactly"""
    phase_models, _, coefficients, temperatures = compound
    phase_name = list(phase_models["phases"].keys())[0]
    datasets = _formation_datasets(compound, ["GM", "HM", "SM", "CPM"])
    dbf = _fit(phase_models, datasets, joint_gibbs_energy_fitting_description, datasets_db)
    assert len(_fitted_symbols(dbf)) == 6
    fitted = _fitted_formation_gibbs_energy(dbf, phase_name)
    expected = _gibbs_energy(coefficients)
    for prop, atol in (("GM", 1.0), ("HM", 1.0), ("SM", 1e-3), ("CPM", 1e-3)):
        assert np.allclose(_evaluate(_property(fitted, prop), temperatures), _evaluate(_property(expected, prop), temperatures), atol=atol), prop


@pytest.mark.parametrize("compound", [CU2MG, CORUNDUM], ids=["CU2MG", "CORUNDUM"])
def test_joint_fit_reproduces_database_from_gm_data_only(datasets_db, compound):
    """GM data at many temperatures resolve all Gibbs energy features"""
    phase_models, _, coefficients, temperatures = compound
    phase_name = list(phase_models["phases"].keys())[0]
    datasets = _formation_datasets(compound, ["GM"])
    # GM data are not used by the sequential fit
    dbf_sequential = _fit(phase_models, datasets, gibbs_energy_fitting_description, datasets_db)
    assert len(_fitted_symbols(dbf_sequential)) == 0
    dbf = _fit(phase_models, datasets, joint_gibbs_energy_fitting_description, datasets_db)
    fitted = _fitted_formation_gibbs_energy(dbf, phase_name)
    expected = _gibbs_energy(coefficients)
    for prop, atol in (("GM", 1.0), ("HM", 5.0), ("SM", 1e-2), ("CPM", 1e-2)):
        assert np.allclose(_evaluate(_property(fitted, prop), temperatures), _evaluate(_property(expected, prop), temperatures), atol=atol), prop


def test_joint_fit_uses_temperature_dependence_of_hm_data(datasets_db):
    """HM data at many temperatures resolve heat capacity features that the sequential fit discards"""
    phase_models, _, coefficients, temperatures = CU2MG
    datasets = _formation_datasets(CU2MG, ["HM"])
    expected_hm = _evaluate(_property(_gibbs_energy(coefficients), "HM"), temperatures)
    dbf_sequential = _fit(phase_models, datasets, gibbs_energy_fitting_description, datasets_db)
    dbf_joint = _fit(phase_models, datasets, joint_gibbs_energy_fitting_description, datasets_db)
    # sequential: only a constant is fit
    assert len(_fitted_symbols(dbf_sequential)) == 1
    sequential_hm = _evaluate(_property(_fitted_formation_gibbs_energy(dbf_sequential, "CU2MG"), "HM"), temperatures)
    assert np.max(np.abs(sequential_hm - expected_hm)) > 100.0
    # joint: the temperature dependence of HM is reproduced, but the entropy (T feature) cannot be resolved by HM data
    fitted = _fitted_formation_gibbs_energy(dbf_joint, "CU2MG")
    assert np.allclose(_evaluate(_property(fitted, "HM"), temperatures), expected_hm, atol=1.0)
    assert np.allclose(_evaluate(_property(fitted, "CPM"), temperatures), _evaluate(_property(_gibbs_energy(coefficients), "CPM"), temperatures), atol=1e-2)
    # without SM or GM data, the entropy is only known up to the (unfit) T coefficient
    sm_difference = _evaluate(_property(fitted, "SM"), temperatures) - _evaluate(_property(_gibbs_energy(coefficients), "SM"), temperatures)
    assert np.allclose(sm_difference, coefficients[1], atol=1e-2)


def test_aicc_penalty_factor_applies_to_heat_capacity_features_resolved_by_hm_data(datasets_db):
    """The CPM AICc penalty factor applies to heat capacity features even when they are resolved by HM data"""
    phase_models, _, _, _ = CU2MG
    datasets = _formation_datasets(CU2MG, ["HM"])
    dbf = _fit(phase_models, datasets, joint_gibbs_energy_fitting_description, datasets_db, aicc_penalty_factor={"CU2MG": {"CPM": 1e6}})
    assert len(_fitted_symbols(dbf)) == 1
    dbf_sequential = _fit(phase_models, datasets, gibbs_energy_fitting_description, datasets_db)
    _assert_equivalent_databases(dbf, dbf_sequential)


def test_joint_fit_matches_sequential_fit_for_disagreeing_hm_data_at_nearby_temperatures(datasets_db):
    """HM data from different sources at nearly the same temperature should not resolve heat capacity features"""
    # Mimics DFT formation energies (assigned to 298.15 K) and calorimetry (at 299 K) that disagree by a few kJ/mol-atom
    phase_models, _, _, _ = CUMG2
    dft, calorimetry = _formation_datasets(CUMG2, ["HM"], temperatures={"HM": [298.15]}) + _formation_datasets(CUMG2, ["HM"], temperatures={"HM": [299.0]})
    calorimetry["values"] = [[[calorimetry["values"][0][0][0] + 3000.0]]]
    dbf_sequential = _fit(phase_models, [dft, calorimetry], gibbs_energy_fitting_description, datasets_db)
    dbf_joint = _fit(phase_models, [dft, calorimetry], joint_gibbs_energy_fitting_description, datasets_db)
    assert len(_fitted_symbols(dbf_joint)) == 1
    _assert_equivalent_databases(dbf_sequential, dbf_joint)


def test_joint_fit_does_not_add_heat_capacity_features_without_heat_capacity(datasets_db):
    """GM and HM data at many temperatures for a compound with zero heat capacity of formation should only fit A + B*T"""
    phase_models, _, coefficients, temperatures = CUMG2
    datasets = _formation_datasets(CUMG2, ["GM", "HM"])
    dbf = _fit(phase_models, datasets, joint_gibbs_energy_fitting_description, datasets_db)
    assert len(_fitted_symbols(dbf)) == 2
    fitted = _fitted_formation_gibbs_energy(dbf, "CUMG2")
    assert np.allclose(_evaluate(fitted, temperatures), _evaluate(_gibbs_energy(coefficients), temperatures), atol=0.1)
    assert np.allclose(_evaluate(_property(fitted, "CPM"), temperatures), 0.0)


def test_joint_fit_gm_mix_data(datasets_db):
    """GM_MIX data (including the ideal mixing contribution) fit the excess Gibbs energy"""
    # FCC_A1 L0 from the Cu-Mg assessment in pycalphad's test databases (cumg.tdb)
    L0 = -22279.28 + 5.868*v.T
    phase_models = {"components": ["CU", "MG", "VA"], "phases": {"FCC_A1": {"sublattice_model": [["CU", "MG"], ["VA"]], "sublattice_site_ratios": [1, 1]}}}
    temperatures = [300, 500, 700, 900]
    x_mg = [0.1, 0.25, 0.5, 0.75, 0.9]
    values = []
    for T in temperatures:
        ideal = [float(v.R)*T*(x*np.log(x) + (1-x)*np.log(1-x)) for x in x_mg]
        excess = [x*(1-x)*float(L0.xreplace({v.T: T})) for x in x_mg]
        values.append([i + e for i, e in zip(ideal, excess)])
    dataset = {
        "components": ["CU", "MG", "VA"],
        "phases": ["FCC_A1"],
        "solver": {
            "mode": "manual",
            "sublattice_site_ratios": [1, 1],
            "sublattice_configurations": [[["CU", "MG"], "VA"] for _ in x_mg],
            "sublattice_occupancies": [[[1-x, x], 1] for x in x_mg],
        },
        "conditions": {"P": 101325, "T": temperatures},
        "output": "GM_MIX",
        "values": [values],
        "reference": "Generated from a CALPHAD database",
    }
    dbf = _fit(phase_models, [dataset], joint_gibbs_energy_fitting_description, datasets_db)
    params = dbf._parameters.search((where('phase_name') == 'FCC_A1') & (where('parameter_type') == 'L'))
    assert len(params) == 1
    assert params[0]['parameter_order'] == 0
    fitted = symengine.S(params[0]['parameter']).xreplace({symengine.Symbol(name): val for name, val in _fitted_symbols(dbf).items()})
    assert np.allclose(_evaluate(fitted, temperatures), _evaluate(L0, temperatures), atol=0.1)


@pytest.mark.parametrize("compound", [CU2MG, CORUNDUM], ids=["CU2MG", "CORUNDUM"])
@pytest.mark.parametrize("fitting_description, outputs, temperatures", [
    (gibbs_energy_fitting_description, ["HM", "SM", "CPM"], {"HM": [298.15], "SM": [298.15]}),
    (gibbs_energy_fitting_description, ["HM", "SM", "CPM"], None),
    (joint_gibbs_energy_fitting_description, ["GM", "HM", "SM", "CPM"], None),
    (joint_gibbs_energy_fitting_description, ["HM"], None),
], ids=["sequential-typical", "sequential", "joint", "joint-HM-only"])
def test_absolute_data_fit_the_same_parameters_as_formation_data(datasets_db, compound, fitting_description, outputs, temperatures):
    """Absolute (SER referenced) data and the equivalent formation data should fit the same parameters"""
    phase_models, _, _, default_temperatures = compound
    rng = np.random.default_rng(1234)
    num_temperatures = {out: len(temperatures.get(out, default_temperatures)) if temperatures is not None else len(default_temperatures) for out in outputs}
    noise_std_dev = {"GM": 50.0, "HM": 50.0, "SM": 0.05, "CPM": 0.05}
    noise = {out: rng.normal(0.0, noise_std_dev[out], num_temperatures[out]) for out in outputs}
    formation_datasets = _formation_datasets(compound, outputs, temperatures=temperatures, noise=noise)
    absolute_datasets = _absolute_datasets(compound, outputs, temperatures=temperatures, noise=noise)
    dbf_formation = _fit(phase_models, formation_datasets, fitting_description, datasets_db)
    dbf_absolute = _fit(phase_models, absolute_datasets, fitting_description, datasets_db)
    assert len(_fitted_symbols(dbf_absolute)) > 0
    _assert_equivalent_databases(dbf_formation, dbf_absolute)


@pytest.mark.parametrize("fitting_description, prop, temperatures", [
    (gibbs_energy_fitting_description, "HM", [298.15]),
    (joint_gibbs_energy_fitting_description, "GM", [300, 500, 700, 900]),
], ids=["sequential-HM", "joint-GM"])
def test_absolute_mixing_data_fit_excess_parameters(datasets_db, fitting_description, prop, temperatures):
    """Absolute data for a solution phase fit the excess Gibbs energy relative to the endmember lattice stabilities"""
    # FCC_A1 L0 from the Cu-Mg assessment in pycalphad's test databases (cumg.tdb)
    L0 = -22279.28 + 5.868*v.T
    unaries = {symengine.Symbol('GHSERCU'): SGTE91Stable['CU'], symengine.Symbol('GHSERMG'): SGTE91Stable['MG']}
    g_fcc_cu = SGTE91[('CU', 'FCC_A1')].xreplace(unaries)
    g_fcc_mg = SGTE91[('MG', 'FCC_A1')].xreplace(unaries)
    phase_models = {"components": ["CU", "MG", "VA"], "phases": {"FCC_A1": {"sublattice_model": [["CU", "MG"], ["VA"]], "sublattice_site_ratios": [1, 1]}}}
    x_mg = [0.1, 0.25, 0.5, 0.75, 0.9]
    values = []
    for T in temperatures:
        row = []
        for x in x_mg:
            gibbs_energy = (1-x)*g_fcc_cu + x*g_fcc_mg + v.R*v.T*(x*symengine.log(x) + (1-x)*symengine.log(1-x)) + x*(1-x)*L0
            row.append(float(_property(gibbs_energy, prop).xreplace({v.T: T})))
        values.append(row)
    dataset = {
        "components": ["CU", "MG", "VA"],
        "phases": ["FCC_A1"],
        "solver": {
            "mode": "manual",
            "sublattice_site_ratios": [1, 1],
            "sublattice_configurations": [[["CU", "MG"], "VA"] for _ in x_mg],
            "sublattice_occupancies": [[[1-x, x], 1] for x in x_mg],
        },
        "conditions": {"P": 101325, "T": temperatures},
        "output": prop,
        "values": [values],
        "reference": "Generated from a CALPHAD database",
    }
    dbf = _fit(phase_models, [dataset], fitting_description, datasets_db)
    params = dbf._parameters.search((where('phase_name') == 'FCC_A1') & (where('parameter_type') == 'L'))
    assert len(params) == 1
    assert params[0]['parameter_order'] == 0
    fitted = symengine.S(params[0]['parameter']).xreplace({symengine.Symbol(name): val for name, val in _fitted_symbols(dbf).items()})
    assert np.allclose(_evaluate(_property(fitted, prop), temperatures), _evaluate(_property(L0, prop), temperatures), atol=0.1)
