"""Fitting Gibbs energy parameters to GM, HM, SM and CPM data together"""

import numpy as np
import pytest
import symengine
from pycalphad import variables as v
from tinydb import where

from espei.paramselect import generate_parameters
from espei.refdata import SGTE91, SGTE91Stable
from espei.parameter_selection.fitting_descriptions import ModelFittingDescription
from espei.parameter_selection.fitting_steps import StepCPM, StepSM, StepHM

from .fixtures import datasets_db

FEATURES = (symengine.S.One, v.T, v.T*symengine.log(v.T), v.T**2, v.T**(-1), v.T**3)
# Gibbs energies (J/mol-formula) from the Cu-Mg assessment in pycalphad's test databases
# (cumg.tdb, M. Palumbo, T. Abe and K. Hashimoto, NIMS 2008), for 298.15 K < T < 923 K
GHSERCU = (-7770.458, 130.485403, -24.112392, -0.00265684, 52478, 1.29223E-07)
GHSERMG = (-8367.34, 143.677875, -26.1849782, 4.858E-04, 78950, -1.393669E-06)
G_CU2MG = (-54690.99, 364.73085, -69.276417, -5.1925E-4, 143502, -5.65953E-6)
# Gibbs energy of formation of Cu2Mg per mole of atoms
CU2MG_FORMATION = sum((g - 2*cu - mg)*f for g, cu, mg, f in zip(G_CU2MG, GHSERCU, GHSERMG, FEATURES)) / 3
CU2MG_PHASE_MODELS = {"components": ["CU", "MG"], "phases": {"CU2MG": {"sublattice_model": [["CU"], ["MG"]], "sublattice_site_ratios": [2, 1]}}}
TEMPERATURES = [298.15] + list(range(300, 901, 50))


def _property(gibbs_energy, prop):
    return {"GM": gibbs_energy, "HM": gibbs_energy - v.T*gibbs_energy.diff(v.T), "SM": -gibbs_energy.diff(v.T), "CPM": -v.T*gibbs_energy.diff(v.T, v.T)}[prop]


def _value(expr, T):
    return float(symengine.S(expr).xreplace({v.T: T}))


def _cu2mg_dataset(prop, temperatures, offset=0.0):
    """CU2MG _FORM data per mole of atoms, as precise as DFT data (standard deviations divided by 100)"""
    return {"components": ["CU", "MG"], "phases": ["CU2MG"], "output": prop + "_FORM", "reference": "Cu-Mg assessment", "weight": 100,
            "solver": {"mode": "manual", "sublattice_site_ratios": [2, 1], "sublattice_configurations": [["CU", "MG"]]},
            "conditions": {"P": 101325, "T": temperatures}, "values": [[[_value(_property(CU2MG_FORMATION, prop), T) + offset] for T in temperatures]]}


def _fitted_formation(dbf, phase_name="CU2MG", atoms=3):
    """Fitted Gibbs energy of formation per mole of atoms, i.e. without the GHSER functions"""
    params = dbf._parameters.search((where('phase_name') == phase_name) & (where('parameter_type') == 'G'))
    assert len(params) == 1
    expr = symengine.S(params[0]['parameter']).xreplace({symengine.Symbol(s): val for s, val in dbf.symbols.items() if s.startswith('VV')})
    return expr.xreplace({x: 0 for x in expr.free_symbols if x.name.startswith('GHSER')}) / atoms


def test_temperature_dependent_enthalpies_fit_the_heat_capacity(datasets_db):
    """HM data at many temperatures fit heat capacity features without CPM data"""
    datasets_db.insert(_cu2mg_dataset("HM", TEMPERATURES))
    datasets_db.insert(_cu2mg_dataset("SM", [298.15]))
    fitted = _fitted_formation(generate_parameters(CU2MG_PHASE_MODELS, datasets_db, 'SGTE91', 'linear'))
    for T in TEMPERATURES:
        assert np.isclose(_value(_property(fitted, "CPM"), T), _value(_property(CU2MG_FORMATION, "CPM"), T), atol=0.01)


def test_decoupled_data_fit_the_same_parameters_as_sequential_steps(datasets_db):
    """CPM data at many temperatures with HM and SM data at 298.15 K fit the same parameters as fitting CPM, then SM, then HM"""
    datasets_db.insert(_cu2mg_dataset("CPM", TEMPERATURES))
    datasets_db.insert(_cu2mg_dataset("HM", [298.15]))
    datasets_db.insert(_cu2mg_dataset("SM", [298.15]))
    together = generate_parameters(CU2MG_PHASE_MODELS, datasets_db, 'SGTE91', 'linear')
    sequential = generate_parameters(CU2MG_PHASE_MODELS, datasets_db, 'SGTE91', 'linear', fitting_description=ModelFittingDescription([StepCPM, StepSM, StepHM]))
    assert {s: val for s, val in together.symbols.items() if s.startswith('VV')} == {s: val for s, val in sequential.symbols.items() if s.startswith('VV')}


def test_gibbs_energy_and_enthalpy_at_one_temperature_fit_the_entropy(datasets_db):
    """GM and HM data at the same temperature determine the entropy, S = (H - G)/T"""
    datasets_db.insert(_cu2mg_dataset("GM", [298.15]))
    datasets_db.insert(_cu2mg_dataset("HM", [298.15]))
    fitted = _fitted_formation(generate_parameters(CU2MG_PHASE_MODELS, datasets_db, 'SGTE91', 'linear'))
    assert np.isclose(_value(_property(fitted, "SM"), 298.15), _value(_property(CU2MG_FORMATION, "SM"), 298.15), atol=1e-3)


def test_enthalpies_of_different_sources_near_298_K_do_not_fit_a_heat_capacity(datasets_db):
    """HM data at 298.15 K and 300 K that disagree do not fit a heat capacity, which would be about 800/(300 - 298.15) J/mol-K"""
    datasets_db.insert(_cu2mg_dataset("HM", [298.15]))
    datasets_db.insert(_cu2mg_dataset("HM", [300], offset=800.0))
    fitted = _fitted_formation(generate_parameters(CU2MG_PHASE_MODELS, datasets_db, 'SGTE91', 'linear'))
    assert _value(_property(fitted, "CPM"), 298.15) == 0.0


def test_absolute_data_fit_the_same_parameters_as_formation_data(datasets_db):
    """Absolute data are formation data plus the GHSER functions of the pure elements"""
    datasets_db.insert(_cu2mg_dataset("HM", TEMPERATURES))
    datasets_db.insert(_cu2mg_dataset("SM", [298.15]))
    formation = generate_parameters(CU2MG_PHASE_MODELS, datasets_db, 'SGTE91', 'linear')
    ser = (2*SGTE91Stable['CU'] + SGTE91Stable['MG']) / 3
    for ds in datasets_db.all():
        prop = ds['output'].split('_')[0]
        datasets_db.update({'output': prop, 'values': [[[val[0] + _value(_property(ser, prop), T)] for val, T in zip(ds['values'][0], ds['conditions']['T'])]]}, doc_ids=[ds.doc_id])
    absolute = generate_parameters(CU2MG_PHASE_MODELS, datasets_db, 'SGTE91', 'linear')
    assert {s: val for s, val in formation.symbols.items() if s.startswith('VV')} == pytest.approx({s: val for s, val in absolute.symbols.items() if s.startswith('VV')}, rel=1e-5)


@pytest.mark.parametrize("prop, temperatures, absolute", [
    ("GM", [300, 500, 700, 900], False),  # GM_MIX, including the ideal mixing contribution
    ("GM", [300, 500, 700, 900], True),  # absolute GM, relative to the lattice stabilities of the endmembers
    ("HM", [298.15], True),
])
def test_mixing_data_fit_the_excess_gibbs_energy(datasets_db, prop, temperatures, absolute):
    """Gibbs energy and enthalpy data of a solution phase fit its excess Gibbs energy"""
    # FCC_A1 L0 from the Cu-Mg assessment in pycalphad's test databases (cumg.tdb)
    L0 = -22279.28 + 5.868*v.T
    unaries = {symengine.Symbol('GHSERCU'): SGTE91Stable['CU'], symengine.Symbol('GHSERMG'): SGTE91Stable['MG']}
    endmembers = (SGTE91[('CU', 'FCC_A1')].xreplace(unaries), SGTE91[('MG', 'FCC_A1')].xreplace(unaries)) if absolute else (0, 0)
    x_mg = [0.1, 0.25, 0.5, 0.75, 0.9]
    values = []
    for T in temperatures:
        gibbs_energies = [(1-x)*endmembers[0] + x*endmembers[1] + v.R*v.T*(x*np.log(x) + (1-x)*np.log(1-x)) + x*(1-x)*L0 for x in x_mg]
        values.append([_value(_property(symengine.S(g), prop), T) for g in gibbs_energies])
    datasets_db.insert({
        "components": ["CU", "MG", "VA"], "phases": ["FCC_A1"], "output": prop if absolute else prop + "_MIX", "reference": "Cu-Mg assessment",
        "solver": {"mode": "manual", "sublattice_site_ratios": [1, 1], "sublattice_configurations": [[["CU", "MG"], "VA"] for _ in x_mg], "sublattice_occupancies": [[[1-x, x], 1] for x in x_mg]},
        "conditions": {"P": 101325, "T": temperatures}, "values": [values],
    })
    phase_models = {"components": ["CU", "MG", "VA"], "phases": {"FCC_A1": {"sublattice_model": [["CU", "MG"], ["VA"]], "sublattice_site_ratios": [1, 1]}}}
    dbf = generate_parameters(phase_models, datasets_db, 'SGTE91', 'linear')
    params = dbf._parameters.search((where('phase_name') == 'FCC_A1') & (where('parameter_type') == 'L'))
    assert len(params) == 1 and params[0]['parameter_order'] == 0
    fitted = symengine.S(params[0]['parameter']).xreplace({symengine.Symbol(s): val for s, val in dbf.symbols.items() if s.startswith('VV')})
    for T in temperatures:
        assert np.isclose(_value(_property(fitted, prop), T), _value(_property(L0, prop), T), atol=0.1)
