"""Fitting Gibbs energy parameters to GM, HM, SM and CPM data in one step"""

import numpy as np
import symengine
from pycalphad import variables as v
from tinydb import where

from espei.paramselect import generate_parameters
from espei.parameter_selection.fitting_descriptions import simultaneous_gibbs_energy_fitting_description as SIMULTANEOUS

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


def _cu2mg_dataset(prop, temperatures, offset=0.0, reference="Cu-Mg assessment"):
    """CU2MG _FORM data per mole of atoms"""
    return {"components": ["CU", "MG"], "phases": ["CU2MG"], "output": prop + "_FORM", "reference": reference,
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
    fitted = _fitted_formation(generate_parameters(CU2MG_PHASE_MODELS, datasets_db, 'SGTE91', 'linear', fitting_description=SIMULTANEOUS))
    for T in TEMPERATURES:
        assert np.isclose(_value(_property(fitted, "CPM"), T), _value(_property(CU2MG_FORMATION, "CPM"), T), atol=0.01)


def test_decoupled_data_fit_the_same_parameters_as_sequential_steps(datasets_db):
    """CPM data at many temperatures with HM and SM data at 298.15 K fit the same parameters as fitting CPM, then SM, then HM"""
    datasets_db.insert(_cu2mg_dataset("CPM", TEMPERATURES))
    datasets_db.insert(_cu2mg_dataset("HM", [298.15]))
    datasets_db.insert(_cu2mg_dataset("SM", [298.15]))
    simultaneous = generate_parameters(CU2MG_PHASE_MODELS, datasets_db, 'SGTE91', 'linear', fitting_description=SIMULTANEOUS)
    sequential = generate_parameters(CU2MG_PHASE_MODELS, datasets_db, 'SGTE91', 'linear')
    assert {s: val for s, val in simultaneous.symbols.items() if s.startswith('VV')} == {s: val for s, val in sequential.symbols.items() if s.startswith('VV')}


def test_enthalpies_of_different_references_do_not_fit_a_heat_capacity(datasets_db):
    """HM data of different references that disagree do not fit a heat capacity"""
    datasets_db.insert(_cu2mg_dataset("HM", [298.15]))
    datasets_db.insert(_cu2mg_dataset("HM", [1273], offset=800.0, reference="another source"))
    datasets_db.insert(_cu2mg_dataset("SM", [298.15]))
    fitted = _fitted_formation(generate_parameters(CU2MG_PHASE_MODELS, datasets_db, 'SGTE91', 'linear', fitting_description=SIMULTANEOUS))
    assert _value(_property(fitted, "CPM"), 298.15) == 0.0
