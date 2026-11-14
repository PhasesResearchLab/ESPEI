"""
Reference states of fixed configuration data in parameter generation and MCMC.

Expected values are computed from pycalphad Models with the definitions of the
Reference States documentation page:
_FORM subtracts the pure elements in their SER phases, _MIX subtracts the
endmembers weighted by the product of their site fractions, and excluded
contributions are removed from both the phase and its reference.
"""

import itertools

import numpy as np
import pytest
from pycalphad import Database, Model, variables as v

from espei.error_functions.non_equilibrium_thermochemical_error import FixedConfigurationPropertyResidual
from espei.paramselect import generate_parameters

from .fixtures import datasets_db


def _per_formula_unit(dbf, comps, phase_name, prop, fractions, T):
    """Property per mole of formula units and moles of atoms per formula unit at fractions {(species, sublattice): y}."""
    mod = Model(dbf, comps, phase_name)
    point = {sf: fractions.get((sf.species.name, sf.sublattice_index), 0.0) for sf in mod.site_fractions}
    atoms = sum(float(mod.moles(el, per_formula_unit=True).xreplace(point)) for el in mod.nonvacant_elements)
    return atoms * float(getattr(mod, prop).xreplace({**point, v.T: T, v.P: 101325.0})), atoms


def _mixing_property(dbf, comps, phase_name, prop, fractions, T):
    """_MIX property per mole of atoms: the endmembers are weighted by the product of their site fractions."""
    value, atoms = _per_formula_unit(dbf, comps, phase_name, prop, fractions, T)
    sublattices = sorted({subl for _, subl in fractions})
    for endmember in itertools.product(*[[(sp, y) for (sp, s), y in fractions.items() if s == subl] for subl in sublattices]):
        endmember_value, _ = _per_formula_unit(dbf, comps, phase_name, prop, {(sp, subl): 1.0 for subl, (sp, _) in zip(sublattices, endmember)}, T)
        value -= np.prod([y for _, y in endmember]) * endmember_value
    return value / atoms


# Magnetic BCC_A2 Fe (SER phase of FE) from Xiong et al. doi:10.1016/j.calphad.2011.05.002
AL_FE_MAGNETIC_BCC_TDB = """
 ELEMENT VA   VACUUM           0.0000E+00  0.0000E+00  0.0000E+00!
 ELEMENT AL   FCC_A1           2.6982E+01  4.5773E+03  2.8322E+01!
 ELEMENT FE   BCC_A2           5.5847E+01  4.4890E+03  2.7280E+01!
 TYPE_DEFINITION % SEQ * !
 TYPE_DEFINITION C GES A_P_D BCC_A2 MAGNETIC -1.0 0.4 !
 PHASE BCC_A2  %C  2 1   3 !
    CONSTITUENT BCC_A2  :AL,FE : VA :  !
   PARAMETER TC(BCC_A2,FE:VA;0)  298.15  1043; 6000 N !
   PARAMETER BMAGN(BCC_A2,FE:VA;0)  298.15  2.22; 6000 N !
"""

# AL2FE is fit before BCC_A2, the SER phase of FE. FCC_A1 is the SER phase of AL.
AL_FE_PHASE_MODELS = {
    "components": ["AL", "FE", "VA"],
    "phases": {
        "AL2FE": {"sublattice_model": [["AL"], ["FE"]], "sublattice_site_ratios": [2, 1]},
        "BCC_A2": {"sublattice_model": [["AL", "FE"], ["VA"]], "sublattice_site_ratios": [1, 3]},
        "FCC_A1": {"sublattice_model": [["AL", "FE"], ["VA"]], "sublattice_site_ratios": [1, 1]},
    },
}


def _al2fe_dataset(output, value):
    return {
        "components": ["AL", "FE"], "phases": ["AL2FE"], "output": output, "reference": "testing",
        "solver": {"mode": "manual", "sublattice_site_ratios": [2, 1], "sublattice_configurations": [["AL", "FE"]]},
        "conditions": {"P": 101325, "T": 298.15}, "values": [[[value]]],
    }


def test_generated_parameters_reproduce_formation_and_mixing_data_with_magnetic_references(datasets_db):
    """Generated parameters reproduce _FORM data relative to magnetic SER Fe and _MIX data relative to magnetic endmembers, with no MCMC residual"""
    datasets_db.insert(_al2fe_dataset("HM_FORM", -27000.0))
    datasets_db.insert(_al2fe_dataset("SM_FORM", -2.0))
    bcc_mix = {
        "components": ["AL", "FE", "VA"], "phases": ["BCC_A2"], "output": "HM_MIX", "reference": "testing",
        "solver": {"mode": "manual", "sublattice_site_ratios": [1, 3],
                   "sublattice_configurations": [[["AL", "FE"], "VA"]], "sublattice_occupancies": [[[0.25, 0.75], 1.0]]},
        "conditions": {"P": 101325, "T": 298.15}, "values": [[[-12000.0]]],
    }
    datasets_db.insert(bcc_mix)
    dbf = generate_parameters(AL_FE_PHASE_MODELS, datasets_db, "SGTE91", "linear", dbf=Database(AL_FE_MAGNETIC_BCC_TDB))

    comps = ["AL", "FE", "VA"]
    for prop, data in (("HM", -27000.0), ("SM", -2.0)):
        compound, compound_atoms = _per_formula_unit(dbf, comps, "AL2FE", prop, {("AL", 0): 1.0, ("FE", 1): 1.0}, 298.15)
        aluminum, _ = _per_formula_unit(dbf, comps, "FCC_A1", prop, {("AL", 0): 1.0, ("VA", 1): 1.0}, 298.15)
        iron, _ = _per_formula_unit(dbf, comps, "BCC_A2", prop, {("FE", 0): 1.0, ("VA", 1): 1.0}, 298.15)
        assert np.isclose((compound - 2*aluminum - iron) / compound_atoms, data, rtol=1e-5)
    mixing = _mixing_property(dbf, comps, "BCC_A2", "HM", {("AL", 0): 0.25, ("FE", 0): 0.75, ("VA", 1): 1.0}, 298.15)
    assert np.isclose(mixing, -12000.0, rtol=1e-5)

    residuals, _ = FixedConfigurationPropertyResidual(dbf, datasets_db, phase_models=None, symbols_to_fit=[]).get_residuals(np.asarray([]))
    assert len(residuals) == 3
    assert np.allclose(residuals, 0.0, atol=0.5)  # fitted parameters are rounded to 6 significant figures


# Endmembers with only vacancies have no atoms
B2_WITH_VACANCIES_TDB = """
ELEMENT VA VACUUM 0 0 0 !
ELEMENT AL FCC_A1 26.982 4577.3 28.322 !
ELEMENT NI FCC_A1 58.69 4787 29.796 !
TYPE_DEFINITION % SEQ * !
PHASE B2 % 2 0.5 0.5 !
CONSTITUENT B2 :AL,NI,VA : AL,NI,VA : !
PARAMETER G(B2,AL:AL;0) 1 -1000 - 2*T; 10000 N !
PARAMETER G(B2,NI:NI;0) 1 -2000 - 3*T; 10000 N !
PARAMETER G(B2,AL:NI;0) 1 -60000 + 10*T; 10000 N !
PARAMETER G(B2,NI:AL;0) 1 -60000 + 10*T; 10000 N !
PARAMETER G(B2,AL:VA;0) 1 10000; 10000 N !
PARAMETER G(B2,VA:AL;0) 1 10000; 10000 N !
PARAMETER G(B2,NI:VA;0) 1 8000; 10000 N !
PARAMETER G(B2,VA:NI;0) 1 8000; 10000 N !
PARAMETER G(B2,VA:VA;0) 1 100*T; 10000 N !
"""
B2_INTERACTION = "PARAMETER L(B2,AL,NI:NI;0) 1 -20000; 10000 N !"


def test_mixing_data_are_relative_to_the_endmembers(datasets_db):
    """_MIX data are relative to the endmembers weighted by their site fractions, also in a phase with endmembers of only vacancies"""
    truth_dbf = Database(B2_WITH_VACANCIES_TDB + B2_INTERACTION)
    comps = ["AL", "NI", "VA"]
    fractions = {("AL", 0): 0.4, ("NI", 0): 0.6, ("NI", 1): 1.0}
    datasets_db.insert({
        "components": comps, "phases": ["B2"], "output": "HM_MIX", "reference": "testing",
        "solver": {"mode": "manual", "sublattice_site_ratios": [0.5, 0.5], "sublattice_configurations": [[["AL", "NI"], "NI"]], "sublattice_occupancies": [[[0.4, 0.6], 1.0]]},
        "conditions": {"P": 101325, "T": 600.0}, "values": [[[_mixing_property(truth_dbf, comps, "B2", "HM", fractions, 600.0)]]],
    })
    residuals, _ = FixedConfigurationPropertyResidual(truth_dbf, datasets_db, phase_models=None, symbols_to_fit=[]).get_residuals(np.asarray([]))
    assert np.allclose(residuals, 0.0, atol=1e-6)

    phase_models = {"components": comps, "phases": {"B2": {"sublattice_model": [["AL", "NI", "VA"], ["AL", "NI", "VA"]], "sublattice_site_ratios": [0.5, 0.5]}}}
    dbf = generate_parameters(phase_models, datasets_db, "SGTE91", "linear", dbf=Database(B2_WITH_VACANCIES_TDB))
    assert np.isclose(dbf.symbols["VV0000"], -20000.0)


SPECIES_TDB = """
ELEMENT VA VACUUM 0 0 0 !
ELEMENT A BCC_A2 10 0 0 !
ELEMENT B BCC_A2 10 0 0 !
SPECIES A2B3 A2B3 !
TYPE_DEFINITION % SEQ * !
PHASE LIQUID % 1 1 !
CONSTITUENT LIQUID :A,A2B3,B : !
PARAMETER G(LIQUID,A;0) 1 -1000 - 2*T; 10000 N !
PARAMETER G(LIQUID,A2B3;0) 1 -50000 + 10*T + T*LN(T); 10000 N !
PARAMETER G(LIQUID,B;0) 1 -3000 + 4*T; 10000 N !
PARAMETER L(LIQUID,A,A2B3;0) 1 -8000; 10000 N !
"""


def test_mixing_data_of_species_are_relative_to_the_species_endmembers(datasets_db):
    """(A, A2B3, B) with [0.25, 0.5, 0.25] is relative to 0.25 of A, 0.5 of A2B3 and 0.25 of B, per mole of formula units"""
    dbf = Database(SPECIES_TDB)
    fractions = {("A", 0): 0.25, ("A2B3", 0): 0.5, ("B", 0): 0.25}
    datasets_db.insert({
        "components": ["A", "B"], "phases": ["LIQUID"], "output": "HM_MIX", "reference": "testing",
        "solver": {"mode": "manual", "sublattice_site_ratios": [1], "sublattice_configurations": [[["A", "A2B3", "B"]]], "sublattice_occupancies": [[[0.25, 0.5, 0.25]]]},
        "conditions": {"P": 101325, "T": 600.0}, "values": [[[_mixing_property(dbf, ["A", "B"], "LIQUID", "HM", fractions, 600.0)]]],
    })
    residuals, _ = FixedConfigurationPropertyResidual(dbf, datasets_db, phase_models=None, symbols_to_fit=[]).get_residuals(np.asarray([]))
    assert np.allclose(residuals, 0.0, atol=1e-6)


def test_formation_data_are_relative_to_ghser_functions_without_ser_phases(datasets_db):
    """Without the SER phases in the Database, _FORM data are relative to the GHSER functions, with a warning"""
    datasets_db.insert(_al2fe_dataset("HM_FORM", -27000.0))
    phase_models = {"components": ["AL", "FE"], "phases": {"AL2FE": {"sublattice_model": [["AL"], ["FE"]], "sublattice_site_ratios": [2, 1]}}}
    with pytest.warns(UserWarning, match="GHSER"):
        dbf = generate_parameters(phase_models, datasets_db, "SGTE91", "linear")
    assert dbf.symbols["VV0000"] == 3 * -27000.0
