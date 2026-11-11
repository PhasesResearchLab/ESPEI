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


def _per_formula_unit(dbf, comps, phase_name, prop, fractions, T, excluded=()):
    """Property per mole of formula units and moles of atoms per formula unit at fractions {(species, sublattice): y}."""
    mod = Model(dbf, comps, phase_name)
    for contrib in excluded:
        mod.models[contrib] = 0
    point = {sf: fractions.get((sf.species.name, sf.sublattice_index), 0.0) for sf in mod.site_fractions}
    atoms = sum(float(mod.moles(el, per_formula_unit=True).xreplace(point)) for el in mod.nonvacant_elements)
    return atoms * float(getattr(mod, prop).xreplace({**point, v.T: T, v.P: 101325.0})), atoms


def _mixing_property(dbf, comps, phase_name, prop, fractions, T, excluded=()):
    """_MIX property per mole of atoms: the endmembers are weighted by the product of their site fractions."""
    value, atoms = _per_formula_unit(dbf, comps, phase_name, prop, fractions, T, excluded)
    sublattices = sorted({subl for _, subl in fractions})
    for endmember in itertools.product(*[[(sp, y) for (sp, s), y in fractions.items() if s == subl] for subl in sublattices]):
        endmember_value, _ = _per_formula_unit(dbf, comps, phase_name, prop, {(sp, subl): 1.0 for subl, (sp, _) in zip(sublattices, endmember)}, T, excluded)
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


def _al2fe_dataset(output, value, T=298.15, excluded=None):
    dataset = {
        "components": ["AL", "FE"], "phases": ["AL2FE"], "output": output, "reference": "testing",
        "solver": {"mode": "manual", "sublattice_site_ratios": [2, 1], "sublattice_configurations": [["AL", "FE"]]},
        "conditions": {"P": 101325, "T": T}, "values": [[[value] for _ in np.atleast_1d(T)]],
    }
    if excluded is not None:
        dataset["excluded_model_contributions"] = excluded
    return dataset


@pytest.mark.parametrize("excluded", [None, [], ["mag"]])
def test_generated_parameters_reproduce_formation_and_mixing_data_with_magnetic_references(datasets_db, excluded):
    """Generated parameters reproduce _FORM data relative to magnetic SER Fe and _MIX data relative to magnetic endmembers, with no MCMC residual"""
    datasets_db.insert(_al2fe_dataset("HM_FORM", -27000.0, excluded=excluded))
    datasets_db.insert(_al2fe_dataset("SM_FORM", -2.0, excluded=excluded))
    bcc_mix = {
        "components": ["AL", "FE", "VA"], "phases": ["BCC_A2"], "output": "HM_MIX", "reference": "testing",
        "solver": {"mode": "manual", "sublattice_site_ratios": [1, 3],
                   "sublattice_configurations": [[["AL", "FE"], "VA"]], "sublattice_occupancies": [[[0.25, 0.75], 1.0]]},
        "conditions": {"P": 101325, "T": 298.15}, "values": [[[-12000.0]]],
    }
    if excluded is not None:
        bcc_mix["excluded_model_contributions"] = excluded
    datasets_db.insert(bcc_mix)
    dbf = generate_parameters(AL_FE_PHASE_MODELS, datasets_db, "SGTE91", "linear", dbf=Database(AL_FE_MAGNETIC_BCC_TDB))

    excluded = excluded or []
    comps = ["AL", "FE", "VA"]
    for prop, data in (("HM", -27000.0), ("SM", -2.0)):
        compound, compound_atoms = _per_formula_unit(dbf, comps, "AL2FE", prop, {("AL", 0): 1.0, ("FE", 1): 1.0}, 298.15, excluded)
        aluminum, _ = _per_formula_unit(dbf, comps, "FCC_A1", prop, {("AL", 0): 1.0, ("VA", 1): 1.0}, 298.15, excluded)
        iron, _ = _per_formula_unit(dbf, comps, "BCC_A2", prop, {("FE", 0): 1.0, ("VA", 1): 1.0}, 298.15, excluded)
        assert np.isclose((compound - 2*aluminum - iron) / compound_atoms, data, rtol=1e-5)
    mixing = _mixing_property(dbf, comps, "BCC_A2", "HM", {("AL", 0): 0.25, ("FE", 0): 0.75, ("VA", 1): 1.0}, 298.15, excluded)
    assert np.isclose(mixing, -12000.0, rtol=1e-5)

    residuals, _ = FixedConfigurationPropertyResidual(dbf, datasets_db, phase_models=None, symbols_to_fit=[]).get_residuals(np.asarray([]))
    assert len(residuals) == 3
    assert np.allclose(residuals, 0.0, atol=0.5)  # fitted parameters are rounded to 6 significant figures


AL2FE_TDB = AL_FE_MAGNETIC_BCC_TDB + """
 FUNCTION GHSERAL 298.15 -7976.15+137.093038*T-24.3671976*T*LN(T)-.001884662*T**2-8.77664E-07*T**3+74092*T**(-1); 700 Y
   -11276.24+223.048446*T-38.5844296*T*LN(T)+.018531982*T**2-5.764227E-06*T**3+74092*T**(-1); 933.47 Y
   -11278.378+188.684153*T-31.748192*T*LN(T)-1.230524E+28*T**(-9); 2900 N !
 FUNCTION GHSERFE 298.15 1225.7+124.134*T-23.5143*T*LN(T)-.00439752*T**2-5.8927E-08*T**3+77359*T**(-1); 1811 Y
   -25383.581+299.31255*T-46*T*LN(T)+2.29603E+31*T**(-9); 6000 N !
 PHASE FCC_A1 % 2 1 1 !
    CONSTITUENT FCC_A1 :AL : VA : !
   PARAMETER G(FCC_A1,AL:VA;0) 298.15 GHSERAL; 6000 N !
   PARAMETER G(BCC_A2,FE:VA;0) 298.15 GHSERFE; 6000 N !
 PHASE AL2FE % 2 2 1 !
    CONSTITUENT AL2FE :AL : FE : !
   PARAMETER G(AL2FE,AL:FE;0) 298.15 2*GHSERAL+GHSERFE-90000+10*T-0.5*T*LN(T); 6000 N !
"""


@pytest.mark.parametrize("excluded", [None, ["mag"]])
def test_mcmc_formation_properties_are_relative_to_magnetic_ser_phases(datasets_db, excluded):
    """HM_FORM, SM_FORM and CPM_FORM in MCMC are relative to the full model of each SER phase, including magnetism"""
    dbf = Database(AL2FE_TDB)
    comps = ["AL", "FE", "VA"]
    temperatures = [300.0, 900.0]
    for prop in ("HM", "SM", "CPM"):
        values = []
        for T in temperatures:
            compound, compound_atoms = _per_formula_unit(dbf, comps, "AL2FE", prop, {("AL", 0): 1.0, ("FE", 1): 1.0}, T, excluded or [])
            aluminum, _ = _per_formula_unit(dbf, comps, "FCC_A1", prop, {("AL", 0): 1.0, ("VA", 1): 1.0}, T, excluded or [])
            iron, _ = _per_formula_unit(dbf, comps, "BCC_A2", prop, {("FE", 0): 1.0, ("VA", 1): 1.0}, T, excluded or [])
            values.append((compound - 2*aluminum - iron) / compound_atoms)
        dataset = _al2fe_dataset(prop + "_FORM", 0.0, T=temperatures, excluded=excluded)
        dataset["values"] = [[[value] for value in values]]
        datasets_db.insert(dataset)
    residuals, _ = FixedConfigurationPropertyResidual(dbf, datasets_db, phase_models=None, symbols_to_fit=[]).get_residuals(np.asarray([]))
    assert len(residuals) == 6
    assert np.allclose(residuals, 0.0, atol=1e-6)


# (A,B)1(A)1 as (FE,NI)1(FE)1, with magnetism only in the FE:FE endmember
TWO_SUBLATTICE_MAGNETIC_TDB = """
ELEMENT VA VACUUM 0 0 0 !
ELEMENT FE BCC_A2 55.847 4489 27.28 !
ELEMENT NI FCC_A1 58.69 4787 29.796 !
TYPE_DEFINITION % SEQ * !
TYPE_DEFINITION C GES A_P_D ALPHA MAGNETIC -1.0 0.4 !
PHASE ALPHA %C 2 1 1 !
CONSTITUENT ALPHA :FE,NI : FE : !
PARAMETER G(ALPHA,FE:FE;0) 1 -10000 - 2*T; 10000 N !
PARAMETER G(ALPHA,NI:FE;0) 1 -20000 + 3*T; 10000 N !
PARAMETER TC(ALPHA,FE:FE;0) 1 1000; 10000 N !
PARAMETER BMAGN(ALPHA,FE:FE;0) 1 2; 10000 N !
"""
TWO_SUBLATTICE_INTERACTION = "PARAMETER L(ALPHA,FE,NI:FE;0) 1 {}; 10000 N !"

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
B2_INTERACTION = "PARAMETER L(B2,AL,NI:NI;0) 1 {}; 10000 N !"


@pytest.mark.parametrize("tdb, interaction, value, phase_name, sublattice_model, site_ratios, configuration, occupancies, excluded", [
    # (A,B)1(A)1 with [[0.25, 0.75], [1.0]] is relative to 0.25 of A:A and 0.75 of B:A
    (TWO_SUBLATTICE_MAGNETIC_TDB, TWO_SUBLATTICE_INTERACTION, 5000.0, "ALPHA", [["FE", "NI"], ["FE"]], [1, 1], [["FE", "NI"], "FE"], [[0.25, 0.75], 1.0], None),
    (TWO_SUBLATTICE_MAGNETIC_TDB, TWO_SUBLATTICE_INTERACTION, 5000.0, "ALPHA", [["FE", "NI"], ["FE"]], [1, 1], [["FE", "NI"], "FE"], [[0.25, 0.75], 1.0], ["mag"]),
    (B2_WITH_VACANCIES_TDB, B2_INTERACTION, -20000.0, "B2", [["AL", "NI", "VA"], ["AL", "NI", "VA"]], [0.5, 0.5], [["AL", "NI"], "NI"], [[0.4, 0.6], 1.0], None),
], ids=["two-sublattice-magnetic", "two-sublattice-magnetic-excluded", "vacancy-endmembers"])
def test_mixing_data_are_relative_to_the_endmembers(datasets_db, tdb, interaction, value, phase_name, sublattice_model, site_ratios, configuration, occupancies, excluded):
    """_MIX data are relative to the endmembers weighted by their site fractions, each with its own physics"""
    truth_dbf = Database(tdb + interaction.format(value))
    comps = sorted({sp for subl in sublattice_model for sp in subl})
    fractions = {}
    for subl, (occupants, occupancy) in enumerate(zip(configuration, occupancies)):
        for species, y in zip(np.atleast_1d(occupants), np.atleast_1d(occupancy)):
            fractions[(species, subl)] = y
    dataset = {
        "components": comps, "phases": [phase_name], "output": "HM_MIX", "reference": "testing",
        "solver": {"mode": "manual", "sublattice_site_ratios": site_ratios, "sublattice_configurations": [configuration], "sublattice_occupancies": [occupancies]},
        "conditions": {"P": 101325, "T": 600.0}, "values": [[[_mixing_property(truth_dbf, comps, phase_name, "HM", fractions, 600.0, excluded or [])]]],
    }
    if excluded is not None:
        dataset["excluded_model_contributions"] = excluded
    datasets_db.insert(dataset)

    residuals, _ = FixedConfigurationPropertyResidual(truth_dbf, datasets_db, phase_models=None, symbols_to_fit=[]).get_residuals(np.asarray([]))
    assert np.allclose(residuals, 0.0, atol=1e-6)

    phase_models = {"components": comps, "phases": {phase_name: {"sublattice_model": sublattice_model, "sublattice_site_ratios": site_ratios}}}
    dbf = generate_parameters(phase_models, datasets_db, "SGTE91", "linear", dbf=Database(tdb))
    assert np.isclose(dbf.symbols["VV0000"], value)


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
