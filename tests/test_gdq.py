from espei.parameter_selection.utils import get_data_quantities
from pycalphad import Database, Model, variables as v
from espei.core_utils import get_samples

import numpy as np
from numpy import array
from sympy import log, Piecewise


def test_get_data_quantities_AL_NI_VA_interaction():
    """Test that an interaction with a VA produces the correct data quantities

    We just have a template database that has the phase defined. We then hot
    patch the Model object to have the GM from the fixed model we printed out
    and the data we printed out. The hot patch is needed because this is
    formation enthalpy data and the model needs to have the lower order terms
    in composition.

    One possible issue is that the new GM in the fixed model does not have any
    individual contributions, so it cannot be used to test excluded model
    contributions. The only excluded model contributions in this data are
    idmix, but the property we are testing is HM_FORM, so the feature transform
    of the idmix property should be zero.

    """
    T = v.T

    data = [{'components': ['AL', 'NI', 'VA'], 'phases': ['BCC_B2'], 'solver': {'mode': 'manual', 'sublattice_occupancies': [[1.0, [0.5, 0.5], 1.0], [1.0, [0.75, 0.25], 1.0]], 'sublattice_site_ratios': [0.5, 0.5, 1.0], 'sublattice_configurations': (('AL', ('NI', 'VA'), 'VA'), ('AL', ('NI', 'VA'), 'VA')), 'comment': 'BCC_B2 sublattice configuration (2SL)'}, 'conditions': {'P': 101325.0, 'T': array([300.])}, 'reference_state': 'SGTE91', 'output': 'HM_FORM', 'values': array([[[-40316.61077, -56361.58554]]]), 'reference': 'C. Jiang 2009 (constrained SQS)', 'excluded_model_contributions': ['idmix']}, {'components': ['AL', 'NI', 'VA'], 'phases': ['BCC_B2'], 'solver': {'mode': 'manual', 'sublattice_occupancies': [[1.0, [0.5, 0.5], 1.0], [1.0, [0.75, 0.25], 1.0]], 'sublattice_site_ratios': [0.5, 0.5, 1.0], 'sublattice_configurations': (('AL', ('NI', 'VA'), 'VA'), ('AL', ('NI', 'VA'), 'VA')), 'comment': 'BCC_B2 sublattice configuration (2SL)'}, 'conditions': {'P': 101325.0, 'T': array([300.])}, 'reference_state': 'SGTE91', 'output': 'HM_FORM', 'values': array([[[-41921.43363, -57769.49473]]]), 'reference': 'C. Jiang 2009 (relaxed SQS)', 'excluded_model_contributions': ['idmix']}]
    NEW_GM = 8.3145*T*(0.5*Piecewise((v.SiteFraction("BCC_B2", 0, "AL")*log(v.SiteFraction("BCC_B2", 0, "AL")), v.SiteFraction("BCC_B2", 0, "AL") > 1.0e-16), (0, True))/(0.5*v.SiteFraction("BCC_B2", 0, "AL") + 0.5*v.SiteFraction("BCC_B2", 0, "NI") + 0.5*v.SiteFraction("BCC_B2", 1, "AL") + 0.5*v.SiteFraction("BCC_B2", 1, "NI")) + 0.5*Piecewise((v.SiteFraction("BCC_B2", 0, "NI")*log(v.SiteFraction("BCC_B2", 0, "NI")), v.SiteFraction("BCC_B2", 0, "NI") > 1.0e-16), (0, True))/(0.5*v.SiteFraction("BCC_B2", 0, "AL") + 0.5*v.SiteFraction("BCC_B2", 0, "NI") + 0.5*v.SiteFraction("BCC_B2", 1, "AL") + 0.5*v.SiteFraction("BCC_B2", 1, "NI")) + 0.5*Piecewise((v.SiteFraction("BCC_B2", 0, "VA")*log(v.SiteFraction("BCC_B2", 0, "VA")), v.SiteFraction("BCC_B2", 0, "VA") > 1.0e-16), (0, True))/(0.5*v.SiteFraction("BCC_B2", 0, "AL") + 0.5*v.SiteFraction("BCC_B2", 0, "NI") + 0.5*v.SiteFraction("BCC_B2", 1, "AL") + 0.5*v.SiteFraction("BCC_B2", 1, "NI")) + 0.5*Piecewise((v.SiteFraction("BCC_B2", 1, "AL")*log(v.SiteFraction("BCC_B2", 1, "AL")), v.SiteFraction("BCC_B2", 1, "AL") > 1.0e-16), (0, True))/(0.5*v.SiteFraction("BCC_B2", 0, "AL") + 0.5*v.SiteFraction("BCC_B2", 0, "NI") + 0.5*v.SiteFraction("BCC_B2", 1, "AL") + 0.5*v.SiteFraction("BCC_B2", 1, "NI")) + 0.5*Piecewise((v.SiteFraction("BCC_B2", 1, "NI")*log(v.SiteFraction("BCC_B2", 1, "NI")), v.SiteFraction("BCC_B2", 1, "NI") > 1.0e-16), (0, True))/(0.5*v.SiteFraction("BCC_B2", 0, "AL") + 0.5*v.SiteFraction("BCC_B2", 0, "NI") + 0.5*v.SiteFraction("BCC_B2", 1, "AL") + 0.5*v.SiteFraction("BCC_B2", 1, "NI")) + 0.5*Piecewise((v.SiteFraction("BCC_B2", 1, "VA")*log(v.SiteFraction("BCC_B2", 1, "VA")), v.SiteFraction("BCC_B2", 1, "VA") > 1.0e-16), (0, True))/(0.5*v.SiteFraction("BCC_B2", 0, "AL") + 0.5*v.SiteFraction("BCC_B2", 0, "NI") + 0.5*v.SiteFraction("BCC_B2", 1, "AL") + 0.5*v.SiteFraction("BCC_B2", 1, "NI")) + Piecewise((v.SiteFraction("BCC_B2", 2, "VA")*log(v.SiteFraction("BCC_B2", 2, "VA")), v.SiteFraction("BCC_B2", 2, "VA") > 1.0e-16), (0, True))/(0.5*v.SiteFraction("BCC_B2", 0, "AL") + 0.5*v.SiteFraction("BCC_B2", 0, "NI") + 0.5*v.SiteFraction("BCC_B2", 1, "AL") + 0.5*v.SiteFraction("BCC_B2", 1, "NI"))) + (45262.9*v.SiteFraction("BCC_B2", 0, "AL")*v.SiteFraction("BCC_B2", 0, "NI")*v.SiteFraction("BCC_B2", 1, "AL")*v.SiteFraction("BCC_B2", 2, "VA") + 45262.9*v.SiteFraction("BCC_B2", 0, "AL")*v.SiteFraction("BCC_B2", 1, "AL")*v.SiteFraction("BCC_B2", 1, "NI")*v.SiteFraction("BCC_B2", 2, "VA"))/(0.5*v.SiteFraction("BCC_B2", 0, "AL") + 0.5*v.SiteFraction("BCC_B2", 0, "NI") + 0.5*v.SiteFraction("BCC_B2", 1, "AL") + 0.5*v.SiteFraction("BCC_B2", 1, "NI")) + (1.0*v.SiteFraction("BCC_B2", 0, "AL")*v.SiteFraction("BCC_B2", 1, "AL")*v.SiteFraction("BCC_B2", 2, "VA")*Piecewise((10083 - 4.813*T, (T >= 298.15) & (T < 2900.0)), (0, True)) + v.SiteFraction("BCC_B2", 0, "AL")*v.SiteFraction("BCC_B2", 1, "NI")*v.SiteFraction("BCC_B2", 2, "VA")*(9.52839e-8*T**3 + 0.00123463*T**2 + 0.000871898*T*log(T) + 1.31471*T - 64435.3 + 23095.2/T) + v.SiteFraction("BCC_B2", 0, "AL")*v.SiteFraction("BCC_B2", 1, "VA")*v.SiteFraction("BCC_B2", 2, "VA")*(10.0*T + 16432.5) + v.SiteFraction("BCC_B2", 0, "NI")*v.SiteFraction("BCC_B2", 1, "AL")*v.SiteFraction("BCC_B2", 2, "VA")*(9.52839e-8*T**3 + 0.00123463*T**2 + 0.000871898*T*log(T) + 1.31471*T - 64435.3 + 23095.2/T) + 1.0*v.SiteFraction("BCC_B2", 0, "NI")*v.SiteFraction("BCC_B2", 1, "NI")*v.SiteFraction("BCC_B2", 2, "VA")*Piecewise((8715.084 - 3.556*T, (T >= 298.15) & (T < 3000.0)), (0, True)) + 32790.6*v.SiteFraction("BCC_B2", 0, "NI")*v.SiteFraction("BCC_B2", 1, "VA")*v.SiteFraction("BCC_B2", 2, "VA") + v.SiteFraction("BCC_B2", 0, "VA")*v.SiteFraction("BCC_B2", 1, "AL")*v.SiteFraction("BCC_B2", 2, "VA")*(10.0*T + 16432.5) + 32790.6*v.SiteFraction("BCC_B2", 0, "VA")*v.SiteFraction("BCC_B2", 1, "NI")*v.SiteFraction("BCC_B2", 2, "VA"))/(0.5*v.SiteFraction("BCC_B2", 0, "AL") + 0.5*v.SiteFraction("BCC_B2", 0, "NI") + 0.5*v.SiteFraction("BCC_B2", 1, "AL") + 0.5*v.SiteFraction("BCC_B2", 1, "NI"))

    dbf = Database("""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    $ Date: 2019-12-08 18:05
    $ Components: AL, NI, VA
    $ Phases: BCC_B2
    $ Generated by brandon (pycalphad 0.8.1.post1)
    $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    ELEMENT AL FCC_A1 26.982 4577.3 28.322 !
    ELEMENT NI FCC_A1 58.69 4787.0 29.796 !
    ELEMENT VA VACUUM 0.0 0.0 0.0 !

    TYPE_DEFINITION % SEQ * !
    DEFINE_SYSTEM_DEFAULT ELEMENT 2 !
    DEFAULT_COMMAND DEFINE_SYSTEM_ELEMENT VA !

    PHASE BCC_B2 %  3 0.5 0.5 1 !
    CONSTITUENT BCC_B2 :AL,NI,VA:AL,NI,VA:VA: !

    """)
    mod = Model(dbf, ['AL', 'NI', 'VA'], 'BCC_B2')
    dd = {ky: 0.0 for ky in mod.models.keys()}
    dd['GM'] = NEW_GM
    mod.models = dd
    print(mod.HM)
    print(get_samples(data))
    # desired_property, fixed_model, fixed_portions, data, samples
    qty = get_data_quantities('HM_FORM', mod, [0], data)
    print(qty)
    assert np.all(np.isclose([-6254.7802775, -5126.1206475, -7458.3974225, -6358.04118875], qty))


def test_get_data_quantities_mixing_entropy():
    """Test that mixing entropy produces correct data quantities.

    """
    data = [{'components': ['AL', 'CR'], 'phases': ['AL11CR2'], 'solver': {'mode': 'manual', 'sublattice_site_ratios': [10.0, 2.0], 'sublattice_configurations': (('AL', ('AL', 'CR')),), 'sublattice_occupancies': [[1.0, [0.5, 0.5]]]}, 'conditions': {'P': 101325.0, 'T': array([300.])}, 'output': 'SM_MIX', 'values': array([[[0.60605556]]]), 'reference': 'text 1 to write down reference for this work', 'comment': 'test 2 to write down comment for this work', 'excluded_model_contributions': ['idmix']}]

    dbf = Database("""
    ELEMENT AL FCC_A1 26.982 4577.3 28.322 !
    ELEMENT CR BCC_A2 51.996 4050.0 23.56 !

    PHASE AL11CR2 %  2 10.0 2.0 !
    CONSTITUENT AL11CR2 :AL:AL,CR: !

    """)
    mod = Model(dbf, ['AL', 'CR'], 'AL11CR2')
    print(get_samples(data))
    # desired_property, fixed_model, fixed_portions, data, samples
    qty = get_data_quantities('SM_MIX', mod, [0], data)
    print(qty)
    assert np.all(np.isclose([7.27266667], qty))

