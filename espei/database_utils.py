"""Provides utilities for creating and working with Databases in ESPEI
"""

import logging
from typing import Dict, Union
from pycalphad import Database, variables as v
import espei.refdata
from espei.phase_models import PhaseModelSpecification
from espei.utils import extract_aliases

_log = logging.getLogger(__name__)

def _get_ser_data(element, ref_state, fallback_ref_state="SGTE91") -> Dict[str, Union[str, float]]:
    """Return a dictionary of the standard element reference (SER) data.

    If no SER data is found, returns an empty dictionary.
    """
    ser_ref_state = ref_state + "SER"
    # Return an empty dict for backwards compatibility, since the SER data may not exist
    ser_dict = getattr(espei.refdata, ser_ref_state, {})
    fallback_ser_ref_state = fallback_ref_state + "SER"
    fallback_ser_dict = getattr(espei.refdata, fallback_ser_ref_state)
    el_ser_data = ser_dict.get(element)
    if el_ser_data is None and ref_state == fallback_ref_state:
        # No data found, no fallback alternative
        _log.warning("%s has no entry in the %s reference data. Fitting formation energies will not be possible.", element, ser_ref_state)
    elif el_ser_data is None:
        # No data found, try the fallback
        el_ser_data = fallback_ser_dict.get(element)
        if el_ser_data is None:
            # No data found in the fallback
            _log.warning("%s has no entry in the %s reference data nor in the %s fallback reference data. Fitting formation energies will not be possible.", element, ser_ref_state, fallback_ser_ref_state)
            return {}
        else:
            # Data found in the fallback
            _log.trace("%s has no entry in the %s reference data, but was available in the %s fallback reference data.", element, ser_ref_state, fallback_ser_ref_state)
    if el_ser_data is not None:
        return el_ser_data
    else:
        return {}


def _missing_reference_data_message(element, ref_state):
    return (
        f"{element} has no reference data in {ref_state}. If {element} is an element, use a reference "
        f"state that defines it or define it in an input database (an ELEMENT line and a GHSER function). "
        f"If {element} is a species, declare it under `species` in the phase models."
    )


def initialize_database(phase_models, ref_state, dbf=None, fallback_ref_state="SGTE91"):
    """Return a Database boostraped with elements, species, phases and unary lattice stabilities.

    Parameters
    ----------
    phase_models : Dict[str, Any]
        Dictionary of components and phases to fit.
    ref_state : str
        String of the reference data to use, e.g. 'SGTE91' or 'SR2016'
    dbf : Optional[Database]
        Initial pycalphad Database that can have parameters that would not be fit by ESPEI
    fallback_ref_state : str
        String of the reference data to use for SER data, defaults to 'SGTE91'

    Returns
    -------
    Database
        A new pycalphad Database object, or a modified one if it was given.

    """
    if dbf is None:
        dbf = Database()
    lattice_stabilities = getattr(espei.refdata, ref_state)
    ser_stability = getattr(espei.refdata, ref_state + "Stable")
    aliases = extract_aliases(phase_models)
    phases = sorted({ph.upper() for ph in phase_models["phases"].keys()})
    elements = {el.upper() for el in phase_models["components"]}
    species = PhaseModelSpecification(**phase_models).get_species(dbf)
    dbf.elements.update(elements)
    dbf.species.update(species)

    # Add SER reference data for this element
    for element in dbf.elements:
        if element in dbf.refstates:
            continue  # Do not clobber user reference states
        el_ser_data = _get_ser_data(element, ref_state, fallback_ref_state=fallback_ref_state)
        if len(el_ser_data) == 0:
            raise ValueError(_missing_reference_data_message(element, ref_state))
        # Try to look up the alias that we are using in this fitting
        el_ser_data["phase"] = aliases.get(el_ser_data["phase"], el_ser_data["phase"])
        # Don't warn if the element is a species with no atoms because per-atom
        # formation energies are not possible (e.g. VA (VACUUM) or /- (ELECTRON_GAS))
        if el_ser_data["phase"] not in phases and v.Species(element).number_of_atoms != 0:
            # We have the Gibbs energy expression that we need in the reference
            # data, but this phase is not a candidate in the phase models. The
            # phase won't be added to the database, so looking up the phases's
            # energy won't work.
            _log.warning(
                "The reference phase for %s, %s, is not in the supplied phase models "
                "and won't be added to the Database phases. Fitting formation "
                "energies will not be possible.", element, el_ser_data["phase"]
            )
        dbf.refstates[element] = el_ser_data

    # Add the phases
    species_names = {sp.name for sp in dbf.species}
    for phase_name, phase_data in phase_models['phases'].items():
        if phase_name not in dbf.phases.keys():  # Do not clobber user phases
            # TODO: Need to support model hints for: magnetic, order-disorder, etc.
            site_ratios = phase_data['sublattice_site_ratios']
            subl_model = phase_data['sublattice_model']
            # Only generate the sublattice model for active constituents
            subl_model = [sorted(set(subl).intersection(species_names)) for subl in subl_model]
            if all(len(subl) > 0 for subl in subl_model):
                dbf.add_phase(phase_name, dict(), site_ratios)
                dbf.add_phase_constituents(phase_name, subl_model)

    # Add the GHSER functions to the Database without clobbering user-provided functions
    for element in dbf.elements:
        if element in ("VA", "/-"):
            continue  # vacancies and electrons have no GHSER functions
        # note that `c.upper()*2)[:2]` returns "AL" for "Al" and "BB" for "B"
        # Using this ensures that GHSER functions will be unique, e.g.
        # GHSERC would be an abbreviation for GHSERCA.
        sym_name = "GHSER" + (element.upper()*2)[:2]
        if sym_name not in dbf.symbols:
            if element not in ser_stability:
                raise ValueError(_missing_reference_data_message(element, ref_state))
            dbf.symbols[sym_name] = ser_stability[element]
    return dbf
